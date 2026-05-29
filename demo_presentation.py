import argparse
from pathlib import Path
from tilelang import tvm as tvm
import tilelang
import tilelang.language as T
from tilelang.layout import make_zz_layout
from tilelang.carver.arch import driver
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from compile_pipeline import compile_test




def matmul_persistent(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.bfloat16, accum_dtype=T.float32):
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), T.MeshShardingPolicy(y=0, x=1), device_mesh_config, accum_dtype, layout=C_layout),
    ):
        with T.Kernel(ncores) as (_cid):
            sharded_M, sharded_K = A.shape
            _, sharded_N = B.shape

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_shared_dist = T.alloc_shared((block_M, block_K * ncols), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            B_shared_dist = T.alloc_shared((block_K * nrows, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            # Each core iterates its own sharded tile grid with plain nested
            # loops. The MeshTensor sharding already distributes A/B/C across
            # the core mesh, so no persistent core-distribution loop is used.
            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        # Stage each A/B tile into a shared buffer, then
                        # all-gather it across the core row / column.
                        # all_gather needs a Buffer source (not an indexed
                        # element) and an explicit concat axis.
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.comm.all_gather(A_shared, A_shared_dist, direction="horizontal", axis=-1)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.comm.all_gather(B_shared, B_shared_dist, direction="vertical", axis=0)
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main

def flashattn(batch, heads, seq_len, dim, groups=1, block_M=64, block_N=64, num_stages=0):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    shard_policy = T.MeshShardingPolicy(y=0, x=2)
    device_mesh_config = driver.get_sunmmio_device_mesh_config()
    nrows, ncols = device_mesh_config
    ncores = nrows * ncols

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, shard_policy, device_mesh_config, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, shard_policy, device_mesh_config, dtype, layout=O_layout),
    ):
        with T.Kernel(ncores) as _cid:
            # Get sharded logical shapes
            sharded_batch = Q.shape[0]
            sharded_heads = Q.shape[2]

            # Declare shared memory buffers
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            # RSRAM-resident dtype-cast staging for the PV gemm input. A
            # Sunmmio DMA copy cannot change dtype, so the fp32->dtype cast
            # is done here by the Tile unit (in RSRAM) and only then DMA'd
            # into acc_s_cast (ASRAM, the PV gemm's A operand).
            acc_s_cast_local = T.alloc_shared([block_M, block_N], dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            # Each core iterates its own sharded work domain with plain
            # nested loops. The MeshTensor sharding already distributes data
            # (batch / heads) across the core mesh, so no persistent
            # core-distribution loop is needed here.
            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(T.ceildiv(seq_len, block_M)):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))

                        # loop over K/V blocks
                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            
                            # Cast fp32 probabilities -> dtype with the Tile
                            # unit in RSRAM, then DMA the same-dtype result
                            # into ASRAM.
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s_cast_local[i, j] = acc_s[i, j]
                            T.copy(acc_s_cast_local, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, choices=["gemm", "flashattn"], default="flashattn")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    target_name = SUNMMIO_TARGET_DESC
    target = tvm.target.Target(target_name)

    with tvm.target.Target(target):
        if args.kernel == "gemm":
            print("Loading GEMM kernel...")
            func = matmul_persistent(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, num_stages=3)
        else:
            print("Loading FlashAttention kernel...")
            func = flashattn(batch=2, heads=8, seq_len=2048, dim=128, groups=2, block_M=64, block_N=64, num_stages=3)

        if args.dump:
            print(f"Compiling...")
            
            log_dir = str(Path(f"pass_logs_{args.kernel}"))
            host_mod, device_mod = compile_test(
                func=func,
                target=target,
                log_pass_output=True,
                log_dir=log_dir,
                log_passes=None,
                split_pass_log=True,
            )
            
            print(f"Logs saved to: {log_dir}")

