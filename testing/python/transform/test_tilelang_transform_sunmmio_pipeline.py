import tilelang
import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.tileview import make_tileview
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tilelang.language.mesh_tensor import MeshShardingPolicy


def matmul(M, N, K, block_M, block_N, block_K, num_stages, dtype="float16", accum_dtype="float"):
    tile_size = (8, 8)
    index_map = (-2, -1)

    @T.prim_func
    def gemm(
        A: T.MeshTensor(
            (M, K),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=dtype,
        ),
        B: T.MeshTensor(
            (K, N),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=dtype,
        ),
        C: T.MeshTensor(
            (M, N),
            sharding_policy=MeshShardingPolicy(cross_mesh_dim=0),
            device_mesh_config=(2, 2),
            hierarchical_dims=(4, 32, 128),
            hierarchical_groups=((0, 2), (2, 3)),
            hierarchical_strides=(32, 1, 4096),
            dtype=accum_dtype,
        ),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            # , scope="shared.asram")
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            # , scope="shared.wsram")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            # , scope="shared.rsram")
            # C_shared = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})

            T.clear(C_shared)
            for k in T.Pipelined(1, T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_fwd(
        Q: T.Tensor(shape, dtype),  # type: ignore
        K: T.Tensor(shape, dtype),  # type: ignore
        V: T.Tensor(shape, dtype),  # type: ignore
        Output: T.Tensor(shape, dtype),  # type: ignore
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            # Q_local = T.alloc_fragment([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            # T.copy(Q_shared, Q_local)
            # for i, j in T.Parallel(block_M, dim):
            #     Q_local[i, j] *= scale
            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


CASES = [
    # matmul(1024, 1024, 1024, 32, 32, 32, 3),
    # matmul(1024, 1024, 1024, 32, 32, 32, 4),
    matmul(1024, 1024, 1024, 32, 32, 32, 5),
    # matmul(1024, 1024, 1024, 64, 64, 64, 3),
    # flashattn_fwd(batch=2, heads=16, seq_len=1024, dim=128, is_causal=False, block_M=32, block_N=32),
]


@pytest.mark.parametrize(
    "kernel",
    CASES,
)
def test_tilelang_transform_sunmmio_pipeline(kernel):
    mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    name = SUNMMIO_TARGET_DESC
    # name = "cuda"
    target = tvm.target.Target(name)

    with target:
        mod = LowerAndLegalize(mod, target)
        if name == SUNMMIO_TARGET_DESC:
            mod = tl.transform.SunmmioPipelinePlanning()(mod)
        elif name == "cuda":
            mod = tl.transform.PipelinePlanning()(mod)

        # mod.show()
        if name == SUNMMIO_TARGET_DESC:
            mod = tl.transform.InjectSunmmioPipeline()(mod)
        elif name == "cuda":
            mod = tl.transform.InjectSoftwarePipeline()(mod)

        # mod.show()
