import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.engine.phase import *
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tilelang.language.mesh_tensor import MeshShardingPolicy
from tilelang.tileview import make_tileview


def strip_layout_map_for_comparison(mod):
    keys_to_remove = {"layout_map", "global_layout_map", "tileview_map"}

    def _filter_annotations(annotations):
        if annotations is None:
            return annotations
        return {k: v for k, v in annotations.items() if k not in keys_to_remove}

    def _postorder(node):
        if isinstance(node, tvm.tir.AttrStmt) and node.attr_key in keys_to_remove:
            return node.body
        if isinstance(node, tvm.tir.For):
            return tvm.tir.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                _filter_annotations(node.annotations),
            )
        if isinstance(node, tvm.tir.Block):
            return tvm.tir.Block(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                _filter_annotations(node.annotations),
            )
        if isinstance(node, tvm.tir.BlockRealize):
            b = node.block
            new_block = tvm.tir.Block(
                b.iter_vars,
                b.reads,
                b.writes,
                b.name_hint,
                b.body,
                b.init,
                b.alloc_buffers,
                b.match_buffers,
                _filter_annotations(b.annotations),
            )
            return tvm.tir.BlockRealize(node.iter_values, node.predicate, new_block)
        return node

    f = mod["main"]
    new_body = tvm.tir.stmt_functor.ir_transform(f.body, None, _postorder, None)
    mod["main"] = f.with_body(new_body)
    return mod


def TestLowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    mod = tir.transform.BindTarget(target)(mod)
    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Infer shared memory SRAM scope
    mod = tilelang.transform.InferSramScope()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)
    # Infer memory layouts for fragments and shared memory
    # mod = tilelang.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    # read TileView metadata and attach them to Tiles loops.
    mod = tilelang.transform.LegalizeTilesLoop()(mod)
    # trans the tiles loop into two layers of loops, the inner one is a vectorized loop and the outer one is a Serial loop
    mod = tilelang.transform.TilesLoop()(mod)
    # Lower l2 persistent map
    mod = tilelang.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)
    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


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
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})

            T.clear(C_shared)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


def matmul_unroll_3(M, N, K, block_M, block_N, block_K, num_stages, dtype="float16", accum_dtype="float"):
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
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})

            T.clear(C_shared)
            T.copy(A[by * block_M : (by + 1) * block_M, 0 * block_K : 1 * block_K], A_shared[0, :, :])
            T.copy(B[0 * block_K : 1 * block_K, bx * block_N : (bx + 1) * block_N], B_shared[0, :, :])
            for k in T.Pipelined(T.floordiv(32, 3)):
                T.gemm(A_shared[0, :, :], B_shared[0, :, :], C_shared)
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 1) * block_K : (k * num_stages + 2) * block_K], A_shared[1, :, :]
                )
                T.copy(
                    B[(k * num_stages + 1) * block_K : (k * num_stages + 2) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[1, :, :]
                )
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 2) * block_K : (k * num_stages + 3) * block_K], A_shared[2, :, :]
                )
                T.copy(
                    B[(k * num_stages + 2) * block_K : (k * num_stages + 3) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[2, :, :]
                )
                T.gemm(A_shared[1, :, :], B_shared[1, :, :], C_shared)
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 3) * block_K : (k * num_stages + 4) * block_K], A_shared[0, :, :]
                )
                T.copy(
                    B[(k * num_stages + 3) * block_K : (k * num_stages + 4) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[0, :, :]
                )
                T.gemm(A_shared[2, :, :], B_shared[2, :, :], C_shared)

            T.gemm(A_shared[0, :, :], B_shared[0, :, :], C_shared)
            T.copy(
                A[by * block_M : (by + 1) * block_M, (10 * num_stages + 1) * block_K : (10 * num_stages + 2) * block_K], A_shared[1, :, :]
            )
            T.copy(
                B[(10 * num_stages + 1) * block_K : (10 * num_stages + 2) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[1, :, :]
            )
            T.gemm(A_shared[1, :, :], B_shared[1, :, :], C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


def matmul_unroll_4(M, N, K, block_M, block_N, block_K, num_stages, dtype="float16", accum_dtype="float"):
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
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, index_map)})
            T.annotate_tileview({B_shared: make_tileview(B_shared, tile_size, index_map)})
            T.annotate_tileview({C_shared: make_tileview(C_shared, tile_size, index_map)})

            T.clear(C_shared)
            T.copy(A[by * block_M : (by + 1) * block_M, 0 * block_K : 1 * block_K], A_shared[0, :, :])
            T.copy(B[0 * block_K : 1 * block_K, bx * block_N : (bx + 1) * block_N], B_shared[0, :, :])
            for k in T.Pipelined(T.floordiv(32, 4) - 1):
                T.gemm(A_shared[0, :, :], B_shared[0, :, :], C_shared)
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 1) * block_K : (k * num_stages + 2) * block_K], A_shared[1, :, :]
                )
                T.copy(
                    B[(k * num_stages + 1) * block_K : (k * num_stages + 2) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[1, :, :]
                )
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 2) * block_K : (k * num_stages + 3) * block_K], A_shared[2, :, :]
                )
                T.copy(
                    B[(k * num_stages + 2) * block_K : (k * num_stages + 3) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[2, :, :]
                )
                T.gemm(A_shared[1, :, :], B_shared[1, :, :], C_shared)
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 3) * block_K : (k * num_stages + 4) * block_K], A_shared[3, :, :]
                )
                T.copy(
                    B[(k * num_stages + 3) * block_K : (k * num_stages + 4) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[3, :, :]
                )
                T.copy(
                    A[by * block_M : (by + 1) * block_M, (k * num_stages + 4) * block_K : (k * num_stages + 5) * block_K], A_shared[0, :, :]
                )
                T.gemm(A_shared[2, :, :], B_shared[2, :, :], C_shared)
                T.copy(
                    B[(k * num_stages + 4) * block_K : (k * num_stages + 5) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[0, :, :]
                )
                T.gemm(A_shared[3, :, :], B_shared[3, :, :], C_shared)

            T.gemm(A_shared[0, :, :], B_shared[0, :, :], C_shared)
            T.copy(A[by * block_M : (by + 1) * block_M, (7 * num_stages + 1) * block_K : (7 * num_stages + 2) * block_K], A_shared[1, :, :])
            T.copy(B[(7 * num_stages + 1) * block_K : (7 * num_stages + 2) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[1, :, :])
            T.copy(A[by * block_M : (by + 1) * block_M, (7 * num_stages + 2) * block_K : (7 * num_stages + 3) * block_K], A_shared[2, :, :])
            T.copy(B[(7 * num_stages + 2) * block_K : (7 * num_stages + 3) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[2, :, :])
            T.gemm(A_shared[1, :, :], B_shared[1, :, :], C_shared)
            T.copy(A[by * block_M : (by + 1) * block_M, (7 * num_stages + 3) * block_K : (7 * num_stages + 4) * block_K], A_shared[3, :, :])
            T.copy(B[(7 * num_stages + 3) * block_K : (7 * num_stages + 4) * block_K, bx * block_N : (bx + 1) * block_N], B_shared[3, :, :])
            T.gemm(A_shared[2, :, :], B_shared[2, :, :], C_shared)
            T.gemm(A_shared[3, :, :], B_shared[3, :, :], C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


def flashattn_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = "float"
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

            # T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            # T.copy(Q_shared, Q_local)
            # for i, j in T.Parallel(block_M, dim):
            #     Q_local[i, j] *= scale
            loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
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
                # T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                # fake operations, retaining read & write buffers
                for i in T.Parallel(block_M):
                    scores_max[i] = acc_s[i, 0]

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
                # T.reduce_sum(acc_s, scores_sum, dim=1)
                # fake operations, retaining read & write buffers
                for i in T.Parallel(block_M):
                    scores_sum[i] = acc_s[i, 0]

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


def flashattn_fwd_unroll_3(batch, heads, seq_len, dim, is_causal, block_M, block_N, num_stages):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    dtype = "float"
    accum_dtype = "float"
    assert is_causal is False
    assert num_stages == 3
    assert seq_len == 1024
    assert block_N == 32

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
            K_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            V_shared = T.alloc_shared([num_stages, block_N, dim], dtype)
            acc_s = T.alloc_shared([num_stages, block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([num_stages, block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([num_stages, block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            T.copy(K[bz, by, 0 * block_N : (0 + 1) * block_N, :], K_shared[0, :, :])
            if not is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[0, i, j] = T.if_then_else(0 * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
            T.copy(V[bz, by, 0 * block_N : (0 + 1) * block_N, :], V_shared[0, :, :])
            trip_count = 10
            for k in T.serial(trip_count):
                T.gemm(Q_shared, K_shared[0, :, :], acc_s[0, :, :], transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_max, scores_max_prev)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[1, i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[2, i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.copy(K[bz, by, k * 96 + 32 : k * 96 + 64, :], K_shared[1, :, :])
                T.copy(K[bz, by, k * 96 + 64 : k * 96 + 96, :], K_shared[2, :, :])
                T.copy(V[bz, by, k * 96 + 32 : k * 96 + 64, :], V_shared[1, :, :])

                for i in T.Parallel(block_M):
                    scores_max[i] = acc_s[0, i, 0]

                T.gemm(Q_shared, K_shared[1, :, :], acc_s[1, :, :], transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                T.copy(V[bz, by, k * 96 + 64 : k * 96 + 96, :], V_shared[2, :, :])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[0, i, j] = T.exp2(acc_s[0, i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s[0, :, :], acc_s_cast[0, :, :])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(scores_max, scores_max_prev)

                T.gemm(acc_s_cast[0, :, :], V_shared[0, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_M):
                    scores_sum[0, i] = acc_s[0, i, 0]
                for i in T.Parallel(block_M):
                    scores_max[i] = acc_s[1, i, 0]

                T.copy(K[bz, by, k * 96 + 96 : k * 96 + 128, :], K_shared[0, :, :])

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                T.copy(V[bz, by, k * 96 + 96 : k * 96 + 128, :], V_shared[0, :, :])

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[0, i]

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[1, i, j] = T.exp2(acc_s[1, i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s[1, :, :], acc_s_cast[1, :, :])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                T.gemm(Q_shared, K_shared[2, :, :], acc_s[2, :, :], transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.copy(scores_max, scores_max_prev)

                for i in T.Parallel(block_M):
                    scores_sum[1, i] = acc_s[1, i, 0]
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[1, i]

                T.gemm(acc_s_cast[1, :, :], V_shared[1, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_M):
                    scores_max[i] = acc_s[2, i, 0]
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[2, i, j] = T.exp2(acc_s[2, i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s[2, :, :], acc_s_cast[2, :, :])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i in T.Parallel(block_M):
                    scores_sum[2, i] = acc_s[2, i, 0]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast[2, :, :], V_shared[2, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[2, i]

            T.gemm(Q_shared, K_shared[0, :, :], acc_s[0, :, :], transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(scores_max, scores_max_prev)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[1, i, j] = T.if_then_else(num_stages * trip_count * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
            T.copy(K[bz, by, num_stages * trip_count * block_N + 32 : num_stages * trip_count * block_N + 64, :], K_shared[1, :, :])
            T.copy(V[bz, by, num_stages * trip_count * block_N + 32 : num_stages * trip_count * block_N + 64, :], V_shared[1, :, :])
            for i in T.Parallel(block_M):
                scores_max[i] = acc_s[0, i, 0]
            T.gemm(Q_shared, K_shared[1, :, :], acc_s[1, :, :], transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i, j in T.Parallel(block_M, block_N):
                acc_s[0, i, j] = T.exp2(acc_s[0, i, j] * scale - scores_max[i] * scale)
            T.copy(acc_s[0, :, :], acc_s_cast[0, :, :])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            T.copy(scores_max, scores_max_prev)
            T.gemm(acc_s_cast[0, :, :], V_shared[0, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(block_M):
                scores_sum[0, i] = acc_s[0, i, 0]
            for i in T.Parallel(block_M):
                scores_max[i] = acc_s[1, i, 0]
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[0, i]
            for i, j in T.Parallel(block_M, block_N):
                acc_s[1, i, j] = T.exp2(acc_s[1, i, j] * scale - scores_max[i] * scale)
            T.copy(acc_s[1, :, :], acc_s_cast[1, :, :])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            T.gemm(acc_s_cast[1, :, :], V_shared[1, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(block_M):
                scores_sum[1, i] = acc_s[1, i, 0]
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[1, i]

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flash_fwd


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H, num_split, num_stages):
    scale = (1.0 / (dim + pe_dim)) ** 0.5 * 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"
    # sm_num = driver.get_num_sms()
    sm_num = 108

    @T.prim_func
    def main_split_persistent(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(sm_num, threads=256) as (block_id):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            # O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_shared([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_H, block_N], dtype)
            acc_o = T.alloc_shared([block_H, dim], accum_dtype)
            scores_max = T.alloc_shared([block_H], accum_dtype)
            scores_max_prev = T.alloc_shared([block_H], accum_dtype)
            scores_scale = T.alloc_shared([block_H], accum_dtype)
            scores_sum = T.alloc_shared([block_H], accum_dtype)
            logsum = T.alloc_shared([block_H], accum_dtype)
            po_local = T.alloc_shared([dim], dtype)
            o_accum_local = T.alloc_shared([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.use_swizzle(10)

            total_tiles = batch * (heads // min(block_H, kv_group_num)) * num_split
            waves = T.ceildiv(total_tiles, sm_num)
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bid = tile_id // ((heads // min(block_H, kv_group_num)) * num_split)
                hid = tile_id // num_split % (heads // min(block_H, kv_group_num))
                sid = tile_id % num_split
                cur_kv_head = hid // (kv_group_num // block_H)

                if bid < batch and hid * VALID_BLOCK_H < heads and sid < num_split:
                    T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
                    T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        kv_start = (seqlen_kv // num_split) * sid + k * block_N
                        kv_end = (seqlen_kv // num_split) * sid + (k + 1) * block_N
                        T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                        T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                        T.clear(acc_s)
                        T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                        T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        # T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_H):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        # T.reduce_sum(acc_s, scores_sum, dim=1)
                        T.copy(acc_s, S_shared)
                        T.copy(S_shared, acc_s_cast)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(logsum, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, sid])
                    # T.copy(acc_o, O_shared)
                    T.copy(acc_o, Output_partial[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, sid, :])

            T.sync_grid()
            waves = T.ceildiv(heads * batch, sm_num)
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                hid = tile_id // batch
                bid = tile_id % batch
                if bid < batch and hid < heads:
                    T.clear(lse_logsum_local)
                    T.clear(o_accum_local)
                    lse_max_local = -T.infinity(accum_dtype)
                    for k in T.serial(num_split):
                        lse_max_local = T.max(lse_max_local, glse[bid, hid, k])
                    for k in T.Pipelined(num_split, num_stages=1):
                        lse_local_split = glse[bid, hid, k]
                        lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
                    lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
                    for k in T.serial(num_split):
                        for i in T.Parallel(dim):
                            po_local[i] = Output_partial[bid, hid, k, i]
                        lse_local_split = glse[bid, hid, k]
                        scale_local = T.exp2(lse_local_split - lse_logsum_local)
                        for i in T.Parallel(dim):
                            o_accum_local[i] += po_local[i] * scale_local
                    for i in T.Parallel(dim):
                        Output[bid, hid, i] = o_accum_local[i]

    return main_split_persistent


CASES = [
    matmul(1024, 1024, 1024, 32, 32, 32, 3),
    #  matmul_unroll_3(1024, 1024, 1024, 32, 32, 32, 3)),
    matmul(1024, 1024, 1024, 32, 32, 32, 4),
    # matmul_unroll_4(1024, 1024, 1024, 32, 32, 32, 4)),
    flashattn_fwd(batch=2, heads=16, seq_len=1024, dim=128, is_causal=False, block_M=32, block_N=32, num_stages=3),
    # flashattn(batch=2, heads=16, kv_head_num=1, seqlen_kv=1024, dim=128, pe_dim=0, block_N=128, block_H=4, num_split=1, num_stages=3),
    # (
    #     flashattn_fwd(batch=2, heads=16, seq_len=1024, dim=128, is_causal=False, block_M=32, block_N=32, num_stages=3),
    #     flashattn_fwd_unroll_3(batch=2, heads=16, seq_len=1024, dim=128, is_causal=False, block_M=32, block_N=32, num_stages=3),
    # ),
    # flashattn(batch=1, heads=32, kv_head_num=1, seqlen_kv=2048, dim=128, pe_dim=0, block_N=128, block_H=4, num_split=1),
    # flashattn(batch=1, heads=32, kv_head_num=1, seqlen_kv=2048, dim=128, pe_dim=0, block_N=128, block_H=4, num_split=1),
]


@pytest.mark.parametrize(
    # "kernel, ref",
    "kernel",
    CASES,
)
def test_tilelang_transform_sunmmio_pipeline(kernel):
    mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    name = SUNMMIO_TARGET_DESC
    target = tvm.target.Target(name)

    with target:
        # mod = TestLowerAndLegalize(mod, target)
        mod = LowerAndLegalize(mod, target)

        if name == SUNMMIO_TARGET_DESC:
            mod = tl.transform.SunmmioPipelinePlanning(debug=False)(mod)
        elif name == "cuda":
            mod = tl.transform.PipelinePlanning()(mod)

        if name == SUNMMIO_TARGET_DESC:
            mod = tl.transform.InjectSunmmioPipeline()(mod)
        elif name == "cuda":
            mod = tl.transform.InjectSoftwarePipeline()(mod)

        mod = tl.transform.Simplify()(mod)
        # mod = strip_layout_map_for_comparison(mod)
        mod.show()
