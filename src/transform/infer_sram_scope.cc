/*!
 * \file infer_sram_scope.cc
 * \brief Infer shared memory SRAM scope
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <queue>
#include <unordered_map>

#include "../layout/utils.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "../tileview/tileview.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/attr.h"
#include "common/loop_fusion_utils.h"
#include "common/remap_buffer_rewriter.h"
#include "common/union_find.h"
#include "layout_reducer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ffi/container/array.h"
#include "tvm/ir/expr.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"

namespace tvm {
namespace tl {

using namespace tir;

/** Creates a compact 0-based region that preserves the original extents. */
Array<Range> MakeCompactRegion(const Array<Range> &region) {
  Array<Range> compact_region;
  compact_region.reserve(region.size());
  for (const Range &range : region) {
    compact_region.push_back(Range::FromMinExtent(0, range->extent));
  }
  return compact_region;
}

/** Creates a temporary buffer that keeps the original buffer shape. */
Buffer makeNewBufferWithScope(const Buffer &buffer, std::string scope) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(buffer->data->name_hint, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, {}, buffer->elem_offset,
                buffer->name, buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

/** Creates a compact temporary buffer whose shape matches the region extents.
 */
Buffer makeNewCompactBufferWithScope(const Buffer &buffer,
                                     const Array<Range> &region,
                                     std::string scope) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type = PointerType(ptr_type->element_type, scope);
  Var new_var = Var(buffer->data->name_hint, new_type);
  Array<PrimExpr> shape;
  shape.reserve(region.size());
  for (const Range &range : region) {
    shape.push_back(range->extent);
  }
  return Buffer(new_var, buffer->dtype, shape, {}, Integer(0), buffer->name,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

class InferSramScopePass : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    InferSramScopePass substituter(&analyzer);

    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "InferSramScopePass: Require the target attribute";

    // Sunmmio specified pass
    if (!TargetIsSunmmio(target.value()))
      return f;

    auto *fptr = f.CopyOnWrite();

    // collect remap info when replace_flag = false
    substituter.replace_flag = false;
    fptr->body = substituter.VisitStmt(f->body);

    substituter.InferUnspecifiedBuffer();

    fptr->body =
        RemapBufferRewriter::Substitute(fptr->body, substituter.buffer_remap_);

    // do remap when replace_flag = true
    substituter.replace_flag = true;
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    BlockRealize block_realize =
        Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
    Block block = block_realize->block;
    // remap shared buffers to rsram buffers by default
    if (!replace_flag) {
      Array<Buffer> alloc_buffers = block->alloc_buffers;
      for (auto buffer : alloc_buffers) {
        if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
          buffers_to_infer.insert(buffer);
        } else if ((buffer.scope() != "shared.asram") &&
                   (buffer.scope() != "shared.wsram") &&
                   (buffer.scope() != "shared.rsram")) {
          // sram type has validated in GEMM node
          ICHECK(0) << "Invalid scope " << buffer.scope() << " of " << buffer
                    << " in Sunmmio.";
        }
      }
      return block_realize;
    }

    if (buffer_remap_.empty()) {
      return block_realize;
    }

    // do block attributes remap
    if (block->annotations.count(attr::kLayoutMap)) {
      auto map = block->annotations.Get(attr::kLayoutMap)
                     ->as<Map<Var, Layout>>()
                     .value();
      Map<Var, Layout> new_map;
      for (const auto &[var, layout] : map) {
        if (var_remap_.count(var)) {
          new_map.Set(var_remap_[var], layout);
        } else {
          new_map.Set(var, layout);
        }
      }
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, new_map);
    }

    if (block->annotations.count(attr::kTileViewMap)) {
      auto map = block->annotations.Get(attr::kTileViewMap)
                     ->as<Map<Var, TileView>>()
                     .value();
      Map<Var, TileView> new_map;
      for (const auto &[var, tileView] : map) {
        if (var_remap_.count(var)) {
          new_map.Set(var_remap_[var], tileView);
        } else {
          new_map.Set(var, tileView);
        }
      }
      block.CopyOnWrite()->annotations.Set(attr::kTileViewMap, new_map);
    }

    // do block->alloc_buffers remap
    Array<Buffer> alloc_buffers = block->alloc_buffers;

    for (auto it : buffer_remap_) {
      bool exists = std::find(alloc_buffers.begin(), alloc_buffers.end(),
                              it.first) != alloc_buffers.end();
      if (!exists) {
        alloc_buffers.push_back(it.first);
      }
    }

    // remove the buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });

    if (!alloc_buffers.same_as(block->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }
    block_realize.CopyOnWrite()->block = block;

    return block_realize;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto loop = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (!replace_flag) {
      return loop;
    }
    return loop;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    // collect remap info in gemm nodes
    if (!replace_flag) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.tileop.gemm")) ||
            call->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
          auto aRegion_ = NormalizeToBufferRegion(call->args[0]);
          auto bRegion_ = NormalizeToBufferRegion(call->args[1]);
          auto cRegion_ = NormalizeToBufferRegion(call->args[2]);

          Array<Stmt> seq;
          std::unordered_map<std::string, PrimExpr> new_args;

          auto buffer = aRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              if (buffer_remap_[buffer].scope() != "shared.asram") {
                // insert a copy stmt
                auto transfer_region = MakeCompactRegion(aRegion_->region);
                auto transfer_buffer = makeNewCompactBufferWithScope(
                    buffer, transfer_region, buffer_remap_[buffer].scope());
                PrimExpr src_region =
                    MakeRegionExpr(buffer, aRegion_->region, /*access_mask=*/1);
                PrimExpr dst_region = MakeRegionExpr(
                    transfer_buffer, transfer_region, /*access_mask=*/2);
                Call copy_call = Call(DataType::Handle(), Copy::Get(),
                                      {src_region, dst_region}, {});
                seq.push_back(Evaluate(copy_call));
                buffer_remap_.Set(
                    transfer_buffer,
                    makeBufferWithScope(transfer_buffer, "shared.asram"));

                new_args.insert(
                    {"A", MakeRegionExpr(transfer_buffer, transfer_region, 1)});
              }
            } else {
              auto remap_buffer =
                  makeBufferWithScope(aRegion_->buffer, "shared.asram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.asram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }

          buffer = bRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              if (buffer_remap_[buffer].scope() != "shared.wsram") {
                // insert a copy stmt
                auto transfer_region = MakeCompactRegion(bRegion_->region);
                auto transfer_buffer = makeNewCompactBufferWithScope(
                    buffer, transfer_region, buffer_remap_[buffer].scope());
                PrimExpr src_region =
                    MakeRegionExpr(buffer, bRegion_->region, /*access_mask=*/1);
                PrimExpr dst_region = MakeRegionExpr(
                    transfer_buffer, transfer_region, /*access_mask=*/2);
                Call copy_call = Call(DataType::Handle(), Copy::Get(),
                                      {src_region, dst_region}, {});
                seq.push_back(Evaluate(copy_call));
                buffer_remap_.Set(
                    transfer_buffer,
                    makeBufferWithScope(transfer_buffer, "shared.wsram"));

                new_args.insert(
                    {"B", MakeRegionExpr(transfer_buffer, transfer_region, 1)});
              }
            } else {
              auto remap_buffer =
                  makeBufferWithScope(bRegion_->buffer, "shared.wsram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.wsram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }

          buffer = cRegion_->buffer;
          if ((buffer.scope() == "shared") ||
              (buffer.scope() == "shared.dyn")) {
            if (buffer_remap_.count(buffer)) {
              if (buffer_remap_[buffer].scope() !=
                  "shared.rsram") { // insert a copy stmt
                auto transfer_region = MakeCompactRegion(cRegion_->region);
                auto transfer_buffer = makeNewCompactBufferWithScope(
                    buffer, transfer_region, buffer_remap_[buffer].scope());
                PrimExpr src_region =
                    MakeRegionExpr(buffer, cRegion_->region, /*access_mask=*/1);
                PrimExpr dst_region = MakeRegionExpr(
                    transfer_buffer, transfer_region, /*access_mask=*/2);
                Call copy_call = Call(DataType::Handle(), Copy::Get(),
                                      {src_region, dst_region}, {});
                seq.push_back(Evaluate(copy_call));
                buffer_remap_.Set(
                    transfer_buffer,
                    makeBufferWithScope(transfer_buffer, "shared.rsram"));

                new_args.insert(
                    {"C", MakeRegionExpr(transfer_buffer, transfer_region, 1)});
              }
            } else {
              auto remap_buffer =
                  makeBufferWithScope(cRegion_->buffer, "shared.rsram");
              buffer_remap_.Set(buffer, remap_buffer);
            }
          } else if (buffer.scope() != "shared.rsram") {
            // incorrect specification
            ICHECK(0) << "Specify invalid scope " << buffer.scope() << " of "
                      << buffer << " in GEMM Sunmmio.";
          }

          if (!seq.empty() && !new_args.empty()) {
            auto gemm_call = call;
            ICHECK(gemm_call->args.size() == 19)
                << "GEMM should have 19 parameters.";
            Array<PrimExpr> new_gemm_args;
            auto it = new_args.find("A");
            if (it != new_args.end()) {
              new_gemm_args.push_back(it->second);
            } else {
              new_gemm_args.push_back(gemm_call->args[0]);
            }
            it = new_args.find("B");
            if (it != new_args.end()) {
              new_gemm_args.push_back(it->second);
            } else {
              new_gemm_args.push_back(gemm_call->args[1]);
            }
            it = new_args.find("C");
            if (it != new_args.end()) {
              new_gemm_args.push_back(it->second);
            } else {
              new_gemm_args.push_back(gemm_call->args[2]);
            }
            for (auto i = 3; i < gemm_call->args.size(); i++) {
              new_gemm_args.push_back(gemm_call->args[i]);
            }
            Call new_gemm(DataType::Handle(), call->op, new_gemm_args);
            seq.push_back(Evaluate(new_gemm));
            return SeqStmt::Flatten(seq);
          }
          return tvm::ffi::GetRef<Stmt>(op);
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (!replace_flag) {
      auto buffer = load->buffer;
      if ((buffer.scope() == "shared") || (buffer.scope() == "shared.dyn")) {
        if (buffer_remap_.count(buffer)) {
          ICHECK(buffer_remap_[buffer].scope() == "shared.rsram")
              << "Buffer " << buffer
              << " used in BufferLoad should be shared.rsram.";
        } else {
          buffers_to_infer.insert(buffer);
        }
      } else if ((buffer.scope() != "shared.rsram") &&
                 (buffer.scope() != "global")) {
        ICHECK(0) << "Invalid scope " << buffer.scope() << " of " << buffer
                  << " in Sunmmio.";
      }
      return load;
    }
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      std::string type_key = op->GetTypeKey();
      return BufferLoad(new_buffer, load->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, load->indices);
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (!replace_flag) {
      return store;
    }
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, store->indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], buffer->dtype, buffer->shape,
          buffer->strides, buffer->elem_offset, buffer->name,
          buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!replace_flag) {
      auto tile_op = ParseOperator(tvm::ffi::GetRef<Call>(op));
      if (auto *copy = tile_op.as<CopyNode>()) {
        return tvm::ffi::GetRef<Call>(op);
      }
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    if (!replace_flag) {
      return std::move(var);
    }
    if (var_remap_.count(var)) {
      auto new_var = var_remap_[var];
      return std::move(new_var);
    }
    return std::move(var);
  }

  Buffer makeBufferWithScope(const Buffer &buffer, std::string scope) {
    const auto *ptr_type =
        TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
    Var new_var;
    if (var_remap_.count(buffer->data)) {
      new_var = var_remap_[buffer->data];
    } else {
      Type new_type = PointerType(ptr_type->element_type, scope);
      new_var = Var(buffer->data->name_hint, new_type);
      var_remap_.Set(buffer->data, new_var);
    }
    return Buffer(new_var, buffer->dtype, buffer->shape, {},
                  buffer->elem_offset, buffer->name, buffer->data_alignment,
                  buffer->offset_factor, buffer->buffer_type);
  }

  void InferUnspecifiedBuffer() {
    for (const auto &buffer : buffers_to_infer) {
      if (!buffer_remap_.count(buffer)) {
        auto remap_buffer = makeBufferWithScope(buffer, "shared.rsram");
        buffer_remap_.Set(buffer, remap_buffer);
      }
    }
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Var, Var> var_remap_;

  std::set<Buffer> buffers_to_infer;

  bool replace_flag = false;
};

tvm::transform::Pass InferSramScope() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InferSramScopePass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferSramScope", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InferSramScope", InferSramScope);
}

} // namespace tl
} // namespace tvm
