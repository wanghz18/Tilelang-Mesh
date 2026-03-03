#include "tvm/ir/expr.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include "../op/region.h"
#include "../op/builtin.h"
#include "../op/utils.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// --- Greedy Scheduler Components ---

bool RegionIntersect(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

class BufferAccessCollector : public ExprVisitor {
public:
  BufferAccessCollector(Map<Var, Buffer> buffer_data_to_buffer)
      : buffer_data_to_buffer_(buffer_data_to_buffer) {}

  Array<BufferRegion> GetReads() const { return reads_; }
  Array<BufferRegion> GetWrites() const { return writes_; }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    auto load_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    auto load_region = BufferRegion(load_buffer, region);
    reads_.push_back(load_region);
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (op->op.same_as(builtin::address_of())) {
      BufferRegion buffer_region;
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        buffer_region = BufferRegion::FullRegion(load->buffer);
      } else if (const auto *var_node = op->args[0].as<VarNode>()) {
        Var data_var = tvm::ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          buffer_region = BufferRegion::FullRegion((*it).second);
        }
      }
      if (buffer_region.defined()) {
        reads_.push_back(buffer_region);
      }
    }
    else if (op->op.same_as(builtin::tvm_access_ptr())) {
      const VarNode *buffer_var = op->args[1].as<VarNode>();
      ICHECK(buffer_var);
      auto it =
      buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(buffer_var)); if (it
      != buffer_data_to_buffer_.end()) {
        const Buffer &buffer = (*it).second;
        const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
        reads_.push_back(buffer_region);
      }
    }
    // else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    //   ICHECK(args[0].as<BufferLoadNode>());
    //   Buffer mbar_buf = args[0].as<BufferLoadNode>()->buffer;
    //   auto buffer_reads =
    //       chain_builder_.mbar_to_buffer_reads_.find(mbar_buf.get());
    //   auto buffer_writes =
    //       chain_builder_.mbar_to_buffer_writes_.find(mbar_buf.get());
    //   if (buffer_reads != chain_builder_.mbar_to_buffer_reads_.end()) {
    //     reads_.insert(reads_.end(), buffer_reads->second.begin(),
    //                   buffer_reads->second.end());
    //   }
    //   if (buffer_writes != chain_builder_.mbar_to_buffer_writes_.end()) {
    //     writes_.insert(
    //         writes_.end(),
    //         chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).begin(),
    //         chain_builder_.mbar_to_buffer_writes_.at(mbar_buf.get()).end());
    //   }
    // }

    else {
      ExprVisitor::VisitExpr_(op);
    }
  }

private:
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

class ASTTraverser: public StmtVisitor {
public:
  ASTTraverser(const PrimFunc &f) {
    for (const auto &[_, buffer] : f->buffer_map) {
      this->buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
  }

  std::pair<Array<BufferRegion>, Array<BufferRegion>> buffer_region_collector(const PrimExpr &expr) {
    auto buf_load_collector = BufferAccessCollector(buffer_data_to_buffer_);
    buf_load_collector(expr);
    Array<BufferRegion> read_regions = buf_load_collector.GetReads();
    Array<BufferRegion> write_regions = buf_load_collector.GetWrites();
    return {read_regions, write_regions};
  }

  void VisitStmt_(const AttrStmtNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) {
    auto [min_read_regions, min_write_regions] =
        buffer_region_collector(op->min);
    for (auto it : min_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : min_write_regions) {
      write_buffer_regions_.insert(it);
    }

    auto [extent_read_regions, extent_write_regions] =
        buffer_region_collector(op->extent);
    for (auto it : extent_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : extent_write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->condition);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssertStmtNode *op) {
    auto [condition_read_regions, condition_write_regions] =
        buffer_region_collector(op->condition);
    for (auto it : condition_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : condition_write_regions) {
      write_buffer_regions_.insert(it);
    }

    auto [message_read_regions, message_write_regions] =
        buffer_region_collector(op->message);
    for (auto it : message_read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : message_write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockRealizeNode *op) {
    auto [read_regions, write_regions] = buffer_region_collector(op->predicate);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) {
    // For a buffer store statement, we need to check the dependencies for the
    // buffer to be stored. For example, in the statement A[i] = B[j] + C[k], we
    // need to check the dependencies for the buffer A.
    Buffer store_buffer = op->buffer;
    Array<PrimExpr> indices = op->indices;
    // convert indices to region
    Array<Range> region;
    for (const auto &index : indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    auto store_region = BufferRegion(store_buffer, region);
    write_buffer_regions_.insert(store_region);

    // For a store statement, we also need to check the read dependencies for
    // the value to be stored. For example, in the statement A[i] = B[j] + C[k],
    // we need to check the read dependencies for the buffers B and C.
    auto [read_regions, write_regions] = buffer_region_collector(op->value);
    for (auto it : read_regions) {
      read_buffer_regions_.insert(it);
    }
    for (auto it : write_regions) {
      write_buffer_regions_.insert(it);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) {
    const CallNode *call = op->value.as<CallNode>();
    if (call->op.same_as(dma_copy())) {
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
      write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
    } else if (call->op.same_as(mma_sunmmio())) {
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
      read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[2]));

      write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[2]));
    // } else if (call->op.same_as(broadcast_())) {
    //   read_buffer_regions_.insert(NormalizeToBufferRegion(call->args[0]));
    //   write_buffer_regions_.insert(NormalizeToBufferRegion(call->args[1]));
    } else {
      auto [read_regions, write_regions] = buffer_region_collector(op->value);
      for (auto it : read_regions) {
        read_buffer_regions_.insert(it);
      }
      for (auto it : write_regions) {
        write_buffer_regions_.insert(it);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    StmtVisitor::VisitStmt_(op);
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
  }

  void clear() {
    read_buffer_regions_.clear();
    write_buffer_regions_.clear();
  }

  void traverse_stmt(Stmt stmt) {
    clear();
    VisitStmt(stmt);
  }

public:
  Map<Var, Buffer> buffer_data_to_buffer_;
  
  std::set<BufferRegion> read_buffer_regions_;
  std::set<BufferRegion> write_buffer_regions_;
};

enum class DeviceType {
  ODMA,
  TensorCore,
  VectorCore,
  Unspecified,
};

class Command {
public:
  int id = -1;
  int iter = -1;
  std::string name = "";
  Stmt stmt;
  std::vector<Command> dependencies;
  DeviceType type = DeviceType::Unspecified;

  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;

  bool finished = false;

public:
  Command(int id, int iter, Stmt stmt)
      : id(id), iter(iter), stmt(stmt),
        name(std::to_string(iter) + "-" + std::to_string(id)) {
    bool set = false;
    if (const auto *block = stmt.as<BlockRealizeNode>()) {
      // set TensorCore for mma_sunmmio
      auto body = block->block->body;
      if (const auto *eval = body.as<EvaluateNode>()) {
        if (const auto *call = eval->value.as<CallNode>()) {
          if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
            type = DeviceType::TensorCore;
            set = true;
          }
        }
      }
      if (!set)
        ICHECK(0) << "Can't identify device type for command " << stmt
                  << ", should be TensorCore.";

    } else if (const auto *eval = stmt.as<EvaluateNode>()) {
      // set ODMA for dma_copy and hlink/vlink, etc
      if (const auto *call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.dma_copy"))) {
          type = DeviceType::ODMA;
          set = true;
        }
      }
      if (!set)
        ICHECK(0) << "Can't identify device type for command " << stmt
                  << ", should be ODMA.";
    } else {
      // set VectorCore for others
      type = DeviceType::VectorCore;
    }
  }

  const float get_delay() {
    if (type == DeviceType::TensorCore)
      return 10;
    else if (type == DeviceType::ODMA)
      return 3;
    else if (type == DeviceType::VectorCore)
      return 2;

    return 1;
  }

  bool blocked(Command cmd) {
    for (auto &incoming_buffer_region : cmd.writes) {
      Buffer incoming_buffer = incoming_buffer_region->buffer;
      Region incoming_region = incoming_buffer_region->region;
      // read after write
      for (auto &read_buffer_region :reads) {
        Buffer read_buffer = read_buffer_region->buffer;
        Region read_region = read_buffer_region->region;
        if (incoming_buffer.same_as(read_buffer) &&
            RegionIntersect(incoming_region, read_region)) {
          return true;
        } 
      }

      // write after write
      for (auto &write_buffer_region :writes) {
        Buffer write_buffer = write_buffer_region->buffer;
        Region write_region = write_buffer_region->region;
        if (incoming_buffer.same_as(write_buffer) &&
            RegionIntersect(incoming_region, write_region)) {
          return true;
        } 
      }
    }
    
    return false; 
  }

  std::vector<Command>
  get_dependent_commands(std::vector<Command> &all_commands) {
    std::vector<Command> dependencies;
    for (auto &cmd : all_commands) {
      if (cmd.name == name || cmd.iter != iter) {
        continue;
      }
      if (!cmd.finished && blocked(cmd)) {
        dependencies.push_back(cmd);
      }
    }
    return dependencies;
  }

  void get_command_reads_writes(std::set<BufferRegion> &read_buffer_regions_,
                                std::set<BufferRegion> &write_buffer_regions_) {
    for (auto &it : read_buffer_regions_) {
      reads.push_back(it);
      LOG(INFO) << name << " read " << it->buffer->name << " " << it->region;
    }
    for (auto &it : write_buffer_regions_) {
      writes.push_back(it);
      LOG(INFO) << name << " write " << it->buffer->name << " " << it->region;
    }
  }

  void set_finished(bool finished) {
    this->finished = finished;
  }

};

class Device {
public:
  DeviceType type;
  bool busy = false;

  Command *current_command = nullptr;
  float command_end_time = std::numeric_limits<float>::max();
  float command_start_time = 0;

public:
  Device(DeviceType type) : type(type) {}

  void assign_command(Command *command, float time) {
    current_command = command;
    busy = true;
    command_end_time = time + command->get_delay();
    command_start_time = time;
  }

  void pass_time(float time) {
    if (busy && time >= command_end_time) {
      current_command->set_finished(true);
      busy = false;
      command_end_time = std::numeric_limits<float>::max();
      command_start_time = 0;
    }
  }
};


class Scheduler {
public:
  std::vector<Command> commands;
  std::vector<Device> devices;

  std::map<std::string, int> b_level;

  Scheduler() {
    devices.push_back(Device(DeviceType::ODMA));
    devices.push_back(Device(DeviceType::TensorCore));
    devices.push_back(Device(DeviceType::VectorCore));
  }

  void AddCommand(Command cmd) { commands.push_back(cmd); }

  float calculate_bottom_level(Command command) {
    if (b_level.find(command.name) != b_level.end()) {
      return b_level[command.name];
    }

    std::vector<Command> dependent_commands;
    for (auto &cmd : commands) {
      if (cmd.name == command.name || cmd.iter != command.iter) {
        continue;
      }
      auto deps = cmd.get_dependent_commands(commands);
      for (auto &dep : deps) {
        if (dep.name == command.name)
          dependent_commands.push_back(cmd);
      }
    }

    float path = 0;

    if (dependent_commands.empty()) {
      path = const_cast<Command &>(command).get_delay();
    } else {
      float max_dependent_level = 0;
      for (auto &dep : dependent_commands) {
        max_dependent_level =
            std::max(max_dependent_level, calculate_bottom_level(dep));
      }
      path = max_dependent_level + const_cast<Command &>(command).get_delay();
    }

    b_level[command.name] = path;
    return path;
  }

  int num_stages_to_iterations(int num_stages) {
    return num_stages;
  }

  std::vector<Command> CriticalPathPipeline() {

    std::vector<Command *> command_queue;
    command_queue.reserve(commands.size());
    for (auto &command : commands) {
      command_queue.push_back(&command);
    }
    float time = 0;
    std::vector<Command> schedule;

    while (!command_queue.empty()) {
      // Try to schedule commands in priority order
      for (auto *command : command_queue) {
        if (command->finished) {
          continue;
        }

        auto deps = command->get_dependent_commands(commands);
        if (deps.empty()) {
          // Find available device of correct type
          for (auto &device : devices) {
            if (device.type == command->type && !device.busy) {
              device.assign_command(command, time);
              LOG(INFO) << command->name << " scheduled at time " << time;
              schedule.push_back(*command);
              break;
            }
          }
        }
      }

      float pass_time = std::numeric_limits<float>::max();
      for (auto &device : devices) {
        pass_time = std::min(pass_time,
                             device.command_end_time - time);
      }
      time += pass_time;

      // Update device states
      for (auto &device : devices)
        device.pass_time(time);

      command_queue.erase(std::remove_if(command_queue.begin(),
                                         command_queue.end(),
                                         [](Command *cmd) {
                                           return cmd->finished;
                                         }),
                          command_queue.end());
    }
    return schedule;
  }
};

class SunmmioPipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f) {
    SunmmioPipelinePlanner substituter(f);
    return substituter.VisitStmt(f->body);
  }

  SunmmioPipelinePlanner(const PrimFunc &f) : traverser(f) {
    traverser.clear();
  }

private:
  Stmt VisitStmt_(const ForNode *loop) final {
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (!num_stages_anno) {
      return StmtExprMutator::VisitStmt_(loop);
    }

    int num_stages = Downcast<IntImm>(num_stages_anno.value())->value;

    Stmt pipeline_body_root{nullptr};
    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        // buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      pipeline_body_root = block->body;
    } else {
      pipeline_body_root = loop->body;
    }

    const SeqStmtNode *pipeline_body_seq = nullptr;
    {
      Stmt current = pipeline_body_root;
      while (true) {
        if (const auto *seq_stmt = current.as<SeqStmtNode>()) {
          pipeline_body_seq = seq_stmt;
          break;
        }
        if (const auto *if_then_else = current.as<IfThenElseNode>()) {
          ICHECK(!if_then_else->else_case.defined())
              << "Pipeline_Planning: Can't handle the body of the loop because "
                 "the IfThenElse node has an else branch";
          current = if_then_else->then_case;
          continue;
        }
        if (const auto *let_stmt = current.as<LetStmtNode>()) {
          current = let_stmt->body;
          continue;
        }
        LOG(FATAL) << "Pipeline_Planning: Can't handle the body of the loop "
                   << "because it is not a SeqStmt, IfThenElse without else, "
                   << "or LetStmt wrapping them, but got "
                   << current->GetTypeKey();
      }
    }
    ICHECK(pipeline_body_seq != nullptr);

    CHECK(num_stages >= 1);
    CHECK(loop->kind == ForKind::kSerial);

    Scheduler scheduler;
    int iterations = scheduler.num_stages_to_iterations(num_stages);
    // LOG(INFO) << loop->loop_var<< loop->min<< loop->extent<< loop->kind<< loop->body<<
    //            loop->thread_binding<< loop->annotations;
    
    // 1. Build Commands and Dependencies
    for (int i = 0; i < iterations; i++) {
      for (size_t j = 0; j < pipeline_body_seq->size(); j++) {
        auto stmt = pipeline_body_seq->seq[j];
        Command cmd(j, i, stmt);
        traverser.traverse_stmt(stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);
        
        scheduler.AddCommand(cmd);
      }
    }

    for (auto &cmd : scheduler.commands) {
      scheduler.calculate_bottom_level(cmd);
    }

    for (auto &it : scheduler.b_level) {
      LOG(INFO) << "Command " << it.first << " bottom level: " << it.second;
    }

    // 2. Do critical path pipeline
    auto result = scheduler.CriticalPathPipeline();

    // 3. Process remaining commands

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, loop->annotations);
  }

private:
  ASTTraverser traverser;
};

tvm::transform::Pass SunmmioPipelinePlanning() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlanner::Substitute(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanning",
                        SunmmioPipelinePlanning);
}

} // namespace tl
} // namespace tvm
