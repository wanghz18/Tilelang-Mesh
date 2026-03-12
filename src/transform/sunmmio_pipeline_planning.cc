#include "../op/builtin.h"
#include "../op/utils.h"
#include "common/ast_traverser.h"
#include "tvm/ir/expr.h"
#include "tvm/node/cast.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <vector>

#define DEBUG true

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

int name2iter(const std::string &name) {
  return std::stoi(name.substr(0, name.find('-')));
}

int name2id(const std::string &name) {
  return std::stoi(name.substr(name.find('-') + 1));
}

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
      for (auto &read_buffer_region : reads) {
        Buffer read_buffer = read_buffer_region->buffer;
        Region read_region = read_buffer_region->region;
        if (incoming_buffer.same_as(read_buffer) &&
            RegionIntersect(incoming_region, read_region)) {
          return true;
        }
      }

      // write after write
      for (auto &write_buffer_region : writes) {
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
      if (cmd.iter != iter) {
        continue;
      }
      if (cmd.name == name) {
        break;
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
    }
    for (auto &it : write_buffer_regions_) {
      writes.push_back(it);
    }
  }

  void set_finished(bool finished) { this->finished = finished; }

  bool operator==(const Command &cmd) const { return name == cmd.name; }
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

  float calculate_bottom_level(Command command) {
    // for Memoization
    std::string key = command.name;
    if (b_level.find(key) != b_level.end()) {
      return b_level[key];
    }

    std::vector<Command> dependent_commands;
    for (auto &cmd : commands) {
      if (cmd.name == key || cmd.iter != command.iter) {
        continue;
      }
      auto deps = cmd.get_dependent_commands(commands);
      for (auto &dep : deps) {
        if (dep.name == key)
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

    b_level[key] = path;
    return path;
  }

  int num_stages_to_iterations(int num_stages) { return num_stages; }

  std::vector<Command> CriticalPathPipeline(std::string log_file_name) {

    std::vector<Command *> command_queue;
    command_queue.reserve(commands.size());
    for (auto &command : commands) {
      command_queue.push_back(&command);
    }
    sort(command_queue.begin(), command_queue.end(),
         [this](Command *a, Command *b) {
           auto a_key = "0-" + std::to_string(name2id(a->name));
           auto b_key = "0-" + std::to_string(name2id(b->name));
           if (b_level[a_key] != b_level[b_key]) {
             return b_level[a_key] > b_level[b_key];
           }
           if (name2iter(a->name) != name2iter(b->name)) {
             return name2iter(a->name) < name2iter(b->name);
           }
           return name2id(a->name) < name2id(b->name);
         });
    float time = 0;
    std::vector<Command> schedule;
    if (DEBUG) {
      LOG(INFO) << "Scheduling starts.";
    }

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
              if (DEBUG) {
                LOG(INFO) << "Command " << command->name
                          << " assigned to device " << int(device.type)
                          << " at time " << time << " with delay "
                          << command->get_delay();
                std::ofstream log_file(log_file_name, std::ios::app);
                if (log_file.is_open()) {
                  log_file << command->name << " " << int(device.type) << " "
                           << time << " " << command->get_delay() << '\n';
                }
              }
              schedule.push_back(*command);
              break;
            }
          }
        }
      }

      float pass_time = std::numeric_limits<float>::max();
      for (auto &device : devices) {
        pass_time = std::min(pass_time, device.command_end_time - time);
      }
      time += pass_time;

      // Update device states
      for (auto &device : devices)
        device.pass_time(time);

      command_queue.erase(
          std::remove_if(command_queue.begin(), command_queue.end(),
                         [](Command *cmd) { return cmd->finished; }),
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

    CHECK(loop->kind == ForKind::kSerial);

    // 1.1 Build Iter0 Commands
    Scheduler scheduler;
    int iterations = scheduler.num_stages_to_iterations(num_stages);
    std::set<Buffer> used_buffers;
    std::vector<Command> iter0_commands;
    for (size_t j = 0; j < pipeline_body_seq->size(); j++) {
      auto stmt = pipeline_body_seq->seq[j];
      Command cmd(j, 0, stmt);
      traverser.traverse_stmt(stmt);
      cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                   traverser.write_buffer_regions_);

      for (auto &read : traverser.read_buffer_regions_) {
        if (read->buffer.scope() == "global") {
          continue;
        }
        used_buffers.insert(read->buffer);
      }
      for (auto &write : traverser.write_buffer_regions_) {
        if (write->buffer.scope() == "global") {
          continue;
        }
        used_buffers.insert(write->buffer);
      }
      iter0_commands.push_back(cmd);
    }
    if (DEBUG) {
      for (auto &it : iter0_commands) {
        LOG(INFO) << it.name;
      }
    }

    // 1.2 Build Prologue Commands with Iter0 Commands
    std::vector<Command> prologue_commands;
    Array<String> prologue_orders;
    std::vector<int> prologue_ids;
    for (auto &cmd : iter0_commands) {
      // only ODMA operations
      if (cmd.type == DeviceType::ODMA) {
        auto deps = cmd.get_dependent_commands(iter0_commands);
        if (deps.empty()) {
          prologue_commands.push_back(cmd);
          prologue_orders.push_back(cmd.name);
          prologue_ids.push_back(name2id(cmd.name));
        }
      }
    }
    if (DEBUG) {
      LOG(INFO) << "Prologue Commands:";
      for (auto &cmd : prologue_commands) {
        LOG(INFO) << cmd.name;
      }
    }
    if (DEBUG) {
      std::ofstream log_file("prologue.log", std::ios::app);
      if (log_file.is_open()) {
        int time = 0;
        for (auto &command : prologue_commands) {
          log_file << command.name << " " << int(command.type) << " " << time
                   << " " << command.get_delay() << '\n';
          time += command.get_delay();
        }
      }
    }

    // 1.3 Build Body Commands
    for (int i = 0; i < iterations; i++) {
      for (size_t j = 0; j < pipeline_body_seq->size(); j++) {
        if (std::find(prologue_ids.begin(), prologue_ids.end(), j) !=
            prologue_ids.end()) {
          continue;
        }
        auto stmt = pipeline_body_seq->seq[j];
        Command cmd(j, i, stmt);
        traverser.traverse_stmt(stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);

        scheduler.commands.push_back(cmd);
      }

      for (auto &command : prologue_commands) {
        Command cmd(name2id(command.name), i + 1, command.stmt);
        traverser.traverse_stmt(cmd.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);

        scheduler.commands.push_back(cmd);
      }
    }

    if (DEBUG) {
      LOG(INFO) << "Body Commands:";
      for (auto &cmd : scheduler.commands) {
        LOG(INFO) << cmd.name;
      }

      ICHECK(scheduler.commands.size() ==
             pipeline_body_seq->size() * iterations);
    }

    // 1.4 Build Epilogue Commands
    Scheduler epilogue_scheduler;
    PrimExpr epilogue_iterations_expr = floormod(loop->extent, iterations);
    int epilogue_iterations = -1;
    if (const auto *mod_int = epilogue_iterations_expr.as<IntImmNode>()) {
      epilogue_iterations = mod_int->value;
    }
    ICHECK(epilogue_iterations != -1)
        << "Can't calculate the epilogue iterations.";

    if (epilogue_iterations == 0) {
      epilogue_iterations = iterations;
    }
    // 1.4.1 Epilogue Loop
    int epilogue_iter = 0;
    for (int i = 0; i < epilogue_iterations - 1; i++) {
      for (size_t j = 0; j < pipeline_body_seq->size(); j++) {
        auto command = scheduler.commands[i * pipeline_body_seq->size() + j];
        int iter = name2iter(command.name);
        int id = name2id(command.name);
        Command cmd(id, iter, command.stmt);
        traverser.traverse_stmt(command.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);
        epilogue_scheduler.commands.push_back(cmd);
        LOG(INFO) << "epilogue add " << cmd.name;
      }
      epilogue_iter += 1;
    }

    // 1.4.2 The End of Epilogue
    for (auto &command : iter0_commands) {
      LOG(INFO) << "iter0 " << command.name;
      if (std::find(prologue_orders.begin(), prologue_orders.end(),
                    command.name) == prologue_orders.end()) {
        int iter = name2iter(command.name);
        int id = name2id(command.name);
        Command cmd(id, epilogue_iter, command.stmt);
        traverser.traverse_stmt(command.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);
        epilogue_scheduler.commands.push_back(cmd);
        LOG(INFO) << "epilogue add " << cmd.name;
      }
    }
    if (DEBUG) {
      LOG(INFO) << "Epilogue Commands:";
      for (auto &it : epilogue_scheduler.commands)
        LOG(INFO) << it.name;
    }

    for (auto &cmd : scheduler.commands) {
      scheduler.calculate_bottom_level(cmd);
    }
    if (DEBUG) {
      LOG(INFO) << "Body Bottom Level:";
      for (auto &it : scheduler.b_level) {
        LOG(INFO) << it.first << ' ' << it.second;
      }
    }

    for (auto &cmd : epilogue_scheduler.commands) {
      epilogue_scheduler.calculate_bottom_level(cmd);
    }
    if (DEBUG) {
      LOG(INFO) << "Epilogue Bottom Level:";
      for (auto &it : epilogue_scheduler.b_level) {
        LOG(INFO) << it.first << ' ' << it.second;
      }
    }

    // 2. Do critical path pipeline
    auto result = scheduler.CriticalPathPipeline("body.log");
    std::vector<Command> epilogue_result =
        epilogue_scheduler.CriticalPathPipeline("epilogue.log");

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages" && key != "used_buffers") {
        annotations.Set(key, value);
      }
    }

    annotations.Set("iterations", iterations);
    annotations.Set("prologue_orders", prologue_orders);
    Array<String> orders;
    for (auto &it : result) {
      orders.push_back(it.name);
    }
    annotations.Set("body_orders", orders);
    orders.clear();
    for (auto &it : epilogue_result) {
      orders.push_back(it.name);
    }
    annotations.Set("epilogue_orders", orders);

    Array<Buffer> used_buffers_array(used_buffers.begin(), used_buffers.end());
    annotations.Set("used_buffers", used_buffers_array);

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, annotations);
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
