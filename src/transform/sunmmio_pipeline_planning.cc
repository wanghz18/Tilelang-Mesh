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
#include <limits>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

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

enum class DeviceType {
  ODMA,
  TensorCore,
  VectorCore,
  Unspecified,
};

enum class Role : uint8_t { kConsumer, kProducer, kBoth, kUndefined };

class SunmmioRoleMarker : public StmtVisitor {
public:
  SunmmioRoleMarker(ASTTraverser &traverser) : traverser_(traverser) {
    traverser.clear();
  }

  Role GetRole(const StmtNode *stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end())
        << " Cannot find role for stmt: " << stmt->GetTypeKey();
    return it->second;
  }

  Role GetRole(const Stmt &stmt) const { return GetRole(stmt.get()); }

  void VisitStmt_(const EvaluateNode *op) final {
    Role role = Role::kConsumer;
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        if (IsGlobalBuffer(src_region->buffer)) {
          role = Role::kProducer;
        }
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    traverser_.traverse_stmt(ffi::GetRef<Stmt>(op));
    Role role = Role::kProducer;
    auto reads = traverser_.read_buffer_regions_;
    for (auto &it : reads) {
      if (!IsGlobalBuffer(it->buffer)) {
        role = Role::kConsumer;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (auto stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined()) {
      auto role_else = GetRole(op->else_case.value());
      if (role != role_else)
        role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType> void HandleBodyStmt(const NodeType *op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AllocateNode *op) final { HandleBodyStmt(op); }
  void VisitStmt_(const DeclBufferNode *op) final { HandleBodyStmt(op); }

private:
  void SetRole(const StmtNode *stmt, Role role) { map_[stmt] = role; }
  std::unordered_map<const StmtNode *, Role> map_;
  ASTTraverser traverser_;
};

class Command {
public:
  int id = -1;
  int iter = -1;
  std::string name = "";
  Stmt stmt;
  std::vector<Command> dependencies;
  DeviceType type = DeviceType::Unspecified;
  Role role = Role::kUndefined;
  bool is_prefetch = false;
  float scheduled_start = -1;
  float scheduled_end = -1;

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

  float get_delay() const {
    // TODO
    if (type == DeviceType::TensorCore)
      return 10;
    else if (type == DeviceType::ODMA)
      return 3;
    else if (type == DeviceType::VectorCore)
      return 2;

    return 1;
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
    command->scheduled_start = time;
    command->scheduled_end = command_end_time;
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

  bool debug_{false};
  std::map<std::string, int> b_level;
  std::unordered_map<std::string, int> name_to_index_;
  std::unordered_map<int, std::string> index_to_name_;
  std::unordered_set<const BufferNode *> versioned_buffers_;
  std::vector<std::vector<int>> predecessors_;
  std::vector<std::vector<int>> successors_;
  std::vector<int> topo_order_;

  Scheduler() {
    devices.push_back(Device(DeviceType::ODMA));
    devices.push_back(Device(DeviceType::TensorCore));
    devices.push_back(Device(DeviceType::VectorCore));
  }

  void SetVersionedBuffers(const Array<Buffer> &versioned_buffers) {
    versioned_buffers_.clear();
    for (const Buffer &buf : versioned_buffers) {
      versioned_buffers_.insert(buf.get());
    }
  }

  void DumpGraph(const std::string &file_name) const {
    if (!debug_) {
      return;
    }
    std::ofstream log_file(file_name, std::ios::out);
    if (!log_file.is_open()) {
      return;
    }
    log_file << "num_commands " << commands.size() << "\n";
    log_file << "nodes\n";
    for (int idx : topo_order_) {
      const auto &cmd = commands[idx];
      int bl = -1;
      auto it = b_level.find(cmd.name);
      if (it != b_level.end()) {
        bl = it->second;
      }
      log_file << idx << " " << cmd.name << " " << cmd.iter << " " << cmd.id
               << " " << int(cmd.type) << " " << int(cmd.role) << " " << bl
               << "\n";
    }
    log_file << "edges\n";
    for (int src = 0; src < static_cast<int>(successors_.size()); ++src) {
      for (int dst : successors_[src]) {
        log_file << src << " " << index_to_name_.at(src) << " -> " << dst << " "
                 << index_to_name_.at(dst) << "\n";
      }
    }
  }

  void BuildDependencyGraph() {
    name_to_index_.clear();
    index_to_name_.clear();
    predecessors_.clear();
    successors_.clear();
    topo_order_.clear();

    int n = static_cast<int>(commands.size());
    predecessors_.resize(n);
    successors_.resize(n);
    topo_order_.resize(n);
    for (int i = 0; i < n; ++i) {
      topo_order_[i] = i;
      name_to_index_[commands[i].name] = i;
      index_to_name_[i] = commands[i].name;
    }
    std::sort(topo_order_.begin(), topo_order_.end(), [&](int a, int b) {
      if (commands[a].iter != commands[b].iter)
        return commands[a].iter < commands[b].iter;
      return commands[a].id < commands[b].id;
    });

    // Track the latest command that wrote to or read from a buffer region
    // Key: Buffer pointer
    // Value: List of (Region, Command Index, AccessType)
    enum AccessType { kRead, kWrite };
    struct AccessRecord {
      Region region;
      int cmd_idx;
      AccessType type;
    };
    std::unordered_map<const BufferNode *, std::vector<AccessRecord>>
        buffer_access_history;

    for (int ii = 0; ii < n; ++ii) {
      int curr_idx = topo_order_[ii];
      const Command &curr_cmd = commands[curr_idx];

      // Check dependencies based on Reads (RAW, WAW handled later by Writes)
      // Current command Reads from Buffer -> Find previous Writes (RAW)
      for (const auto &read_region : curr_cmd.reads) {
        const BufferNode *buf = read_region->buffer.get();
        if (buffer_access_history.find(buf) == buffer_access_history.end())
          continue;

        auto &history = buffer_access_history[buf];
        // Iterate backwards to find the latest dependency
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
          if (versioned_buffers_.count(buf) &&
              commands[it->cmd_idx].iter != curr_cmd.iter) {
            continue;
          }
          if (it->type == kWrite &&
              RegionIntersect(read_region->region, it->region)) {
            // Found a RAW dependency: it->cmd_idx -> curr_idx
            bool exists = false;
            for (int pred : predecessors_[curr_idx]) {
              if (pred == it->cmd_idx) {
                exists = true;
                break;
              }
            }
            if (!exists) {
              predecessors_[curr_idx].push_back(it->cmd_idx);
              successors_[it->cmd_idx].push_back(curr_idx);
            }
            // Since we found the latest write that covers/intersects this read,
            // we can stop searching backwards for this specific buffer region
            // interaction assuming total ordering of writes.
            break;
          }
        }
      }

      // Check dependencies based on Writes (WAR, WAW)
      // Current command Writes to Buffer -> Find previous Reads (WAR) and
      // Writes (WAW)
      for (const auto &write_region : curr_cmd.writes) {
        const BufferNode *buf = write_region->buffer.get();
        if (buffer_access_history.find(buf) == buffer_access_history.end())
          continue;

        auto &history = buffer_access_history[buf];
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
          if (versioned_buffers_.count(buf) &&
              commands[it->cmd_idx].iter != curr_cmd.iter) {
            continue;
          }
          if (versioned_buffers_.count(buf) && it->type == kRead) {
            continue;
          }
          if (RegionIntersect(write_region->region, it->region)) {
            // Dependency: it->cmd_idx -> curr_idx
            bool exists = false;
            for (int pred : predecessors_[curr_idx]) {
              if (pred == it->cmd_idx) {
                exists = true;
                break;
              }
            }
            if (!exists) {
              predecessors_[curr_idx].push_back(it->cmd_idx);
              successors_[it->cmd_idx].push_back(curr_idx);
            }

            // If we hit a Write (WAW), we can stop because any earlier
            // Reads/Writes would be dependent on that Write.
            if (it->type == kWrite) {
              break;
            }
            // If we hit a Read (WAR), we continue searching backwards because
            // there might be other Reads that we also need to wait for,
            // until we hit a Write.
          }
        }
      }

      // Update history with current command's accesses
      for (const auto &read_region : curr_cmd.reads) {
        if (buffer_access_history.find(read_region->buffer.get()) ==
            buffer_access_history.end()) {
          buffer_access_history[read_region->buffer.get()] = {};
        }
        buffer_access_history[read_region->buffer.get()].push_back(
            {read_region->region, curr_idx, kRead});
      }
      for (const auto &write_region : curr_cmd.writes) {
        if (buffer_access_history.find(write_region->buffer.get()) ==
            buffer_access_history.end()) {
          buffer_access_history[write_region->buffer.get()] = {};
        }
        buffer_access_history[write_region->buffer.get()].push_back(
            {write_region->region, curr_idx, kWrite});
      }
    }
  }

  void CalculateBottomLevels() {
    b_level.clear();
    for (auto it = topo_order_.rbegin(); it != topo_order_.rend(); ++it) {
      int idx = *it;
      int max_succ = 0;
      for (int succ : successors_[idx]) {
        max_succ = std::max(max_succ, b_level[commands[succ].name]);
      }
      b_level[commands[idx].name] =
          static_cast<int>(commands[idx].get_delay()) + max_succ;
    }
  }

  int num_stages_to_iterations(int num_stages) { return num_stages; }

  std::vector<Command> CriticalPathPipeline(std::string log_file_name) {
    std::ofstream log_file;
    if (debug_) {
      log_file.open(log_file_name, std::ios::out);
    }

    std::vector<Command *> primary_queue;
    std::vector<Command *> prefetch_queue;
    primary_queue.reserve(commands.size());
    prefetch_queue.reserve(commands.size());
    for (auto &command : commands) {
      command.finished = false;
      command.scheduled_start = -1;
      command.scheduled_end = -1;
      if (command.is_prefetch) {
        prefetch_queue.push_back(&command);
      } else {
        primary_queue.push_back(&command);
      }
    }
    auto primary_cmp = [this](Command *a, Command *b) {
      if (b_level[a->name] != b_level[b->name]) {
        return b_level[a->name] > b_level[b->name];
      } else {
        if (a->iter != b->iter) {
          return a->iter < b->iter;
        } else {
          return a->id < b->id;
        }
      }
    };
    sort(primary_queue.begin(), primary_queue.end(), primary_cmp);
    float time = 0;
    std::vector<Command> schedule;
    if (debug_) {
      LOG(INFO) << "Scheduling starts.";
    }

    // Phase 1: schedule only primary commands (prefetch commands do not
    // participate in priority scheduling).
    while (!primary_queue.empty()) {
      for (auto *command : primary_queue) {
        if (command->finished) {
          continue;
        }

        bool ready = true;
        auto it = name_to_index_.find(command->name);
        ICHECK(it != name_to_index_.end());
        int idx = it->second;
        for (int pred : predecessors_[idx]) {
          if (!commands[pred].finished) {
            ready = false;
            break;
          }
        }
        if (ready) {
          // Find available device of correct type
          for (auto &device : devices) {
            if (device.type == command->type && !device.busy) {
              device.assign_command(command, time);
              if (debug_) {
                LOG(INFO) << "Command " << command->name
                          << " assigned to device " << int(device.type)
                          << " at time " << time << " with delay "
                          << command->get_delay();
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

      primary_queue.erase(
          std::remove_if(primary_queue.begin(), primary_queue.end(),
                         [](Command *cmd) { return cmd->finished; }),
          primary_queue.end());
    }

    // Phase 2: insert prefetch commands into the fixed primary schedule.
    // The primary schedule timing is kept unchanged; prefetch commands are
    // placed into device idle gaps.
    struct Interval {
      float start;
      float end;
    };
    std::unordered_map<DeviceType, std::vector<Interval>> busy;
    for (const auto &cmd : schedule) {
      busy[cmd.type].push_back({cmd.scheduled_start, cmd.scheduled_end});
    }
    for (auto &kv : busy) {
      auto &v = kv.second;
      std::sort(v.begin(), v.end(), [](const Interval &a, const Interval &b) {
        return a.start < b.start;
      });
    }

    std::sort(prefetch_queue.begin(), prefetch_queue.end(),
              [](Command *a, Command *b) {
                if (a->iter != b->iter)
                  return a->iter < b->iter;
                return a->id < b->id;
              });

    auto insert_interval = [](std::vector<Interval> &v, Interval x) {
      v.push_back(x);
      std::sort(v.begin(), v.end(), [](const Interval &a, const Interval &b) {
        return a.start < b.start;
      });
    };

    for (auto *cmd : prefetch_queue) {
      float duration = cmd->get_delay();

      float ready_time = 0;

      auto &v = busy[cmd->type];
      float t = ready_time;
      bool placed = false;
      for (size_t i = 0; i <= v.size(); i++) {
        float gap_start = t;
        float gap_end =
            (i < v.size()) ? v[i].start : std::numeric_limits<float>::max();
        if (gap_end - gap_start >= duration) {
          float start = gap_start;
          float end = start + duration;
          cmd->scheduled_start = start;
          cmd->scheduled_end = end;
          cmd->finished = true;
          insert_interval(v, {start, end});
          schedule.push_back(*cmd);
          placed = true;
          break;
        }
        if (i < v.size()) {
          t = std::max(t, v[i].end);
        }
      }
      if (!placed) {
        LOG(FATAL) << "Failed to insert prefetch command " << cmd->name;
      }
    }

    std::sort(schedule.begin(), schedule.end(),
              [](const Command &a, const Command &b) {
                if (a.scheduled_start != b.scheduled_start)
                  return a.scheduled_start < b.scheduled_start;
                return a.name < b.name;
              });

    if (debug_ && log_file.is_open()) {
      for (const auto &cmd : schedule) {
        log_file << (cmd.is_prefetch ? "p:" : "") << cmd.name << " "
                 << int(cmd.type) << " " << cmd.scheduled_start << " "
                 << cmd.get_delay() << '\n';
      }
    }
    return schedule;
  }
};

class SunmmioPipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool debug) {
    SunmmioPipelinePlanner substituter(f, debug);
    return substituter.VisitStmt(f->body);
  }

  SunmmioPipelinePlanner(const PrimFunc &f, bool debug)
      : traverser(f), debug_(debug) {
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

    // 1.1 Build Iter0 Commands for Analysis
    std::set<Buffer> used_buffers;
    std::vector<Role> roles;
    Array<Array<BufferRegion>> reads, writes;
    std::vector<Command> iter0_commands;

    // Use SunmmioRoleMarker to determine roles
    SunmmioRoleMarker role_marker(traverser);

    for (size_t j = 0; j < pipeline_body_seq->size(); j++) {
      auto stmt = pipeline_body_seq->seq[j];
      Command cmd(j, 0, stmt);
      role_marker(stmt);
      cmd.role = role_marker.GetRole(stmt); // Set role
      roles.push_back(cmd.role);
      traverser.traverse_stmt(stmt);
      cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                   traverser.write_buffer_regions_);

      for (auto &read : traverser.read_buffer_regions_) {
        if (!IsGlobalBuffer(read->buffer)) {
          used_buffers.insert(read->buffer);
        }
      }
      reads.push_back(
          Array<BufferRegion>(traverser.read_buffer_regions_.begin(),
                              traverser.read_buffer_regions_.end()));
      for (auto &write : traverser.write_buffer_regions_) {
        if (!IsGlobalBuffer(write->buffer)) {
          used_buffers.insert(write->buffer);
        }
      }
      writes.push_back(
          Array<BufferRegion>(traverser.write_buffer_regions_.begin(),
                              traverser.write_buffer_regions_.end()));
      iter0_commands.push_back(cmd);
    }

    std::unordered_set<const BufferNode *> consumer_used, producer_used;
    std::unordered_map<const BufferNode *, size_t> first_write_index;
    std::unordered_map<const BufferNode *, Array<size_t>> write_indexes;
    std::unordered_map<const BufferNode *, size_t> first_read_index;
    std::unordered_map<const BufferNode *, size_t> last_read_index;
    Array<Buffer> versioned_buffers;
    // detect versioned buffers with TileLang's role pattern

    auto is_copy_stage = [&](size_t idx) {
      bool has_shared_write = false;
      for (const BufferRegion &wr : writes[idx]) {
        if (IsSunmmioSharedBuffer(wr->buffer)) {
          has_shared_write = true;
          break;
        }
      }
      if (!has_shared_write)
        return false;
      for (const BufferRegion &rd : reads[idx]) {
        if (IsGlobalBuffer(rd->buffer)) {
          return true;
        }
      }
      return false;
    };
    for (size_t i = 0; i < pipeline_body_seq->size(); i++) {
      bool copy_stage = is_copy_stage(i);
      bool is_producer = roles[i] == Role::kProducer ||
                         (roles[i] == Role::kBoth && copy_stage);
      bool is_consumer = roles[i] == Role::kConsumer ||
                         (roles[i] == Role::kBoth && !copy_stage);
      if (is_producer) {
        for (BufferRegion br : writes[i]) {
          producer_used.insert(br->buffer.get());
        }
      }
      if (is_consumer) {
        for (BufferRegion br : reads[i]) {
          consumer_used.insert(br->buffer.get());
        }
      }
      for (BufferRegion br : writes[i]) {
        const BufferNode *buf = br->buffer.get();
        if (!first_write_index.count(buf)) {
          first_write_index[buf] = i;
        }
        if (!write_indexes.count(buf)) {
          write_indexes[buf] = Array<size_t>();
        }
        write_indexes[buf].push_back(i);
      }
      for (BufferRegion br : reads[i]) {
        const BufferNode *buf = br->buffer.get();
        if (!first_read_index.count(buf)) {
          first_read_index[buf] = i;
        }
        last_read_index[br->buffer.get()] = i;
      }
    }

    for (const Buffer &buffer : used_buffers) {
      if (consumer_used.count(buffer.get()) &&
          producer_used.count(buffer.get())) {
        versioned_buffers.push_back(buffer);
        continue;
      }
      // Fallback: if we saw a write before a later read, the buffer spans
      // multiple stages even if role classification missed one side.
      auto it_w = first_write_index.find(buffer.get());
      auto it_r = last_read_index.find(buffer.get());
      if (it_w != first_write_index.end() && it_r != last_read_index.end() &&
          it_w->second < it_r->second) {
        if (!is_copy_stage(it_w->second))
          continue;
        versioned_buffers.push_back(buffer);
      }
    }

    bool update = true;
    while (update) {
      update = false;
      for (const Buffer &buffer : used_buffers) {
        if (std::find(versioned_buffers.begin(), versioned_buffers.end(),
                      buffer) != versioned_buffers.end())
          continue;
        const auto &write_index = write_indexes[buffer.get()];
        bool can_propogate = true;
        if (write_index.empty()) {
          continue;
        }
        for (auto idx : write_index) {
          for (const BufferRegion &rd : reads[idx]) {
            if (!IsGlobalBuffer(rd->buffer) &&
                std::find(versioned_buffers.begin(), versioned_buffers.end(),
                          rd->buffer) == versioned_buffers.end()) {
              can_propogate = false;
              break;
            }
          }
          if (!can_propogate) {
            break;
          }
        }
        if (can_propogate) {
          versioned_buffers.push_back(buffer);
          update = true;
        }
      }
    }

    if (debug_) {
      LOG(INFO) << versioned_buffers;
    }

    // 1.2 Build Prologue Commands
    Scheduler prologue_scheduler;
    std::vector<Command> prologue_commands;
    Array<String> prologue_orders;
    std::vector<int> prologue_ids;
    for (auto &cmd : iter0_commands) {
      if (cmd.role == Role::kProducer) {
        prologue_scheduler.commands.push_back(cmd);
        prologue_commands.push_back(cmd);
        prologue_orders.push_back(cmd.name);
        prologue_ids.push_back(cmd.id);
      }
    }

    // 1.3 Build Body Commands
    Scheduler body_scheduler;
    int iterations = body_scheduler.num_stages_to_iterations(num_stages);
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

        body_scheduler.commands.push_back(cmd);
      }

      for (auto &command : prologue_commands) {
        Command cmd(command.id, i + 1, command.stmt);
        // prologue of next iteration doesn't participate in pipeline scheduling
        if (i == iterations - 1) {
          cmd.is_prefetch = true;
        }
        traverser.traverse_stmt(cmd.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);

        body_scheduler.commands.push_back(cmd);
      }
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
        auto command =
            body_scheduler.commands[i * pipeline_body_seq->size() + j];
        int iter = command.iter;
        int id = command.id;
        Command cmd(id, iter, command.stmt);
        traverser.traverse_stmt(command.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);
        epilogue_scheduler.commands.push_back(cmd);
      }
      epilogue_iter += 1;
    }

    // 1.4.2 The End of Epilogue
    for (auto &command : iter0_commands) {
      if (std::find(prologue_orders.begin(), prologue_orders.end(),
                    command.name) == prologue_orders.end()) {
        int iter = command.iter;
        int id = command.id;
        Command cmd(id, epilogue_iter, command.stmt);
        traverser.traverse_stmt(command.stmt);
        cmd.get_command_reads_writes(traverser.read_buffer_regions_,
                                     traverser.write_buffer_regions_);
        epilogue_scheduler.commands.push_back(cmd);
      }
    }

    prologue_scheduler.SetVersionedBuffers(versioned_buffers);
    prologue_scheduler.debug_ = debug_;
    prologue_scheduler.BuildDependencyGraph();
    prologue_scheduler.CalculateBottomLevels();

    body_scheduler.SetVersionedBuffers(versioned_buffers);
    body_scheduler.debug_ = debug_;
    body_scheduler.BuildDependencyGraph();
    body_scheduler.CalculateBottomLevels();
    if (debug_) {
      body_scheduler.DumpGraph("body_graph.log");
      LOG(INFO) << "Body Bottom Level:";
      for (auto &it : body_scheduler.b_level) {
        LOG(INFO) << it.first << ' ' << it.second;
      }
    }

    epilogue_scheduler.SetVersionedBuffers(versioned_buffers);
    epilogue_scheduler.debug_ = debug_;
    epilogue_scheduler.BuildDependencyGraph();
    epilogue_scheduler.CalculateBottomLevels();
    if (debug_) {
      LOG(INFO) << "Epilogue Bottom Level:";
      for (auto &it : epilogue_scheduler.b_level) {
        LOG(INFO) << it.first << ' ' << it.second;
      }
    }

    // 2. Do critical path pipeline
    auto prologue_result =
        prologue_scheduler.CriticalPathPipeline("prologue.log");
    auto body_result = body_scheduler.CriticalPathPipeline("body.log");
    auto epilogue_result =
        epilogue_scheduler.CriticalPathPipeline("epilogue.log");

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages" && key != "versioned_buffers") {
        annotations.Set(key, value);
      }
    }

    annotations.Set("iterations", iterations);
    Array<String> orders;
    for (auto &it : prologue_result) {
      orders.push_back(it.name);
    }
    annotations.Set("prologue_orders", orders);
    orders.clear();
    for (auto &it : body_result) {
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

    annotations.Set("versioned_buffers", versioned_buffers);

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, annotations);
  }

private:
  ASTTraverser traverser;
  bool debug_{false};
};

tvm::transform::Pass SunmmioPipelinePlanning(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlanner::Substitute(f, debug);
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
