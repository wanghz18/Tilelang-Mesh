import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import os
import re
import argparse

parser = argparse.ArgumentParser(description="Analyze buffer lifespans and memory allocation.")
parser.add_argument('--kernel', type=str, default='gemm', help="Kernel name (e.g., gemm, flashattn)")
args = parser.parse_args()

print('\n--- Buffer Lifespan Visualization ---')
print(f"Visualization saved to buffer_new.png")

method = args.kernel
all_buffer_size = {
    'gemm': {
        "A_shared": 4096,
        "B_shared": 16384,
        "C_shared": 16384,
        "A_shared_dist": 16384,
        "B_shared_dist": 16384,
    },
    'flashattn': {
        "K_shared": 8192,
        "V_shared": 8192,
        "acc_s": 4096,
        "acc_s_cast": 4096,
        "acc_s_cast_local": 4096,
        "Q_shared": 8192,
        "acc_o": 8192,
        "logsum": 64,
        "scores_max": 64,
        "scores_max_acc": 64,
        "scores_max_prev": 64,
        "scores_max_res": 64,
        "scores_scale": 64,
        "scores_sum": 64,
        "scores_sum_acc": 64,
    }
}

all_versioned_buffer = {
    'gemm': ['A_shared', 'B_shared', 'A_shared_dist', 'B_shared_dist'],
    'flashattn': ['K_shared', 'V_shared', 'acc_s', 'acc_s_cast', 'acc_s_cast_local'],
}

versioned_buffer = all_versioned_buffer[method]
FLOAT16_BYTES = 2
BUFFER_SIZE_ELEMENTS = all_buffer_size.get(method, {})
BUFFER_SIZE_FILE = 'buffer_sizes.json'

def get_base_name(buf_name):
    return re.sub(r'\d+$', '', buf_name)

def load_buffer_sizes():
    buffer_sizes = dict(BUFFER_SIZE_ELEMENTS)
    if os.path.exists(BUFFER_SIZE_FILE):
        with open(BUFFER_SIZE_FILE, 'r', encoding='utf-8') as file:
            buffer_sizes.update(json.load(file))
    return buffer_sizes

def get_buffer_size_elements(buf_name, base_name, buffer_sizes):
    if buf_name in buffer_sizes:
        return buffer_sizes[buf_name]
    return buffer_sizes.get(base_name)

def format_size_kb(num_bytes):
    return f'{num_bytes / 1024:.2f} KB'

def extract_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return None
        
    sunmmio_schedule = {}
    costs = [0] * 100
    devices = [0] * 100
    max_time = 0
    
    with open(log_path, 'r') as f:
        schedule = [line.strip() for line in f.readlines() if line.strip()]
        
    for it in schedule:
        items = it.split(' ')
        key = items[0]
        key = items[0] if ':' not in items[0] else items[0].split(':')[1]
        
        sunmmio_schedule[key] = (items[1], items[2], items[3])
        id_str = items[0].split('-')[1]
        id = int(id_str)
        costs[id] = int(items[3])
        devices[id] = int(items[1])
        max_time = max(max_time, int(items[2]) + int(items[3]))
        
    return sunmmio_schedule, max_time

# You can modify the paths here
command_info_path = 'command_info.log'
body_log_path = 'body.log'

if not os.path.exists(command_info_path) or not os.path.exists(body_log_path):
    print("Log files command_info.log or body.log not found. The script requires these files to generate the graph.")
    print("Please generate the pipeline schedule logs first.")
    exit(1)

with open(command_info_path, 'r') as f:
    data = f.readlines()

data = [(data[3 * i].strip(), data[3 * i + 1].strip().split(';')[:-1], data[3 * i + 2].strip().split(';')[:-1]) for i in range(len(data) // 3)]
data = [(it[0], [item.split('[')[0] for item in it[1]], [item.split('[')[0] for item in it[2]]) for it in data]
new_data = {}

for it in data:
    item = [it[0], [], []]
    iter = int(it[0].split('-')[0])

    for buffer in it[1]:
        if buffer in versioned_buffer:
            item[1].append(f'{buffer}{iter}')
        else:
            item[1].append(f'{buffer}')
    for buffer in it[2]:
        if buffer in versioned_buffer:
            item[2].append(f'{buffer}{iter}')
        else:
            item[2].append(f'{buffer}')
    new_data[item[0]] = [item[1], item[2]]

buffer_data = new_data
sunmmio_schedule, max_time = extract_log(body_log_path)

first_read_id = {}
last_read_id = {}
last_write_id = {}
first_write_id = {}

for key in sunmmio_schedule:
    if key not in buffer_data:
        continue
    for read in buffer_data[key][0]:
        last_read_id[read] = key
        if read not in first_read_id:
            first_read_id[read] = key
    for write in buffer_data[key][1]:
        last_write_id[write] = key
        if write not in first_write_id:
            first_write_id[write] = key

buffers = list(set(first_read_id.keys()) | set(last_read_id.keys()) | set(first_write_id.keys()) | set(last_write_id.keys()))

class BufferLifespan:
    def __init__(self, name, iteration):
        self.name = name
        self.iteration = iteration
        self.birth_time = 0
        self.death_time = max_time
        self.writer = None
        self.readers = []

    def __repr__(self):
        return f"{self.name}_iter{self.iteration} [{self.birth_time}-{self.death_time}]"

logical_bufs = []
for buf in buffers:
    has_digits = any(char.isdigit() for char in buf)
    if has_digits:
        # Match trailing digits
        m = re.search(r'\d+$', buf)
        iteration = int(m.group(0)) if m else -1
    else:
        iteration = -1
        
    logical_bufs.append((buf, iteration, BufferLifespan(buf, iteration)))
    
    if buf in first_write_id and first_write_id[buf] in sunmmio_schedule:
        sched = sunmmio_schedule[first_write_id[buf]]
        logical_bufs[-1][-1].birth_time = int(sched[2]) + int(sched[1])
    if buf in last_read_id and last_read_id[buf] in sunmmio_schedule:
        sched = sunmmio_schedule[last_read_id[buf]]
        logical_bufs[-1][-1].death_time = int(sched[2]) + int(sched[1])
    else:
        logical_bufs[-1][-1].death_time = max_time

buffer_sizes = load_buffer_sizes()
base_groups = {}
for buf_name, iteration, lifespan in logical_bufs:
    if iteration != -1:
        base_name = get_base_name(buf_name)
    else:
        base_name = buf_name
        
    if base_name not in base_groups:
        base_groups[base_name] = []
    base_groups[base_name].append(lifespan)

num_logical = len(logical_bufs)
num_physical = 0
allocation_info = {}
logical_memory_bytes = 0
physical_memory_bytes = 0
missing_size_buffers = []

print("\n--- Memory Allocation Analysis ---")
for base_name, group in base_groups.items():
    group.sort(key=lambda x: x.birth_time)
    
    slots = [] 
    for lifespan in group:
        allocated = False
        for slot_id in range(len(slots)):
            if slots[slot_id] <= lifespan.birth_time:
                slots[slot_id] = lifespan.death_time
                allocation_info[lifespan.name] = f"{base_name}_slot{slot_id}"
                allocated = True
                break
        
        if not allocated:
            slots.append(lifespan.death_time)
            slot_id = len(slots) - 1
            allocation_info[lifespan.name] = f"{base_name}_slot{slot_id}"
            
    physical_needed = len(slots)
    num_physical += physical_needed
    buffer_size_elements = get_buffer_size_elements(group[0].name, base_name, buffer_sizes)
    
    if buffer_size_elements is None:
        missing_size_buffers.append(base_name)
        size_desc = 'elements=unknown'
    else:
        buffer_size_bytes = buffer_size_elements * FLOAT16_BYTES
        logical_memory_bytes += len(group) * buffer_size_bytes
        physical_memory_bytes += physical_needed * buffer_size_bytes
        size_desc = (
            f'elements={buffer_size_elements} | '
            f'size={format_size_kb(buffer_size_bytes)} | '
            f'logical_mem={format_size_kb(len(group) * buffer_size_bytes)} | '
            f'physical_mem={format_size_kb(physical_needed * buffer_size_bytes)}'
        )

    print(
        f"Base Buffer: {base_name:15s} | Logical Versions: {len(group):2d} | "
        f"Physical Slots Needed: {physical_needed} | {size_desc}"
    )

print("-" * 50)
print(f"Total Logical Buffers : {num_logical}")
print(f"Total Physical Buffers: {num_physical}")
if missing_size_buffers:
    missing_items = ', '.join(sorted(set(missing_size_buffers)))
    print(f"Missing Buffer Sizes (elements): {missing_items}")
else:
    memory_reduction = (
        (logical_memory_bytes - physical_memory_bytes) / logical_memory_bytes * 100
        if logical_memory_bytes > 0 else 0
    )
    print(f"Logical Memory Footprint : {format_size_kb(logical_memory_bytes)}")
    print(f"Physical Memory Footprint: {format_size_kb(physical_memory_bytes)}")
    print(f"Memory Reduction         : {memory_reduction:.1f}%")

fig, ax = plt.subplots(figsize=(15, max(6, len(buffers) * 0.8)))
cmap = plt.get_cmap('Set3')

logical_bufs.sort(key=lambda x: x[0])

idx = 0
for buf_name, iteration, lifespan in logical_bufs:
    y = len(logical_bufs) - 1 - idx
    idx += 1
    if iteration == -1:
        color = '#cccccc'
    else:
        color = cmap(iteration % 12)
    duration = lifespan.death_time - lifespan.birth_time
    rect = mpatches.Rectangle((lifespan.birth_time, y - 0.4), duration, 0.8,
                            linewidth=1, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)

font_path = 'SourceHanSansSC-Regular.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
else:
    font_name = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ax.set_ylim(-1, len(logical_bufs))
ax.set_xlim(0, max_time * 1.05)
ax.set_yticks(range(len(logical_bufs)))
y_labels = [name for name, _, _ in reversed(logical_bufs)]
ax.set_yticklabels(y_labels, fontsize=18, fontweight="bold")

# ax.set_title("FlashAttention Variables Liveness Map", fontsize=20, fontweight="bold", pad=20)
ax.set_xlabel('时间（时钟周期）', fontsize=20, fontweight='bold', labelpad=10)
ax.set_ylabel('缓冲区', fontsize=20, fontweight='bold', labelpad=10)


# 放大刻度标签字体
ax.tick_params(axis='x', labelsize=14)


ax.grid(True, axis='x', linestyle='--', alpha=0.5)

unique_iterations = sorted({iteration for _, iteration, _ in logical_bufs if iteration != -1})
legend_elements = [
    mpatches.Patch(
        facecolor=cmap(iteration % 12),
        edgecolor='black',
        label=f'版本 {iteration}',
    )
    for iteration in unique_iterations
]
if any(iteration == -1 for _, iteration, _ in logical_bufs):
    legend_elements.append(
        mpatches.Patch(
            facecolor='#cccccc',
            edgecolor='black',
            label='非多版本',
        )
    )

if legend_elements:
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=18,
        title_fontsize=18,
        frameon=True,
        facecolor='white',
        framealpha=0.95,
    )

plt.tight_layout()
plt.savefig('buffer_new.png', dpi=300)

