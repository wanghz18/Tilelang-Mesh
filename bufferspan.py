import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import os
import re

method = 'nas'
all_buffer_size = {
    'gemm' : {
    "A_shared": 4096,
    "B_shared": 16384,
    "C_shared": 16384,
},
'flashattn': {
    "K_shared": 8192,
    "V_shared": 8192,
    "acc_s": 4096,
    "acc_s_cast": 4096,
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
},
'flashdecoding': {
    "mask_local": 128,
    "K_shared": 16384,
    "V_shared": 16384,
    "acc_s": 8192,
    "acc_s_cast": 8192,
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
},
'flashmladecode': {
    "K_pe_shared": 4096,
    "acc_s": 4096,
    "acc_s_cast": 4096,
    "Q_shared": 64*512,
    "acc_o": 64*512,
    "logsum": 64,
    "scores_max": 64,
    "scores_max_acc": 64,
    "scores_max_prev": 64,
    "scores_max_res": 64,
    "scores_scale": 64,
    "scores_sum": 64,
    "scores_sum_acc": 64,
    'S_shared': 64*64,
    'Q_pe_shared': 64*64,
    'KV_shared': 8192 * 4,
},
'nas': {
    "K_shared": 1024,
    "V_shared": 1024,
    "acc_s": 1024,
    "acc_s_cast": 1024,
    "Q_shared": 1024,
    "acc_o": 1024,
    "logsum": 32,
    "scores_max": 32,
    "scores_max_acc": 32,
    "scores_max_prev": 32,
    "scores_max_res": 32,
    "scores_scale": 32,
    "scores_sum": 32,
    "scores_sum_acc": 32,
},
}

all_versioned_buffer = {
    'gemm' : ['A_shared', 'B_shared'],
    'flashattn': ['K_shared', 'V_shared', 'acc_s', 'acc_s_cast'],
    'flashdecoding': ['K_shared', 'V_shared', 'mask_local', 'acc_s', 'acc_s_cast'],
    'flashmladecode': ['KV_shared', 'K_pe_shared'],
    'nas': ['K_shared', 'V_shared', 'acc_s', 'acc_s_cast'],
}

versioned_buffer = all_versioned_buffer[method]

FLOAT16_BYTES = 2

# Fill buffer sizes as element counts here, or provide them in `buffer_sizes.json`.
# All buffers are assumed to be float16, so bytes = element_count * 2.
# Exact logical names have higher priority than base names.
BUFFER_SIZE_ELEMENTS = all_buffer_size[method]

BUFFER_SIZE_FILE = 'buffer_sizes.json'


def get_base_name(buf_name):
    """Return the base buffer name by removing the trailing version suffix."""
    return re.sub(r'\d+$', '', buf_name)


def load_buffer_sizes():
    """Load buffer sizes as element counts from the in-file table and an optional JSON file."""
    buffer_sizes = dict(BUFFER_SIZE_ELEMENTS)
    if os.path.exists(BUFFER_SIZE_FILE):
        with open(BUFFER_SIZE_FILE, 'r', encoding='utf-8') as file:
            buffer_sizes.update(json.load(file))
    return buffer_sizes


def get_buffer_size_elements(buf_name, base_name, buffer_sizes):
    """Resolve the buffer size as element count using exact-name then base-name lookup."""
    if buf_name in buffer_sizes:
        return buffer_sizes[buf_name]
    return buffer_sizes.get(base_name)


def format_size_kb(num_bytes):
    """Format a byte size using KB only."""
    return f'{num_bytes / 1024:.2f} KB'


with open('command_info.log', 'r') as f:
    data = f.readlines()
    # data = [it.strip() for it in data]

data = [(data[3 * i].strip(), data[3 * i + 1].strip().split(';')[:-1], data[3 * i + 2].strip().split(';')[:-1]) for i in range(len(data) // 3)]
data = [it for it in data]
data = [(it[0], [item.split('[')[0] for item in it[1]], [item.split('[')[0] for item in it[2]]) for it in data]
new_data = {}
for it in data:
    item = [it[0], [], []]
    iter = int(it[0].split('-')[0])
    # if iter == num_stages:
    #     iter = 0

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

# print(new_data)
buffer_data = new_data

with open('body.log', 'r') as f:
    schedule = f.readlines()
    schedule = [it.strip() for it in schedule]
sunmmio_schedule = {}
costs = [0] * 100
devices = [0] * 100

max_time = 0
for it in schedule:
    items = it.split(' ')
    key = items[0]
    key = items[0] if ':' not in items[0] else items[0].split(':')[1]
    
    sunmmio_schedule[key] = (items[1], items[2], items[3])
    id = int(items[0].split('-')[1])
    costs[id] = int(items[3])
    devices[id] = int(items[1])
    max_time = max(max_time, int(items[2]) + int(items[3]))

print(sunmmio_schedule)
first_read_id = {}
last_read_id = {}
last_write_id = {}
first_write_id = {}
for key in sunmmio_schedule:
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
        self.birth_time = 0  # When the buffer is written
        self.death_time = max_time  # When the buffer is last read
        self.writer = None
        self.readers = []

    def __repr__(self):
        return f"{self.name}_iter{self.iteration} [{self.birth_time}-{self.death_time}]"

logical_bufs = []
for buf in buffers:
    has_digits = any(char.isdigit() for char in buf)
    if has_digits:
        iteration = int(buf[-1])
    else:
        iteration = -1
    logical_bufs.append((buf, iteration, BufferLifespan(buf, iteration)))
    if buf in first_write_id:
        logical_bufs[-1][-1].birth_time = int(sunmmio_schedule[first_write_id[buf]][2]) + int(sunmmio_schedule[first_write_id[buf]][1])
    if buf in last_read_id:
        logical_bufs[-1][-1].death_time = int(sunmmio_schedule[last_read_id[buf]][2]) + int(sunmmio_schedule[last_read_id[buf]][1])
    else:
        logical_bufs[-1][-1].death_time = max_time
    

# --- Greed Graph Coloring for Memory Allocation ---
# Group buffers by their base names
buffer_sizes = load_buffer_sizes()
base_groups = {}
for buf_name, iteration, lifespan in logical_bufs:
    # If it's a versioned buffer, extract base name by removing trailing digits
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

print("\n--- Memory Allocation (Graph Coloring) ---")
for base_name, group in base_groups.items():
    # Sort intervals by birth time
    group.sort(key=lambda x: x.birth_time)
    
    # physical slots tracking end times
    slots = [] 
    for lifespan in group:
        allocated = False
        for slot_id in range(len(slots)):
            # If the current slot is free (its previous occupant died before/when this one is born)
            if slots[slot_id] <= lifespan.birth_time:
                slots[slot_id] = lifespan.death_time
                allocation_info[lifespan.name] = f"{base_name}_slot{slot_id}"
                allocated = True
                break
        
        # If no free slot found, allocate a new one
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
print("-" * 50)
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(15, max(6, len(buffers) * 0.8)))

# Use Set3 colormap to match pipeline visualization
cmap = plt.get_cmap('Set3')
buffer_colors = {}

logical_bufs.sort(key=lambda x: x[0])

idx = 0
for buf_name, iteration, lifespan in logical_bufs:
    y = len(logical_bufs) - 1 - idx
    idx += 1
    # Top to bottom
    if iteration == -1:
        color = '#cccccc' # Neutral color for accumulated buffers
    else:
        color = cmap(iteration % 12)
    duration = lifespan.death_time - lifespan.birth_time
    rect = mpatches.Rectangle((lifespan.birth_time, y - 0.4), duration, 0.8,
                            linewidth=1, edgecolor='black', facecolor=color, alpha=0.9)
    ax.add_patch(rect)

    label = f"{buf_name} ({allocation_info.get(buf_name, '')})"
    # ax.text(lifespan.birth_time + duration/2, y, label,
    #         ha='center', va='center', fontsize=9, fontweight='bold', clip_on=True)


font_path = 'SourceHanSansSC-Regular.otf'

# 注册字体
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    # 获取字体名称
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    print(f"注册字体: {font_name}")
else:
    print(f"字体文件不存在: {font_path}")
    font_name = 'DejaVu Sans'  # 回退

# 设置全局字体
plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ax.set_ylim(-1, len(logical_bufs))
ax.set_xlim(0, max_time * 1.05)

ax.set_yticks(range(len(logical_bufs)))
y_labels = [name for name, _, _ in reversed(logical_bufs)]
ax.set_yticklabels(y_labels, fontsize=10)

# ax.set_xlabel('Time (Clock Cycles)', fontsize=12, fontweight='bold')
ax.set_xlabel('时间（时钟周期）', fontsize=12, fontweight='bold')
ax.set_ylabel('缓冲区', fontsize=12, fontweight='bold')

reduction = ((num_logical - num_physical) / num_logical * 100) if num_logical > 0 else 0
if missing_size_buffers:
    title_suffix = (
        f'复用前缓冲区数量: {num_logical} | '
        f'复用后缓冲区数量: {num_physical} | '
        f'数量节省: {reduction:.1f}%'
    )
else:
    memory_reduction = (
        (logical_memory_bytes - physical_memory_bytes) / logical_memory_bytes * 100
        if logical_memory_bytes > 0 else 0
    )
    title_suffix = (
        f'复用前内存: {format_size_kb(logical_memory_bytes)} | '
        f'复用后内存: {format_size_kb(physical_memory_bytes)} | '
        f'内存节省: {memory_reduction:.1f}%'
    )

# ax.set_title(
#     f'变量生命周期分析与内存复用\n{title_suffix}',
#     fontsize=14,
#     fontweight='bold',
#     pad=15,
# )

ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Add legend for iteration colors used by multi-version buffers.
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
        # title='颜色编码',
        loc='upper right',
        fontsize=10,
        title_fontsize=10,
        frameon=True,
        facecolor='white',
        framealpha=0.95,
    )

plt.tight_layout()
plt.savefig('buffer.png', dpi=300)
print(f"Visualization saved to buffer.png")
