import json



# method = 'gemm'
# method = 'flashattn'
method = 'flashdecoding'
# method = 'flashmladecode'
# method = 'native_sparse_attention'


with open('body.log', 'r') as f:
    data = f.readlines()
    data = [it.strip() for it in data]

sunmmio_schedule = []
costs = [0] * 100
devices = [0] * 100
for it in data:
    items = it.split(' ')
    sunmmio_schedule.append((items[0], items[1], items[2], items[3]))
    id = int(items[0].split('-')[1])
    costs[id] = int(items[3])
    devices[id] = int(items[1])

iterations = 64

# tilelang
for num_stages in range(2, 6):
    with open(f'experiment/{method}_numstages_{num_stages}_data.json', 'r') as f:
        tilelang_info = json.load(f)

    prologue_nums = max(tilelang_info['software_pipeline_stage'])
    ids = []
    for i in range(len(tilelang_info['software_pipeline_stage'])):
        if tilelang_info['software_pipeline_stage'][i] == 0:
            ids.append(i)
    prologue_cost = sum([costs[it] for it in ids]) * prologue_nums
    single_body_cost = sum([costs[it] for it in range(len(tilelang_info['software_pipeline_stage'])) if it not in ids])
    vector_cost = sum([costs[it] for it in range(len(tilelang_info['software_pipeline_stage'])) if devices[it] == 2 and it not in ids])
    tensor_cost = sum([costs[it] for it in range(len(tilelang_info['software_pipeline_stage'])) if devices[it] == 1 and it not in ids])
    print(f'efficiency: {vector_cost / single_body_cost:.4f}, {tensor_cost / single_body_cost * 100:.2f}')
    body_cost = single_body_cost * iterations
    tl = body_cost + prologue_cost
    print(tl)
    ours = 182144
    print(f'{tl / ours:.3f}, {(1 - ours/tl) * 100:.2f}')



# ours
s = {}
max_time = 0
for it in sunmmio_schedule:
    if it[1] not in s:
        s[it[1]] = [(int(it[2]), int(it[2]) + int(it[3]))]
    else:
        s[it[1]].append((int(it[2]), int(it[2]) + int(it[3])))

    max_time = max(max_time, int(it[2]) + int(it[3]))

for key in s:
    used_time = 0
    for it in s[key]:
        used_time += it[1] - it[0]
    print(f'{key}: {used_time}, {used_time / max_time:.4f}, {max_time}')
        
# print(s)