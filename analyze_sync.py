import glob
import re
import sys
import os

def count_sync_primitives(filepath):
    if not os.path.exists(filepath):
        return 0, 0, 0
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    wait_tokens = len(re.findall(r'T\.wait_token', content))
    sync_null_tokens = len(re.findall(r'T\.sync_null_token', content))
    sync_token_ids = len(re.findall(r'T\.sync_token_id', content))
    barrier_inits = len(re.findall(r'T\.barrier_init', content))
    barrier_arrive_and_waits = len(re.findall(r'T\.barrier_arrive_and_wait', content))
    
    return wait_tokens, sync_null_tokens, sync_token_ids, barrier_inits, barrier_arrive_and_waits

def analyze(kernel_name):
    log_dir = f"pass_logs_{kernel_name}"
    files = sorted(glob.glob(f"{log_dir}/*.py"))
    
    final_file = None
    for f in reversed(files):
        if "DeviceMod" in f:
            final_file = f
            break
            
    if not final_file:
        print(f"Error: Could not find final DeviceMod file in {log_dir}")
        return
        
    w, sn, st, bi, baw = count_sync_primitives(final_file)
    total = w + sn + st + bi + baw
    
    print(f"\n--- Automatic Synchronization Injection Analysis ---")
    
    print(f"{'Synchronization Primitive':<30} | {'Injected Count':>14}")
    print("-" * 30 + "-+-" + "-" * 14)
    print(f"{'T.sync_token_id':<30} | {st:>14}")
    print(f"{'T.wait_token':<30} | {w:>14}")
    print(f"{'T.sync_null_token':<30} | {sn:>14}")
    print(f"{'T.barrier_init':<30} | {bi:>14}")
    print(f"{'T.barrier_arrive_and_wait':<30} | {baw:>14}")
    print("-" * 30 + "-+-" + "-" * 14)
    print(f"{'TOTAL':<30} | {total:>14}\n")
    print("--- Data Dependency Graph Visualization ---")
    # print(f"\n*Conclusion: The compiler automatically injected {total} underlying hardware synchronization primitives into the backend for the concise frontend code, completely freeing developers from this cognitive burden.*")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default=None, help="Kernel to analyze (e.g. gemm, flashattn)")
    args = parser.parse_args()
    
    if args.kernel:
        analyze(args.kernel)
    else:
        for k in ["flashattn", "gemm"]:
            analyze(k)
