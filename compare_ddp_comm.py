import json
import glob
import os
from collections import defaultdict
from pathlib import Path

def parse_trace_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    comm_time_us = 0
    total_time_us = 0
    # cat_dict = dict()
    for evt in events:
        if evt.get("ph") != "X":
            continue

        dur = evt.get("dur", 0)
        name = evt.get("name", "").lower()
        cat = evt.get("cat", "").lower()
        if "nccl" in name and ("allreduce" in name or "broadcast" in name or "all_reduce" in name):
            comm_time_us += dur
        if cat != "cpu_op" and cat != "user_annotation":
            # if cat not in cat_dict:
            #     cat_dict[cat] = 1
            # else:
            #     cat_dict[cat] +=1
            total_time_us += dur
    # print(cat_dict)

    return comm_time_us, total_time_us

def analyze_logs(log_root):
    results = defaultdict(lambda: {"comm": 0, "total": 0})
    log_root = Path(log_root)

    for config_path in log_root.iterdir():
        if not config_path.is_dir():
            continue
        config_name = config_path.name

        for rank_dir in config_path.glob("rank_*"):
            for file in rank_dir.glob("*.pt.trace.json"):
                print(file)
                comm, total = parse_trace_json(file)
                results[config_name]["comm"] += comm
                results[config_name]["total"] += total

    return results

def print_report(results):
    print(f"{'Config':<20} | {'Comm Time (ms)':>15} | {'Total Time (ms)':>15} | {'Comm %':>8}")
    print("-" * 65)
    for config, vals in sorted(results.items()):
        comm_ms = vals['comm'] / 1000
        total_ms = vals['total'] / 1000
        percent = (comm_ms / total_ms * 100) if total_ms > 0 else 0
        print(f"{config:<20} | {comm_ms:15.2f} | {total_ms:15.2f} | {percent:8.2f}%")

if __name__ == "__main__":
    log_dir = "/work/hdd/beih/yuli9/log_pipe"
    results = analyze_logs(log_dir)
    print_report(results)
