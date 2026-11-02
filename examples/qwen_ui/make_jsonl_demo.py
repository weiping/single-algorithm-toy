"""把内置的 UI 任务导出成 JSONL 目录，演示 data_jsonl.py 的数据格式，也可以用来测试 --data-dir 路径。
用法  python make_jsonl_demo.py demo_data"""
import json
import os
import random
import sys

import ui_task


def write(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"prompt": r["prompt"], "target": r["target"]}, ensure_ascii=False) + "\n")


def main(out="demo_data", n_sft=3000, n_rl=600):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(0)
    sets = ui_task.build_eval_sets(48)
    write(os.path.join(out, "sft.jsonl"), [ui_task.demo_stream(rng)() for _ in range(n_sft)])
    write(os.path.join(out, "rl.jsonl"), [ui_task.gen_task(rng, "held_pair") for _ in range(n_rl)])
    write(os.path.join(out, "eval_held.jsonl"), sets["held_pair"])
    write(os.path.join(out, "eval_seen.jsonl"), sets["seen_pair"])
    write(os.path.join(out, "eval_single.jsonl"), sets["single"])
    write(os.path.join(out, "retention.jsonl"), ui_task.build_retention_set(64))
    print("已写入", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["demo_data"]))
