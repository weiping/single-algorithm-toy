"""JSONL 数据与验证器。把你自己的 UI 编辑数据按下面的格式放进一个目录即可接入训练器。

目录结构
  sft.jsonl           SFT 示范，每行 {"prompt": ..., "target": ...}
  rl.jsonl            RL 提示，每行 {"prompt": ..., "target": ...}，target 供验证器使用
  eval_<名字>.jsonl   评测集，格式同 rl.jsonl，可以有多个，例如 eval_held.jsonl、eval_seen.jsonl
  retention.jsonl     可选，留出的示范，用来度量保留性（每行 {"prompt":..., "target":...}）

prompt 是模型看到的完整输入文本（UI 状态加编辑指令），target 是标准输出。
训练器会把评测集与 retention 的 prompt 从 sft.jsonl、rl.jsonl 中剔除，保证留出。
"""
import json
import os
import random


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        assert "prompt" in r and "target" in r, f"{path} 的记录必须含 prompt 与 target"
    return recs


class Cycle:
    """打乱后循环取样的流。exclude 中的提示会被剔除。"""

    def __init__(self, records, rng, exclude=()):
        ex = set(exclude)
        self.recs = [r for r in records if r["prompt"] not in ex]
        assert self.recs, "剔除后没有剩余样本"
        self.rng, self.order = rng, []

    def next(self):
        if not self.order:
            self.order = list(range(len(self.recs)))
            self.rng.shuffle(self.order)
        return self.recs[self.order.pop()]

    __call__ = next


def load_dir(path):
    out = {"sft": load_jsonl(os.path.join(path, "sft.jsonl")),
           "rl": load_jsonl(os.path.join(path, "rl.jsonl")), "eval": {}, "retention": []}
    for f in sorted(os.listdir(path)):
        if f.startswith("eval_") and f.endswith(".jsonl"):
            out["eval"][f[5:-6]] = load_jsonl(os.path.join(path, f))
    rp = os.path.join(path, "retention.jsonl")
    if os.path.exists(rp):
        out["retention"] = load_jsonl(rp)
    assert out["eval"], "至少需要一个 eval_*.jsonl"
    return out


# ---------------- 验证器，返回 0 或 1，不给部分分 ----------------
def exact(task, text):
    return int(text.strip() == task["target"].strip())


def lines_equal(task, text):
    a = [l.strip() for l in text.strip().split("\n")]
    b = [l.strip() for l in task["target"].strip().split("\n")]
    return int(a == b)


def json_equal(task, text):
    """把输出与标准答案都解析成 JSON 后做结构比较，键顺序无关。解析失败记 0。"""
    try:
        return int(json.loads(text) == json.loads(task["target"]))
    except (ValueError, TypeError):
        return 0


VERIFIERS = {"exact": exact, "lines_equal": lines_equal, "json_equal": json_equal}
