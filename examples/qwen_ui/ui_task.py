"""可程序化验证的 UI 编辑任务（示例）。

UI 状态是若干行 `名称 类型 x y 颜色 尺寸`，例如 `b1 btn 30 50 red m`。
编辑指令是一到两个原子操作，用 ` ; ` 连接，例如 `move b1 right 20 ; color t1 blue`。
模型要输出编辑后的完整 UI 状态。奖励由程序判定，输出与标准答案逐行完全一致才算对（0/1）。

两种操作的先后组合（例如先 move 后 delete）只有一部分出现在 SFT 示范里，
其余留给 RL 与评测，用来测组合泛化，对应论文的受控实验设计。
真实的 UI 编辑任务把这里换成你自己的状态表示和验证器，比如 DOM 或布局 JSON 的结构化比较、
约束检查、渲染后的像素差，只要能给出对错判定即可接入。
"""
import itertools
import random

COLORS = ["red", "blue", "green", "gray", "black"]
SIZES = ["s", "m", "l"]
NAMES = ["b1", "b2", "t1", "t2", "i1", "i2"]
TYPE_OF = {"b": "btn", "t": "txt", "i": "inp"}
DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
OP_TYPES = ["move", "color", "size", "delete"]
ALL_PAIRS = list(itertools.product(OP_TYPES, repeat=2))
HELD_PAIRS = {("move", "delete"), ("delete", "color"), ("size", "move"),
              ("color", "delete"), ("delete", "size"), ("size", "color")}
SEEN_PAIRS = [p for p in ALL_PAIRS if p not in HELD_PAIRS]


def render(state):
    return "\n".join(f"{e['n']} {TYPE_OF[e['n'][0]]} {e['x']:02d} {e['y']:02d} {e['c']} {e['s']}"
                     for e in state)


def gen_state(rng):
    names = sorted(rng.sample(NAMES, 3))
    return [{"n": n, "x": 10 * rng.randint(0, 9), "y": 10 * rng.randint(0, 9),
             "c": rng.choice(COLORS), "s": rng.choice(SIZES)} for n in names]


def sample_op(rng, state, kind):
    """在当前状态上采样一个合法的原子操作，返回 (指令文本, 新状态)，失败返回 None。"""
    e = rng.choice(state)
    new = [dict(x) for x in state]
    t = next(x for x in new if x["n"] == e["n"])
    if kind == "move":
        d = rng.choice(list(DIRS))
        amt = rng.choice([10, 20, 30])
        nx, ny = t["x"] + DIRS[d][0] * amt, t["y"] + DIRS[d][1] * amt
        if not (0 <= nx <= 90 and 0 <= ny <= 90):
            return None
        t["x"], t["y"] = nx, ny
        return f"move {e['n']} {d} {amt}", new
    if kind == "color":
        c = rng.choice([c for c in COLORS if c != e["c"]])
        t["c"] = c
        return f"color {e['n']} {c}", new
    if kind == "size":
        s = rng.choice([s for s in SIZES if s != e["s"]])
        t["s"] = s
        return f"size {e['n']} {s}", new
    if len(state) < 2:
        return None
    return f"delete {e['n']}", [x for x in new if x["n"] != e["n"]]


def gen_task(rng, kind):
    """kind 为 single、seen_pair 或 held_pair。返回 dict(prompt, target)。"""
    while True:
        if kind == "single":
            ops = [rng.choice(OP_TYPES)]
        elif kind == "seen_pair":
            ops = list(rng.choice(SEEN_PAIRS))
        else:
            ops = list(rng.choice(sorted(HELD_PAIRS)))
        state0 = gen_state(rng)
        state, texts = state0, []
        for k in ops:
            r = sample_op(rng, state, k)
            if r is None:
                break
            texts.append(r[0])
            state = r[1]
        else:
            prompt = f"ui\n{render(state0)}\nedit {' ; '.join(texts)}\nout\n"
            return {"prompt": prompt, "target": render(state), "kind": kind}


def reward(task, text):
    """二元奖励，逐行完全一致才算对。"""
    return int([l.strip() for l in text.strip().split("\n")] == task["target"].split("\n"))


def build_eval_sets(n, seed=12345):
    rng = random.Random(seed)
    out = {}
    for kind in ("single", "seen_pair", "held_pair"):
        seen, items = set(), []
        while len(items) < n:
            t = gen_task(rng, kind)
            if t["prompt"] not in seen:
                seen.add(t["prompt"])
                items.append(t)
        out[kind] = items
    return out


class Stream:
    """任务流，排除给定的提示（用于 RL 提示流与评测集去重）。"""

    def __init__(self, kind, rng, exclude=()):
        self.kind, self.rng, self.exclude = kind, rng, set(exclude)

    def next(self):
        while True:
            t = gen_task(self.rng, self.kind)
            if t["prompt"] not in self.exclude:
                return t


def demo_stream(rng, exclude=()):
    """SFT 示范流，一半单操作，一半已见组合，响应是标准答案。exclude 中的提示被剔除。"""
    ex = set(exclude)

    def f():
        while True:
            t = gen_task(rng, "single" if rng.random() < 0.5 else "seen_pair")
            if t["prompt"] not in ex:
                return t
    return f


def build_retention_set(n, seed=777):
    """留出的示范，用来度量 SFT 能力的保留（对应论文的留出语料损失）。"""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        t = gen_task(rng, "single" if rng.random() < 0.5 else "seen_pair")
        if t["prompt"] not in seen:
            seen.add(t["prompt"])
            out.append(t)
    return out
