"""任务与数据。

受控实验设计与论文第 5.1 节同构，规模缩到 CPU 可跑。

- 中训示范：原子操作与 10 种操作组合，解法由程序逐步求值（oracle）。
- 强化学习与评测：6 种中训从未出现的操作组合，分成 A、B 两组。A 组用于第一段 RL，
  B 组用于接在后面的第二段 RL（对应论文的持续学习实验）。
- 预训练语料：另一种表面形式的三操作数中缀算式。语句按哈希切成训练集与留出集，
  留出集的损失才真正是留出的，用来度量通用能力的保留。
- 评测题与 RL 提示流严格去重。
"""
import itertools
import random

NUM_MAX = 27
OPS = ["ADD", "SUB", "MAX", "MIN"]
SPECIALS = ["PAD", "BOS", "EOS"]
SYMBOLS = ["(", ")", "=", "+", "-"]
TOKENS = SPECIALS + SYMBOLS + OPS + [f"N{i}" for i in range(NUM_MAX + 1)]
T2I = {t: i for i, t in enumerate(TOKENS)}
PAD, BOS, EOS = T2I["PAD"], T2I["BOS"], T2I["EOS"]
EQ = T2I["="]
VOCAB = len(TOKENS)

HELD_A = {("MAX", "SUB"), ("MIN", "SUB"), ("ADD", "MIN")}
HELD_B = {("SUB", "MAX"), ("MIN", "MAX"), ("SUB", "MIN")}
HELD_OUT = HELD_A | HELD_B
SEEN_COMBOS = sorted((o, i) for o in OPS for i in OPS if (o, i) not in HELD_OUT)


def num_tok(v):
    return T2I[f"N{v}"]


# ---------------- 表达式 ----------------
# 原子表达式 (op, a, b)；组合表达式 (outer, 子表达式或数, 子表达式或数)
def expr_value(e):
    if isinstance(e, int):
        return e
    op, x, y = e
    vx, vy = expr_value(x), expr_value(y)
    return {"ADD": vx + vy, "SUB": vx - vy,
            "MAX": max(vx, vy), "MIN": min(vx, vy)}[op]


def _atomic(rng, op):
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    if op == "SUB" and a < b:
        a, b = b, a
    return (op, a, b)


def gen_atomic(rng):
    return _atomic(rng, rng.choice(OPS))


def gen_composed(rng, outer, inner):
    """外层 op 作用于「内层 op 的结果」与一个一位数；结果为负的重采样。"""
    while True:
        sub, c = _atomic(rng, inner), rng.randint(0, 9)
        cand = (outer, sub, c) if rng.random() < 0.5 else (outer, c, sub)
        if expr_value(cand) >= 0:
            return cand


def tokenize_expr(e):
    if isinstance(e, int):
        return [num_tok(e)]
    op, x, y = e
    return [T2I[op], T2I["("]] + tokenize_expr(x) + tokenize_expr(y) + [T2I[")"]]


def key(e):
    return tuple(tokenize_expr(e))


def sub_values(e):
    """从内到外的子表达式值，最后一个是整体值。"""
    if isinstance(e, int):
        return []
    _, x, y = e
    return sub_values(x) + sub_values(y) + [expr_value(e)]


def make_demo(e):
    """SFT / 中训样本。prompt = BOS + 表达式；response = 逐步求值轨迹。"""
    resp = []
    for v in sub_values(e):
        resp += [EQ, num_tok(v)]
    return [BOS] + tokenize_expr(e), resp + [EOS]


def gen_demo(rng):
    """中训示范：一半原子，一半已见组合。"""
    if rng.random() < 0.5:
        return make_demo(gen_atomic(rng))
    o, i = rng.choice(SEEN_COMBOS)
    return make_demo(gen_composed(rng, o, i))


def parse_answer(gen_ids):
    """最后一个 '=' 之后的数字；没有则 None。"""
    last = -1
    for i, t in enumerate(gen_ids):
        if t == EQ:
            last = i
    if last < 0 or last + 1 >= len(gen_ids):
        return None
    name = TOKENS[gen_ids[last + 1]]
    return int(name[1:]) if name.startswith("N") else None


def is_correct(gen_ids, target):
    """奖励：必须以 EOS 结束（触到长度上限不算对），且最终答案正确。"""
    return bool(gen_ids) and gen_ids[-1] == EOS and parse_answer(gen_ids) == target


# ---------------- 评测集与 RL 提示流 ----------------
def sample_expr(rng, kind):
    if kind == "atomic":
        return gen_atomic(rng)
    if kind == "seen":
        o, i = rng.choice(SEEN_COMBOS)
        return gen_composed(rng, o, i)
    combos = sorted(HELD_A if kind == "A" else HELD_B)
    o, i = rng.choice(combos)
    return gen_composed(rng, o, i)


def build_eval_sets(n, seed=12345):
    """返回 {'A': [...], 'B': [...], 'base': [...]}，每组内部去重。"""
    rng = random.Random(seed)

    def uniq(kinds, count):
        out, seen = [], set()
        while len(out) < count:
            e = sample_expr(rng, rng.choice(kinds))
            if key(e) not in seen:
                seen.add(key(e))
                out.append(e)
        return out
    return {"A": uniq(["A"], n), "B": uniq(["B"], n),
            "base": uniq(["atomic", "seen"], n)}


class RLStream:
    """RL 提示流：只出指定组合，且排除评测题。"""

    def __init__(self, kind, exclude_keys, rng):
        self.kind, self.exclude, self.rng = kind, set(exclude_keys), rng

    def next(self):
        while True:
            e = sample_expr(self.rng, self.kind)
            if key(e) not in self.exclude:
                return e


# ---------------- 预训练语料（中缀算式，哈希切分） ----------------
def _stmt_value(a, o1, b, o2, c):
    v1 = a + b if o1 == "+" else a - b
    if not 0 <= v1 <= NUM_MAX:
        return None
    v2 = v1 + c if o2 == "+" else v1 - c
    return v2 if 0 <= v2 <= NUM_MAX else None


def _is_held(a, o1, b, o2, c):
    return (a * 37 + b * 11 + c * 5 + (o1 == "-") * 3 + (o2 == "-") * 7) % 10 == 0


ALL_STMTS = []
for _a, _b, _c in itertools.product(range(10), repeat=3):
    for _o1, _o2 in itertools.product("+-", repeat=2):
        _r = _stmt_value(_a, _o1, _b, _o2, _c)
        if _r is not None:
            ALL_STMTS.append((_a, _o1, _b, _o2, _c, _r))
HELD_STMTS = [s for s in ALL_STMTS if _is_held(*s[:5])]
TRAIN_STMTS = [s for s in ALL_STMTS if not _is_held(*s[:5])]


def stmt_tokens(s):
    a, o1, b, o2, c, r = s
    return [BOS, num_tok(a), T2I[o1], num_tok(b), T2I[o2], num_tok(c),
            EQ, num_tok(r), EOS]
