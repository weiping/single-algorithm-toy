"""四个阶段，一个算法：用一份 CPU 上几分钟能跑完的代码，演示论文的公式怎样落成程序。

预训练、中训、SFT、RL 共用同一个损失函数 loss_fn。四个阶段的区别只在三处，
喂什么数据，各占多少 token，学习率多大。数据来源与配比写在文件末尾的 PLANS 里，
分阶段训练与统一训练是同一份代码的两份配比方案。

  python four_stages.py --plan staged     # 四个阶段依次走完，阶段边界处配比跳变
  python four_stages.py --plan unified    # 配比连续变化，来源重叠，RL 期间继续回放 SFT 数据
  python four_stages.py --self-test       # 只验证公式的程序实现，几秒

公式与代码的对应关系
  式 (1)  样本是 (z, w)          Sample，lm_sample / sft_sample / rl_sample
  式 (2)  加权似然损失           loss_fn
  式 (3)  MaxRL 权重 N/k         maxrl_weights
  式 (4)  混合速率 ∝ f_s / T_s   rates_from_fractions
  式 (5)  累计取整的混入         Accountant
"""
import argparse
import random
import time

import torch
import torch.nn.functional as F

import task
from model import TinyGPT


# ---------------- 式 (1)，样本 (z, w) ----------------
class Sample:
    """ids 是 token 序列 z，w 是逐 token 的权重，counts 标出哪些 token 计入损失分母。"""
    __slots__ = ("ids", "w", "counts")

    def __init__(self, ids, w, counts):
        self.ids, self.w, self.counts = ids, w, counts

    def loss_tokens(self):
        return sum(self.counts[1:])           # 位置 0 没有前文，不被预测


def lm_sample(ids):
    """预训练与中训。每个 token 权重 1，提示和回答不作区分。"""
    return Sample(ids, [1.0] * len(ids), [True] * len(ids))


def sft_sample(prompt, response):
    """SFT。提示部分权重 0，只学怎么回答，不学怎么提问。"""
    n = len(prompt)
    return Sample(prompt + response, [0.0] * n + [1.0] * len(response),
                  [False] * n + [True] * len(response))


def rl_sample(prompt, generation, weight):
    """RL。提示部分权重 0，回答部分乘上 MaxRL 权重。权重为 0 的回答 token 仍计入分母。"""
    n = len(prompt)
    return Sample(prompt + generation, [0.0] * n + [weight] * len(generation),
                  [False] * n + [True] * len(generation))


# ---------------- 式 (2)，一个损失处理所有来源 ----------------
def loss_fn(model, batch):
    """L = -(1/T_B) Σ_样本 Σ_位置 w · log π(z_ℓ | z_<ℓ)，T_B 为批内计入分母的 token 总数。"""
    L = max(len(s.ids) for s in batch)
    ids = torch.full((len(batch), L), task.PAD, dtype=torch.long)
    w = torch.zeros(len(batch), L)
    counts = torch.zeros(len(batch), L)
    for i, s in enumerate(batch):
        n = len(s.ids)
        ids[i, :n] = torch.tensor(s.ids)
        w[i, :n] = torch.tensor(s.w)
        counts[i, :n] = torch.tensor([float(c) for c in s.counts])
    logp = F.log_softmax(model(ids)[:, :-1], dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (nll * w[:, 1:]).sum() / counts[:, 1:].sum().clamp(min=1.0)


# ---------------- 式 (3)，MaxRL 权重 ----------------
def maxrl_weights(correct):
    """一道题采样 N 条，答对 k 条。正确回答权重 1/p̂ = N/k，错误回答 0，全错整体记零。"""
    n, k = len(correct), sum(correct)
    if k == 0:
        return [0.0] * n
    return [n / k if c else 0.0 for c in correct]


# ---------------- 式 (4)(5)，按 token 占比混合 ----------------
def rates_from_fractions(fracs, mean_len):
    """要让来源 s 占 token 的 f_s，按条数抽样的速率要正比于 f_s / T_s，T_s 是它每个样本的平均损失 token 数。"""
    r = {s: f / mean_len[s] for s, f in fracs.items() if f > 0}
    z = sum(r.values())
    return {s: v / z for s, v in r.items()}


class Accountant:
    """RL 步骤里，示范数据的 token 数要等于 f/(1-f) 乘以 RL 回答的 token 数。
    按累计取整，只补欠账，所以占比很小、每步不足一条时，实测占比也逼近目标。"""

    def __init__(self):
        self.want, self.got = 0.0, 0

    def top_up(self, ratio, anchor_tokens, draw):
        self.want += ratio * anchor_tokens
        out = []
        while self.got < self.want:
            s = draw()
            out.append(s)
            self.got += s.loss_tokens()
        return out


# ---------------- 数据来源，四个阶段各用一种 ----------------
class Sources:
    def __init__(self, seed, eval_sets):
        self.rng = random.Random(seed)
        exclude = [task.key(e) for e in eval_sets["A"]]
        self.rl_stream = task.RLStream("A", exclude, random.Random(seed + 1))
        self.sources = {
            "web": lambda: lm_sample(task.stmt_tokens(self.rng.choice(task.TRAIN_STMTS))),   # 预训练语料
            "mid": lambda: lm_sample(sum(task.make_demo(task.gen_atomic(self.rng)), [])),    # 中训，原子操作的解题文本，全部 token 学
            "sft": lambda: sft_sample(*task.make_demo(task.gen_composed(self.rng, *self.rng.choice(task.SEEN_COMBOS)))),  # SFT，已见组合，只学回答
        }
        self.mean_len = {k: sum(f().loss_tokens() for _ in range(300)) / 300 for k, f in self.sources.items()}


@torch.no_grad()
def evaluate(model, eval_sets, k=8):
    def pass_at(exprs):
        prompts = [[task.BOS] + task.tokenize_expr(e) for e in exprs]
        gens = model.generate([p for p in prompts for _ in range(k)], max_new=8, eos=task.EOS)
        ok = [[int(task.is_correct(g, task.expr_value(e))) for g in gens[i * k:(i + 1) * k]]
              for i, e in enumerate(exprs)]
        return sum(map(sum, ok)) / (len(ok) * k), sum(any(r) for r in ok) / len(ok)
    stmts = [lm_sample(task.stmt_tokens(s)) for s in task.HELD_STMTS]
    nll = loss_fn(model, stmts).item()
    return {"heldout_loss": nll, "A": pass_at(eval_sets["A"]), "base": pass_at(eval_sets["base"])}


# ---------------- 训练主循环，四个阶段都走这一个循环 ----------------
def run(plan, args):
    torch.manual_seed(args.seed)
    model = TinyGPT(task.VOCAB)
    eval_sets = task.build_eval_sets(args.eval_n)
    src = Sources(args.seed, eval_sets)
    mix_rng = random.Random(args.seed + 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for ph in plan:
        for g in opt.param_groups:
            g["lr"] = ph["lr"]
        realized = {}
        t0, first, last = time.time(), None, None
        acct = Accountant()
        fr = ph["fracs"]
        for step in range(ph["steps"]):
            if "rl" in fr:                                         # RL 步骤，先做 rollout
                prompts_e = [src.rl_stream.next() for _ in range(args.rl_prompts)]
                prompts = [[task.BOS] + task.tokenize_expr(e) for e in prompts_e]
                gens = model.generate([p for p in prompts for _ in range(args.rl_n)], max_new=8, eos=task.EOS)
                batch = []
                for i, e in enumerate(prompts_e):
                    group = gens[i * args.rl_n:(i + 1) * args.rl_n]
                    corr = [int(task.is_correct(g, task.expr_value(e))) for g in group]
                    for g, a in zip(group, maxrl_weights(corr)):
                        batch.append(rl_sample(prompts[i], g, a))
                rl_tok = sum(s.loss_tokens() for s in batch)
                realized["rl"] = realized.get("rl", 0) + rl_tok
                for name, f in fr.items():                         # 其余来源补到各自的 token 占比
                    if name == "rl":
                        continue
                    extra = acct.top_up(f / fr["rl"], rl_tok, src.sources[name])
                    batch += extra
                    realized[name] = realized.get(name, 0) + sum(s.loss_tokens() for s in extra)
            else:                                                   # 非 RL 步骤，按 f_s/T_s 抽样
                rates = rates_from_fractions(fr, src.mean_len)
                names = list(rates)
                batch = [src.sources[n]() for n in
                         mix_rng.choices(names, weights=[rates[n] for n in names], k=args.batch)]
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
        print(f"[{ph['name']}] {ph['steps']} 步，学习率 {ph['lr']:g}，损失 {first:.3f} -> {last:.3f}，"
              f"用时 {time.time() - t0:.0f}s", flush=True)
        r = evaluate(model, eval_sets)
        print(f"    评测  留出语句损失 {r['heldout_loss']:.3f}  留出组合 pass@1 {r['A'][0]:.3f} pass@8 {r['A'][1]:.3f}"
              f"  Base pass@1 {r['base'][0]:.3f}", flush=True)


# ---------------- 两份配比方案 ----------------
PLANS = {
    # 分阶段，每个阶段只有一个来源，阶段边界处配比跳变
    "staged": [
        dict(name="预训练", steps=300, lr=1e-3, fracs={"web": 1.0}),
        dict(name="中训", steps=100, lr=3e-4, fracs={"mid": 1.0}),
        dict(name="SFT", steps=100, lr=3e-4, fracs={"sft": 1.0}),
        dict(name="RL", steps=100, lr=5e-5, fracs={"rl": 1.0}),
    ],
    # 统一，同一个循环，来源重叠。预训练后段混入中训与 SFT 数据，RL 期间继续回放 15% 的 SFT 数据
    "unified": [
        dict(name="预训练", steps=150, lr=1e-3, fracs={"web": 1.0}),
        dict(name="预训练+中训+SFT 混合", steps=350, lr=1e-3, fracs={"web": 0.5, "mid": 0.25, "sft": 0.25}),
        dict(name="RL+SFT 回放", steps=100, lr=5e-5, fracs={"rl": 0.85, "sft": 0.15}),
    ],
}


def self_test():
    w = maxrl_weights([1, 0, 1, 0, 0, 0, 1, 0])
    assert abs(sum(w) - 8) < 1e-9 and w[0] == 8 / 3 and w[1] == 0
    assert maxrl_weights([0] * 8) == [0.0] * 8
    ex = sft_sample([1, 5, 6], [7, 8, 2])
    assert ex.w == [0, 0, 0, 1, 1, 1] and ex.loss_tokens() == 3
    model = TinyGPT(task.VOCAB)
    good, bad = rl_sample([1, 5], [7, 2], 2.0), rl_sample([1, 5], [7, 2], 0.0)
    l1, l2 = loss_fn(model, [good]).item(), loss_fn(model, [good, bad]).item()
    assert abs(l2 - l1 / 2) < 1e-5, "权重为零的回答 token 也要计入分母"
    mean_len = {"a": 4.0, "b": 20.0}
    p = rates_from_fractions({"a": 0.7, "b": 0.3}, mean_len)
    assert abs(p["a"] * 4 / (p["a"] * 4 + p["b"] * 20) - 0.7) < 1e-9
    acct, got, rl = Accountant(), 0, 0
    f = 0.005
    for _ in range(400):
        rl += 4500
        got += sum(s.loss_tokens() for s in acct.top_up(f / (1 - f), 4500, lambda: lm_sample([1] * 56)))
    assert abs(got / (got + rl) - f) < 5e-4
    print("自测通过。MaxRL 权重 N/k 与全错记零，SFT 提示掩码，零权重回答计入分母，f/T 混合速率，累计取整的混入占比")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS), default="staged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rl-prompts", type=int, default=24)
    ap.add_argument("--rl-n", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test() if a.self_test else run(PLANS[a.plan], a)
