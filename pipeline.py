"""单循环训练器：一个循环、一个损失，阶段只是配置里的几条曲线。

对应论文的核心观点。预训练、中训、SFT、RL 不是四套算法，而是同一个加权似然损失，
区别只在每个数据源在每个时刻占多少 token。这里把「配比轨迹」落成代码：

  configs/*.json 里每个来源有一条随优化步变化的 token 占比曲线（分段线性，
  相同横坐标的两个点表示跳变），学习率也是一条曲线。
  分阶段训练就是占比只取 0 或 1、阶段边界处跳变的特例，统一训练是占比连续变化、
  多个来源长期重叠的一般情形，训练代码完全相同。

约定与限制：
- 每一步按曲线取各来源的占比并归一化。没有 RL 来源时，按速率 ∝ f_s/T_s 抽一批样本。
- 有 RL 来源时，先用（落后的）生成器副本做一批 rollout，以它的损失 token 数为基准，
  其余来源补到各自占比对应的 token 数。同一时刻最多一个 RL 来源。
- 优化器全程不重置（阶段式训练通常会重置，这是与 train.py 中 staged 的差别）。
"""
import argparse
import copy
import json
import random
import time

import torch

import task
import unified as U
from model import TinyGPT
from train import evaluate


class Curve:
    """分段线性曲线。points 为 [[step, value], ...]，相同 step 的点表示跳变。"""

    def __init__(self, points):
        self.pts = sorted(points, key=lambda p: p[0])   # 稳定排序，保留同 step 的先后顺序

    def __call__(self, t):
        pts, idx = self.pts, -1
        for i, (x, _) in enumerate(pts):
            if x <= t:
                idx = i
        if idx < 0:
            return pts[0][1]
        if idx == len(pts) - 1:
            return pts[idx][1]
        (x0, y0), (x1, y1) = pts[idx], pts[idx + 1]       # x1 > t >= x0
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


class Schedule:
    def __init__(self, spec, scale=1.0):
        s = lambda pts: [[round(x * scale), y] for x, y in pts]
        self.total = round(spec["total_steps"] * scale)
        self.kinds = {n: v["kind"] for n, v in spec["sources"].items()}
        self.prompts = {n: v.get("prompts") for n, v in spec["sources"].items()}
        self.curves = {n: Curve(s(v["weight"])) for n, v in spec["sources"].items()}
        self.lr = Curve(s(spec["lr"]))
        self.eval_at = [round(x * scale) for x in spec["eval_at"]]
        self.eval_names = spec.get("eval_names", [str(x) for x in spec["eval_at"]])

    def fractions(self, t):
        raw = {n: max(0.0, c(t)) for n, c in self.curves.items()}
        tot = sum(raw.values())
        assert tot > 0, f"第 {t} 步所有来源占比为零"
        return {n: v / tot for n, v in raw.items()}


def draw_mixed_batch(fracs, sources, T, rng, batch, tally=None):
    """没有 RL 的步骤：速率 ∝ f_s / T_s，共抽 batch 条。tally 非空时记录各来源实际抽到的损失 token 数。"""
    names = [n for n, f in fracs.items() if f > 0]
    rates = [fracs[n] / T[n] for n in names]
    out = []
    for _ in range(batch):
        s = rng.choices(names, weights=rates)[0]
        ex = sources[s]()
        if tally is not None:
            tally[s] = tally.get(s, 0) + U.loss_tokens(ex)
        out.append(ex)
    return out


def run(spec, seed, args, trace=None):
    """trace 为 list 时，每一步追加一行运行记录，供 verify.py 核对实现与目标是否一致。"""
    sched = Schedule(spec, args.steps_scale)
    torch.manual_seed(seed)
    model = TinyGPT(task.VOCAB)
    meter = U.Meter(sum(p.numel() for p in model.parameters()))
    rng_pre, rng_demo, rng_mix, rng_a, rng_b = (random.Random(seed * 100 + i)
                                                for i in range(1, 6))
    eval_sets = task.build_eval_sets(args.eval_n)
    eval_keys = {task.key(e) for k in ("A", "B", "base") for e in eval_sets[k]}
    sources, T, rl_names = {}, {}, []
    for name, kind in sched.kinds.items():
        if kind == "lm":
            sources[name] = lambda: U.lm_example(task.stmt_tokens(rng_pre.choice(task.TRAIN_STMTS)))
        elif kind == "sft":
            sources[name] = lambda: U.sft_example(*task.gen_demo(rng_demo))
        elif kind == "rl":
            which = sched.prompts[name]
            exclude = [task.key(e) for e in eval_sets[which]]
            sources[name] = task.RLStream(which, exclude, rng_a if which == "A" else rng_b)
            rl_names.append(name)
        else:
            raise ValueError(kind)
    for name in sources:
        if name not in rl_names:
            T[name] = sum(U.loss_tokens(sources[name]()) for _ in range(300)) / 300

    opt = torch.optim.AdamW(model.parameters(), lr=sched.lr(0), weight_decay=0.0)
    gen_model, rl_updates, last_sync = copy.deepcopy(model), 0, 0
    stale, log = [], []
    t0 = time.time()
    for t in range(sched.total + 1):
        if t in sched.eval_at:
            r = evaluate(model, eval_sets, args.eval_k)
            r["flops"] = meter.flops
            name = sched.eval_names[sched.eval_at.index(t)]
            log.append((name, t, r))
            print(f"  [eval {name} @step {t}] A p@1 {r['A']['pass1']:.3f} B p@1 {r['B']['pass1']:.3f} "
                  f"Base p@1 {r['base']['pass1']:.3f} 留出损失 {r['loss']:.3f}", flush=True)
        if t == sched.total:
            break
        fr = sched.fractions(t)
        for g in opt.param_groups:
            g["lr"] = sched.lr(t)
        row = {"t": t, "fr": fr, "tok": {}}
        active = [n for n in rl_names if fr.get(n, 0) > 0]
        assert len(active) <= 1, "同一时刻最多一个 RL 来源"
        if active:
            n = active[0]
            if rl_updates % args.sync_every == 0:
                gen_model.load_state_dict(model.state_dict())
                last_sync = rl_updates
            stale.append(rl_updates - last_sync)
            rl_updates += 1
            exprs = [sources[n].next() for _ in range(args.rl_prompts)]
            examples, st = U.generate_rl_batch(gen_model, exprs, args.rl_n, args.objective, meter)
            rl_tok = sum(U.loss_tokens(e) for e in examples)
            row["tok"][n] = rl_tok
            row["p_hat"] = st["p_hat"]
            row["stale"] = stale[-1]
            row["eval_overlap"] = sum(task.key(e) in eval_keys for e in exprs)
            row["group_weight_sums"] = [
                sum(max(ex.w) for ex in examples[g * args.rl_n:(g + 1) * args.rl_n])
                for g in range(len(exprs))]
            total_tok = rl_tok / fr[n]
            for s, f in fr.items():
                if s == n or f <= 0:
                    continue
                got = 0
                while got < f * total_tok:
                    ex = sources[s]()
                    examples.append(ex)
                    got += U.loss_tokens(ex)
                row["tok"][s] = got
        else:
            examples = draw_mixed_batch(fr, sources, T, rng_mix, args.batch, row["tok"])
        loss = U.step(model, opt, examples, meter)
        row["loss"] = loss
        if trace is not None:
            trace.append(row)
        if (t + 1) % args.log_every == 0:
            mix = " ".join(f"{k}:{v:.2f}" for k, v in fr.items() if v > 0)
            print(f"    step {t + 1}/{sched.total} loss {loss:.3f} lr {sched.lr(t):.0e} 占比 {mix}",
                  flush=True)
    print(f"  完成，用时 {time.time() - t0:.0f}s，RL 更新 {rl_updates} 次，"
          f"平均策略滞后 {sum(stale) / max(1, len(stale)):.2f} 次更新", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-scale", type=float, default=1.0, help="整体缩放所有步数，用于冒烟测试")
    ap.add_argument("--objective", default="maxrl", choices=["maxrl", "reinforce"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rl-prompts", type=int, default=24)
    ap.add_argument("--rl-n", type=int, default=8)
    ap.add_argument("--sync-every", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--eval-k", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    run(json.load(open(args.config)), args.seed, args)


if __name__ == "__main__":
    main()
