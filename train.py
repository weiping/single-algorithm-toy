"""分阶段训练 vs 统一训练的玩具对照（对应论文第 5 节）。

三种练法共享同一个随机初始化与同样的数据流（按种子固定）：
  staged       预训练 -> 中训 -> RL(A 组) -> RL(B 组)，阶段之间配比突然跳变
  unified      预训练后段混入中训数据，RL(A 组) 期间继续混入，之后 RL(B 组) 不含中训数据
  unified_rl0  只在预训练后段混入，RL 期间不混（论文的 0% 对照，用来隔离 RL 阶段混合的作用）
RL(A) 之后再接 RL(B)，两段 RL 对所有练法完全相同，对应论文的持续学习实验。

用法见 README.md。
"""
import argparse
import json
import random
import statistics
import time

import torch

import task
import unified as U
from model import TinyGPT

ARMS = ["staged", "unified", "unified_rl0"]
POINTS = ["pre_rl", "post_rl", "post_cl"]


def evaluate(model, eval_sets, k):
    r = {}
    for name in ["A", "B", "base"]:
        p1, pk = U.pass_at_k(model, eval_sets[name], k)
        r[name] = {"pass1": p1, "passk": pk}
    r["loss"] = U.heldout_loss(model, task.HELD_STMTS)
    return r


def run_arm(arm, seed, cfg, eval_sets):
    t0 = time.time()
    torch.manual_seed(seed)
    model = TinyGPT(task.VOCAB)
    n_params = sum(p.numel() for p in model.parameters())
    meter = U.Meter(n_params)
    rng_pre, rng_demo, rng_mix, rng_rl, rng_cl = (
        random.Random(seed * 100 + i) for i in range(1, 6))
    demo_count = [0]

    def stmt_src():
        return U.lm_example(task.stmt_tokens(rng_pre.choice(task.TRAIN_STMTS)))

    def demo_src():
        demo_count[0] += 1
        return U.sft_example(*task.gen_demo(rng_demo))

    adam = lambda lr: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    out = {"arm": arm, "seed": seed, "params": n_params}

    if arm == "staged":
        U.likelihood_phase(model, adam(cfg.lr),
                           lambda: [stmt_src() for _ in range(cfg.batch)],
                           cfg.p_steps, meter, "pretrain")
        U.likelihood_phase(model, adam(cfg.mid_lr),
                           lambda: [demo_src() for _ in range(cfg.m_batch)],
                           cfg.m_steps, meter, "midtrain")
    else:
        opt = adam(cfg.lr)
        pure = int(cfg.p_steps * (1 - cfg.mix_portion))
        U.likelihood_phase(model, opt, lambda: [stmt_src() for _ in range(cfg.batch)],
                           pure, meter, "pretrain")
        sampler = U.MixtureSampler({"stmt": stmt_src, "demo": demo_src},
                                   {"stmt": 1 - cfg.mix_frac, "demo": cfg.mix_frac},
                                   rng_mix)
        out["mixture_check"] = sampler.empirical_token_fracs(5000)
        U.likelihood_phase(model, opt, lambda: [sampler.draw() for _ in range(cfg.batch)],
                           cfg.p_steps - pure, meter, "pretrain+mid")
    out["demos_before_rl"] = demo_count[0]

    out["pre_rl"] = evaluate(model, eval_sets, cfg.eval_k)
    out["pre_rl"]["flops"] = meter.flops
    print(f"  [{arm} s{seed}] RL 前 A p@1 {out['pre_rl']['A']['pass1']:.3f} "
          f"损失 {out['pre_rl']['loss']:.3f}", flush=True)

    # 第一段 RL（A 组）
    stream_a = task.RLStream("A", [task.key(e) for e in eval_sets["A"]], rng_rl)
    use_mix = arm == "unified"
    st = U.rl_phase(model, adam(cfg.rl_lr), cfg, stream_a, meter, cfg.rl_updates,
                    demo_src if use_mix else None, cfg.rl_mix if use_mix else 0.0, "rl-A")
    out["staleness"] = st["mean_staleness"]
    out["demos_after_rl"] = demo_count[0]
    out["post_rl"] = evaluate(model, eval_sets, cfg.eval_k)
    out["post_rl"]["flops"] = meter.flops

    # 第二段 RL（B 组），所有练法完全相同，不含中训数据
    stream_b = task.RLStream("B", [task.key(e) for e in eval_sets["B"]], rng_cl)
    U.rl_phase(model, adam(cfg.rl_lr), cfg, stream_b, meter, cfg.cl_updates,
               None, 0.0, "rl-B")
    out["post_cl"] = evaluate(model, eval_sets, cfg.eval_k)
    out["post_cl"]["flops"] = meter.flops
    out["seconds"] = round(time.time() - t0, 1)
    print(f"  [{arm} s{seed}] 完成 B p@1 {out['post_cl']['B']['pass1']:.3f} "
          f"p@{cfg.eval_k} {out['post_cl']['B']['passk']:.3f} 用时 {out['seconds']}s",
          flush=True)
    return out


def ms(vals):
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"


def summarize(results):
    runs = results["runs"]
    arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]
    k = results["config"]["eval_k"]
    n_seed = len({r["seed"] for r in runs})
    print(f"\n种子数 {n_seed}，单元格为均值±标准差（单种子时只给均值）。\n")
    cols = [("A 组 pass@1", lambda d: d["A"]["pass1"]),
            (f"A 组 pass@{k}", lambda d: d["A"]["passk"]),
            ("B 组 pass@1", lambda d: d["B"]["pass1"]),
            (f"B 组 pass@{k}", lambda d: d["B"]["passk"]),
            ("Base pass@1", lambda d: d["base"]["pass1"]),
            ("留出语句损失", lambda d: d["loss"]),
            ("算力 (1e12 FLOPs)", lambda d: d["flops"] / 1e12)]
    for point, title in zip(POINTS, ["RL(A) 之前", "RL(A) 之后", "RL(B) 之后（持续学习终点）"]):
        print(f"### {title}\n")
        print("| 练法 | " + " | ".join(c[0] for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for a in arms:
            rs = [r[point] for r in runs if r["arm"] == a]
            print(f"| {a} | " + " | ".join(ms([c[1](r) for r in rs]) for c in cols) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--objective", default="maxrl", choices=["maxrl", "reinforce"])
    ap.add_argument("--p-steps", type=int, default=400)
    ap.add_argument("--m-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--m-batch", type=int, default=32)
    ap.add_argument("--mix-portion", type=float, default=0.5, help="预训练后段混入的比例")
    ap.add_argument("--mix-frac", type=float, default=0.5, help="混入段内中训 token 占比")
    ap.add_argument("--rl-updates", type=int, default=150)
    ap.add_argument("--cl-updates", type=int, default=100)
    ap.add_argument("--rl-prompts", type=int, default=24)
    ap.add_argument("--rl-n", type=int, default=8)
    ap.add_argument("--rl-mix", type=float, default=0.15, help="RL 期间中训 token 占比")
    ap.add_argument("--sync-every", type=int, default=4, help="生成器每隔多少次更新同步权重")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mid-lr", type=float, default=3e-4)
    ap.add_argument("--rl-lr", type=float, default=5e-5)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--eval-k", type=int, default=16)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--summarize", default=None, help="只汇总已有的 results.json")
    cfg = ap.parse_args()
    if cfg.summarize:
        summarize(json.load(open(cfg.summarize)))
        return
    eval_sets = task.build_eval_sets(cfg.eval_n)
    results = {"config": vars(cfg), "runs": []}
    for seed in cfg.seeds:
        print(f"===== seed {seed} =====", flush=True)
        for arm in cfg.arms:
            results["runs"].append(run_arm(arm, seed, cfg, eval_sets))
            json.dump(results, open(cfg.out, "w"), ensure_ascii=False, indent=1)
    summarize(results)


if __name__ == "__main__":
    main()
