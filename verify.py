"""运行验证。让两份配置各跑一遍（可缩短步数），把每一步的运行记录与算法的约定逐项核对。

check.py 验证的是公式与小模块，verify.py 验证的是整条训练流水线在真实运行里是否
按配置和论文的约定行事。每一项给出通过与否和实测数值，全部通过才退出码为 0。
这里不比较效果，不涉及与论文数值的对照。
"""
import argparse
import json
import platform
import sys
from types import SimpleNamespace

import torch

import task
from pipeline import Schedule, run


def windows(trace):
    """把连续且目标占比相同的步骤合成窗口。"""
    out = []
    for r in trace:
        key = tuple(sorted((k, round(v, 6)) for k, v in r["fr"].items() if v > 0))
        if out and out[-1][0] == key:
            out[-1][1].append(r)
        else:
            out.append((key, [r]))
    return out


def verify(cfg_path, args):
    spec = json.load(open(cfg_path))
    ns = SimpleNamespace(objective="maxrl", batch=64, rl_prompts=24, rl_n=8,
                         sync_every=4, eval_n=24, eval_k=4, log_every=10 ** 9,
                         steps_scale=args.steps_scale)
    trace = []
    run(spec, args.seed, ns, trace)
    checks = []

    # 1. 实测 token 占比与目标占比
    worst, detail = 0.0, []
    for key, rows in windows(trace):
        tot = sum(sum(r["tok"].values()) for r in rows)
        for name, target in key:
            real = sum(r["tok"].get(name, 0) for r in rows) / tot
            worst = max(worst, abs(real - target))
            detail.append(f"{name} 目标 {target:.2f} 实测 {real:.3f}")
    checks.append(("各配比窗口的实测 token 占比与曲线目标一致（最大偏差 ≤ 0.03）",
                   worst <= 0.03, f"最大偏差 {worst:.4f}；" + "，".join(detail[:6]) + ("…" if len(detail) > 6 else "")))

    # 2. 分阶段：每步只有一个来源
    if "staged" in cfg_path:
        multi = sum(1 for r in trace if sum(1 for v in r["tok"].values() if v > 0) != 1)
        checks.append(("分阶段配置下每一步的数据只来自一个来源", multi == 0, f"违反步数 {multi} / {len(trace)}"))
    else:
        overlap = sum(1 for r in trace if sum(1 for v in r["tok"].values() if v > 0) > 1)
        checks.append(("统一配置下存在多个来源同一步混合的步骤", overlap > 0, f"混合步数 {overlap} / {len(trace)}"))

    # 3. RL 每道题所有回答的权重之和为 0 或 N
    rl_rows = [r for r in trace if "group_weight_sums" in r]
    bad, zero, total = 0, 0, 0
    for r in rl_rows:
        for w in r["group_weight_sums"]:
            total += 1
            zero += w == 0
            if not (abs(w) < 1e-9 or abs(w - ns.rl_n) < 1e-6):
                bad += 1
    checks.append(("RL 每道题的权重之和只可能是 0（全错）或 N（有正确回答）", bad == 0 and total > 0,
                   f"共 {total} 道题，其中全错 {zero} 道，违反 {bad} 道"))

    # 4. RL 提示与评测题无重叠
    ov = sum(r["eval_overlap"] for r in rl_rows)
    checks.append(("RL 提示流中没有评测题", ov == 0, f"重叠 {ov} 道，共抽取 {len(rl_rows) * ns.rl_prompts} 道"))

    # 5. 策略滞后与同步周期一致
    R = len(rl_rows)
    expect = sum(u % ns.sync_every for u in range(R)) / R
    got = sum(r["stale"] for r in rl_rows) / R
    checks.append((f"策略滞后的均值与同步周期（每 {ns.sync_every} 次更新同步）的理论值一致",
                   abs(expect - got) < 1e-9, f"理论 {expect:.3f}，实测 {got:.3f}，RL 更新 {R} 次"))

    # 6. 预训练损失确实下降
    pre = [r["loss"] for r in trace if r["fr"].get("web", 0) > 0 and "p_hat" not in r]
    k = max(1, len(pre) // 10)
    first, last = sum(pre[:k]) / k, sum(pre[-k:]) / k
    checks.append(("预训练阶段的训练损失下降（末段均值 < 0.7 × 首段均值）", last < 0.7 * first,
                   f"首段 {first:.3f}，末段 {last:.3f}"))

    # 观察项，不判定
    obs = ""
    a_rows = [r for r in rl_rows if r["fr"].get("rl_A", 0) > 0]
    if len(a_rows) >= 8:
        q = len(a_rows) // 4
        obs = (f"RL(A) 训练通过率，首四分之一 {sum(r['p_hat'] for r in a_rows[:q]) / q:.3f}，"
               f"末四分之一 {sum(r['p_hat'] for r in a_rows[-q:]) / q:.3f}（观察项，不判定）")
    return checks, obs, len(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/staged.json", "configs/unified.json"])
    ap.add_argument("--steps-scale", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"环境 Python {platform.python_version()}，PyTorch {torch.__version__}，种子 {args.seed}，"
          f"步数缩放 {args.steps_scale}\n")
    ok_all = True
    for path in args.configs:
        checks, obs, n = verify(path, args)
        print(f"\n### {path}（{n} 步）\n")
        print("| 验证项 | 结果 | 实测 |")
        print("|---|---|---|")
        for desc, ok, detail in checks:
            ok_all &= ok
            print(f"| {desc} | {'通过' if ok else '未通过'} | {detail} |")
        if obs:
            print(f"\n{obs}")
    print("\n全部通过" if ok_all else "\n存在未通过项")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
