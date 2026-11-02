"""基准测试。对一个检查点在若干评测集上报告准确率、覆盖率和保留性，带置信区间。

指标与论文一致。每题独立采样 n 条（温度 1，不截断），pass@1 取 c/n 的均值，
pass@k 用无偏估计（Chen et al., 2021）。论文每题采 64 条，报 pass@1 与 pass@16。
保留性用留出示范上每个回答 token 的负对数似然，对应论文的留出语料损失。
置信区间是按题重采样的 95% 区间，题目少时区间会很宽，报告它才不会把噪声当成差距。

用法
  python benchmark.py --tiny --init sft.pt --eval-samples 64 --eval-k 16
  python benchmark.py --model <Qwen 路径> --data-dir my_data --verifier json_equal --init rl.pt ...
"""
import argparse
import json
import random

import torch

import hf_unified as H
import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--eos-token", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--verifier", default="lines_equal")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--eval-k", type=int, default=16)
    ap.add_argument("--retention-n", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.tiny or args.model
    torch.manual_seed(args.seed)
    tok = H.make_tok(args)
    model = H.build_model(args, tok)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.to(args.device)
    data = H.setup_data(args, random.Random(1), random.Random(2))
    r = H.evaluate(model, tok, data["sets"], args.eval_samples, args.eval_k, args.max_new, data["verify"])
    n, k = args.eval_samples, args.eval_k
    print(f"\n每题采样 {n} 条，pass@{k} 为无偏估计，括号内为按题重采样的 95% 置信区间。\n")
    print(f"| 评测集 | 题数 | 准确率 pass@1 | 覆盖率 pass@{k} |")
    print("|---|---|---|---|")
    out = {}
    for name, v in r.items():
        p1 = [c / n for c in v["c"]]
        pk = [metrics.pass_at_k(n, c, k) for c in v["c"]]
        (a1, b1), (ak, bk) = metrics.bootstrap_ci(p1), metrics.bootstrap_ci(pk)
        print(f"| {name} | {len(p1)} | {v['pass1']:.3f} ({a1:.3f}, {b1:.3f}) | {v['passk']:.3f} ({ak:.3f}, {bk:.3f}) |")
        out[name] = {"questions": len(p1), "pass1": v["pass1"], "pass1_ci": [a1, b1],
                     f"pass{k}": v["passk"], f"pass{k}_ci": [ak, bk]}
    if data["retention"]:
        out["_retention_loss"] = H.retention_loss(model, tok, data["retention"])
        print(f"\n留出示范损失（每个回答 token 的 NLL，越低保留越好）{out['_retention_loss']:.3f}，"
              f"共 {len(data['retention'])} 条")
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
