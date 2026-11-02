"""在 Hugging Face 因果语言模型（含 Qwen 系列）上实现统一训练：SFT 与 RL 共用一个加权似然损失。

与仓库根目录的 pipeline.py 同一思路，模型换成 transformers 的 AutoModelForCausalLM。
- 样本是 (z, w)，提示权重 0，SFT 回答权重 1，RL 回答权重取 MaxRL 的 N/k 或 0。
- 损失除以批内非提示 token 总数，全错回答的 token 也计入分母。
- 配比由配置文件给出，可以是按阶段的曲线，也可以是论文式的 mix_progress（占训练进度的函数）。
- 混入示范数据按累计 token 取整，小比例（例如 0.5%）下实测占比也不会偏高。
- RL 阶段从新的优化器开始，生成器可用落后的权重副本（--sync-every 大于 1）。
- 数据来自内置的 UI 任务，或 --data-dir 指向的 JSONL 目录（格式见 data_jsonl.py）。

用法见 README.md。--tiny 用随机初始化的小型 Qwen2 结构加字符级分词器，无需下载权重。
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from unified import Example, loss_tokens, rl_weights   # 仓库根目录的算法模块

import data_jsonl
import metrics
import ui_task


# ---------------- 分词器 ----------------
class CharTokenizer:
    """字符级分词器，仅用于 --tiny。真实模型用 HFTok 包装 AutoTokenizer。"""
    use_chat = False

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789 \n;")
        self.itos = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.vocab_size = len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i > 2:
                out.append(self.itos[i])
        return "".join(out)

    def format_prompt(self, text):
        return text


class HFTok:
    """包装 AutoTokenizer。chat=True 时用聊天模板包装提示。
    eos_token 可指定回答的结束符，例如聊天模型的 <|im_end|>，训练与采样都以它结束。"""

    def __init__(self, tok, chat=False, eos_token=None):
        self.tok, self.use_chat = tok, chat
        self.eos_id = tok.convert_tokens_to_ids(eos_token) if eos_token else tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos_id
        self.bos_id = None if chat else tok.bos_token_id

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids):
        ids = list(ids)
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        return self.tok.decode(ids, skip_special_tokens=True)

    def format_prompt(self, text):
        if not self.use_chat:
            return text
        return self.tok.apply_chat_template([{"role": "user", "content": text}],
                                            tokenize=False, add_generation_prompt=True)


def prompt_ids(tok, text):
    ids = tok.encode(tok.format_prompt(text))
    return ([tok.bos_id] if tok.bos_id is not None else []) + ids


# ---------------- 样本与损失 ----------------
def sft_example(tok, task):
    p = prompt_ids(tok, task["prompt"])
    r = tok.encode(task["target"]) + [tok.eos_id]
    return Example(p + r, [0.0] * len(p) + [1.0] * len(r), [False] * len(p) + [True] * len(r))


def rl_example(p, gen, weight):
    return Example(p + gen, [0.0] * len(p) + [weight] * len(gen), [False] * len(p) + [True] * len(gen))


def weighted_loss(model, examples, pad_id):
    """式 (2)。分母是批内非提示 token 总数，权重为零的回答 token 也计入。"""
    dev = next(model.parameters()).device
    L = max(len(e.ids) for e in examples)
    B = len(examples)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros(B, L, dtype=torch.long)
    w = torch.zeros(B, L)
    resp = torch.zeros(B, L)
    for i, e in enumerate(examples):
        n = len(e.ids)
        ids[i, :n] = torch.tensor(e.ids)
        att[i, :n] = 1
        w[i, :n] = torch.tensor(e.w)
        resp[i, :n] = torch.tensor([float(r) for r in e.resp])
    ids, att, w, resp = ids.to(dev), att.to(dev), w.to(dev), resp.to(dev)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    nll = -F.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    denom = resp[:, 1:].sum().clamp(min=1.0)
    return (nll * w[:, 1:]).sum() / denom


@torch.no_grad()
def retention_loss(model, tok, records, chunk=32):
    """留出示范上每个回答 token 的负对数似然，度量 SFT 能力的保留。"""
    model.eval()
    tot, cnt = 0.0, 0
    for s in range(0, len(records), chunk):
        exs = [sft_example(tok, r) for r in records[s:s + chunk]]
        n = sum(loss_tokens(e) for e in exs)
        tot += weighted_loss(model, exs, tok.pad_id).item() * n
        cnt += n
    return tot / cnt


# ---------------- 采样与判分 ----------------
@torch.no_grad()
def generate(model, tok, prompts, max_new, chunk=128):
    """prompts 为 token id 列表的列表，返回生成部分（含 EOS）。左填充批量采样，
    温度 1，不做 top-k 与 top-p 截断（这样损失与计数不需要处理被截掉的 token）。"""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for s in range(0, len(prompts), chunk):
        part = prompts[s:s + chunk]
        L = max(len(p) for p in part)
        ids = torch.full((len(part), L), tok.pad_id, dtype=torch.long)
        att = torch.zeros(len(part), L, dtype=torch.long)
        for i, p in enumerate(part):
            ids[i, L - len(p):] = torch.tensor(p)
            att[i, L - len(p):] = 1
        g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True,
                           top_k=0, top_p=1.0, temperature=1.0, max_new_tokens=max_new,
                           pad_token_id=tok.pad_id, eos_token_id=tok.eos_id)[:, L:].tolist()
        for row in g:
            outs.append(row[:row.index(tok.eos_id) + 1] if tok.eos_id in row else row)
    return outs


def is_correct(tok, task, gen, verify):
    """奖励为二元。触到长度上限而没有结束符的回答记 0（论文的约定）。"""
    return bool(gen) and gen[-1] == tok.eos_id and verify(task, tok.decode(gen)) == 1


def rl_batch(gen_model, tok, tasks, n, max_new, verify):
    prompts = [prompt_ids(tok, t["prompt"]) for t in tasks]
    flat = [p for p in prompts for _ in range(n)]
    gens = generate(gen_model, tok, flat, max_new)
    examples, p_sum, weight_sums = [], 0.0, []
    for i, t in enumerate(tasks):
        group = gens[i * n:(i + 1) * n]
        corr = [int(is_correct(tok, t, g, verify)) for g in group]
        p_sum += sum(corr) / n
        ws = rl_weights(corr, "maxrl")
        weight_sums.append(sum(ws))
        for g, a in zip(group, ws):
            examples.append(rl_example(prompts[i], g, a))
    return examples, {"p_hat": p_sum / len(tasks), "weight_sums": weight_sums,
                      "gen_tokens": sum(len(p) + len(g) for p, g in zip(flat, gens))}


def evaluate(model, tok, sets, n, k, max_new, verify):
    """每题采样 n 条，pass@1 取 c/n，pass@k 用无偏估计。同时返回逐题的正确条数，供置信区间使用。"""
    out = {}
    for name, items in sets.items():
        prompts = [prompt_ids(tok, t["prompt"]) for t in items]
        gens = generate(model, tok, [p for p in prompts for _ in range(n)], max_new)
        cs = [sum(is_correct(tok, t, g, verify) for g in gens[i * n:(i + 1) * n])
              for i, t in enumerate(items)]
        out[name] = {"pass1": sum(c / n for c in cs) / len(cs),
                     "passk": sum(metrics.pass_at_k(n, c, k) for c in cs) / len(cs), "c": cs, "n": n, "k": k}
    return out


# ---------------- 配比曲线与混入记账 ----------------
class Curve:
    """分段线性曲线，横坐标相同的两点表示跳变。"""

    def __init__(self, pts):
        self.pts = sorted(pts, key=lambda p: p[0])

    def __call__(self, t):
        idx = -1
        for i, (x, _) in enumerate(self.pts):
            if x <= t:
                idx = i
        if idx < 0:
            return self.pts[0][1]
        if idx == len(self.pts) - 1:
            return self.pts[idx][1]
        (x0, y0), (x1, y1) = self.pts[idx], self.pts[idx + 1]
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


class MixAccountant:
    """按累计 token 取整地混入示范数据。目标是示范 token 数 = ratio × RL 回答 token 数，
    ratio = f/(1-f)，f 为示范 token 在二者之和中的占比。每步只补欠账，所以 f 很小
    （例如 0.5%，每步不足一条）时，实测占比也能逼近目标，不会因为每步至少加一条而偏高。"""

    def __init__(self):
        self.want, self.got = 0.0, 0

    def add(self, ratio, rl_tok, draw):
        self.want += ratio * rl_tok
        out = []
        while self.got < self.want:
            ex = draw()
            out.append(ex)
            self.got += loss_tokens(ex)
        return out


# ---------------- 数据来源 ----------------
def setup_data(args, rng_sft, rng_rl):
    if args.data_dir:
        d = data_jsonl.load_dir(args.data_dir)
        excl = [r["prompt"] for v in d["eval"].values() for r in v] + [r["prompt"] for r in d["retention"]]
        return dict(sets=d["eval"], retention=d["retention"], verify=data_jsonl.VERIFIERS[args.verifier],
                    sft_next=data_jsonl.Cycle(d["sft"], rng_sft, excl).next,
                    rl_next=data_jsonl.Cycle(d["rl"], rng_rl, excl).next)
    sets = ui_task.build_eval_sets(args.eval_n)
    retention = ui_task.build_retention_set(args.retention_n)
    stream = ui_task.Stream("held_pair", rng_rl, [t["prompt"] for t in sets["held_pair"]])
    return dict(sets=sets, retention=retention, verify=ui_task.reward,
                sft_next=ui_task.demo_stream(rng_sft, [t["prompt"] for t in retention]),
                rl_next=stream.next)


# ---------------- 训练主循环 ----------------
def build_model(args, tok):
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    if args.tiny:
        cfg = Qwen2Config(vocab_size=tok.vocab_size, hidden_size=128, intermediate_size=384,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=512, tie_word_embeddings=True,
                          bos_token_id=tok.bos_id, eos_token_id=tok.eos_id, pad_token_id=tok.pad_id)
        return Qwen2ForCausalLM(cfg)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)


def make_tok(args):
    if args.tiny:
        return CharTokenizer()
    from transformers import AutoTokenizer
    return HFTok(AutoTokenizer.from_pretrained(args.model), args.chat_template, args.eos_token)


def run(spec, args, trace=None):
    torch.manual_seed(args.seed)
    scale = args.steps_scale
    S = lambda pts: [[round(x * scale), y] for x, y in pts]
    total = round(spec["total_steps"] * scale)
    progress = "mix_progress" in spec
    if progress:       # 论文式调度，f 是示范 token 占比，自变量是训练进度 u ∈ [0, 1]
        mix_curve = Curve(spec["mix_progress"])
    else:
        cur = {n: Curve(S(v)) for n, v in spec["weight"].items()}
    lr_curve = Curve(S(spec["lr"]))
    eval_at = [round(x * scale) for x in spec["eval_at"]]
    names = spec["eval_names"]

    tok = make_tok(args)
    model = build_model(args, tok)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    rng_sft, rng_rl = random.Random(args.seed * 100 + 1), random.Random(args.seed * 100 + 2)
    data = setup_data(args, rng_sft, rng_rl)
    verify = data["verify"]
    sft_ex = lambda: sft_example(tok, data["sft_next"]())

    opt = torch.optim.AdamW(model.parameters(), lr=lr_curve(0), weight_decay=args.weight_decay)
    gen_model = model if args.sync_every == 1 else copy.deepcopy(model)
    rl_updates, last_sync, stale, log, flops = 0, 0, [], [], 0.0
    acct = MixAccountant()
    print(f"模型参数 {n_params:,}，总步数 {total}", flush=True)
    t0 = time.time()
    for t in range(total + 1):
        if t in eval_at:
            r = evaluate(model, tok, data["sets"], args.eval_samples, args.eval_k, args.max_new, verify)
            if data["retention"]:
                r["_retention_loss"] = retention_loss(model, tok, data["retention"])
            name = names[eval_at.index(t)]
            log.append((name, t, r, flops))
            print(f"  [eval {name} @{t}] " + "  ".join(
                f"{k} p@1 {v['pass1']:.3f} p@{args.eval_k} {v['passk']:.3f}" for k, v in r.items() if k[0] != "_")
                + (f"  留出示范损失 {r['_retention_loss']:.3f}" if "_retention_loss" in r else "")
                + f"  算力 {flops / 1e12:.2f}e12", flush=True)
        if t == total:
            break
        if progress:
            f_sft = min(max(mix_curve(t / total), 0.0), 0.999)
            f_rl = 1.0 - f_sft
        else:
            f_sft, f_rl = max(0.0, cur["sft"](t)), max(0.0, cur["rl"](t))
            z = f_sft + f_rl
            f_sft, f_rl = f_sft / z, f_rl / z
        for g in opt.param_groups:
            g["lr"] = lr_curve(t)
        row = {"t": t, "fr": {"sft": f_sft, "rl": f_rl}, "tok": {}}
        if f_rl > 0:
            if rl_updates % args.sync_every == 0 and gen_model is not model:
                gen_model.load_state_dict(model.state_dict())
                last_sync = rl_updates
            stale.append(rl_updates - last_sync)
            rl_updates += 1
            tasks = [data["rl_next"]() for _ in range(args.rl_prompts)]
            examples, st = rl_batch(gen_model, tok, tasks, args.rl_n, args.max_new, verify)
            flops += 2 * n_params * st["gen_tokens"]                 # 生成 2N·token
            rl_tok = sum(loss_tokens(e) for e in examples)
            row.update(p_hat=st["p_hat"], weight_sums=st["weight_sums"], stale=stale[-1])
            row["tok"]["rl"] = rl_tok
            if f_sft > 0:
                extra = acct.add(f_sft / f_rl, rl_tok, sft_ex)
                examples.extend(extra)
                row["tok"]["sft"] = sum(loss_tokens(e) for e in extra)
        else:
            examples = [sft_ex() for _ in range(args.batch)]
            row["tok"]["sft"] = sum(loss_tokens(e) for e in examples)
        model.train()
        loss = weighted_loss(model, examples, tok.pad_id)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        flops += 6 * n_params * sum(len(e.ids) for e in examples)   # 训练 6N·token
        row["loss"] = loss.item()
        if trace is not None:
            trace.append(row)
        if (t + 1) % args.log_every == 0:
            extra_s = f" 训练通过率 {row['p_hat']:.3f}" if "p_hat" in row else ""
            print(f"    step {t + 1}/{total} loss {row['loss']:.3f} 示范占比目标 {f_sft:.3f}{extra_s}", flush=True)
    print(f"  完成，用时 {time.time() - t0:.0f}s，累计示范 token 占比（示范/(示范+RL)）"
          f"{acct.got / max(1, acct.got + sum(r['tok'].get('rl', 0) for r in (trace or []))):.4f}"
          if trace else f"  完成，用时 {time.time() - t0:.0f}s", flush=True)
    if args.save:
        torch.save(model.state_dict(), args.save)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="Hugging Face 模型名或本地路径，例如你选定的 Qwen 模型")
    ap.add_argument("--tiny", action="store_true", help="随机初始化小型 Qwen2 结构，字符级分词，无需下载")
    ap.add_argument("--chat-template", action="store_true", help="用分词器的聊天模板包装提示")
    ap.add_argument("--eos-token", default=None, help="回答结束符，聊天模型常用 <|im_end|>")
    ap.add_argument("--data-dir", default=None, help="JSONL 数据目录，格式见 data_jsonl.py；缺省用内置 UI 任务")
    ap.add_argument("--verifier", default="lines_equal", choices=sorted(data_jsonl.VERIFIERS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rl-prompts", type=int, default=16)
    ap.add_argument("--rl-n", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--sync-every", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=48, help="内置 UI 任务每个评测集的题数")
    ap.add_argument("--eval-samples", type=int, default=16, help="每题评测采样条数 n")
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument("--retention-n", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--init", default=None, help="从保存的权重继续（例如 SFT 检查点）")
    ap.add_argument("--save", default=None, help="训练结束后保存权重")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.tiny or args.model, "需要 --tiny 或 --model"
    log = run(json.load(open(args.config)), args)
    if args.out:
        json.dump([{"name": n, "step": s, "result": r, "flops": f} for n, s, r, f in log],
                  open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
