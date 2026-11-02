"""示例的自检（不训练到收敛，几十秒）。python test_ui.py"""
import random

import torch

import hf_unified as H
import ui_task as U


def test_verifier():
    rng = random.Random(0)
    for kind in ("single", "seen_pair", "held_pair"):
        t = U.gen_task(rng, kind)
        assert U.reward(t, t["target"]) == 1
        assert U.reward(t, t["target"] + "\nextra") == 0
        assert U.reward(t, "") == 0
    assert not (set(U.SEEN_PAIRS) & U.HELD_PAIRS) and len(U.SEEN_PAIRS) + len(U.HELD_PAIRS) == 16
    print("ok  验证器：标准答案判对，多余行或空输出判错；已见组合与留出组合不相交")


def test_data_hygiene():
    sets = U.build_eval_sets(48)
    held = {t["prompt"] for t in sets["held_pair"]}
    stream = U.Stream("held_pair", random.Random(1), held)
    drawn = {stream.next()["prompt"] for _ in range(2000)}
    assert not (drawn & held)
    demo = U.demo_stream(random.Random(2))
    for _ in range(500):
        t = demo()
        assert t["kind"] in ("single", "seen_pair")        # SFT 示范里没有留出组合
    print("ok  数据：RL 提示流与评测集去重，SFT 示范不含留出组合")


def test_loss_matches_manual():
    tok = H.CharTokenizer()
    class Args: tiny = True
    model = H.build_model(Args, tok)
    t = U.gen_task(random.Random(3), "single")
    ex = H.sft_example(tok, t)
    got = H.weighted_loss(model, [ex], tok.pad_id).item()
    ids = torch.tensor([ex.ids])
    logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], -1)
    nll = -logp[range(len(ex.ids) - 1), ids[0, 1:]]
    resp = torch.tensor(ex.resp[1:], dtype=torch.float)
    manual = (nll * resp).sum() / resp.sum()
    assert abs(got - manual.item()) < 1e-4
    # 全错回答（权重 0）的 token 仍计入分母
    p = H.prompt_ids(tok, t["prompt"])
    good, bad = H.rl_example(p, [5, 6, tok.eos_id], 2.0), H.rl_example(p, [5, 6, tok.eos_id], 0.0)
    l1 = H.weighted_loss(model, [good], tok.pad_id).item()
    l2 = H.weighted_loss(model, [good, bad], tok.pad_id).item()
    assert abs(l2 - l1 / 2) < 1e-4
    print("ok  损失：与手算的加权 NLL 一致；权重为零的回答 token 仍计入分母")


def _args(**kw):
    from types import SimpleNamespace
    d = dict(tiny=True, model=None, device="cpu", dtype="float32", seed=0, steps_scale=1.0, batch=8,
             rl_prompts=8, rl_n=8, sync_every=2, max_new=64, eval_n=4, eval_samples=2, eval_k=2,
             retention_n=4, log_every=10 ** 9, init=None, save=None, weight_decay=0.0, data_dir=None,
             verifier="lines_equal", chat_template=False, eos_token=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_pass_at_k():
    import metrics
    assert abs(metrics.pass_at_k(5, 2, 1) - 0.4) < 1e-12
    assert metrics.pass_at_k(5, 2, 5) == 1.0 and metrics.pass_at_k(5, 0, 3) == 0.0
    assert abs(metrics.pass_at_k(4, 1, 2) - 0.5) < 1e-12
    print("ok  指标：无偏 pass@k 与手算一致")


def test_mix_accountant():
    """小比例（0.5%）下，每步不足一条示范，累计取整后实测占比仍逼近目标。"""
    from unified import lm_example
    acct = H.MixAccountant()
    ex = lambda: lm_example([1] * 56)                    # 55 个损失 token
    f = 0.005
    got, rl = 0, 0
    for _ in range(400):
        rl += 4500
        got += sum(55 for _ in acct.add(f / (1 - f), 4500, ex))
    frac = got / (got + rl)
    assert abs(frac - f) < 0.0005, frac
    print(f"ok  混入记账：目标占比 0.005，每步约 0.4 条示范，实测累计占比 {frac:.4f}")


def test_verifiers_and_jsonl(tmp="/tmp/_ui_jsonl_test"):
    import data_jsonl as D
    import make_jsonl_demo
    t = {"prompt": "x", "target": '{"a": 1, "b": [1, 2]}'}
    assert D.json_equal(t, '{"b": [1, 2], "a": 1}') == 1 and D.json_equal(t, "not json") == 0
    assert D.exact({"target": " a "}, "a\n") == 1 and D.lines_equal({"target": "a\nb"}, " a \n b") == 1
    make_jsonl_demo.main(tmp, n_sft=300, n_rl=120)
    d = D.load_dir(tmp)
    assert set(d["eval"]) == {"held", "seen", "single"} and len(d["retention"]) == 64
    spec = {"total_steps": 4, "weight": {"sft": [[0, 1], [2, 1], [2, 0.15]], "rl": [[0, 0], [2, 0], [2, 0.85]]},
            "lr": [[0, 1e-3]], "eval_at": [4], "eval_names": ["end"]}
    log = H.run(spec, _args(data_dir=tmp, eval_samples=2, eval_k=2))
    assert "_retention_loss" in log[0][2] and set(log[0][2]) >= {"held", "seen", "single"}
    print("ok  数据：JSONL 目录可加载，验证器行为正确，--data-dir 路径可训练并评测")


def test_pipeline_fractions():
    import json
    spec = json.load(open("configs/rl_mixed.json"))
    spec["total_steps"], spec["eval_at"] = 6, [0]
    trace = []
    H.run(spec, _args(), trace)
    tok = sum(sum(r["tok"].values()) for r in trace)
    real = sum(r["tok"].get("sft", 0) for r in trace) / tok
    assert abs(real - 0.15) < 0.03, real
    for r in trace:
        assert all(w == 0.0 or abs(w - 8) < 1e-9 for w in r["weight_sums"])
    print(f"ok  训练循环：RL 期间示范 token 占比目标 0.15，实测 {real:.3f}；每题权重和为 0 或 N")


def test_progress_schedules():
    import json
    for name, f0 in (("full_anneal", 0.01), ("growth", 0.0), ("half_anneal", 0.02), ("0p5_const", 0.005)):
        spec = json.load(open(f"configs/rl_sched_{name}.json"))
        c = H.Curve(spec["mix_progress"])
        assert abs(c(0) - f0) < 1e-12
        avg = sum(c(u / 1000) for u in range(1000)) / 1000
        assert abs(avg - 0.005) < 2e-4, (name, avg)
    print("ok  调度：论文四种混入调度（恒定、全程退火、线性增长、半程退火）平均占比都是 0.5%")


if __name__ == "__main__":
    test_verifier()
    test_data_hygiene()
    test_loss_matches_manual()
    test_pass_at_k()
    test_mix_accountant()
    test_progress_schedules()
    test_pipeline_fractions()
    test_verifiers_and_jsonl()
    print("全部通过")
