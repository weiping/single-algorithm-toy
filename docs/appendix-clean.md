## 附录，用一个小 Transformer 把四个阶段写成同一个循环

这个附录用约八十万参数的小 Transformer，在单核 CPU 上把预训练、中训、监督微调和强化学习写成同一个训练循环。四个阶段共用一个损失函数，差别只剩两处，样本的权重怎么给，以及每个数据来源在每一步占多大比例。代码是仓库根目录的 `four_stages.py`，配套 `task.py` 和 `model.py`，只依赖 PyTorch，仓库地址 https://github.com/weiping/single-algorithm-toy 。它的目的是让每个公式对上几行代码，不追论文的数值。

任务是组合算术。原子运算有加、减、取大、取小，组合是外层运算作用在内层结果与一个一位数上。十种组合进示范数据，六种留给强化学习和评测，用来测组合泛化。四个阶段各用一种数据。

| 阶段 | 数据 | 权重 | 学到什么 |
|---|---|---|---|
| 预训练 | 中缀算式 | 所有 token 为 1 | 通用语料，留出损失衡量遗忘 |
| 中训 | 原子运算的求值文本 | 所有 token 为 1 | 运算本身 |
| 监督微调 | 已见组合的示范 | 提示为 0，回答为 1 | 格式与组合 |
| 强化学习 | 模型对留出组合的作答 | 提示为 0，回答为 N/k 或 0 | 新组合 |

### 五处核心实现

第一，样本是一段 token 加上逐 token 的权重。三类来源只差权重的给法。

```python
def lm_sample(ids):                       # 预训练与中训
    return Sample(ids, [1.0] * len(ids), [True] * len(ids))

def sft_sample(prompt, response):         # SFT，提示权重 0
    n = len(prompt)
    return Sample(prompt + response, [0.0] * n + [1.0] * len(response),
                  [False] * n + [True] * len(response))

def rl_sample(prompt, generation, weight):  # RL，回答权重乘 MaxRL 权重
    n = len(prompt)
    return Sample(prompt + generation, [0.0] * n + [weight] * len(generation),
                  [False] * n + [True] * len(generation))
```

第二，统一损失。对每个 token 取负对数似然，乘上它的权重，全部加总，再除以批内计入损失的 token 总数。

```python
logp = F.log_softmax(model(ids)[:, :-1], dim=-1)
nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
return (nll * w[:, 1:]).sum() / counts[:, 1:].sum().clamp(min=1.0)
```

分母要多看一眼。强化学习里答错回答的权重为零，它的 token 仍然计入分母，全错的题才不会因为分母变小而放大其余样本的更新。四个阶段的批次都走这个函数。

第三，MaxRL 的权重。一道题采样若干条回答，答对的回答权重取通过率的倒数，答错的取零，全错的整组取零。这样每道题所有回答的权重之和等于采样条数，难题与易题得到同样的总权重。

```python
def maxrl_weights(correct):
    n, k = len(correct), sum(correct)
    if k == 0:
        return [0.0] * n
    return [n / k if c else 0.0 for c in correct]
```

第四，混合速率。要让某个来源占到目标的 token 比例，按条数抽样的速率要正比于目标比例除以该来源每个样本的平均损失 token 数。漏掉这个除法，一条很短的算式和一条很长的回答按条数混合，token 占比就会偏离设计的配比。

```python
def rates_from_fractions(fracs, mean_len):
    r = {s: f / mean_len[s] for s, f in fracs.items() if f > 0}
    z = sum(r.values())
    return {s: v / z for s, v in r.items()}
```

第五，强化学习步骤里混入示范数据，按累计 token 取整。目标占比很小时，每步不足一条，如果每步至少补一条，实测占比会偏高。`Accountant` 只补累计的欠账，自测要求目标 0.5% 时实测偏差不超过 0.05 个百分点。

### 两份配比方案

阶段只存在于配置里，循环本身不知道自己处在哪个阶段。分阶段是占比取 0 或 1、边界处跳变的特例，统一是占比连续变化、来源重叠的一般情形。

```python
"staged": [
    dict(name="预训练", steps=300, lr=1e-3, fracs={"web": 1.0}),
    dict(name="中训",   steps=100, lr=3e-4, fracs={"mid": 1.0}),
    dict(name="SFT",    steps=100, lr=3e-4, fracs={"sft": 1.0}),
    dict(name="RL",     steps=100, lr=5e-5, fracs={"rl": 1.0}),
],
"unified": [
    dict(name="预训练", steps=150, lr=1e-3, fracs={"web": 1.0}),
    dict(name="预训练+中训+SFT 混合", steps=350, lr=1e-3,
         fracs={"web": 0.5, "mid": 0.25, "sft": 0.25}),
    dict(name="RL+SFT 回放", steps=100, lr=5e-5, fracs={"rl": 0.85, "sft": 0.15}),
],
```

### 同一个循环

每一步先看配比里有没有强化学习。有就用当前模型做 rollout，算出 MaxRL 权重，再把其余来源补到各自的 token 占比。没有就按目标比例除以平均长度抽样。之后都是同一个损失、同一套反向传播。

```python
if "rl" in fr:
    ...                                    # rollout，maxrl_weights，rl_sample
    batch += acct.top_up(f / fr["rl"], rl_tok, src.sources[name])
else:
    rates = rates_from_fractions(fr, src.mean_len)
    batch = [src.sources[n]() for n in mix_rng.choices(names, weights=..., k=args.batch)]
loss = loss_fn(model, batch)
loss.backward(); opt.step()
```

### 运行与验证

```bash
pip install torch
python four_stages.py --self-test          # 公式的程序实现，几秒
python four_stages.py --plan staged        # 约两分钟
python four_stages.py --plan unified
```

`--self-test` 检查 MaxRL 权重之和为 N 且全错记零、SFT 的提示掩码、权重为零的回答仍计入分母、混合速率、小比例下的累计占比。仓库根目录的 `check.py` 另外对一个六结果的小策略做精确枚举，验证两件事。一是对数通过率的梯度，等于只含正确答案的分布上的似然梯度。二是采样条数有限时，这个估计的期望梯度等于对数通过率的梯度，乘上一个系数，系数是一减去该题采样全错的概率。两者都已通过。

### 结果

两份方案各跑三个种子，下表是强化学习结束时的均值（括号内为三个种子的总体标准差），原始输出在 `docs/four_stages_runs.md`。

| 指标 | 分阶段 | 统一 |
|---|---|---|
| 留出语句损失 | 5.005（0.338） | 1.364（0.040） |
| 留出组合 pass@1 | 0.265（0.008） | 0.573（0.019） |
| 留出组合 pass@8 | 0.712（0.013） | 0.868（0.042） |
| Base pass@1 | 0.180（0.032） | 0.621（0.014） |

三个种子里统一在四项指标上都更好。轨迹上最清楚的一点是，分阶段的留出语句损失在中训开始的那一刻从 1.30 升到 5.25，之后没有回来，统一路径一直停在 1.4 左右，遗忘的账在中训阶段就记下了。

这个对照能说明的有限。两份方案的总步数相同，但统一方案在整个后半程持续看到中训和 SFT 数据，累计曝光量、学习率的安排都与分阶段不同，所以它回答的是同样的步数下按曲线混合还是按阶梯切换，不能推出相同数据量下统一更好。模型极小，三个种子只够看方向，论文关于算力最优、持续学习和异步生成的结论这里都没有验证。
