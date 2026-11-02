# single-algorithm-toy：统一训练的玩具复现

用一个约 80 万参数的 Transformer，在 CPU 上复现 *Training Foundation Models with a Single Algorithm*（arXiv 2610.40564）的核心机制与实验结构。规模比论文小两个数量级，复现的是机制和对照设计，不是论文的数值。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python check.py                       # 几秒，验证论文的核心等式与实现约定（七条）
.venv/bin/python verify.py --steps-scale 0.25   # 两份配置各跑一遍，核对运行记录与算法约定（约三分钟）
.venv/bin/python pipeline.py --config configs/unified.json   # 单循环训练器，阶段由配置里的曲线决定
.venv/bin/python pipeline.py --config configs/staged.json    # 分阶段只是占比取 0/1 的特例
.venv/bin/python train.py --seeds 0 1 2         # 三种练法并排对照（含零混合），结果写入 results.json
.venv/bin/python train.py --summarize results.json
```

加 `--objective reinforce` 可把 RL 权重换成 REINFORCE。`--sync-every 1` 为同策略，默认 4 模拟异步生成的策略滞后。

## 论文机制与代码位置

| 论文 | 代码 |
|---|---|
| 成功条件分布恒等式 ∇log p = E_{π+}[∇log π]（3.1） | `check.py: test_likelihood_identity` |
| MaxRL 有限样本权重 1/p̂，全错记零（3.1、4.2） | `unified.py: rl_weights` |
| 有限 N 的期望为 [1-(1-p)^N]·∇log p（4.2） | `check.py: test_finite_n_expectation`（对全部结果精确枚举） |
| 样本为 (z, w)，提示掩码，统一加权似然损失（4.1） | `unified.py: Example, weighted_loss` |
| 分母为批内非提示 token 数，全错回答也计入（附录 C.3） | `check.py: test_loss_denominator` |
| 混合速率 ∝ f_s / T_s 的长度校正（4.2） | `unified.py: MixtureSampler`，`check.py: test_mixture_sampler` |
| 生成与训练解耦、策略滞后（4.3、附录 D.6） | `unified.py: rl_phase`（`--sync-every`） |
| 新组合测泛化的受控任务（5.1） | `task.py`（6 个留出组合） |
| 持续学习：先 RL(A) 再 RL(B)，B 阶段不含中训数据（5.2） | `train.py: run_arm` |
| 零混合对照（附录 D.6） | 练法 `unified_rl0` |
| 阶段只是配比曲线的特例（全文） | `pipeline.py: Curve, Schedule`，`configs/*.json` |

## 与论文的差异

生成与训练同进程，用落后的权重副本模拟异步。模型用学习式位置嵌入与 LayerNorm。算力为按 6·P·token（训练）与 2·P·token（生成）的粗略估计。RL 学习率 5e-5，高于论文的 1e-5，因为 150 次更新内要看到效应。混合比例（预训练后段 50%、RL 期间 15%）远大于论文的 0.5% 到 5%，同样为了在短训练里放大效应。

## 数据卫生

预训练语句按哈希切成训练集与留出集，损失在留出集上计算。评测题与 RL 提示流去重。三种练法共享同一随机初始化与按种子固定的数据流。

## 目录

- `unified.py` 是算法本体（样本格式、加权损失、MaxRL 权重、混合采样、策略滞后）。`pipeline.py` 是读配比曲线的单循环训练器，`configs/` 是分阶段与统一两份配置。`train.py` 是三种练法的并排对照。`check.py` 是七条自检。`task.py`、`model.py` 是任务与模型。
- `verify.py` 为运行级验证，`docs/verification.md` 为它和 `check.py` 的实际运行记录。
- `docs/blog-revised-marked.md` 为带编辑标记的博客修订稿，`docs/appendix-revised.md` 为按本代码重写的附录。

- `examples/qwen_ui/` 是在 Hugging Face 因果语言模型（含 Qwen 系列）上做 UI 编辑 SFT + RL 的示例，含可验证的任务、训练器、自检和一次小模型的运行记录，详见该目录的 README。
- `docs/appendix-qwen-ui.md` 是用 Qwen 做 UI 编辑模型后训练的独立附录，覆盖思路、管线、数据格式、算法实现、配比轨迹、调参和基准测试。
- `four_stages.py` 是最短的入口，用一份 CPU 上几分钟跑完的代码演示五个公式与四个阶段的套用，运行记录见 `docs/four_stages_runs.md`。
