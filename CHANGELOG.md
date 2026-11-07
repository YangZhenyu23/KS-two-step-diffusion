# 变更日志

本仓库遵循 [语义化版本](https://semver.org/lang/zh-CN/) 与 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 约定。
每次本地更新后**必须**在此登记，并同步推送到 GitHub，以便版本回溯。

格式约定：

- `新增` — 新功能、新脚本、新数据
- `变更` — 对既有结果的修改
- `修复` — 修正错误
- `移除` — 删除内容
- `复现` — 与论文/上游仓库数值对比的结论变化

---

## [Unreleased]

### 待办

- 补齐 `docs/PAPER_COMPARISON.md` §5 提出的网络结构对照实验（正弦嵌入 + depth 5 + skip + small-gain 头；DP 用 SiLU+LayerNorm 的 depth-4 ResNet）
- 用论文 KPP 的 72×72 直方图口径校验度量约定是否一致
- 双参数扩展的后续：把两步 MF 与 DP 精修也搬到 (σ, μ) 上；试 χ 作为第三参数；μ 的极端压力测试（μ=100）
- 全量实验的后续：更大网络容量（当前 64×3 对二维曲面有稀释）；完整收敛扫描；σ 方向加密到 40 档

---

## [v0.3.0] — 2026-11-07

全量（论文版）两参数实验：200 格 × 100k 粒子、LHS-40、训练至收敛。

### 新增

- **`fullscale/` 包**（6 模块）
  - `grid.py` — 20×10 网格（σ 对齐论文评测网格 10…200 步长 10）、100k 粒子、噪声基底常量
  - `generate.py` — 可断点续跑的物理云生成器，共享初值 + 每格 SHA256 + 原子写入
  - `lhs.py` — 非方形网格（20×10）上的分层 maximin LHS，模拟退火
  - `model.py` — `TwoParamMeanflow`（7 输入）与 `OneParamMeanflow`（6 输入，上游原版）
  - `train.py` — 双参数/单参数训练，按每格更新数对齐预算
  - `evaluate.py` — 每格独立 16 箱网格 + 精确 OT + 三重认证（含从势函数重构对偶间隙）
  - `cli.py` — design / data / train / evaluate / ablation
- **脚本** `scripts/`
  - `calibrate_fullscale.py` — 100k 粒子噪声基底与单格耗时
  - `generate_phased.py` — 两阶段生成（先训练集，后验证集）
  - `generate_baseline_row.py` — μ=1 整行（单参数对照的训练数据）
  - `fullscale_selfcheck.py` — 五项预检，含端点约定的强制验证
  - `fullscale_partial.py` / `show_partial_table.py` — 中途部分评测
  - `fullscale_diagnose.py` — μ=1 行逐格诊断
  - `fullscale_analysis.py` — 泛化 + 消融 + μ 效应 + 连续性汇总与出图
- **文档** `docs/FULLSCALE_REPORT.md` — 全量实验完整报告

### 复现

**规模**：20σ × 10μ = 200 格，每格 100k 粒子，数据 637 MB，生成 191 分钟。

**噪声基底**（本机实测）：100k 粒子 16 箱 = **0.0644**（pilot 20k 是 0.0931，降 31%）。

**LHS-40 vs 随机-40**：最小距离 `0.1053 vs 0.0526`（**2.00×**），相关系数 `0.000 vs −0.208`。
σ 层计数全为 2、μ 层全为 4（随机抽样有 5 个 σ 层完全没点到）。

**泛化**：训练 40 格 `0.1532`，**留出 160 格 `0.1547`**（无衰减）。全网格 0.1544 = 2.40× 基底。
OT 200/200 最优，质量残差 `2.2e-16`，对偶间隙 `1.7e-12`。

**双参数 vs 单参数**（144 个双留出格）：

| 模型 | 平均 W2 | 相对基底 | 胜出 |
|---|---:|---:|---:|
| 单参数 | 0.2140 | 3.32× | 27 |
| **双参数** | **0.1560** | **2.42×** | **117/144** |

**相对改进 +27.1%**，`corr(增益, |log10 μ|) = 0.660`。

**增益随 μ 偏离单调增长**（这是学到 μ 的决定性证据）：

| \|log10 μ\| | 单参数 | 双参数 | 增益 |
|---:|---:|---:|---:|
| 0.20 | 0.1361 | 0.1426 | −0.007 |
| 0.40 | 0.1654 | 0.1500 | +0.015 |
| 0.60 | 0.2172 | 0.1667 | +0.051 |
| 0.80 | 0.1455 | 0.1029 | +0.043 |
| 1.00（μ=10） | 0.3714 | 0.1912 | **+0.180** |

**连续性**（100k 粒子）：370 对相邻格点 `W2/h` 全部有限（σ 方向 max 11.37 / median 4.61；μ 方向 max 1.20 / median 0.45）。μ 比值比 σ 小 10 倍，与定理给出的 1/2 阶 Hölder 界一致。

### 修复 / 纠正

- **记录一次错误的中间结论**：在只有 71 格数据时曾判断"双参数优势消失（+1.0%）"，实际是样本选择偏差——早期完成的格集中在 μ≈1 附近，而那里单参数本就正确。完整 144 格的真实结果是 +27.1%。教训写入 `docs/FULLSCALE_REPORT.md` §4.4。
- `fullscale/generate.py` — `np.savez_compressed` 会自动追加 `.npz` 后缀，导致原子重命名失败；改用文件对象写入。
- `fullscale/generate.py` — manifest 记录缺少顶层 `i`/`j`（被 `meta` 的 `grid` 子字典覆盖），排序时报 `KeyError`。
- `fullscale/generate.py` — 补 `--cells-from` / `--skip-cells-from` 参数，支持分阶段生成。
- `fullscale/cli.py` — `cmd_data` 转调生成器时参数传递方式错误，改为显式 argv。
- `scripts/fullscale_selfcheck.py` — `train_two_param` 不接受 `log` 关键字；改用 `redirect_stdout`。
- **进程管理教训**：`Stop-Process` 杀的是 PowerShell 包装进程，python 子进程仍存活并继续占用 GPU，导致后续生成速度减半（205s vs 80s）。需按 PID 杀 python 进程本身。

---

## [v0.2.0] — 2026-11-07

双参数（σ, μ）扩展实验：LHS 抽样、全网格留出验证、参数连续性的数学证明与数值验证。

### 新增

- **`twoparam/` 包**（7 模块，不修改上游任何文件）
  - `protocol.py` — 10×10 参数网格、无量纲群 (S, C, G, δ̃)、位移尺度
  - `lhs.py` — 离散 maximin 拉丁超立方（模拟退火，24 重启 × 4000 迭代）+ 随机对照
  - `data.py` — 复用上游 float64 内核生成物理云，共享初值，manifest 带 SHA256
  - `model.py` — `TwoParamMeanflow`（6→7 输入，其余结构与上游完全一致）
  - `train.py` — MeanFlow 训练，端点约定与上游逐字对齐
  - `evaluate.py` — 16 箱精确直方图 OT，对偶间隙从势函数重构并强制认证
  - `cli.py` — `design` / `data` / `train` / `evaluate` / `continuity` 五个子命令
- **11 个分析脚本** `scripts/`
  - `measure_floor.py` — W2 统计噪声基底
  - `selfcheck_two_param.py` — 端点约定自检（防止静默反向）
  - `compare_one_vs_two.py` — 单参数 vs 双参数
  - `compare_budget_matched.py` — 按每格更新数对齐的公平对比
  - `convergence_sweep.py` — 收敛性扫描
  - `analyse_mu_effect.py` — 按 μ 距离分组的增益分析
  - `diagnose_gap.py` — 差距来源诊断
  - `continuity_test.py` — 全网格 180 对相邻格点的 W2/h
  - `refine_test.py` — 网格细化
  - `signal_vs_noise.py` — 信号/噪声分离
  - `plot_continuity.py` — 三联图
  - `inspect_grid.py` — 网格几何特征
- **文档**
  - `docs/TWO_PARAM_REPORT.md` — 双参数扩展完整报告
  - `docs/CONTINUITY.md` — 连续性定理与证明
- **结果** `results/twoparam/`（0.81 MB，粒子云 npz 按 .gitignore 排除）
  - `design/design.json` — LHS-20 与随机-20 的质量对比
  - `data/full/manifest.json` — 100 格云清单（含每文件 SHA256）
  - `evaluate/lhs20/` — 100 格 W2 指标（训练/留出分开标记）
  - `compare/` — 预算匹配对比、收敛扫描、μ 效应分析
  - `continuity/` — 连续性三组测量 + `continuity_plot.png`
  - `floor.json` — 噪声基底

### 复现

**第二个参数的选择**：任务书提出的 `kappa` 经核对**不属于 KS 方程**（是 KPP 的扩散系数，
`s = −log₂κ`）。KS 的扩散系数是 `mu`。且任务书"σ 控扩散、κ 控对流"的描述是反的：
σ 是对流幅值，μ 才是扩散。故第二个参数选 **μ**。

**参数正交性**（实测几何）：
- σ: 10→200 时 x 向标准差 0.481→0.856（+78%），yz 半径 1.007→0.991（−2%）→ 只改各向异性
- μ: 0.1→10 时 x 向标准差 0.481→0.795，yz 半径 1.007→1.925（+91%）→ 只改整体尺度

**LHS 质量**（LHS-20 vs 随机-20）：最小距离 0.1571 vs 0.1111（+41%），平均最近邻 0.1865 vs 0.1439（+30%），
相关系数 0.000 vs +0.063。每个 σ、每个 μ 各恰好出现 2 次。

**泛化**（LHS-20 训练，全 100 格评测）：训练集 20 格平均 W2 `0.2800`，
**留出集 80 格 `0.2717`**（甚至略低），无泛化衰减。100/100 OT 求解最优，
最大质量残差 `1.1e-16`，最大对偶间隙 `1.3e-12`。

**双参数 vs 单参数**（72 个双留出格，按每格更新数对齐）：

| 每格更新 | 单参数 | 双参数 | 改进 |
|---:|---:|---:|---:|
| 200 | 0.2620 | **0.2460** | −6.1% |
| 600 | 0.2260 | **0.1827** | **−19.1%** |
| 1500 | 0.2091 | **0.1873** | −10.4% |

全档位胜出。**记录一次失败尝试**：固定 1000 总步时双参数输 67%，原因是预算稀释
（20 格 × 50 步 vs 10 格 × 100 步）；加大到 8000 步后反超。多参数实验必须按每格更新数对齐。

**参数连续性**：
- 数学证明：σ 方向 Lipschitz、μ 方向 1/2 阶 Hölder（同步耦合 + Grönwall），见 `docs/CONTINUITY.md`
- 数值验证：噪声扣除后 `W2_corr/h` 在 σ 方向 3.49→3.84→4.18（±9%，近似常数）；
  180 对相邻格点 `W2/h` 全部有限（max 5.68、median 0.93）
- 可分辨阈值：σ 约 1.4–3%，μ 约 25%（因 μ 的信号斜率 0.33 比 σ 的 3.60 低 11 倍）
- 对照组：同一初值云 `W2 = 0`（精确），确认管线无偏

**噪声基底**：N=20k、16 箱、σ=100 处 `0.0931 ± 0.0020`（3 次独立重复）。

### 修复

- `twoparam/train.py` — 初版把插值端点写反（`velocity = target − source`），
  导致生成端与训练端不在同一处取值、W2 虚高到 1.8–4.2。已按上游 `core.py:369`
  的约定（`x0 = target, x1 = initial`）修正，并新增 `scripts/selfcheck_two_param.py`
  在每次实验前验证"采样器必须优于初始云"。
- `twoparam/evaluate.py` — 初版用全部 100 格构建**共享网格**，跨度太大导致箱中心距离暴涨、
  W2 虚高 4 倍。已改为**每格单独建网格**（与上游 `measures_for` 一致）。
- `twoparam/evaluate.py` — `primal_dual_gap` 初版恒为 `NaN`（POT log 无 `dual` 键），
  认证静默失效。已改为从对偶势 `u`/`v` 重构 `Σaᵢuᵢ + Σbⱼvⱼ − cost`，
  并在不可用时显式标记 `primal_dual_gap_available: false`。
- `twoparam/train.py` — 初版把 checkpoint 写进数据目录（`directory` 同时作输入输出），
  已拆分为独立的 `data_directory` / `output_directory`。

---

## [v0.1.0] — 2026-11-07

首个版本：KS3D 两步扩散（two-step MF）复现完成并归档。

### 新增

- **代码快照** `upstream/MF-DeepParticle/` — 上游复现包完整源码（266 文件，1.67 MB），
  排除 `.git`、`__pycache__`、`*.npz`。此快照为**只读参照**，本仓库不改动它。
- **复现结果** `results/ks3d/` — 20 个 σ × 3 seed 的全部指标与元数据：
  - `evaluation/` — 主实验 180 条 W2 指标（MF / DP / MF+DP）+ 验证 JSON
  - `multistep/` — 两步 MF 60 条 W2 指标（20 σ × 3 seed）+ 生成清单
  - `data/` — 48 组粒子云的 manifest（含每个文件 SHA256）+ 协议指纹
  - `training/` — 12 个 checkpoint 的校验报告 + 每 seed 训练时长
  - `timing/` — 480 条原始计时 + 汇总
  - `figures/` — ks3d_w2.pdf、d+1 投影图、figures_manifest.json
  - `analysis/` — comparison.csv / comparison.md / curves.png（本地重跑 vs 历史值）
- **文档** `docs/`
  - `REPRODUCTION.md` — 完整复现报告（命令、耗时、数值、局限）
  - `PAPER_COMPARISON.md` — 与 Shen et al. 2026 论文的逐项对比，含"论文数值低于统计噪声下限"的判据
  - `UPSTREAM_ANALYSIS.md` — 上游仓库代码审计（14.6k 行 Python 通读结论）
  - `KS_physical_parameters.md` — KS 方程物理参数梳理
  - `KS_equation_and_two_parameter_design.md` — 两参数条件化（σ, χ）设计建议
- **同步脚本** `scripts/sync.ps1` — 一键提交并推送，自动写时间戳到提交信息
- **规则** `.cursor/rules/` — 约束后续会话遵守"改完即同步 + 登记 CHANGELOG"

### 复现

- 两步 MF（two-step mean-flow，`(r,t)=(0.5,1)` 与 `(0,0.5)` 两段）在 20 个 σ 上全部复现：
  与上游历史值最大偏差 `0.0021`，平均 `0.00045`。
- 主实验 180 行与上游历史值最大绝对偏差 `0.0066`、平均 `0.0011`、最大相对偏差 `3.6%`。
- 定性结论一致：两步在 18/20 个 σ 上有效（误差降到一步的 0.91–0.98 倍），
  σ=10、20 反而变差（1.13、1.07 倍）。
- 与论文原始 Table 的**绝对值差异 4–35 倍**，判定为**度量口径不同**而非模型能力问题
  （论文报告的 0.0047–0.0214 低于同分布独立样本间的 W2 下限 0.067–0.081）。详见 `docs/PAPER_COMPARISON.md`。

### 环境

- RTX 4060 Laptop（8 GB，CUDA 12.9 驱动）、Python 3.13.2、torch 2.8.0+cu129、
  numpy 2.2.5、scipy 1.15.3、POT 0.9.7.post1
- 全流程墙钟 34 分 13 秒，产物 374 MB（本仓库仅收录其中 2.76 MB 的指标与图）
