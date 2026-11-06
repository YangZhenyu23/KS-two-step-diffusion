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

### 文档

- 补充仓库地址链接到 `README.md` 头部

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
