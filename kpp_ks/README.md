# KPP 与 KS 基础项目

这里是并入本仓库的**自写基础项目**，和上面的 KS3D 全量复现不是同一条管线。

| 部分 | 目录 | 做什么 |
|---|---|---|
| KPP 2D / 3D | `two_step_2d/`、`two_step_3d/`、`KPP data/` | 论文 4.3 / 4.4 的两步扩散：参考测度、MeanFlow、DP 修正、W2 |
| KS 基础训练 | `example/` | 3D Keller–Segel 层流的数据生成和两步训练（标准 MeanFlow 或 Improved MeanFlow） |
| 文稿 | `kpp_report/`、`overleaf_kpp_4_3_4_4/`、`midterm_report.tex` | KPP 4.3/4.4 摘录和中期报告 |

上游 `paper_reproduction` 的 KS3D 结果在仓库的 `results/ks3d/` 与 `results/fullscale/`。两边的分箱、粒子数和训练预算不同，**数字不能横比**。

在 `kpp_ks/` 里运行。`two_step_2d/data_io.py` 用自身上一级目录当项目根，所以 `KPP data/` 必须和 `two_step_2d/` 平级。

## KPP

参考云不在仓库里（根目录 `.gitignore` 排除 `*.npz`）。在本地重新生成：

```powershell
cd kpp_ks
python "KPP data/kpp_common.py"
python "KPP data/kpp3d_kolmogorov.py"
```

默认 `M = 20000`、`T = 4096`、`seed = 666`，缓存落在 `kpp2d_cache/`、`kpp3d_cache/`。高分辨率 2D 参考用 `KPP data/kpp2d_gpu.py`（默认 20 万粒子，只跑给定的 log2 σ）。

训练和评测：

```powershell
python -m two_step_2d.run_2d --mode smoke --device cuda
python -m two_step_3d.run_3d --mode smoke --device cuda
python -m two_step_2d.run_2d --mode all --run-name full_2d --device cuda
python -m two_step_3d.run_3d --mode all --run-name full_3d --device cuda
```

`--mode` 可以是 `smoke`、`train-stage1`、`train-stage2`、`evaluate`、`all`。

报告用的定稿运行：

- 2D：`two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/`
- 3D：`two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/`

同目录下还有后续的 stage-1 对照（端点项、自适应权重、加宽网络）和 72-bin / 高分辨率 W2 微调。权重 `.pt`、`config.json`、损失和 `metrics.json` 都在对应 `runs/` 里。

出图：

```powershell
python kpp_report/plot_paper_figures.py
cd kpp_report
latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex
```

## KS 基础

```powershell
cd example
python quick_test.py
python generate_ks_twostep_data.py
python train_twostep.py --output-dir ks_mf
python train_twostep.py --output-dir ks_imf --imf
```

`generate_ks_twostep_data.py` 写出的 `*.npy` 同样被仓库忽略，只留在本机。

## 依赖

Python 3.10+、PyTorch（CUDA 可选）。KPP 评测用到 NumPy、Matplotlib。`example/meanflow.py` 是这里的 MeanFlow 实现；旁边的 `meanflow/` 是图像 MeanFlow 的 MIT 源码快照，KPP/KS 训练不导入它。`DP/` 是 DeepParticle 的参考实现。

## 没有放进来的东西

这些仍留在本地 `project/` 工作目录，不进本仓库：

- 粒子云 `*.npz` / `*.npy`，以及 `kpp3d_old_backup`、`remote_results`
- `runs_backup_20260829`、三个 zip、LaTeX 中间文件、`.ai/`
- 原论文 PDF
- `meanflow/assets/` 里的演示 gif
