# MF-DeepParticle 仓库分析报告

- 仓库：`https://github.com/fire-lit-band/MF-DeepParticle`
- 本地位置：`C:\Users\90592\Desktop\KS two-step diffusion\MF-DeepParticle`（只读分析，未改动任何原文件，`git status` 干净）
- 分析时间：2026-11-02
- 分析方式：通读源码（约 14.6k 行 Python）+ 只读测试（不训练、不模拟、不改仓库）

---

## 0. 结论（先看这里）

1. **这不是论文的源码库，而是一个"数值复现包"**：单次 commit、2.2 MB、266 个文件，内容是代码 + 小 JSON/CSV；所有 `.npz` 数据和 `.pt` 权重被 `.gitignore` 排除，需从外部 1.03 GB 归档安装。
2. **代码本身质量相当高**（防御式校验极多：协议指纹、SHA256、OT 对偶证书、种子溯源）。我实际跑了 19 个单元测试全过（13.2 s，纯 CPU），4D 稀疏精确 OT 与 POT 稠密解对拍到 2e-16。
3. **但结构上是"两层皮"**：`paper_reproduction/` 只是一个薄壳，真正的数值内核是从 `output/`（本应是结果目录）里用 `importlib` 按绝对路径动态加载的历史脚本——`paper_reproduction/*.py` 里共有 38 处 `output/` 路径引用（含 docstring），其中 8 个 `.py` 真被当成模块加载。干净 checkout 下"冻结图/严格重放"这类命令会直接抛 `FileNotFoundError`。
4. **最关键的声明无法被外部验证**：README/VALIDATION 声称"新训练出的 60 个 checkpoint 与历史模型逐位相同"，但这需要 224 个 NPZ + 60 个 PT（合计 1.03 GB），仓库里没有，README 也没给下载地址。外部读者只能走 `--from-scratch` 重训，无法验证"逐位相同"。
5. **发现两处实质性不一致**（详见 §5）：
   - `deepparticle/particle_pipeline/physics.py` 的 3D KPP 势能符号、反应项放置、重采样间隔，与论文 KPP 数据实际使用的存档 GPU 加速器**不同**；论文链路绕开了这段代码，所以论文数字不受影响，但单独使用这套"统一库"会得到另一个物理模型。
   - `VALIDATION.json` 记录的"已验证训练内核"哈希指向 `output/.../runtime/training/core.py`，而实际被 `import` 的是 `deepparticle/training/core.py`——两者已经分叉（写入原子性差异）。
6. 对你在做的 **KS 两步扩散**：这套代码里"两步 MF"（`[1→0.5] + [0.5→0]` 两段 mean-flow）是现成可借鉴的最小实现；它自带的实验数据显示，**两步在 20 个 σ 里有 18 个优于一步（约为一步误差的 0.91–0.98 倍），但在 σ=10/20 更差（1.13/1.06 倍）**。

---

## 1. 仓库概况

| 项目 | 数值 |
|---|---|
| 文件数 / 体积 | 266 个 / 1.76 MB（git pack 468 KiB） |
| Python | 70 个文件、14,642 行 |
| JSON / CSV | 178 个（539 KB）/ 13 个（403 KB） |
| commit | 1 个：`6a93268` "Add validated source-only numerical reproduction"（2026-10-26） |
| 依赖 | `numpy==2.2.5`、`scipy==1.15.3`、`matplotlib==3.10.7`、`torch==2.8.0`、`POT==0.9.7.post1`、`psutil==7.2.0` |
| 声明环境 | Windows 10、Python 3.10.19、torch 2.8.0+cu129、RTX 5080（Linux/其他硬件"未验证"） |
| 许可 / 打包 / CI | 无 LICENSE、无 pyproject/setup.py、无 CI、无 Dockerfile |

代码洁癖指标不错：0 个 TODO/FIXME、0 个裸 `except:`、0 处硬编码本机绝对路径。

---

## 2. 目录结构与调用链

```
MF-DeepParticle/
├── README.md                 # 112 行，用法 + 物理细节 + 发布说明
├── VALIDATION.json           # 552 行，声明"source-only 全流程 79.5 分钟通过"
├── install_data.py           # 校验并安装外部 1.03 GB 数据归档
├── deepparticle/             # 7.3k 行：可独立使用的"统一"粒子管线库
│   ├── particle_pipeline/    #   physics / protocol / models / engine / geometry /
│   │                         #   data / cli / bdp / fk_batch / plotting
│   ├── training/             #   core.py（历史 notebook 内核）+ protocol.py
│   ├── non_kpp/              #   cubic/arrhenius 反应扩散：physics/experiment/sweep/ot
│   ├── data/                 #   只有 manifest.json + KPP 源码快照（NPZ 全被 gitignore）
│   └── tests/                #   19 个单元/集成测试
├── paper_reproduction/       # 4.1k 行：论文级复现 CLI（argparse 子命令）
│   ├── __main__.py           #   check/data/train/generate/evaluate/reproduce/
│   │                         #   timing/multistep/warm-start/references/figures
│   ├── experiments.py        #   ★用 importlib 从 output/ 加载"训练内核"
│   ├── ks_workflows.py       #   KS 生成+评价（含两步 MF）
│   ├── kpp_workflows.py      #   KPP 生成/评价/热启动/参考解
│   ├── cubic_experiment.py   #   cubic 训练与评价
│   ├── ks_data.py / kpp_data.py / cubic_data.py   # 从零生成物理数据
│   ├── ks.py / kpp.py / cubic.py / result_figures.py  # 21 张图
│   └── checks.py / portability.py / runtime.py / timing.py
└── output/                   # 5.2k 行 Python + 结果 JSON/CSV（★同时是代码依赖！）
    ├── mf_matched_dp_20261016/experiment.py        # 训练总控（e.dataset/e.args/sample）
    ├── dp_dplus1_four_benchmarks_20261016/…        # d+1 输入消融 + experiment.py
    ├── dp_dplus1_manuscript_20261016/measure.py    # 重测/重放/延迟/热启动（含两步采样）
    ├── ks_full_histogram_100k_bins16_20261022/     # KS 指标内核（run_evaluation + exact_sparse）
    ├── other_histogram_100k_bins16_20261023/       # KPP/cubic 指标内核（evaluate.py）
    ├── kpp_original_sampling_20261014/             # KPP 物理源：float64 GPU 加速器
    └── dp_alpha01_refinement_alpha10_20261017/     # 论文选定的 checkpoint manifest + 计时
```

**关键调用链**（以训练为例）：

```
python -m paper_reproduction train
  → experiments._training_kernel()          # 临时改 sys.path，importlib 加载
      → output/dp_dplus1_four_benchmarks_20261016/experiment.py   (注册为 sys.modules["experiment"])
          → importlib 加载 output/mf_matched_dp_20261016/experiment.py  (变量 e)
              → deepparticle/particle_pipeline/engine.py  (train_mf / train_dp)
      → output/dp_dplus1_manuscript_20261016/measure.py
  → measure.training(strict_replay=…)
```

`paper_reproduction/__init__.py` 把 `deepparticle/` 插到 `sys.path[0]`，所以 `training`、`particle_pipeline`、`non_kpp` 这些**顶层短模块名**会被解析到 `deepparticle/` 下。

---

## 3. 科学内容拆解

### 3.1 三个物理模型

| | KS（Keller–Segel） | KPP（反应-扩散 FK） | cubic（非 KPP） |
|---|---|---|---|
| 维数 | 3D / 4D | 2D / 3D | 3D |
| 粒子数 | 训练 15k、评测 10k、参考 100k | 训练/评测 20k、参考 100k | 5,000,000 |
| 方程 | `dx=σ e^{-\|y\|²}dt − (χ/N)Σ∇K_δ dt`，`dy=√(2μ)dW` | 环面 `[0,2π)^d`，漂移 + 势能加权（Feynman–Kac） | `u_t=DΔu + w·∇u + r(u)`，`r(u)=u²(1−u)` |
| 时间 | `T=0.02`，`dt=0.005`（4 步） | `T=4096` 周期 × 256 子步，`dt=2⁻⁸` | `T=20`，`dt=0.5`（40 步） |
| 核 / 参数 | `K_δ(r)=r/(4π(r²+δ²))`（软化牛顿核），`χ=16π`、`δ=0.1`、`μ=1` | `λ=2`，`s=−log₂κ`，势能 `κλ²+λv₁` | 初值半径 5 的球，`100³` 网格，`[-60,60]³` |
| 积分 | 剪切 → 相互作用（O(N²) 分块）→ 噪声，float64 | 传播 → 势能加权 → 每周期多项式重采样 | 速度 → 噪声 → 网格重构 → Newton 反应 → 重采样 |

KS4D 用的是"同一个 3D 核在 4 个坐标上求值"的**扩展**，代码里明确标注了这是 extension，不是物理上自洽的 4D 核。

### 3.2 三个生成模型

| 方法 | 结构 | 训练预算 | 采样代价 |
|---|---|---|---|
| **MF**（MeanFlow） | MLP `d+3 → 64 → 64 → 64 → d`，SELU；输入 `(z, t, r, condition)`；对 `(x,r,t)` 做 `jvp` 求 `∂u/∂t`，损失 `‖u − (v − (t−r)∂ₜu)‖²` | 1000 步 | 1 次网络前向（一步生成） |
| **DP**（Corrector） | `x + α·MLP(x, cond)`，α=0.1，零初始化输出头（初始即恒等映射） | 1000 步 | 1 次前向 + 精确 OT 匹配 |
| **MF+DP** | MF 500 步初始化 + DP 500 步精修，α=10，残差惩罚 0.1 | 1000 步 | **2 次前向** |
| **两步 MF**（本文档重点） | 同一个 MF 网络、同一权重，按 `[1→0.5]`、`[0.5→0]` 两段欧拉：`x ← x − (t−r)·u_θ(x,r,t,c)` | 不额外训练 | 2 次前向 |

- 训练目标里 DP 部分是"到 OT 匹配后的目标云"的平方距离 + 残差惩罚；OT 用 `ot.emd`（network simplex）每 500 步刷新一次匹配。
- `condition` 在 KPP/cubic 是 `protocol.condition(level)`，在 KS 是 `log10 σ`（`condition_mode="log10_sigma"`，且刻意复刻了 notebook 的 float32 舍入）。
- 全部 3 个 seed（88/89/90）× 5 个数据集，固定最终步数、无早停、无测试集调参。

### 3.3 评价指标：直方图 OT

- 100,000 粒子 → 每轴 16 箱的直方图 → 对**箱中心**做无权（按质量加权）精确二次 OT，报 `W2 = √cost`。
- KS3D/KS4D：网格锚定在**参考云**的逐坐标最小值，箱宽 = `ptp(ref)/16`，越界的生成点**不裁剪**、落在外延格点上。4D 用稀疏列生成 + 全局对偶证书（`exact_sparse.py`）。
- KPP：环面固定网格 `2π/16`，周期平方距离。
- 每次求解都校验：质量残差 `<1e-9`、原始-对偶间隙 `<1e-8`、对偶可行性 `<1e-8`，并记录 `solver_result_code`。

> ⚠️ 这是**直方图之间的 W2**，不是经验测度之间的真 W2：箱内输运免费，绝对数值随网格变化；跨论文/跨网格比较只能看相对趋势。代码注释里对此是诚实的（"finite-time, finite-particle estimate"）。

### 3.4 实验矩阵

| 数据集 | 参数点 | 协议 |
|---|---|---|
| ks3d / ks4d | σ = 10…200，20 点 | 8 个训练条件 × 15k、20 个评测 × 10k、20 个 100k 参考 |
| kpp2d / kpp3d | s = 2…4.5，11 点 | 8 × 20k 训练、11 × 20k 评测、11 × 100k 参考（T=4096） |
| cubic | D ∈ {.25,.375,.5,.75,1,1.5,2,2.5,3} | 只在 D∈{.25,.5,1,2} 训练，其余外推/内插 |
| 附加 | 两步 MF（KS3D 全部 20 个 σ）、KPP 热启动（30 组配对 FK，128 周期）、`bdp` 基线 | — |

总计：360 条 KS 指标 + 198 条 KPP + 90 条 cubic + 60 条两步 MF + 480 行 FK trace + 21 张图。

---

## 4. 代码质量：值得肯定的地方

1. **防呆做得很足**：`fresh_directory` 拒绝写入 `output/`、`deepparticle/`、`.git`；`resolve_recorded_path` 拒绝路径穿越与逃逸 symlink；输出目录必须为空。
2. **可追溯性**：每个 checkpoint 带 `protocol` 元数据 + `dataset_signature` + 原始源码 SHA256；加载时逐项比对，避免"拿旧模型冒充新模型"。
3. **OT 证书**：稀疏 4D 求解器用 KD 树/格点做 c-transform 定价 + 分量对偶修复，只在全局对偶可行时返回；我实测与 POT 稠密解相差 `2e-16`。
4. **统计口径克制**：多处明确写"3 个 seed 的 SD 不是置信区间""参数点不是独立重复""计时依赖硬件，不可用于跨机比较"。
5. **测试是"真测试"**：19 个用例里有小规模端到端训练、checkpoint 重载、FK 冷/热启动配对、并行与串行数据一致性、缓存篡改检测。

---

## 5. 问题清单（按严重性）

### P0 — 影响"能否复现结论"

| # | 问题 | 证据 |
|---|---|---|
| 1 | **数据/权重不在仓库，且无下载入口** | `.gitignore` 排除 `*.npz`/`*.pt`；`retained_data.json` 记录 468 个文件 1.03 GB，干净 clone 上 284 个缺失（224 NPZ + 60 PT）；README 只说 `install_data.py /path/to/reproduction-data-v1.zip`，没有 URL |
| 2 | **核心声明不可外部验证**："新训练参数与历史权重逐位相同" | `VALIDATION.json` 的 `new_model_parameter_tensors_bitwise_equal_to_historical: true` 只在"已装归档"的机器上成立；干净 clone 下 `--strict-replay` 直接失败 |
| 3 | 单 commit、无历史、无 LICENSE | `git log` 只有 1 条；作者/论文标题/arXiv 号都没有，只有目录名暗示 KS/KPP/cubic 三组实验 |

### P1 — 结构与一致性

| # | 问题 | 证据 |
|---|---|---|
| 4 | **把 `output/` 当代码目录**：`paper_reproduction/*.py` 中 38 处 `output/` 路径引用，其中 8 个 `.py` 被 `importlib` 按路径加载；还用 `sys.modules["experiment"] = runner` 这种别名劫持 | `paper_reproduction/experiments.py:98-122`；`measure.py:22`；`kpp_workflows.py:479` |
| 5 | **两套训练内核分叉**：`deepparticle/training/core.py`（被 import）与 `output/total_budget_refresh50_20261003/runtime/training/core.py`（VALIDATION 记录的哈希）不一致，差在写盘原子性与里程碑落盘顺序 | SHA256 `EFEFC6…` vs `FBB1B7…`；`git diff --no-index` 14 增 5 删 |
| 6 | **两套 KPP 物理约定**：`particle_pipeline/physics.py` 的 3D 势能用 `+λv₁`，存档加速器用 `−λv₁`（源码注释："3D negative potential sign is deliberate fidelity to the executed source"）；反应项一边放进权重、一边事后加 `+κ`；重采样一边每子步、一边每周期 256 步 | `physics.py:42,89`；`original_gpu_accelerator.py:65,131`；`evaluate.py:328` 里的期望值 `kappa*lambda^2 - lambda*v1` |
| 7 | 干净 clone 下 `figures --out …`（"重绘 21 张冻结图"）抛未捕获 `FileNotFoundError` | 实测：`output/dp_alpha01_refinement_alpha10_20261017/benchmarks/ks3d/evaluation/particles_15.npz` 不存在 |

> 说明：#6 不影响论文数字——`kpp_data.prepare` 走的是存档加速器，不是 `physics.kpp_fk`；受影响的是 `deepparticle` 独立使用路径和 `engine.evaluate` 的 FK 诊断（它把 `physics.kpp_fk` 的 `c_hat` 与由加速器生成的 `meta["c_hat"]` 直接比较）。

### P2 — 方法与可移植性

| # | 问题 | 证据 |
|---|---|---|
| 8 | 指标是"16 箱/轴直方图 OT"，箱内输运免费，绝对数值依赖网格 | `run_evaluation.py:139-144`；`evaluate.py:243-245` |
| 9 | KS4D 用"3D 核套 4 坐标"，非物理自洽 4D 核（已标注为 extension） | `protocol.py:117-120` |
| 10 | 只验证 Windows + CUDA；`config.get("gpu")`、`torch.cuda` 分支遍布，CPU 只能跑 `deepparticle` 单测 | `VALIDATION.json limits`；`experiments.py:142` 硬性要求 CUDA |
| 11 | 79.5 分钟/1.85 GiB 的声明绑定 RTX 5080，不可外推 | `VALIDATION.json`；本机是 RTX 4060 Laptop 8 GB |
| 12 | 文件名/目录名用未来日期（2026-10-xx），内部 JSON 带 `material_passport: origin_skill "academic-research-suite / experiment-agent"`，看得出是自动化研究 agent 产物，但仓库里没有任何生成过程说明 | `output/mf_matched_dp_20261016/experiment.py:96`；`run_evaluation.py:183` |

### P3 — 工程细节

- 无 LICENSE（README 自己也说"Choose a project license before public distribution"）。
- 无 `pyproject.toml`/`setup.py`，只能用 `python -m paper_reproduction`，不能 `pip install`。
- `requirements.txt` 只钉了版本没有 hash，torch 装法依赖用户自己看 README。
- 大量"历史遗留"变量名（`e`、`r`、`ABLATION`、`BASE`、`COMPACT`），阅读成本高。
- 文档是英文，注释密度高但偏"审计日志"风格，缺少一张架构图。

---

## 6. 我实际执行的验证（不改仓库、不跑论文实验）

| # | 操作 | 结果 |
|---|---|---|
| 1 | `python -m paper_reproduction check --source-only` | **passed**：70 个源码语法检查、物理源 SHA 校验、0 个二进制文件被读取 |
| 2 | `python -m unittest tests.test_particle_pipeline tests.test_non_kpp_histogram_ot` | **19/19 通过，13.178 s**（Python 3.13.2 / torch 2.15.0+cpu / numpy 2.5.4 / POT 0.9.7.post1）——比声明的 3.10/torch 2.8 新两个大版本，说明纯管线代码对版本漂移不敏感 |
| 3 | 加载复现层三套"内核" | 成功；打印出训练超参：`mf_steps=500, dp_steps=500, lr 1e-3/1e-4, width 64, dp_blocks 4, alpha .1, hybrid_alpha 10, residual .1, batch 2000, seeds 88/89/90` |
| 4 | 4D 稀疏精确 OT vs POT 稠密解（随机格点，64×80 质量） | 3D：差 `2.2e-16`；4D：差 `1.8e-15`；对偶间隙 ≤ `4.4e-16`，质量残差 ~`8e-17` |
| 5 | KPP 周期直方图求解器 vs 独立稠密 POT（300×260 bins，3D 环面） | `W2 = 0.8130850711` 完全一致，对偶违反 `1.1e-16` |
| 6 | KS 稠密直方图求解器 vs 独立稠密 POT（500×480） | 完全一致，对偶间隙 `8.0e-15` |
| 7 | `python -m paper_reproduction figures --out <临时目录>`（"重绘 21 张冻结图"） | **失败**：`FileNotFoundError: …/particles_15.npz`（需 1 GB 归档）；失败前已写出 3 个 PDF |
| 8 | 统计：`retained_data.json` 覆盖度 | 468 条记录中 **284 条在干净 clone 上缺失**（224 NPZ + 60 PT），合计 1.029 GB |

> 未执行：`data`/`train`/`generate`/`evaluate`/`reproduce`/`timing`/`multistep`/`warm-start`/`references`——它们需要 CUDA、100k 粒子 O(N²) 模拟或数小时级训练，且会写出 1.85 GiB 结果。

**测试环境**：我在仓库**外**新建了 `C:\Users\90592\Desktop\KS two-step diffusion\.venv-analysis`（uv + Python 3.13，699 MB）用于跑测试；仓库本身 `git status` 始终干净，没有任何 `.pyc`/缓存写入。

---

## 7. 复现成本估算（若要真跑）

| 阶段 | 内容 | 量级 |
|---|---|---|
| data | KS 100k 参考 × 20×2、KPP 4096 周期 × 100k × 22、cubic 5M 粒子 × 9 | 主要开销；声明中整条 `--from-scratch` 共 79.5 min（RTX 5080） |
| training | 5 数据集 × 3 seed × 4 方法 ≈ 60 个 checkpoint | 每 seed 约 1000+1000+500 步 |
| evaluation | 360 + 198 + 90 条精确 OT | 4D 走稀疏列生成 |
| timing | 100k 粒子 ×40 次重复 × 预热 10 次 | 数分钟 |
| 输出 | 约 1.85 GiB | — |

在你的机器上（RTX 4060 Laptop，8 GB）：KS3D 100k 的相互作用分块（chunk=256）峰值约 1.5–2 GB，内存能过，但 1.6×10¹² 量级的粒子对计算会明显更慢，预计 **数小时到十几小时**，且 `fk_batch` 会按剩余显存自适应 batch，容易 OOM 重试。**建议不要在本地跑全流程。**

---

## 8. 对"KS 两步扩散"的直接价值

### 可直接拿走的 4 块代码

| 文件 | 价值 |
|---|---|
| `deepparticle/particle_pipeline/models.py`（60 行） | 最小可读的 MeanFlow（含 `jvp` 训练所需的输入顺序 `z,t,r,cond`）与"恒等初始化"的 Corrector；`generate()` 里 MF/DP/MF+DP 三种组合的统一接口 |
| `deepparticle/particle_pipeline/engine.py:72-147` | `train_mf` 的完整实现：时间对 `(r,t)` 采样（`1/(1+exp(−N(−0.4,1)))` + 一半对角线置 `r=t`）、`jvp` 求 `∂ₜu`、`desired = v − (t−r)∂ₜu` |
| `output/dp_dplus1_manuscript_20261016/measure.py:121-139` | **两步采样**的全部逻辑：同一个网络、`(r,t)=(0.5,1)` 与 `(0,0.5)` 两次欧拉，分块 10k |
| `output/ks_full_histogram_100k_bins16_20261022/exact_sparse.py` | 稀疏精确 OT + 全局对偶证书，可直接用于高维直方图 W2（比 POT 稠密解快且可证明最优） |

### 一个可直接引用的数据点（来自仓库自带 `summary.csv`，非我计算）

KS3D，100k 粒子、16 箱/轴直方图 W2，3 seed 均值：

| σ | MF 一步 | MF 两步 | 两步/一步 |
|---|---|---|---|
| 10 | 0.3406 | 0.3853 | **1.131** |
| 20 | 0.1547 | 0.1645 | **1.064** |
| 30 | 0.1784 | 0.1625 | 0.911 |
| … | … | … | 0.91–0.98 |
| 160 | 0.4829 | 0.4415 | 0.914 |
| 200 | 0.8518 | 0.7975 | 0.936 |

**读法**：两步在 18/20 个 σ 上把误差降低约 2–9%，但在低 σ（10、20，分布最尖锐、单步误差最小）反而变差 6–13%。也就是说，"两步"不是无条件更准，收益集中在难样本区间。这与"一步 MF 在简单情形已经很准、第二步引入额外离散误差"的直觉一致。

### 不要直接依赖的部分

- `particle_pipeline` 的 KPP 物理约定（3D 势能符号、反应项、重采样间隔）与论文数据源不一致（§5 #6）。要做 KPP 相关实验，应以 `output/kpp_original_sampling_20261014/original_gpu_accelerator.py` 为准。
- 任何依赖 `output/` 绝对路径的入口（`experiments.py`、`measure.py`、`kpp_workflows.warm_start`）。
- `paper_reproduction` 的 `figures`（冻结图）与 `--strict-replay`：干净 checkout 不可用。

---

## 9. 如果继续用这个仓库，建议的三条路

1. **只想读方法 / 抄结构** → 读 §8 的 4 个文件即可，`deepparticle/tests` 是很好的可执行说明书（13 秒跑完）。
2. **想在本地跑通** → 需要外部 1 GB 归档（仓库没给地址）或 `--from-scratch`（需要 ~16 GB 显存级 GPU + 数小时 + 2 GiB 磁盘）。建议先用 `--models kpp2d` 单模型试水。
3. **想验证它的核心声明** → 只能 `--from-scratch` 重训后再与 `output/*/metrics_per_seed.csv` 对比；KS3D 的期望差 ≤ 0.0022（VALIDATION 自述，因为评测样本重新抽样），其余应完全一致。

---

## 附录 A：关键文件索引

| 文件 | 作用 | 行数 |
|---|---|---|
| `deepparticle/particle_pipeline/engine.py` | MF/DP/混合训练 + 评估主循环 | 507 |
| `deepparticle/particle_pipeline/physics.py` | KS 模拟、KPP FK、单位球初值 | 147 |
| `deepparticle/particle_pipeline/geometry.py` | 周期/欧氏代价、精确 OT、直方图 W2 | 109 |
| `deepparticle/particle_pipeline/models.py` | MeanFlow / Corrector / generate | 60 |
| `deepparticle/non_kpp/physics.py` | cubic/arrhenius 的 SGIP（PyTorch 移植） | 212 |
| `paper_reproduction/experiments.py` | 从 `output/` 加载"内核"的胶水层 | 412 |
| `paper_reproduction/ks_workflows.py` | KS 生成/评价/两步 MF | 535 |
| `output/ks_full_histogram_100k_bins16_20261022/exact_sparse.py` | 稀疏精确 OT + 对偶证书 | 241 |
| `output/mf_matched_dp_20261016/experiment.py` | 训练总控（超参、数据集、sample） | 447 |

## 附录 B：本次执行的命令

```powershell
git clone https://github.com/fire-lit-band/MF-DeepParticle.git      # 经本地代理 127.0.0.1:7897 完成
python -m paper_reproduction check --source-only
python -m unittest tests.test_particle_pipeline tests.test_non_kpp_histogram_ot -v
python -m paper_reproduction figures --out <临时目录>                 # 预期内失败（缺 NPZ）
# 另有两个只读探针脚本（放在 %TEMP%，不在仓库内）：内核加载 + OT 对拍
```
