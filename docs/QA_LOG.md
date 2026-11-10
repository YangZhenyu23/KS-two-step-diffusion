# 问答记录

> 本文件记录用户提出的问题与基于已有实验内容的回答。
> 回答只依据仓库内已完成的实验与代码事实，不做新实验、不改动已有结果。

---

## Q1. 粒子云的生成原理、保存形式，以及训练/评测中如何使用

### 提问

> （1）我理解粒子云是用粒子交互法，把一个均匀分布通过满足 KS 方程对应随机微分方程的动力学性质，运输到目标分布。这个理解对不对？
> （2）生成的粒子云以什么数学形式保存、后面如何调用？
> （3）一共生成了哪些粒子云？训练数据集有哪些、分别什么格式？
> （4）训练时用了什么？评测时又用了哪些？如何调用？

---

### A1.1 对理解的第一处核对：基本正确，但有两处需澄清

**正确的部分**：

- ✅ **粒子交互法**：`simulate_original` 求解的是 $N$ 个相互作用粒子的随机微分方程。
- ✅ **从均匀分布出发**：初值在单位球内均匀抽样。实现用 $r^{1/3}$ 变换保证球内均匀（`unit_ball`，`fullscale/generate.py:56`）。
- ✅ **通过 SDE 动力学到达目标分布**：跑 4 个时间步后取终态。

**需要澄清的第一点：方程不是"KS 方程"，而是 KS 的粒子系统**

代码实际求解的（`upstream/MF-DeepParticle/deepparticle/training/protocol.py:40`）：

$$
dX_j = -\frac{\chi}{N}\sum_{i\neq j}\nabla K_\delta(X_j-X_i)\,dt
\;+\;\sigma\,g(X_j)\,dt
\;+\;\sqrt{2\mu}\,dW_j,
$$

其中 $g(x)=(e^{-y^2-z^2},0,0)$，$K_\delta(z)=\lvert z\rvert/\big(4\pi(\lvert z\rvert^2+\delta^2)\big)$。

三项分别是**趋化相互作用**（$O(N^2)$）、**剪切流**、**布朗扩散**。这是 KS 偏微分方程的粒子离散，不是 PDE 本身。

**需要澄清的第二点："运输"不是最优传输意义下的映射**

这里没有求解 Monge/Brenier 最优映射。严格说法是**推前测度**：初值分布 $\mu_0$ 被 SDE 的随机流 $\Phi$ 推前成终态分布 $\nu=\Phi_\#\mu_0$。"运输"是直觉描述，不是最优传输的术语。

（顺带：最优传输只在**评测**时出现，用于计算两团云之间的 $W_2$。）

**数值格式**：4 步 Euler–Maruyama，$dt=0.005$，$T=0.02$；`points += dt*force + sqrt(2*mu*dt)*noise`。浮点用 float64 计算、float32 存储。

---

### A1.2 粒子云的数学形式

**数学上是一组加权 Dirac 测度**：

$$
\hat\mu_N=\frac{1}{N}\sum_{i=1}^N\delta_{X_i},\qquad X_i\in\mathbb{R}^3,\ N=10^5 .
$$

所有粒子**等权** $1/N$，没有额外权重。

**磁盘上**每格一个 `.npz`（`np.savez_compressed`），含 **4 个字段**：

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `meta` | 标量字符串 | — | JSON：σ、μ、粒子数、噪声种子、初值 SHA256、耗时、无量纲数 |
| `initial` | (100000, 3) | float32 | 单位球初值云 |
| `source` | (100000, 3) | float32 | **与 `initial` 逐元素相同** |
| `reference` | (100000, 3) | float32 | 物理真值（终态云）|

`initial` 与 `source` 内容重复，是刻意的：为了让文件与上游 `ks_cloud_v1` 格式兼容。上游协议中 MeanFlow 总从新抽的单位球出发，故"待输运的云"即初值本身。

**路径布局**：

```
results/fullscale/data/full/
├── initial.npy                共享初值 (100000,3)，独立一份
├── manifest.json              200 格清单，每格含 file / sha256 / bytes / 参数 / 耗时
└── clouds/
    └── s{ii}_m{jj}.npz        共 200 个，ii=σ索引, jj=μ索引
```

命名示例：`s09_m05` 对应 $\sigma=100$（第 10 个 σ）、$\mu=1.0$（第 6 个 μ）。

---

### A1.3 生成了哪些粒子云

#### 主数据集（200 格，用于全部主结果）

| 项 | 数值 |
|---|---|
| 格点数 | $20\times10=200$ |
| 每格粒子 | $10^5$ |
| 总坐标数 | $200\times10^5\times3=6000$ 万 |
| 磁盘 | 637 MB |
| 生成耗时 | 191 分钟（单格约 80 秒）|

**σ 网格**（20 个，等距，对齐上游评测网格）：
$10,20,30,\dots,200$

**μ 网格**（10 个，几何级数跨两个数量级，$\mu=1$ 居中）：
$0.1,\,0.1585,\,0.2512,\,0.3981,\,0.6310,\,1.0,\,1.585,\,2.512,\,3.981,\,10$

#### 底噪用的额外云（未入库）

| 用途 | 数量 | 是否保存云 |
|---|---|---|
| 直方图口径底噪（`local_floor`）| 5 点 × 2–3 次 ≈ 15 条 | ❌ 只存 JSON 结果 |
| 粒子级口径底噪（`particle_metric`）| 5 点 × 3 次 = 18 条 | ❌ 只存 JSON 结果 |

**这是一个已知疏漏**：底噪实验的云未保存，导致后续若要改用其他子样本量或扩展参数点，必须重新模拟（每点约 80 秒）。

#### 早期 pilot 数据（另一套，与主结果无关）

`results/twoparam/data/full/` — 20k 粒子 × 100 格，pilot 阶段使用。

---

### A1.4 训练时使用了什么

#### 加载代码（`fullscale/train.py` 的 `load_training_tensors`）

```python
initial = None
targets = []
for cell in cells:                              # 40 个 LHS 选出的格点
    payload = load_cell(cell)
    start  = payload["clouds"]["initial"]       # 只读 initial
    target = payload["clouds"]["reference"]     # 只读 reference
    if initial is None:
        initial = start
    elif not np.array_equal(initial, start):
        raise ValueError("cells do not share one initial cloud")   # 强制校验共享初值
    targets.append(target)

return (torch.from_numpy(initial).to(device),                   # (100000, 3)
        torch.from_numpy(np.stack(targets)).to(device))         # (40, 100000, 3)
```

**训练只使用 `initial` 与 `reference` 两个字段。**

`source` 在训练中**完全未使用**。

#### 显存布局

| 张量 | 形状 | 大小 |
|---|---|---|
| `initial` | (100000, 3) | 1.2 MB |
| `targets` | (40, 100000, 3) | 48 MB |

合计约 96 MB（含副本），全部常驻显存。

#### 每步取数

```python
env = torch.randint(len(cells), (1,)).expand(batch_size)   # 随机选一格
idx = torch.randint(N, (batch_size,))                      # 随机选粒子
x_target = targets[env, idx]     # (4096, 3)
x_source = initial[idx]          # (4096, 3)  ← 对任何 env 都相同
```

**关键点**：`initial[idx]` 与 `env` 无关，正是"共享初值"设计在训练中的体现。

#### 训练目标（与云的格式无关）

$$
\mathcal{L}=\mathbb{E}\Big\|u_\theta(z,r,t,c)-\big(v-(t-r)\tfrac{d}{dt}u_\theta(z,r,t,c)\big)\Big\|^2,
$$

其中 $v=x_{\text{source}}-x_{\text{target}}$，$z=(1-t)x_{\text{target}}+t\,x_{\text{source}}$，$c=(\log_{10}\sigma,\log_{10}\mu)$。

**逐粒子的均方误差，不涉及任何直方图或 OT。**

---

### A1.5 评测时使用了什么

#### 主实验（143 个留出格）

```python
payload   = load_cell(cell)
source    = payload["clouds"]["source"]       # 推理输入
reference = payload["clouds"]["reference"]    # 真值

prediction = sample_two_param(model, torch.from_numpy(source), cell.sigma, cell.mu)
#            ↑ 一次前向：x_gen = x − u(x, r=0, t=1, cond)

w2 = score_pair(reference, prediction, bins=16)     # 直方图口径
# 或 exact_w2(prediction[idx], reference[idx])      # 粒子级口径（3000 点子样本）
```

因为 `source ≡ initial`，**200 个格点推理时喂的是同一团初值云**，仅条件 $(\sigma,\mu)$ 不同。

#### 三种评测的字段使用对照

| 评测类型 | 读 `source` | 读 `reference` | 是否需新模拟 |
|---|---|---|---|
| 主实验（模型 vs 真值）| ✅ 作推理输入 | ✅ 作真值 | ❌ |
| 底噪（真值 vs 真值）| ❌ | ✅ 作第一个样本 | ✅ 需第二条独立云 |
| 粒子级重测 | ✅ | ✅ | ❌（复用 5 点底噪）|

#### 网格必须逐格建立

评测时用 `shared_edges([reference, prediction])` 为**每一格单独**构建 16 箱网格。若用全局网格，箱宽由整个参数域跨度决定，$W_2$ 会虚高约 4 倍（早期实现的缺陷，已修正）。

---

### A1.6 一句话总结

| 环节 | 内容 |
|---|---|
| **生成** | 单位球均匀初值 → SDE 走 4 步 → 终态云（推前测度）|
| **保存** | 每格一个 `.npz`，4 字段：`meta` / `initial` / `source` / `reference`，float32 |
| **训练** | 只用 `initial`（共享，1.2 MB）+ `reference`（每格一份，48 MB 共 40 格），随机抽 4096 粒子，MSE 损失 |
| **评测** | 用 `source`（≡initial）作输入、`reference` 作真值，评分函数可换（直方图 / 粒子级）|

**训练与评测的耦合点只有一个：评测读取训练好的 checkpoint。云数据本身在两边独立使用。**
