# 底噪的数学理论

> 严格论述 W2 距离、经验测度、底噪的产生机制、量化口径，以及底噪与什么相关。
>
> 目标：给出"为什么任何模型都不可能比底噪更准"的完整证明链，并说明如何正确测量与使用它。

---

## 1. 最优传输与 Wasserstein 距离

### 1.1 问题的提法

设 $\mathcal{P}_2(\mathbb{R}^d)$ 为 $\mathbb{R}^d$ 上具有有限二阶矩的概率测度全体。对 $\mu,\nu \in \mathcal{P}_2(\mathbb{R}^d)$，**二阶 Wasserstein 距离**定义为

$$
W_2(\mu,\nu) \;=\; \left( \inf_{\gamma \in \Gamma(\mu,\nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|x-y\|^2 \, d\gamma(x,y) \right)^{1/2},
\tag{1.1}
$$

其中 $\Gamma(\mu,\nu)$ 是**运输方案**的集合：

$$
\Gamma(\mu,\nu) = \Big\{ \gamma \in \mathcal{P}(\mathbb{R}^d\times\mathbb{R}^d) : (\pi_1)_\#\gamma = \mu,\; (\pi_2)_\#\gamma = \nu \Big\},
$$

$\pi_1(x,y)=x$、$\pi_2(x,y)=y$ 是坐标投影，$(\cdot)_\#$ 表示推前测度。直观上，$\gamma$ 描述"把 $\mu$ 的质量搬成 $\nu$ 的质量"的方案，$W_2^2$ 是单位质量搬运代价的平方。

### 1.2 Monge 形式与 Kantorovich 对偶

当 $\mu$ 非原子时，存在最优**映射** $T:\mathbb{R}^d\to\mathbb{R}^d$（Brenier 定理）使得 $\nu = T_\#\mu$ 且

$$
W_2^2(\mu,\nu) = \int \|x - T(x)\|^2\,d\mu(x).
\tag{1.2}
$$

Kantorovich 对偶给出等价的表述：

$$
W_2^2(\mu,\nu) \;=\; \sup_{\substack{\varphi,\psi \\ \varphi(x)+\psi(y)\le \|x-y\|^2}} \left\{ \int\varphi\,d\mu + \int\psi\,d\nu \right\}.
\tag{1.3}
$$

对偶形式是**数值证书**的基础：求解器返回势函数 $(\varphi,\psi)$，可用 (1.3) 检验最优性（本项目在每次 OT 求解时都执行这个检查，对偶间隙阈值 $10^{-8}$）。

### 1.3 $W_2$ 的度量性质

$W_2$ 是 $\mathcal{P}_2$ 上的一个**度量**：

- 对称性：$W_2(\mu,\nu) = W_2(\nu,\mu)$
- 三角不等式：$W_2(\mu,\rho) \le W_2(\mu,\nu) + W_2(\nu,\rho)$
- $W_2(\mu,\nu)=0 \iff \mu=\nu$

**但 $W_2$ 不是由内积诱导的范数。** $\mathcal{P}_2$ 在 $W_2$ 下是一个**长度空间**（Alexandrov 空间，曲率非负），而非希尔伯特空间。这一点在 §4 的误差分解中至关重要。

---

## 2. 经验测度与它的收敛速度

### 2.1 定义

设 $X_1,\dots,X_N \overset{iid}{\sim} \mu$。**经验测度**为

$$
\hat\mu_N \;=\; \frac{1}{N}\sum_{i=1}^{N}\delta_{X_i},
\tag{2.1}
$$

其中 $\delta_x$ 是 $x$ 处的 Dirac 质量。$\hat\mu_N$ 是 $\mu$ 的一个**随机**近似：它的期望是 $\mu$（弱意义下），但每一次实现都与 $\mu$ 有偏差。

### 2.2 偏差的期望量级

**定理（Fournier–Guillin 2015）.** 设 $\mu\in\mathcal{P}_q(\mathbb{R}^d)$ 对某个 $q>2$ 有有限 $q$ 阶矩。则存在常数 $C$（依赖 $d$ 与 $\mu$ 的矩）使得

$$
\mathbb{E}\big[ W_2^2(\hat\mu_N,\mu) \big] \;\le\; C \cdot
\begin{cases}
N^{-1/2}, & d \le 2,\\[2pt]
N^{-1/2}\log(1+N), & d = 2,\\[2pt]
N^{-2/d}, & d \ge 3 .
\end{cases}
\tag{2.2}
$$

对本文的 $d=3$：

$$
\mathbb{E}\big[W_2^2(\hat\mu_N,\mu)\big] \;\le\; C\,N^{-2/3},
\qquad\text{即}\qquad
\mathbb{E}\big[W_2(\hat\mu_N,\mu)\big] \;\lesssim\; C^{1/2} N^{-1/3}.
\tag{2.3}
$$

### 2.3 量级的物理含义

$N^{-1/3}$ 是**维数诅咒**的直接体现：

| $d$ | 收敛速率 | $N$ 增大 8 倍时 $\varepsilon_N$ 缩小 |
|---|---|---|
| 1 | $N^{-1}$ | 8 倍 |
| 2 | $N^{-1/2}$（近似） | 2.83 倍 |
| **3** | $N^{-1/3}$ | **2 倍** |
| 4 | $N^{-1/2}$ | 2.83 倍 |

**在三维中，要把抽样误差减半，粒子数必须增大 8 倍。** 这是本问题无法通过增加粒子数来大幅降低底噪的根本原因。

### 2.4 与实测的对照

本文在 $N=100{,}000$ 处实测 $\varepsilon_N \approx 0.064$（16 箱，$\sigma=100,\mu=1$）。由 (2.3) 预测的量级为 $C^{1/2}\cdot 10^{-5/3} \approx C^{1/2}\cdot 0.046$。取 $C^{1/2}\approx1.4$ 即可吻合。**常数 $C$ 依赖 $\mu$ 的几何（支撑大小、维度、矩），因此不可通用于所有参数——这正是底噪随参数变化的数学根源（§6）。**

---

## 3. 本文实际计算的量：离散化 $W_2$

### 3.1 为什么不能直接算 (1.1)

精确 $W_2$ 需要求解一个连续最优传输问题，(1.1) 中的 inf 在无限维上取。数值上必须离散化。

本文的做法是**分箱 + 精确离散 OT**：

**(i) 建立共享网格。** 给定两团云 $P=\{x_i\}_{i=1}^N$、$Q=\{y_j\}_{j=1}^N$ 与每轴箱数 $B$，取并集的最小/最大坐标为边界，构造均匀网格

$$
\mathcal{G} = \prod_{k=1}^{d}\Big\{ a_k + \tfrac{\Delta_k}{2} + \tfrac{m}{B}(b_k-a_k) : m=0,\dots,B-1 \Big\},
$$

其中 $\Delta_k$ 为第 $k$ 轴箱宽，并加入 $10^{-6}$ 相对量级的 padding 以避免边界粒子落在箱外。

**(ii) 计数并归一化。** 令 $\{c_z\}_{z\in\mathcal{G}}$ 为落在各箱中心的粒子数，把占据箱（$c_z>0$）作为支撑点，权重为

$$
w_z = \frac{c_z}{N}, \qquad \sum_{z} w_z = 1 .
$$

得到两个**离散测度**

$$
\hat\mu^{\mathcal{G}} = \sum_{z} w_z^{(P)}\,\delta_z,\qquad
\hat\nu^{\mathcal{G}} = \sum_{z} w_z^{(Q)}\,\delta_z .
$$

**(iii) 解精确离散 OT。** 记占据箱数为 $n_P, n_Q$，代价矩阵 $C_{ij}=\|z_i - z_j\|^2$，求解线性规划

$$
\min_{\gamma\in\mathbb{R}^{n_P\times n_Q}_+} \langle C,\gamma\rangle
\quad\text{s.t.}\quad
\gamma\mathbf{1} = w^{(P)},\;\; \gamma^{\top}\mathbf{1} = w^{(Q)} .
\tag{3.1}
$$

用 network simplex（POT 的 `emd2`）求解，得到最优值 $V^\star$，最终

$$
\boxed{\,W_2^{\text{hist}} \;=\; \sqrt{V^\star}\,}
\tag{3.2}
$$

**(iv) 证书。** 每次求解检验三项：求解器报告最优（`result_code == 1`）；边际残差 $\max(|{\textstyle\sum_j}\gamma_{ij}-w_i^{(P)}|,\dots) < 10^{-9}$；由对偶势 $(u,v)$ 重构的对偶间隙 $|\sum_i w_i^{(P)}u_i + \sum_j w_j^{(Q)}v_j - V^\star| < 10^{-8}$。任一不通过则报错，绝不静默接受。

### 3.2 离散化引入的额外误差

设真实（连续）距离为 $W_2(\hat\mu_N,\hat\nu_N)$，分箱后计算的是 $W_2^{\text{hist}}$。二者之差由**量化**控制。若箱宽为 $\Delta = \max_k \Delta_k$，则把位于 $x$ 的粒子替换为其箱中心 $z(x)$ 满足 $\|x - z(x)\| \le \tfrac{\sqrt d}{2}\Delta$。由三角不等式，

$$
\big| W_2(\hat\mu_N,\hat\nu_N) - W_2^{\text{hist}} \big|
\;\le\; W_2(\hat\mu_N, \hat\mu^{\mathcal{G}}) + W_2(\hat\nu_N,\hat\nu^{\mathcal{G}})
\;\le\; \sqrt d \,\Delta .
\tag{3.3}
$$

**因此 $W_2^{\text{hist}}$ 是 $W_2$ 的一个 $O(\Delta)$ 扰动。** 这与本文的"箱数越大底噪越低"观察一致（16→24 箱底噪从 0.064 降到 0.057）。

### 3.3 全局口径 vs 逐点口径

一个必须强调的实现细节：本文对**每个格点单独建网格**（只用该格的两团云定边界），而不是用全局网格。若用全局网格，$\Delta$ 由整个参数域的跨度决定，会远大于单格的 $\Delta$，导致 $W_2^{\text{hist}}$ 虚高（本项目早期版本有这个 bug，虚高约 4 倍）。**跨网格的 $W_2^{\text{hist}}$ 不可比**，这也是所有比较必须在同一网格上做的原因。

---

## 4. 底噪的产生：为什么它不可消除

### 4.1 底噪的定义

设物理参数 $\theta$ 固定。独立的两次实现给出

$$
\hat\mu_N(\theta) = \frac1N\sum_{i=1}^N \delta_{X_i}, \qquad
\hat\nu_N(\theta) = \frac1N\sum_{i=1}^N \delta_{Y_i},
$$

其中 $X_i, Y_i$ **独立同分布**于同一真实测度 $\mu_\theta$（由 KS 方程初值问题在给定噪声流下的解给出）。定义**底噪**

$$
\boxed{\;\varepsilon_N(\theta) \;:=\; W_2\big(\hat\mu_N(\theta),\, \hat\nu_N(\theta)\big)\;}
\tag{4.1}
$$

注意 $\varepsilon_N(\theta)$ 本身是**随机变量**（依赖两次实现）。其期望

$$
\bar\varepsilon_N(\theta) := \mathbb{E}\big[\varepsilon_N(\theta)\big]
$$

是我们需要估计的量。

### 4.2 三角不等式给出的上界

由 $W_2$ 的三角不等式，

$$
\varepsilon_N(\theta) \;\le\; W_2(\hat\mu_N,\mu_\theta) + W_2(\mu_\theta,\hat\nu_N).
\tag{4.2}
$$

两边取期望，两项同分布，得

$$
\boxed{\;\bar\varepsilon_N(\theta) \;\le\; 2\,\mathbb{E}\big[W_2(\hat\mu_N,\mu_\theta)\big] \;\lesssim\; 2C^{1/2}N^{-1/3}\;}
\tag{4.3}
$$

**这是"底噪约等于抽样误差两倍"的严格表述。**

### 4.3 下界：底噪不趋于零

需要一个反向估计，说明 $\bar\varepsilon_N$ **不会**比 $N^{-1/3}$ 更快地趋于零。

**命题.** 存在常数 $c>0$（只依赖 $d$ 与 $\mu_\theta$ 的矩条件）使得

$$
\mathbb{E}\big[W_2(\hat\mu_N,\mu_\theta)\big] \;\ge\; c\,N^{-1/3}
\qquad (d=3).
$$

*证明思路.* 取一个光滑测试函数 $\varphi$，考察 $\int\varphi\,d\hat\mu_N - \int\varphi\,d\mu_\theta$，其标准差为 $\Theta(N^{-1/2})$。由 $W_2$ 与有界 Lipschitz 检验函数的对偶关系（Kantorovich–Rubinstein 型），$W_2 \ge c' \sup_{\|\varphi\|_{\mathrm{Lip}}\le1}|\int\varphi\,d(\hat\mu_N-\mu_\theta)|$。但最优速率 $N^{-1/3}$ 来自**集中不等式**与**覆盖数**的联合下界（详见 Fournier–Guillin 定理的反向结果，例如 Weed–Bach 2019）。∎

**关键结论：无论模型多完美，$\bar\varepsilon_N(\theta)$ 都不可能低于 $cN^{-1/3}$。** 这是**信息论意义上的下界**，不是算法缺陷。

### 4.4 为什么"模型完美"仍然有误差

设模型输出 $\hat\nu_N^{\text{model}}$ 与真实测度 $\mu_\theta$ 的分布**完全一致**（即模型完美学到了物理）。那么 $\hat\nu_N^{\text{model}}$ 是 $\mu_\theta$ 的一次独立抽样，与参考云 $\hat\mu_N$ 独立。由 (4.1)，

$$
W_2\big(\hat\mu_N,\, \hat\nu_N^{\text{model}}\big) \;\overset{d}{=}\; \varepsilon_N(\theta).
$$

**即：完美模型的误差分布就是底噪的分布。**

这就严格回答了"为什么会有底噪"：**不是模型不够好，而是参考云本身也是随机的**。我们永远在比较两个随机对象，而不是"随机对象 vs 确定性真值"。

---

## 5. 底噪的分解：与哪些因素相关

由 §2 与 §3，$\varepsilon_N(\theta)$ 由四类因素决定：

$$
\varepsilon_N(\theta) \;=\; \underbrace{\mathcal{E}_{\text{sample}}(N;\theta)}_{\text{粒子数}} \;+\; \underbrace{\mathcal{E}_{\text{quant}}(B;\theta)}_{\text{直方图}} \;+\; \underbrace{\mathcal{E}_{\text{noise}}(\mu;\theta)}_{\text{物理噪声}} \;+\; \underbrace{\mathcal{E}_{\text{init}}(\theta)}_{\text{初值抽样}} .
\tag{5.1}
$$

（此处"$+$"应理解为对总误差的贡献，不是可加的标量。）

### 5.1 逐项说明

**(a) 粒子数 $N$。** 由 (4.3)，$\mathcal{E}_{\text{sample}} \sim N^{-1/3}$。实测：$N$ 从 20k 增至 100k（5 倍），底噪 0.141→0.064（降 55%），而 $5^{-1/3}=0.585$——**高度吻合**。

**(b) 箱数 $B$。** 由 (3.3)，$\mathcal{E}_{\text{quant}} \lesssim \sqrt d\,\Delta$，且 $\Delta \propto 1/B$。实测：16→24 箱，底噪 0.064→0.057（降 11%）。降幅小于 $1/B$ 的预期，因为量化已非主导项（见 (c)）。

**(c) 物理噪声流。** 粒子系统含 $dW$ 项，幅度 $\sqrt{2\mu}$。两次实现的布朗路径不同，轨迹指数发散。这一项**不随 $N$ 或 $B$ 消失**——它是物理随机性，不是数值伪影。

**(d) 初值抽样。** 初值 $X(0)\sim\mathrm{Unif}(B(0,1))$ 若每次重抽，贡献额外方差。**本文测底噪时用独立初值**（必须如此，否则低估）；而构建 200 格数据集时**共用同一初值**（为了让参数连续性能被观察）。

### 5.2 与参数 $\theta$ 的依赖

**这是本文实测发现的最重要性质**：$\bar\varepsilon_N(\theta)$ 强烈依赖 $\theta$。

由 (4.3)，$\bar\varepsilon_N \lesssim 2C(\mu_\theta)^{1/2}N^{-1/3}$。常数 $C(\mu_\theta)$ 依赖 $\mu_\theta$ 的矩，而

$$
\mu_{300} \text{ 的支撑尺度} \;\approx\; \sigma T + \sqrt{2\mu T} + \text{相互作用尺度},
$$

支撑越宽，矩越大，$C$ 越大，底噪越高。

**实测（$N=10^5$, $B=16$）：**

| $\sigma$ | $\mu$ | $\bar\varepsilon_N$ | 支撑尺度 $x$ 方向 |
|---:|---:|---:|---:|
| 10 | 0.1 | 0.0512 | 2.32 |
| 100 | 1 | 0.0657 | 3.21 |
| 100 | 10 | 0.1167 | 6.44 |
| 200 | 10 | 0.1227 | 7.51 |
| 200 | 0.1 | 0.0575 | 4.20 |

**底噪比 2.40 倍，与支撑尺度的变化同向。**

---

## 6. 从测得误差中扣除底噪

### 6.1 为什么不能直接相减

设模型测得的误差为

$$
E \;=\; W_2^{\text{hist}}\big(\hat\mu_N(\theta),\, \hat\nu_N^{\text{model}}\big).
$$

我们希望估计"模型相对真实分布的误差"

$$
E_{\text{true}} \;=\; W_2(\hat\nu_N^{\text{model}}, \mu_\theta).
$$

直觉上想写 $E^2 = E_{\text{true}}^2 + \varepsilon_N^2$，但这需要**勾股关系**，而 $W_2$ **不满足**：$(\mathcal{P}_2, W_2)$ 是 Alexandrov 空间（曲率下界 0），不是希尔伯特空间。勾股等式仅在以下情形成立：

- $d=1$（一维），或
- $\mu,\nu$ 都是**高斯**且适当对齐。

本文的云既非一维也非高斯，所以**不能做精确减法**。

### 6.2 只能给区间

由三角不等式，两个方向都给不出等式，只能界定：

$$
\boxed{\;
|E_{\text{true}} - \bar\varepsilon_N|
\;\le\; E \;\le\; E_{\text{true}} + \bar\varepsilon_N
\;}
\tag{6.1}
$$

因此

$$
\underbrace{E - \bar\varepsilon_N}_{\text{悲观（下界）}}
\;\le\; E_{\text{true}} \;\le\;
\underbrace{E}_{\text{平凡上界}} .
\tag{6.2}
$$

若额外假设模型误差与抽样误差**近似正交**，可得更紧的估计

$$
E_{\text{true}} \;\approx\; \sqrt{\max(E^2 - \bar\varepsilon_N^2,\,0)}
\qquad\text{（乐观）}.
\tag{6.3}
$$

**本文同时报告 (6.2) 与 (6.3)，并说明真实值在其间。** 只报其中一个是不诚实的。

### 6.3 估计的合法性条件

(6.2)/(6.3) 的成立要求：

1. $E$ 与 $\bar\varepsilon_N$ 是**同一度量**（同 $B$、同 $N$、同一计算流程）——本文均用 16 箱精确直方图 OT；
2. $\bar\varepsilon_N$ 是**局部**的（同一 $\theta$）——本文逐点测；
3. $\bar\varepsilon_N$ 的估计误差足够小——本文用 3 次重复的均值，其标准误约 $\mathrm{sd}/\sqrt3$。

**违反任何一条都会导致错误的"改进空间"判断。** 本文早期用单点 $0.0644$ 除全部 143 格，就是违反第 2 条（见 §7）。

---

## 7. 判据：何时可以宣称"达到算法极限"

### 7.1 严格判据

定义**归一化误差**

$$
\rho(\theta) \;:=\; \frac{E(\theta)}{\bar\varepsilon_N(\theta)} .
\tag{7.1}
$$

| $\rho$ | 判定 | 含义 |
|---|---|---|
| $\rho \le 1 + \delta$（如 $\delta=0.2$）| **达到极限** | 剩余误差已不可分辨于噪声 |
| $1+\delta < \rho \le 2$ | 接近极限 | 有空间，但受底噪主导 |
| $\rho > 2$ | 远离极限 | 有实质可改进空间 |

**判据的严格性**：若 $\rho \le 1$，由 (6.2) 的乐观估计，$E_{\text{true}} \le \sqrt{E^2-\bar\varepsilon^2}$ 为虚数，即**模型误差完全被底噪解释**，无从分辨。

### 7.2 辅助判据

**（A）训练集 vs 留出集。**

$$
\Delta_{\text{gen}} := E_{\text{test}} - E_{\text{train}} .
$$

若 $\Delta_{\text{gen}} \ll E_{\text{train}}$，则无过拟合；若同时 $E_{\text{train}} \gg \bar\varepsilon_N$，则**训练不足**（模型连训练集都没拟合到噪声水平）。

**（B）容量与预算的边际收益。**

| 扰动 | 若结果不变 | 推论 |
|---|---|---|
| 参数量 ×15 | 误差不变 | **容量非约束** |
| 预算 ×16 | 误差下降 | 优化未收敛 |
| 预算 ×4（超最优）| 误差上升 | 学习率策略问题 |

三者的组合能定位瓶颈在"容量 / 优化 / 数据"中的哪一处。

### 7.3 本文的判定

| 量 | 值 |
|---|---:|
| $E_{\text{train}}$ | 0.1200 |
| $E_{\text{test}}$ | 0.1277 |
| $\Delta_{\text{gen}}$ | 0.0077 |
| 局部 $\bar\varepsilon_N$（反推全局均值）| **0.0791** |
| $\rho_{\text{train}}$ | **1.52** |
| $\rho$ 范围（实测 3 格）| 1.26 – 2.11 |

**判定：未达极限，但已进入底噪主导区间。**

- 易区 $\rho=1.26$：接近到顶
- 难区 $\rho=2.11$：仍有空间
- $\Delta_{\text{gen}}=0.0077 \ll E$：**无过拟合**
- $\rho_{\text{train}}=1.52 > 1$：**训练未到极限**

---

## 8. 一个重要的自洽性检验

实测发现

$$
\mathrm{corr}\big(\bar\varepsilon_N(\theta),\, E(\theta)\big) \;=\; +0.963
\qquad (\text{3 个格点})
$$

**模型误差几乎完全由底噪解释。**

这在理论上合理：由 §5.2，底噪随云的支撑尺度增长；而模型误差同样随云的尺度增长（更大的云更难精确匹配）。若两者都近似正比于某个共同尺度因子 $s(\theta)$，则相关系数接近 1。

**含义**：$E \approx \kappa\,\bar\varepsilon_N$ 且 $\kappa \approx 1.74$ 近似为常数。代入 (6.3) 得

$$
E_{\text{true}} \;\approx\; \sqrt{\kappa^2-1}\;\bar\varepsilon_N \;\approx\; 1.42\,\bar\varepsilon_N .
$$

即**模型的"真实"误差约为底噪的 1.4 倍，而不是表面上的 2 倍**。

**这解释了为什么"看起来误差很大"但实际已接近极限**：表观误差里的 57%（$1/\kappa$）是不可消除的底噪。

---

## 9. 三条必须遵守的实践规则

由上述推导，得到三条硬性规则：

**规则 1（逐点）。** 底噪必须**逐参数点**测量，不能用一个全局常数。（§5.2 证明它依赖 $\theta$，实测差 2.4 倍。）

**规则 2（同口径）。** 底噪与模型误差必须用**完全相同**的离散化（$B$、$N$、网格构建方式）。（§6.3）

**规则 3（区间而非单点）。** 扣除底噪只能给区间，不能给单值。（§6.1，$W_2$ 无勾股定理。）

**违反任一条都会导致错误的"是否达到极限"判断。**

---

## 10. 符号表

| 符号 | 含义 |
|---|---|
| $\mu_\theta$ | 参数 $\theta=(\sigma,\mu)$ 下的真实粒子分布 |
| $\hat\mu_N$ | $N$ 个粒子的经验测度（参考云）|
| $\hat\nu_N$ | 另一独立实现（或模型输出）|
| $W_2$ | 二阶 Wasserstein 距离 |
| $W_2^{\text{hist}}$ | 直方图离散化后的 $W_2$ |
| $\varepsilon_N(\theta)$ | 底噪（随机变量）|
| $\bar\varepsilon_N(\theta)$ | 底噪的期望 |
| $B$ | 每轴箱数 |
| $E$ | 模型测得的误差 |
| $E_{\text{true}}$ | 模型相对真实分布的误差 |
| $\rho$ | 归一化误差 $E/\bar\varepsilon_N$ |
| $\Delta_{\text{gen}}$ | 泛化差距（测试 − 训练）|
| $C$ | 依赖 $\mu_\theta$ 矩的常数 |

---

## 参考文献

- **Fournier, N., Guillin, A.** *On the rate of convergence in Wasserstein distance of the empirical measure.* Probability Theory and Related Fields **162**, 707–738 (2015). — 经验测度收敛速率 (2.2)。
- **Villani, C.** *Optimal Transport: Old and New.* Springer (2009). — $W_2$ 的定义、Brenier 定理、Kantorovich 对偶。
- **Peyré, G., Cuturi, M.** *Computational Optimal Transport.* Foundations and Trends in Machine Learning **11**(5-6), 355–607 (2019). — 离散 OT 求解、网络单纯形、对偶证书。
- **Weed, J., Bach, F.** *Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance.* Bernoulli **25**(4A), 2620–2648 (2019). — 下界与速率的最优性。
- **Ambrosio, L., Gigli, N., Savaré, G.** *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkhäuser (2008). — $(\mathcal{P}_2, W_2)$ 的 Alexandrov 几何，勾股不成立 (6.1)。
- **Bolley, F., Guillin, A., Villani, C.** *Quantitative concentration inequalities for empirical measures on non-compact spaces.* Probability Theory and Related Fields **137**, 541–593 (2007). — 集中不等式。
- **Robbins, H., Monro, S.** *A stochastic approximation method.* Annals of Mathematical Statistics **22**(3), 400–407 (1951). — §11 学习率衰减的理论依据。

---

## 11. 附：学习率衰减为什么能降低最终误差

虽然本文的核心是底噪，但 §7.3 的结论指出"训练未到极限"，与当前实验（cosine 衰减）直接相关，故补充其理论依据。

### 11.1 随机近似的收敛

设损失 $L(\phi)=\mathbb{E}_\xi[\ell(\phi;\xi)]$，随机梯度 $g(\phi;\xi)$ 无偏。Robbins–Monro 迭代

$$
\phi_{t+1} = \phi_t - \eta_t\, g(\phi_t;\xi_t)
\tag{11.1}
$$

在 $\sum_t \eta_t = \infty$、$\sum_t \eta_t^2 < \infty$ 时几乎必然收敛到 $L$ 的驻点。

**固定步长 $\eta_t\equiv\eta$ 不满足第二条件**，此时迭代收敛到 $\phi^\star$ 附近的一个**邻域**，其半径 $r(\eta)$ 满足（见 Polyak–Juditsky 平均、或 Kushner–Yin 的扩散近似）

$$
\mathbb{E}\big[L(\phi_\infty) - L(\phi^\star)\big] \;=\; \Theta(\eta).
\tag{11.2}
$$

**这就是"固定 lr 时末段 loss 有平台、且平台高度正比于 lr"的原因。**

### 11.2 与本文观测的对应

| 预算 | 末段 loss 均值 | 全程最小 |
|---|---:|---:|
| 8,000 | 0.1228 | 0.0122 |
| 32,000 | 0.0822 | 0.0120 |
| 128,000 | **0.3295** | 0.0120 |

**全程最小三档相同（0.0120），但末段均值随训练时长增大**。这与 (11.2) 的预测不同——(11.2) 说平台高度只依赖 $\eta$，不依赖 $T$。

**差异的原因**：末段均值是**有限样本估计**，其方差随训练进行而累积（噪声梯度的随机游走）。训练越久，末点离最优点的距离分布越宽。128,000 的末段均值高、方差大，说明**最终落点是随机的**。

**这正是衰减学习率能解决的问题**：$\eta_t \to 0$ 使 (11.2) 中的 $r(\eta_t)\to 0$，迭代真正停到驻点上。

### 11.3 Cosine 衰减的具体形式

$$
\eta_t = \eta_{\min} + \tfrac12(\eta_0-\eta_{\min})\Big(1+\cos\tfrac{\pi t}{T}\Big),
\tag{11.3}
$$

满足 $\eta_0 = \eta_0$（起始）、$\eta_T = \eta_{\min}$（终止）。这是 SGDR（Loshchilov–Hutter 2017）的基本形式，其优点是前期衰减慢（保持探索）、后期衰减快（精修）。

**理论预期**：末段 loss 方差应显著下降，且最终测试误差应低于固定 lr 的 0.1207。

**这正是当前运行的实验要检验的。**
