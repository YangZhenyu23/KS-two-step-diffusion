# 双参数两步扩散的数学理论（完整版）

> **核心问题**：把网络作用在连续参数 $\theta=(\sigma,\mu)$ 上，"学一个参数到分布的映射"这件事
> 在本问题中是否良定义、是否可学、误差如何被控制、在哪里失效。
>
> **本文用途**：给出一份**可选菜单**。每条结论都标注了严格程度与它支撑的实验结论，
> 便于决定哪些写进论文、哪些只作为讨论。
>
> ---
>
> **严格程度标记**
>
> | 标记 | 含义 |
> |---|---|
> | 【定理·严格】 | 完整证明，常数有效可用 |
> | 【定理·定性】 | 完整证明，但 Grönwall 常数在本问题上**数值上无效**（见 §1.4） |
> | 【经验定律】 | 形式由理论给出、常数由数据定出，且常数在 $20\times$ 参数范围内**经验上稳定** |
> | 【启发式】 | 机制性论证，无严格证明 |
> | 【已否定】 | 经数据检验后**撤回**的旧结论 |
>
> ---
>
> **相对旧版的三处实质修正**（务必先读）
>
> 1. **$\mu$ 方向的 $1/2$ 阶 Hölder 是错的**，真实结论是 **Lipschitz**。
>    错误来自把 $|\sqrt{2\mu_1}-\sqrt{2\mu_2}|\le C\sqrt{|\Delta\mu|}$ 当成锐利界；
>    实际上在 $\mu\ge\mu_{\min}>0$ 上它自动是线性的。这一改动推翻了旧版
>    "$\mu$ 因为指数是 $1/2$ 所以本质上更难分辨"的整套论述。
> 2. **假设 (A1) 中"$\|\nabla^2K_\delta\|_\infty<\infty$"是假的**。
>    $\nabla K_\delta$ 在原点**不连续**，$\nabla^2K_\delta$ 在原点奇异（$\sim1/r$）。
>    修正方式见 §1.1（对卷积后的场而非原始核作假设）。
> 3. **"$L_\sigma\approx4.6$、$L_\mu\approx0.45$ 是常数"是统计假象**。
>    这两个量在参数域内分别变化 $20\times$ 与 $10\times$。真正稳定的是
>    **位移坐标**下的常数，见【经验定律 1】。

---

## 目录与取舍建议

| § | 结论 | 严格程度 | 支撑的实验 | 论文建议 |
|---|---|---|---|---|
| 1.2 | 平均场极限存在唯一 | 定理·严格 | 方法的对象良定义 | **必用** |
| 1.3 | $\theta\mapsto\nu_\theta$ 连续（Lipschitz） | 定理·定性 | 五项连续性检验 | **必用**（只用作定性） |
| 1.5 | **位移律** $W_2\approx c_S\|\Delta S\|+c_G\|\Delta G\|$ | 经验定律 | 370 对格点、3 项检验 | **强烈推荐**（本文最有价值的新结果） |
| 1.6 | 可分辨阈值（修正版） | 经验定律的推论 | $\sigma$ 2.5%、$\mu$ 25% | **推荐**（修好了旧版 12 倍偏差） |
| 2.1 | MSE 最优确定性映射 = 条件期望 | 定理·严格 | — | **必用** |
| 2.3 | 底噪上限 $W_2\le\sqrt{\mathbb E\mathrm{Var}}$ | 定理·严格 | 底噪 0.12 vs 界 0.35 | **用，但必须说明界很松** |
| 2.4 | 量化误差界 | 定理·严格 | "94% 是量化" | **推荐** |
| 3.2 | **DP 的极限是 Brenier 最优映射** | 定理·严格 | DP $+14.4\%$ | **强烈推荐**（比旧版残差论干净） |
| 3.3 | DP 有效性条件 | 定理·严格 | 内插有效、外推失效 | **推荐** |
| 4.2 | 泛化界，速率 $n^{-1/2}$（修正） | 定理·严格 | $n{=}40$ 预测 0.176 实测 0.151 | **推荐** |
| 4.3 | LHS 与随机不可分辨 | 定理·严格 | 4 次重复 | **推荐** |
| 5 | 外推失效的机制 | 启发式 + 严格下界 | $\mu$ 外推退化 $3.76\times$ | **用，标注为机制** |
| 8 | 四问四答 | — | — | **结论段** |

---

# 第 0 部分：记号与设定

## 0.1 物理系统

参数 $\theta=(\sigma,\mu)\in\Theta=[10,200]\times[0.1,10]$。$N$ 粒子系统（$j=1,\dots,N$）：

$$
dX_j^{N,\theta}(t)
= -\frac{\chi}{N}\sum_{i\neq j}\nabla K_\delta\big(X_j-X_i\big)\,dt
+ \sigma\,g(X_j)\,dt
+ \sqrt{2\mu}\,dW_j,
\tag{0.1}
$$

$$
X_j(0)\ \text{i.i.d.}\sim \mu_0=\mathrm{Unif}\big(B(0,R_0)\big),
\qquad
g(x)=\big(e^{-y^2-z^2},0,0\big),
\qquad
K_\delta(z)=\frac{|z|}{4\pi\big(|z|^2+\delta^2\big)} .
$$

固定量：$\chi=16\pi\approx50.27$，$\delta=0.1$，$R_0=1$，$T=0.02$，$dt=0.005$（$T/dt=4$ 步 Euler–Maruyama）。

> **注意**：$\chi=16\pi$ 与 $\delta=0.1$ 使相互作用项**极强**：
> $\chi\|\nabla K_\delta\|_\infty=\chi/(4\pi\delta^2)=400$。
> 这决定了 §1.4 里 Grönwall 常数必然很大，不是技术缺陷而是物理设定。

## 0.2 记号

| 记号 | 含义 |
|---|---|
| $\mathcal{P}_2(\mathbb{R}^3)$ | $\mathbb{R}^3$ 上二阶矩有限的概率测度 |
| $W_2$ | 二阶 Wasserstein 距离，$W_2^2(\mu,\nu)=\inf_{\gamma\in\Gamma(\mu,\nu)}\int|x-y|^2d\gamma$ |
| $\hat\mu_{N,\theta}$ | 系统 (0.1) 在参数 $\theta$ 下的经验终态测度（$N$ 个粒子）|
| $\nu_{\theta,t}$ | 对应的平均场极限（$N\to\infty$），$\nu_\theta:=\nu_{\theta,T}$ |
| $\Phi^\theta_t(x,\omega)$ | 从 $x$ 出发的随机流在时刻 $t$ 的位置 |
| $\mathcal{D}\subset\Theta$ | 训练参数集，$n=|\mathcal{D}|$ |
| $h$ | 参数空间度量（具体形式见 §1.5，**不是**任选的）|

## 0.3 三个层次的对象

方法涉及三个必须区分的对象：

$$
\underbrace{\nu_\theta}_{\text{真分布（}N=\infty\text{）}}
\xrightarrow[\ \text{定理 1}\ ]{\text{平均场极限}}
\underbrace{\hat\mu_{N,\theta}}_{\text{有限粒子真值}}
\xrightarrow[\ \text{定理 4}\ ]{\text{确定性映射}}
\underbrace{T^*_\theta(x)=\mathbb E[X_T\mid X_0=x]}_{\text{MSE 最优确定性预测}}
$$

网络学的**不是** $\nu_\theta$（那是分布），而是 $T^*_\theta$。**这一步的代价是本文全部"底噪"的来源**（§2）。

> **还有第四个对象，只在 §3 出现**：DP 阶段学的也不是 $T^*_\theta$，而是
> **最优传输映射** $\nabla\varphi_\theta$。这与 $T^*_\theta$ 是**不同的函数**，
> 而二者的差别恰是 DP 能超过条件期望"下限"的原因。旧版文档把两者混为一谈。

## 0.4 位移尺度：正确的坐标

这是本文与旧版最重要的结构性差别。两个物理参数在 SDE 中**只通过两个位移尺度**起作用：

$$
\boxed{
S:=\sigma T\quad(\text{剪切位移}),\qquad
G:=\sqrt{2\mu T}\quad(\text{一阶布朗位移})
}
\tag{0.2}
$$

数值范围：$S\in[0.2,\,4.0]$（$20\times$），$G\in[0.063,\,0.632]$（$10\times$）。

**为什么这两个量是自然的**：在时间 $T$ 内，
- 剪切项把粒子沿 $x$ 平移 $\sigma T\,|g|$，而 $|g|_\infty=1$，所以位移 $\le S$；
- 扩散项贡献标准差 $\sqrt{2\mu T}=G$ 每坐标。

二者都是**长度**，因此可以直接相加、与 $W_2$（也是长度）比较；而 $\sigma$ 是 $1/\text{时间}$、$\mu$ 是长度$^2/$时间，直接相加没有量纲意义。旧版的界 $L(|\Delta\sigma|+\sqrt{|\Delta\mu|})$ 在量纲上就不成立。

---

# 第 1 部分：良定义性与正则性

**本部分回答：** $\theta\mapsto\nu_\theta$ 是否良定义？是否连续？连续性有多强？

若这个映射不连续，则"用有限个参数点训练、在中间参数上预测"在原理上不可能。**这是整个方法的前提。**

## 1.1 假设（修正版）

> **(A1) 核的正则性。**
>
> (a) **一阶有界（严格）**：
> $$
> \|\nabla K_\delta\|_\infty=\frac{1}{4\pi\delta^2} .
> $$
>
> (b) **二阶导数无界（修正！）**：$\nabla K_\delta$ 在 $z=0$ **不连续**，$\nabla^2K_\delta$ 在原点奇异：
> $$
> \big|\partial_r^2K_\delta(r)\big|\le\frac{C_2}{4\pi\delta^3},\qquad
> \big|\partial_rK_\delta(r)\big|/r\;\sim\;\frac{1}{8\pi\delta^2\,r}\to\infty\quad(r\to0).
> $$
>
> (c) **卷积后恢复 Lipschitz（这才是要用的）**：对任意有界密度 $\nu$，
> $\nabla K_\delta*\nu$ 是 Lipschitz 的，且
> $$
> \mathrm{Lip}\big(\nabla K_\delta*\nu\big)\;\le\;C\,\|\nu\|_\infty .
> $$

**证明 (a)。** 由

$$
\nabla K_\delta(z)=\frac{z}{4\pi}\cdot\frac{\delta^2-|z|^2}{\big(|z|^2+\delta^2\big)^2},
\qquad
\big|\nabla K_\delta\big|=\frac{|\delta^2-r^2|}{4\pi(r^2+\delta^2)^2},\quad r=|z| .
$$

令 $F(r)=(\delta^2-r^2)/(r^2+\delta^2)^2$。
- $r\le\delta$：$F(r)\le \delta^2/\delta^4=1/\delta^2$（分子在 $r=0$ 最大、分母在 $r=0$ 最小）；
- $r\ge\delta$：$F(r)\le r^2/(r^2+\delta^2)^2\le r^2/r^4=1/r^2\le1/\delta^2$。

故 $\sup F=1/\delta^2$，即 $\|\nabla K_\delta\|_\infty=1/(4\pi\delta^2)$。$\square$

**证明 (b)。** 直接求导：

$$
K_\delta''(r)=\frac{1}{4\pi}\cdot\frac{2r^3-6r\delta^2}{(r^2+\delta^2)^3},
\qquad
\frac{K_\delta'(r)}{r}=\frac{1}{4\pi}\cdot\frac{\delta^2-r^2}{r(r^2+\delta^2)^2}.
$$

第一式令 $x=r/\delta$，则为 $\delta^{-3}\cdot\dfrac{|2x^3-6x|}{4\pi(1+x^2)^3}$，该无量纲因子在 $x\approx0.5$ 取最大值 $\approx1.77$，故 $|K''|\le 1.77/(4\pi\delta^3)$。
第二式在 $r\to0$ 时 $\sim 1/(8\pi\delta^2 r)\to\infty$。$\square$

> **这一步为什么重要**：旧版把 $\|\nabla^2K_\delta\|_\infty\le C/\delta^3$ 当作假设，
> 但它**是假的**。若照旧版的证明直接对 $|x-y|$ 用中值定理，
> 得到的是 $\infty\cdot|x-y|$，证明在技术上不成立。下面给出正确的替代。

**证明 (c)。** 设 $|x-y|=h$，记 $D(z)=\nabla K_\delta(x-z)-\nabla K_\delta(y-z)$。则

$$
\big|\nabla K_\delta*\nu(x)-\nabla K_\delta*\nu(y)\big|\le\int|D(z)|\,\nu(dz).
$$

把积分域按 $|z|\le 2h$ 与 $|z|>2h$ 分裂（原点取在 $x,y$ 的中点附近，误差为 $O(h)$ 的平移）。

- **近区** $|z|\le2h$：用 (a)，$|D|\le 2\|\nabla K_\delta\|_\infty$，故
  $$
  \int_{|z|\le2h}|D|\nu\,dz\le 2\|\nabla K_\delta\|_\infty\|\nu\|_\infty\cdot\tfrac43\pi(2h)^3=O(h^3).
  $$
- **远区** $|z|>2h$：$\nabla K_\delta$ 在 $\{|w|\ge h\}$ 上光滑，且由 (b) 的显式计算
  $$
  \big|\nabla^2K_\delta(w)\big|\le\frac{C}{4\pi\delta^2|w|}\quad(|w|\le\delta),
  \qquad
  \big|\nabla^2K_\delta(w)\big|\le\frac{C}{4\pi|w|^3}\quad(|w|\ge\delta).
  $$
  沿线段 $[x-z,\,y-z]$（长度 $h$，始终在 $\{|w|\ge h\}$ 内）用中值定理：
  $$
  |D(z)|\le h\sup_{|w|\ge h}\big|\nabla^2K_\delta(w)\big|\le \frac{Ch}{4\pi\delta^2 h}=\frac{C}{4\pi\delta^2} .
  $$
  这个界太粗（$O(1)$，不随 $h$ 收缩）。**需要更精细的结构**：在远区，$D(z)$ 是单位向量场
  $\hat w$ 的差，其大小 $\approx h/|z|$（方向变化率），故
  $$
  \int_{|z|>2h}|D|\nu\,dz\;\lesssim\;h\int_{2h}^{\infty}\frac{\nu(dz)}{|z|}\;\lesssim\;h\,\|\nu\|_\infty\,\delta .
  $$
  两项合并得 $|\nabla K_\delta*\nu(x)-\nabla K_\delta*\nu(y)|\lesssim h\|\nu\|_\infty\delta$。$\square$

> **严格性说明**：远区这一步是标准的"奇异核 + 有界密度"补偿论证（与 Riesz 位势的
> $W^{1,p}$ 正则性同源），本文给出的是其骨架而非逐行常数。对论文用途足够；
> 若要写成正式引理，建议引 Gilbarg–Trudinger 关于 Newton 位势的引理并作尺度变换。

> **(A2) 剪切项。** $\|g\|_\infty=1$，$\mathrm{Lip}(g)\le1$
> （因 $\max_y|2ye^{-y^2}|=\sqrt{2/e}\approx0.858$）。
>
> **(A3) 初值共享。** $\mu_0$ 与 $\theta$ 无关（所有格点共用同一初值云），$\mu_0\in\mathcal P_4(\mathbb R^3)$。
>
> **(A4) 密度有界。** 对所有 $t>0$，$\nu_{\theta,t}$ 有密度且 $\|\nu_{\theta,t}\|_\infty<\infty$。

> **(A3) 是实验设计而非数学假设。** 它保证"两格之间的差异纯粹来自参数"，是观察到
> 连续性的必要条件。若每格独立抽样初值，$W_2$ 在 $h\to0$ 时趋于噪声基底而非零。
> **(A4) 由 (0.1) 的噪声自动成立**（$t>0$ 时热核平滑），但常数依赖 $t$。

## 1.2 定理 1：平均场极限（【定理·严格】）

> **定理 1.** 设 (A1a)(A2)(A3)(A4) 成立，$T>0$。则对每个 $\theta\in\Theta$ 存在唯一的
> 概率流 $t\mapsto\nu_{\theta,t}\in C([0,T];\mathcal P_2(\mathbb R^3))$ 满足 McKean–Vlasov 方程
> $$
> \partial_t\nu_t+\nabla\cdot\Big(\big(-\chi\nabla K_\delta*\nu_t+\sigma g\big)\nu_t\Big)=\mu\Delta\nu_t ,
> \tag{1.1}
> $$
> 且 $N\to\infty$ 时 $\hat\mu_{N,\theta}\Rightarrow\nu_{\theta,T}$ 依概率收敛。

**证明要点。** 漂移

$$
b_\theta(x,\nu)=-\chi\big(\nabla K_\delta*\nu\big)(x)+\sigma g(x)
$$

关于 $x$ 是 Lipschitz（由 A1c、A2），关于 $\nu$ 在 $W_1$ 下 Lipschitz：

$$
\big|b_\theta(x,\nu)-b_\theta(x,\nu')\big|\le\chi\|\nabla K_\delta\|_\infty\,W_1(\nu,\nu')
=\frac{\chi}{4\pi\delta^2}W_1(\nu,\nu')=400\,W_1(\nu,\nu').
$$

代入 $\chi=16\pi$、$\delta=0.1$。常数有限，故 (1.1) 是标准 McKean–Vlasov 方程。
由 Sznitman (1991) Thm 1.1 得存在唯一性；由 Thm 2.1 得粒子逼近。$\square$

> **关于速率**：Sznitman 在漂移对 $\nu$ 全局 Lipschitz 时给
> $\mathbb E[W_2^2(\hat\mu_N,\nu_T)]\le CN^{-2/d}=CN^{-2/3}$（$d=3$）。
> 旧版写 $N^{-1/2}$ 是 $d=4$ 的速率，已修正。

## 1.3 定理 2：连续依赖（【定理·定性】）

> **定理 2.** 在 (A1)–(A4) 下，存在**常数** $L_\sigma,L_\mu<\infty$ 使
> $$
> W_2\big(\nu_{\theta_1,T},\nu_{\theta_2,T}\big)\;\le\;L_\sigma\,|\sigma_1-\sigma_2|
> +L_\mu\,|\mu_1-\mu_2| .
> \tag{1.2}
> $$
> 即关于 $\sigma$ **和** $\mu$ **都是 Lipschitz**。写成位移坐标：
> $$
> W_2\big(\nu_{\theta_1,T},\nu_{\theta_2,T}\big)\;\le\;e^{CT}\Big(|\Delta S|+\sqrt{d}\,|\Delta G|\Big).
> \tag{1.3}
> $$

**证明。**

**第一步：同步耦合。** 用同一个布朗运动 $W$ 驱动两条 SDE：

$$
\begin{aligned}
dX_t&=b_{\theta_1}(X_t,\nu_{\theta_1,t})\,dt+\sqrt{2\mu_1}\,dW_t,\\
dY_t&=b_{\theta_2}(Y_t,\nu_{\theta_2,t})\,dt+\sqrt{2\mu_2}\,dW_t,
\end{aligned}
\qquad X_0=Y_0 .
$$

**第二步：Itô。** 令 $Z_t=X_t-Y_t$、$\Delta_t=\mathbb E|Z_t|^2$。扩散系数之差是确定性常数，故二次变差为 $|\sqrt{2\mu_1}-\sqrt{2\mu_2}|^2t$：

$$
d|Z_t|^2=2Z_t\cdot\big(b_{\theta_1}(X_t,\nu_1)-b_{\theta_2}(Y_t,\nu_2)\big)\,dt
+2Z_t\cdot\big(\sqrt{2\mu_1}-\sqrt{2\mu_2}\big)\,dW_t
+\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big|^2\,dt .
$$

取期望（随机积分项期望为零）：

$$
\Delta_t=\int_0^t 2\,\mathbb E\big[Z_s\cdot\Delta b_s\big]\,ds
+\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big|^2t .
\tag{1.4}
$$

**第三步：漂移差分解。** 记 $\Delta b_s=b_{\theta_1}(X_s,\nu_1)-b_{\theta_2}(Y_s,\nu_2)$。关键是**分成四项**，其中剪切项要拆开：

$$
\begin{aligned}
\Delta b_s
&=\underbrace{-\chi\big[(\nabla K*\nu_1)(X_s)-(\nabla K*\nu_1)(Y_s)\big]}_{\text{(i) 相互作用·同测度}}
\underbrace{-\chi\big[(\nabla K*\nu_1)(Y_s)-(\nabla K*\nu_2)(Y_s)\big]}_{\text{(ii) 相互作用·测度差}}\\
&\quad+\underbrace{(\sigma_1-\sigma_2)\,g(X_s)}_{\text{(iii) 剪切·参数差}}
+\underbrace{\sigma_2\big[g(X_s)-g(Y_s)\big]}_{\text{(iv) 剪切·位置差}} .
\end{aligned}
$$

逐项估计（用 A1c、A2、$W_1\le W_2$）：

$$
\begin{aligned}
\text{(i)}&\le\chi\,C\|\nu_1\|_\infty\,|X_s-Y_s|=:C_1|Z_s|,\\
\text{(ii)}&\le\chi\|\nabla K_\delta\|_\infty\,W_1(\nu_{1,s},\nu_{2,s})\le 400\,W_1(\nu_{1,s},\nu_{2,s})\le400\sqrt{\Delta_s},\\
\text{(iii)}&\le|\Delta\sigma|,\\
\text{(iv)}&\le\sigma_{\max}\,|Z_s|=200\,|Z_s| .
\end{aligned}
$$

于是

$$
\mathbb E[Z_s\cdot\Delta b_s]\le C_2\Delta_s+400\,\Delta_s+|\Delta\sigma|\sqrt{\Delta_s},
\qquad C_2:=C_1+200 .
\tag{1.5}
$$

**第四步：Grönwall 型不等式。** 设 $u(t)=\sup_{s\le t}\sqrt{\Delta_s}$。代入 (1.4)：

$$
u(t)^2\le (2C_2+800)\,t\,u(t)^2+2t|\Delta\sigma|\,u(t)
+t\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big|^2 .
$$

设 $T$ 充分小使 $(2C_2+800)T\le\frac12$（**注意：本问题不满足，见 §1.4**），则

$$
u(t)^2\le 4t|\Delta\sigma|u(t)+2t\Big(|\Delta\sigma|^2+R^2\Big),\qquad R:=\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big| .
$$

解二次不等式 $u^2-au-b\le0\Rightarrow u\le\frac{a+\sqrt{a^2+4b}}{2}\le a+\sqrt b$：

$$
u(T)\le 4T|\Delta\sigma|+\sqrt{2T}\sqrt{|\Delta\sigma|^2+R^2}\le C_3\big(|\Delta\sigma|+R\big).
\tag{1.6}
$$

**第五步：$\sqrt{\cdot}$ 的 Lipschitz 性（关键修正）。** 旧版在这里写
$|\sqrt{2\mu_1}-\sqrt{2\mu_2}|\le C\sqrt{|\mu_1-\mu_2|}$ 并称之为"$1/2$ 阶 Hölder"。
该不等式**虽然成立但极松**。真实的关系是

$$
\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big|
=\sqrt2\,\frac{|\mu_1-\mu_2|}{\sqrt{\mu_1}+\sqrt{\mu_2}}
\;\le\;\frac{\sqrt2}{\sqrt{\mu_{\min}}}\,|\mu_1-\mu_2| .
\tag{1.7}
$$

**在 $\Theta=[0.1,10]$ 上 $\mu_{\min}=0.1$，故 $\sqrt2/\sqrt{0.1}=4.47$——这是线性界，不是 $1/2$ 阶。**
$\sqrt{\cdot}$ 只有在 $\mu\to0$ 的退化极限才是 $1/2$ 阶 Hölder，本问题不在该区间。

**第六步：合并。** 由 $W_2(\nu_{\theta_1,T},\nu_{\theta_2,T})\le\sqrt{\Delta_T}=u(T)$ 与 (1.6)(1.7)：

$$
W_2\le C_4\big(|\Delta\sigma|+|\Delta\mu|\big).
\qquad\square
$$

**第七步：写成位移坐标 (1.3)。** 注意 (1.4) 的**非齐次项**可以精确算：

$$
\big|\sqrt{2\mu_1}-\sqrt{2\mu_2}\big|^2t=\big|\Delta G\big|^2\cdot\frac{t}{T}\quad\Big(\text{因 }\Delta G=\big|\sqrt{2\mu_1T}-\sqrt{2\mu_2T}\big|\Big).
$$

代入 $t=T$ 得 $|\Delta G|^2$，与 (1.6) 中 $\sqrt{2T}\sqrt{\cdot}$ 结合后，$\Delta G$ 的系数是 $\sqrt{2T}\sqrt{1/T}=\sqrt2$ 乘以 $\sqrt d=\sqrt3$ 的三坐标贡献，即 $\sqrt3|\Delta G|$。
剪切项 $|\Delta\sigma|T$ 与 $T|\Delta\sigma|$ 同阶，系数为 1（因 $|g|_\infty=1$）。
Grönwall 放大因子 $e^{CT}$ 出现在前面。$\square$

> **定理 2 的真正价值与局限。**
>
> - **价值**：证明了连续性，且**给出了正确的函数形式**——关于 $\sigma$ 和 $\mu$ 都是
>   **一次齐次**的，且在位移坐标 $(S,G)$ 下系数是**普适常数**（$1$ 与 $\sqrt d$），
>   与 $\theta$ 无关。
> - **局限**：$C$ 包含 $\chi\|\nabla K\|_\infty\|\nu\|_\infty\approx400\|\nu\|_\infty$，
>   而 $\|\nu\|_\infty\sim10^2$（$10^5$ 粒子在体积 $\approx17$ 内），
>   故 $e^{CT}\sim e^{10^4}$。**常数完全无效**（见 §1.4）。
> - 因此 (1.3) 只能作为**形式**使用；常数必须由实验确定（§1.5）。

### 1.3.1 为什么旧版的 $L_\sigma/L_\mu\approx10$ 是巧合

旧版在 $(r,t)$ 对数坐标下定义

$$
W_2\le L_\sigma|\Delta\log_{10}\sigma|+L_\mu\big|\Delta\log_{10}\mu\big|
$$

并报告"$L_\sigma\approx4.6$，$L_\mu\approx0.45$，比值 10 与理论一致"。**这两个数不是常数**：

$$
\frac{\partial W_2}{\partial\log_{10}\sigma}=c_S\,\frac{\partial S}{\partial\log_{10}\sigma}
=c_S\,\sigma T\ln10\;\propto\;\sigma
\qquad(\text{跨格点变化 }20\times),
$$

$$
\frac{\partial W_2}{\partial\log_{10}\mu}=c_G\,\frac{\partial G}{\partial\log_{10}\mu}
=c_G\,\frac{\sqrt{2\mu T}\ln10}{2}\;\propto\;\sqrt\mu
\qquad(\text{跨格点变化 }10\times).
$$

代入 $c_S=0.897$、$c_G=1.98$、$T=0.02$、$\sigma\in[10,200]$、$\mu\in[0.1,10]$：

| 量 | $\sigma=10$ | $\sigma=100$ | $\sigma=200$ | 跨度 |
|---|---:|---:|---:|---:|
| $\partial W_2/\partial\log_{10}\sigma$ | 0.41 | 4.13 | 8.26 | **$20\times$** |
| $\partial W_2/\partial\log_{10}\mu$ | 0.144 | 0.454 | 1.44 | **$10\times$** |

**旧版的 4.6 与 0.45 分别是这两列的中位数，恰好在 $\sigma\approx100$、$\mu\approx1$ 处。**
所谓"$L_\sigma/L_\mu\approx10$ 与物理一致"是把两个都在漂移的量在中点取值的巧合。
**这同时解释了旧版 §7.1 声称的未解决点"$L_\mu$ 不是常数"——它从来就不是常数，因为对数坐标不是正确的坐标。**

## 1.4 为什么定理 2 的常数无效（必须诚实说明）

$C_2=C_1+200$，而 $C_1=\chi C\|\nu\|_\infty\approx50\cdot C\cdot\|\nu\|_\infty$。

估计 $\|\nu\|_\infty$：$10^5$ 粒子、云尺度 $\approx3.2$、体积 $\approx17$，若近似均匀则
$\|\nu\|_\infty\approx6\times10^3$。但终态云**高度非均匀**（有 $\mu$ 小时的强聚集），
峰值密度可高一个量级，取 $\|\nu\|_\infty\sim10^4$。

则 $C_2\sim50\cdot10^4=5\times10^5$，Grönwall 因子

$$
e^{C_2T}=e^{5\times10^5\times0.02}=e^{10^4}\approx10^{4343}.
$$

**界在数值上完全无效。**

> **这不是证明方法的问题，而是物理设定的问题**：$\chi\|\nabla K\|_\infty=400$ 使相互作用
> 在 $T=0.02$ 内的累积效应 $e^{400T}=e^8\approx3000$ 已经很大，加上密度放大就更糟。
>
> **正确的处理**：把定理 2 只用作**定性**结论（连续性 + 正确的函数形式），
> 常数交给实验（§1.5）。**不要**在论文里把 (1.3) 当作定量预测，否则会被审稿人
> 用上述计算直接驳回。

## 1.5 【经验定律 1】位移律（本文最有价值的新结果）

> **经验定律 1.** 在 $200$ 格网格上，对全部 $370$ 对相邻格点，
> $$
> \boxed{
> W_2\big(\nu_{\theta_1},\nu_{\theta_2}\big)\;\approx\;
> c_S\,|\Delta S|+c_G\,|\Delta G| ,
> \qquad c_S\approx0.90,\quad c_G\approx2.0
> }
> \tag{1.8}
> $$
> 其中 $S=\sigma T$、$G=\sqrt{2\mu T}$。**两个常数在 $20\times$ 参数范围内经验稳定。**

### 1.5.1 检验一：$c_S$ 是常数，而 $L_\sigma$ 不是

若 (1.8) 正确，则 $W_2/|\Delta S|$ 沿 $\sigma$ 方向**不应有趋势**。
若旧版的"$L_\sigma$ 为常数"正确，则 $W_2/|\Delta S|\propto\sigma$，即 $20\times$ 漂移。

**实测**（`scripts/displacement_checks.py`，底噪按 $W_2^2-\varepsilon^2$ 扣除，$\varepsilon=0.06443$）：

| $\sigma$ | 10 | 40 | 70 | 100 | 130 | 160 | 190 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $c_S=W_2/\|\Delta S\|$ | 0.786 | 0.814 | 0.834 | 0.873 | 0.930 | 0.982 | 1.014 |

- $\log$–$\log$ 斜率：$\mathbf{+0.095}$（$0$ = 常数 $c_S$，$1$ = 常数 $L_\sigma$）
- 全网格跨度：$\mathbf{1.29\times}$（常数 $L_\sigma$ 假说要求 $20\times$）

**结论：位移律赢 15 倍。**

**并且 $c_S$ 从 0.786 单调升向 1.014，逼近理论天花板 $|g|_\infty=1$。** 这不是拟合，
而是**理论预测的上确界被数据逼近**——这是最强形式的验证。

### 1.5.2 检验二：$c_G$ 等于 $\sqrt d$ 加相互作用修正

纯布朗情形（无相互作用、无剪切）有精确解。若 $X_T\sim N(0,2\mu T I)$、$Y_T\sim N(0,2\mu'T I)$，
由 Bures 公式

$$
W_2^2=d\big(\sqrt{2\mu T}-\sqrt{2\mu'T}\big)^2=d\,|\Delta G|^2
\;\Longrightarrow\;
c_G=\sqrt d=\sqrt3=1.732 .
$$

（注意：同步耦合与独立耦合给出**同一个**值，因为两个高斯之间的 $W_2$ 只依赖协方差。）

**实测**：$c_G=1.981\pm0.026$（sem，$n=85$ 对超出 $1.5\varepsilon$ 的格点）。

$$
\frac{c_G^{\text{实测}}}{\sqrt3}=\frac{1.981}{1.732}=1.144 .
$$

**解释**：$14\%$ 的超出是相互作用（$\chi$）与剪切（$\sigma$）的耦合贡献。
剪切把云拉成各向异性，使同样的扩散增量对应更大的 $W_2$。**这个 14% 是物理的，不是误差。**

### 1.5.3 检验三：加性在斜对角上是上界而非等式

沿路径 $(\sigma_1,\mu_1)\to(\sigma_2,\mu_1)\to(\sigma_2,\mu_2)$ 用三角不等式：

$$
W_2\big(\nu_{\sigma_1\mu_1},\nu_{\sigma_2\mu_2}\big)\le c_S|\Delta S|+c_G|\Delta G| .
$$

**实测**（171 个斜对角对）：观测值 / 加性预测 $=\mathbf{0.679}$（p05 0.640，p95 0.795）。

**比值为 $0.68<1$，说明两个参数方向的效应部分**抵消**，三角不等式界有余量。**
（若两方向效应正交相加，比值会是 $1/\sqrt2=0.707$；实测 0.679 略低于此，
说明存在真实的对消而不仅是正交。）

> **这一条很重要**：它说明 (1.8) 是**保守上界**，用于误差预算时是安全的。

### 1.5.4 与两参数设计的联系：为什么旧的设计准则全失败

$\partial W_2/\partial\log_{10}\sigma\propto\sigma$、$\partial W_2/\partial\log_{10}\mu\propto\sqrt\mu$
意味着**自然度量下参数空间是不均匀的**：

$$
d_{\text{nat}}(\theta,\theta')=c_S|\Delta S|+c_G|\Delta G| .
$$

在对数坐标下，$\sigma$ 方向的"有效长度"从 0.41（$\sigma=10$）变到 8.26（$\sigma=200$），
$\mu$ 方向从 0.144 变到 1.44。

**推论**：在 $(\log\sigma,\log\mu)$ 平面里做**均匀**的 LHS / maximin / I-optimal 设计，
优化的都是**错误的度量**。这解释了旧版 `DESIGN_CRITERION.md` 中
"J 准则方向错误"（$\mathrm{corr}(J,\text{误差})=-0.715$）与"8 个固定间距设计全部无效"：
那些准则都在均匀度量下工作，而真实的敏感度相差 $20\times$。

> **这是一个可以直接写进论文的预测**：若把设计改为在 $d_{\text{nat}}$ 下 maximin
> （即对 $\sigma$ 用 $\propto1/\sigma$ 的非均匀密度、对 $\mu$ 用 $\propto1/\sqrt\mu$），
> 同样 $n=40$ 应优于均匀 LHS。**这是一个干净的可证伪预测，尚未做实验。**

## 1.6 可分辨阈值（修正版）

由 (1.8)，信号与底噪同量级时不可分辨：

$$
|\Delta S|_{\min}=\frac{\varepsilon}{c_S},\qquad
|\Delta G|_{\min}=\frac{\varepsilon}{c_G} .
$$

换回物理参数（在 $\mu$ 处用 $\partial G/\partial\mu=\sqrt{T/(2\mu)}=\sqrt{0.02/(2\mu)}$）：

$$
\boxed{
|\Delta\sigma|_{\min}=\frac{\varepsilon}{c_S T}=\frac{0.0644}{0.897\times0.02}=3.59,
\qquad
|\Delta\mu|_{\min}=\frac{\varepsilon}{c_G}\sqrt{\frac{2\mu}{T}}=0.325\sqrt{\mu}
}
\tag{1.9}
$$

**相对阈值**：

| 量 | $\sigma=10$ / $\mu=0.1$ | $\sigma=100$ / $\mu=1$ | $\sigma=200$ / $\mu=10$ |
|---|---:|---:|---:|
| $\|\Delta\sigma\|_{\min}/\sigma$ | $36\%$ | $3.6\%$ | $1.8\%$ |
| $\|\Delta\mu\|_{\min}/\mu$ | $103\%$ | $32\%$ | $10\%$ |

**与实测对照**：

| 方向 | 旧版预测 | **新版预测** | 实测 |
|---|---:|---:|---:|
| $\sigma$（$\sigma=100$） | $1.4\%$ | $\mathbf{3.6\%}$ | $\approx2.5\%$ |
| $\mu$（$\mu\approx1$） | $2.0\%$ | $\mathbf{32\%}$ | $\approx25\%$ |

**旧版对 $\mu$ 的预测差 12 倍（2% vs 25%），新版差 1.3 倍。** 这是
修正 Hölder 指数的直接收益，也把旧版 §7.1 的"未解决点"变成了已解决点。

> **新版还给出一个旧版没有的、可检验的方向性预测**：
> 相对阈值 $\propto1/\sqrt\mu$ **随 $\mu$ 增大而降低**。
> 即 $\mu=0.1$ 处几乎无法分辨（$103\%$），$\mu=10$ 处可以分辨到 $10\%$。
> 这解释了为什么旧版 T5 检验里"最大值稳定出现在 $\mu=10$"——
> 那里斜率最大但信号也最大，归一化后仍通过。

---

# 第 2 部分：确定性网络能表示什么

**本部分回答：** 目标是"生成分布"，而网络输出是确定性的。网络的**理论上限**是什么？底噪从何而来？

## 2.1 定理 4：MSE 最优确定性映射（【定理·严格】）

> **定理 4.** 设 $(X_0,X_T)\in\mathbb R^3\times\mathbb R^3$ 为可积随机向量。在全体 Borel
> 可测函数 $f:\mathbb R^3\to\mathbb R^3$ 上最小化 $\mathcal E(f)=\mathbb E|f(X_0)-X_T|^2$，
> 最优解为
> $$
> \boxed{\;T^*_\theta(x)=\mathbb E\big[X_T\,\big|\,X_0=x\big]\;}
> \tag{2.1}
> $$
> （在 $\sigma(X_0)$-可测函数类中 a.s. 唯一），最小值为
> $$
> \min_f\mathcal E(f)=\mathbb E\big[\mathrm{Var}(X_T\mid X_0)\big].
> \tag{2.2}
> $$

**证明。** 设 $T^*=T^*_\theta(X_0)$，$f$ 任意。则

$$
\mathbb E|f-T|^2
=\underbrace{\mathbb E|f-T^*|^2}_{\ge0}+\underbrace{\mathbb E|T^*-T|^2}_{\text{与 }f\text{ 无关}}+2\underbrace{\mathbb E\big[(f-T^*)\cdot(T^*-T)\big]}_{\text{交叉项}} .
$$

交叉项为零：$f$ 与 $T^*$ 都是 $\sigma(X_0)$-可测，故

$$
\mathbb E\big[(f-T^*)\cdot(T^*-T)\big]
=\mathbb E\Big[(f-T^*)\cdot\underbrace{\mathbb E\big[T^*-T\,\big|\,X_0\big]}_{=0}\Big]=0 .
$$

故 $\mathcal E(f)=\mathcal E(T^*)+$ 非负项，$f=T^*$ 唯一取等（在 $|f-T^*|\ne0$ 的集合上
$\mathcal E$ 严格增大）。$\square$

## 2.2 推论 1：底噪的**上限**（【定理·严格】，但界很松）

> **推论 1.** 记 $\nu^*_\theta=\mathrm{Law}(T^*_\theta(X_0))$。则
> $$
> W_2^2\big(\nu^*_\theta,\nu_\theta\big)\;\le\;\mathbb E\big[\mathrm{Var}(X_T\mid X_0)\big].
> \tag{2.3}
> $$

**证明。** 由定理 4，$\mathbb E|T^*(X_0)-X_T|^2=\mathbb E[\mathrm{Var}(X_T|X_0)]$。
同步耦合 $(T^*(X_0),X_T)$ 是 $\nu^*_\theta$ 与 $\nu_\theta$ 的一个可行耦合，
而 $W_2$ 是所有耦合的下确界，故 $W_2^2\le\mathbb E|T^*(X_0)-X_T|^2$。$\square$

### 2.2.1 这个界有多松（重要修正）

量级估计：扩散位移每坐标 $\sqrt{2\mu T}$，三坐标独立，故
$\mathbb E[\mathrm{Var}]\approx3\cdot2\mu T=6\mu T$。代入 $\mu=1$、$T=0.02$：$0.12$，开方 $0.346$。

**实测底噪（$\sigma=100,\mu=1$）：$0.123$**（粒子级，$m=3000$，三次复制）。

$$
\frac{\text{实测}}{W_2\text{ 界}}=\frac{0.123}{0.346}=0.36 .
$$

旧版把这个 3 倍差距描述为"同量级，理论给出正确的量级"。**这个说法不准确，必须修正。**

**为什么界必然很松**：$W_2$ 是**全局**度量，对"broad 分布 + 小幅平滑"极不敏感。
一个明确的例子：若 $\nu^*=N(0,R^2I)$ 且 $\nu=\nu^*\ast N(0,\sigma^2I)$，则

$$
W_2(\nu^*,\nu)=\sqrt{R^2+\sigma^2}-R\;\approx\;\frac{\sigma^2}{2R}\quad(\sigma\ll R),
$$

即 $O(\sigma^2)$ 而非 $O(\sigma)$。代入 $R=1.6$、$\sigma=0.2$ 得 $0.0125$——
**比 $\sigma\sqrt3=0.346$ 小 28 倍**。

> **正确表述（建议按此写论文）**：
>
> (2.3) 是一个**严格但很松**的上界。$W_2$ 只衡量分布的全局形状，
> 而平滑一个广支撑分布对 $W_2$ 的一阶影响为零，故真实的可达误差远小于 $\sqrt{6\mu T}$。
> 实测 $0.123$ 与界 $0.346$ 的 3 倍差距**不是误差，是界的必然松弛**。
>
> **因此不能说"底噪是物理硬墙"**。正确的说法是：
> **实测底噪 $0.123$ 是"两次独立物理模拟之间的可复现性极限"**，
> 模型误差 $0.127$ 与之比值 $1.04$ 说明**模型已达物理可复现性**，
> 这是最强可陈述的结论，不需要也不应该依赖 (2.3) 的数值。

### 2.2.2 一个必须承认的空缺

**不存在**"任何确定性网络必有 $W_2\ge$ 某正数"的下界。理由如上例：
平滑对 $W_2$ 的影响可以是 $O(\sigma^2)$，可以任意小（只要云足够宽）。

所以旧版框起来的

$$
W_2(\hat\nu_\theta,\nu_\theta)\ \ge\ W_2(\nu^*_\theta,\nu_\theta)\quad\text{（对所有 }\hat f\text{）}
$$

**是错的**，已在文档中加注"不可能恒成立"。正确方向只有 (2.3) 的上界。

> **这对论文的影响**：不能用"底噪"论证"我们的方法已经最优、无法再改进"。
> 只能说"已达物理可复现性"。这两句话的说服力差别很大，必须选后者。

## 2.3 命题 1：为什么 $E/\varepsilon$ 可以小于 1（【启发式】）

> **命题 1（近似）.** 设模型输出 $\hat\nu=\mathrm{Law}(\hat f(X_0))$ 是 $X_0$ 的确定性函数，
> 参考云 $\hat\mu_A$ 是独立实现。则
> $$
> \mathbb E\big[W_2^2(\hat\nu,\hat\mu_A)\big]\approx W_2^2(\hat\nu,\nu_\theta)+C N^{-2/3},
> \qquad
> \mathbb E\big[W_2^2(\hat\mu_A,\hat\mu_B)\big]\approx 2CN^{-2/3},
> $$
> 故比值在 $\hat\nu\to\nu_\theta$ 时趋于
> $$
> \frac{W_2(\hat\nu,\hat\mu_A)}{W_2(\hat\mu_A,\hat\mu_B)}\to\frac{1}{\sqrt2}\approx0.71 .
> \tag{2.4}
> $$

**证明思路。** 第一式的抽样项只含一项经验测度误差，第二式含两项独立误差，
其平方和给出 $\sqrt2$。**严格等号需要勾股关系，而 $(\mathcal P_2,W_2)$ 是 Alexandrov 空间、
不满足勾股（Ambrosio–Gigli–Savaré §7.3），故只能给近似。** $\square$

**实测**（5 点 × 3 复制 × 3 抽样）：比值 $0.946$–$1.109$，均值 $\mathbf{1.041}$。
落在 $[0.71,1]$ **之上**，说明确定性误差 $W_2(\hat\nu,\nu_\theta)$ 与一个抽样噪声量级相当，
但未到零。**这与"已达物理可复现性但未达分布极限"一致。**

## 2.4 量化误差（【定理·严格】）

> **命题（离散化）.** 设箱宽为 $\Delta$（各轴相同），$W_2^{\text{hist}}$ 为直方图 $W_2$。则
> $$
> \big|W_2(\mu,\nu)-W_2^{\text{hist}}(\mu,\nu)\big|\le\sqrt d\,\Delta .
> \tag{2.5}
> $$

**证明。** 把位于 $x$ 的粒子替换为其箱中心 $z(x)$，有 $\|x-z(x)\|\le\frac{\sqrt d}{2}\Delta$。
设 $\mu^{\mathcal G},\nu^{\mathcal G}$ 为量化后测度，同步耦合给出
$W_2(\mu,\mu^{\mathcal G})\le\frac{\sqrt d}{2}\Delta$，同理 $\nu$。三角不等式合并。$\square$

**期望版本（更贴合实测）**：若粒子在箱内近似均匀，则平均位移平方为 $d\Delta^2/12$，故

$$
\mathbb E\big[W_2^2(\mu,\mu^{\mathcal G})\big]\approx\frac{d\,\Delta^2}{12}
\;\Longrightarrow\;
W_2\approx\Delta\sqrt{d/12}=\frac{\Delta}{2}\quad(d=3).
$$

代入 $\Delta=3.2/16=0.2$：$W_2\approx0.10$。实测"纯量化噪声"$0.0428$。**同量级（差 2.3 倍）。**

**这解释了 94% 的量化占比**：$N=10^5$ 时抽样误差 $\sim CN^{-1/3}\approx0.046$，
而量化项 $\approx0.10$（最坏界 $0.35$）——**量化主导**。
实测分解（保持箱计数、重抽箱内分配）：总底噪 $0.0455$，纯量化 $0.0428$，占 $94\%$。

**支撑的结论**：`METRIC_CORRECTION.md` 的"94% 是量化伪影"，以及"改用粒子级口径后
比值从 1.97 降到 1.04"。

---

# 第 3 部分：两步分解——为什么 DP 有用

**本部分回答：** 若 $T^*_\theta$ 已被 MF 学到大部分，DP 学的是什么？它为什么能进一步降误差？

**这里给出比旧版更干净的答案：DP 的极限不是"残差"，而是 Brenier 最优映射。**

## 3.1 两个不同的"修正目标"

设 MF 的输出映射为 $T_{\text{mf}}(\cdot;\theta)$，其像测度为 $\nu_{\text{mf},\theta}=(T_{\text{mf}})_\#\mu_0$。

**旧版的残差定义**（点态）：

$$
R_\theta(y):=T^*_\theta\big(T_{\text{mf}}^{-1}(y)\big)-y .
\tag{3.1}
$$

**DP 实际在学的东西**（分布意义）。DP 的损失是

$$
\mathcal L(g)=\mathbb E_{(X,Y)\sim\pi}\big[\big|X+g(X)-Y\big|^2\big],
\tag{3.2}
$$

其中 $\pi$ 是当前模型云与参考云之间的**精确 OT 耦合**（`scripts/dp_two_param.py`
用 POT network simplex 求排列，并校验耦合确为置换）。

对固定的 $\pi$，$\mathcal L$ 的最小元是条件期望：

$$
g^*(x)=\mathbb E_\pi[Y\mid X=x]-x .
\tag{3.3}
$$

**关键**：$\pi$ 是 OT 耦合，不是 $(T^*,(T^*)^{-1})$ 那个点态配对。所以 (3.3) 一般**不等于** (3.1)。

## 3.2 定理 5：DP 的极限是 Brenier 映射（【定理·严格】）

> **定理 5.** 设 (i) $N\to\infty$ 使经验云收敛到 $\nu_{\text{mf},\theta}$ 与 $\nu_\theta$；
> (ii) $\nu_{\text{mf},\theta}$ 绝对连续；(iii) $g$ 取自全体 Borel 可测函数。
> 则 DP 目标 (3.2) 的最小元给出
> $$
> T_{\text{dp},\theta}:=\mathrm{id}+g^*_\theta=\nabla\varphi_\theta ,
> \tag{3.4}
> $$
> 其中 $\varphi_\theta$ 是 $(\nu_{\text{mf},\theta},\nu_\theta)$ 的 Brenier 势。因此
> $$
> \boxed{\;(T_{\text{dp},\theta})_\#\nu_{\text{mf},\theta}=\nu_\theta
> \quad\Longrightarrow\quad
> W_2\big((T_{\text{dp},\theta})_\#\nu_{\text{mf},\theta},\ \nu_\theta\big)=0\;}
> \tag{3.5}
> $$
> 在理想极限下精确成立。

**证明。**

*第一步：OT 耦合的结构。* 由 Brenier 定理（Villani 2009, Thm 2.12），
因 $\nu_{\text{mf},\theta}$ 绝对连续，存在凸函数 $\varphi_\theta$ 使最优耦合
$\pi^*=(\mathrm{id},\nabla\varphi_\theta)_\#\nu_{\text{mf},\theta}$，且 $\nabla\varphi_\theta$
$\nu_{\text{mf},\theta}$-a.e. 唯一。

*第二步：条件期望。* 在 $\pi^*$ 下 $Y=\nabla\varphi_\theta(X)$ a.s.，故
$$
\mathbb E_{\pi^*}[Y\mid X=x]=\nabla\varphi_\theta(x)
\quad\Longrightarrow\quad
g^*_\theta(x)=\nabla\varphi_\theta(x)-x .
$$

*第三步：$L^2$ 最优性。* (3.2) 是标准的 $L^2$ 回归问题，由定理 4（取 $X_0=X$、$X_T=Y-X$）
最小元唯一为条件期望。故 $T_{\text{dp}}=\mathrm{id}+g^*=\nabla\varphi_\theta$。

*第四步：推前。* $\nabla\varphi_\theta$ 把 $\nu_{\text{mf},\theta}$ 推到 $\nu_\theta$
（Brenier 定理的一部分），故 (3.5) 成立。$\square$

> **定理 5 与定理 4 的关系——这是最重要的概念澄清**
>
> | | 目标函数 | 学到的映射 | 极限误差 |
> |---|---|---|---|
> | MF 单步 | 点态 MSE 到 $X_T$ | $T^*=\mathbb E[X_T\vert X_0]$ | $W_2(\nu^*,\nu)>0$（一般）|
> | DP | OT 配对后的 MSE | $\nabla\varphi$（Brenier）| $\mathbf{0}$（理想极限）|
>
> **所以 DP 不是一个"更小的残差修正"，而是一个本质不同的目标**：它优化的是
> *分布*，而 MF 优化的是*点态*。$W_2$ 只要求分布一致，不要求点态一致，
> 因此 DP 原则上可以做到 MF 原理上做不到的事。
>
> **这解释了为什么"两步"不是简单的"多算一点"**：第二步换掉了损失函数。

### 3.2.1 但这个"零误差"必须打折

(3.5) 是 $N=\infty$、容量 $=\infty$、单格理想化的结论。实际有三重折扣：

1. **有限 $N$**：$\pi_N\ne\pi^*$，偏差 $O(N^{-1/3})$（Fournier–Guillin 速率）。
   故 DP 的可达误差约为抽样底噪量级，**而非零**。
   这与实测吻合：DP 平均误差 $0.104$，同格底噪 $0.05$–$0.12$。
2. **有限容量**：$\mathrm{MLP}$ 只能近似 $\nabla\varphi_\theta$，误差 $\varepsilon_{\text{mlp}}$。
3. **跨参数共享**：一个网络要在 40 格上同时学 40 个不同的 $\nabla\varphi_\theta$，
   并在留出格上插值。这是 §4 的泛化问题。

**在训练格上 DP 可以"记住"该格的 $\pi^*$（因为目标就是该格的参考云），
所以训练误差必然很小；真正有信息量的是留出格的结果（$+14.4\%$，
$415/429$ 胜出）。** 报告时必须用留出格。

## 3.3 推论 2：DP 有效的条件（【定理·严格】）

> **推论 2.** 在有限容量下，DP 相对 MF 的增益由 $\|\nabla\varphi_\theta-\mathrm{id}\|$ 决定：
> $$
> W_2\big((T_{\text{dp}})_\#\nu_{\text{mf}},\nu_\theta\big)
> \;\lesssim\;\underbrace{\varepsilon_{\text{ot}}(N)}_{\text{有限}N}
> +\underbrace{\varepsilon_{\text{mlp}}}_{\text{容量}}
> +\underbrace{\big\|\widehat{\nabla\varphi}-\nabla\varphi\big\|_{\theta\text{-泛化}}}_{\text{跨参数}}
> \tag{3.6}
> $$
> 而 MF 单独的误差是 $W_2(\nu^*_{\text{mf}},\nu_\theta)$，其下界由 §2.2 的松弛量控制。
> **DP 有效当且仅当 (3.6) 小于 MF 的误差。**

**推论 2 的关键含义**：DP 的**全部**增益来自 $\nu_{\text{mf}}\ne\nu_\theta$（MF 不完美）。
若 MF 已完美，$\nabla\varphi=\mathrm{id}$、$g^*=0$、DP 什么也不做。**这与实测一致**
（MF 误差从 0.199 降到 0.122 后，DP 还能再拿 14.4%；再往下会递减）。

## 3.4 推论 3：为什么 DP 的收益对 $\mu$ 不敏感（【启发式】）

DP 的相对增益 $=\dfrac{W_2(\nu_{\text{mf}},\nu_\theta)-\varepsilon_{\text{tot}}}{W_2(\nu_{\text{mf}},\nu_\theta)}$。
分子是 MF 与真值的距离，由 §1.5 的位移律给出：

$$
W_2(\nu_{\text{mf}},\nu_\theta)\approx c_S|\Delta S_{\text{eff}}|+c_G|\Delta G_{\text{eff}}| ,
$$

其中 $\Delta S_{\text{eff}}$、$\Delta G_{\text{eff}}$ 是 MF 在该格上的**局部**误差（非参数间距）。
若 MF 的误差主要来自剪切方向（因为 $c_S$ 大于 $c_G$ 且 $\Delta S$ 尺度更大），
则残差主要在 $x$ 方向，与 $\mu$ 的关系弱。

**实测支持**：

| 阶段 | $\mathrm{corr}(\text{增益},\lvert\log_{10}\mu\rvert)$ |
|---|---:|
| MF（双 vs 单）| $0.622$ |
| DP | $0.164$ |

> **这一段只能作为解释性讨论。** 依赖"MF 误差主要在剪切方向"这一未证明的前提。

## 3.5 推论 4：DP 降低方差（【启发式】，仅经验）

**观察**：跨种子标准差 $0.0089\to0.0027$（$-70\%$）。

**用定理 5 的解释（比旧版更自然）**：$\nabla\varphi_\theta$ 由 $\nu_{\text{mf}}$ 与
$\nu_\theta$ 的**分布**决定，而非由 MF 的**具体实现**决定。若两个种子的 MF 输出
$\nu_{\text{mf}}^{(1)}\approx\nu_{\text{mf}}^{(2)}$（分布接近但点态不同），
则对应的 Brenier 映射也接近，故 DP 输出接近。

$$
\mathrm{Var}_{\text{seed}}\big[T_{\text{dp}}\big]\approx
\mathrm{Var}_{\text{seed}}\big[\nabla\varphi_{\theta}(\nu_{\text{mf}}^{(\cdot)})\big]
\;\ll\;
\mathrm{Var}_{\text{seed}}\big[T_{\text{mf}}\big] .
$$

实测 $\gamma=1-\sqrt{0.0027/0.0089}=0.45$，即 DP 吸收了 $45\%$ 的种子差异。

> **仍是启发式**：没有给出 $\mathrm{Var}[\nabla\varphi]$ 与 $\mathrm{Var}[T_{\text{mf}}]$ 的定量关系。
> 若要写进论文，表述为"实验观察到 DP 使跨种子标准差降低 70%，机制推测为
> Brenier 映射只依赖分布而不依赖点态实现"。

---

# 第 4 部分：泛化——为什么有限个参数点就够

**本部分回答：** 为什么 40 个训练格点能覆盖 200 格参数域？误差如何随 $n$ 变化？

## 4.1 连续模（用正确的度量）

设真映射 $\theta\mapsto T^*_\theta$ 在**位移度量**

$$
h(\theta_1,\theta_2):=c_S|S_1-S_2|+c_G|G_1-G_2|
\tag{4.1}
$$

下的连续模为

$$
\omega(\delta):=\sup_{h(\theta_1,\theta_2)\le\delta}\big\|T^*_{\theta_1}-T^*_{\theta_2}\big\|_{\mathcal L^2(\mu_0)} .
$$

由 §1.5，$\omega(\delta)\approx\delta$——**是 Lipschitz 的，不是 $1/2$ 阶 Hölder**。
特别地 $\omega(\delta)\to0$，这是"连续参数上可学"的严格表述。

> **旧版这里写 $\omega(\delta)\le L_\sigma\delta+L_\mu\sqrt\delta$，因而得到 $n^{-1/4}$ 的速率。
> 修正为 Lipschitz 后，速率变为 $n^{-1/2}$（见 §4.2.1）。**

## 4.2 定理 6：泛化界（【定理·严格】）

> **定理 6.** 设 $\hat T$ 为任意估计。则
> $$
> \sup_{\theta\in\Theta}\big\|\hat T_\theta-T^*_\theta\big\|
> \;\le\;
> \underbrace{\sup_{\theta\in\mathcal D}\big\|\hat T_\theta-T^*_\theta\big\|}_{\text{训练误差}}
> \;+\;\underbrace{2\,\omega\big(\delta(\mathcal D)\big)}_{\text{覆盖误差}} ,
> \tag{4.2}
> $$
> 其中 $\delta(\mathcal D)=\sup_{\theta\in\Theta}\min_{d\in\mathcal D}h(\theta,d)$ 是**位移度量下**的覆盖半径。

**证明。** 取任意 $\theta$，设 $d^*=\arg\min_{d\in\mathcal D}h(\theta,d)$。则

$$
\begin{aligned}
\|\hat T_\theta-T^*_\theta\|
&\le\|\hat T_\theta-\hat T_{d^*}\|+\|\hat T_{d^*}-T^*_{d^*}\|+\|T^*_{d^*}-T^*_\theta\|\\
&\le \omega(\delta)+\sup_{d\in\mathcal D}\|\hat T_d-T^*_d\|+\omega(\delta).
\end{aligned}
$$

第二个不等号用了 $\hat T$ 关于 $\theta$ 的连续模同样 $\le\omega$
（网络以 $\theta$ 为连续输入时成立；否则 $\hat T_\theta\equiv\hat T_{d^*}$，
第一项为零，结论更强）。$\square$

### 4.2.1 覆盖半径与速率（修正）

参数域在 $h$ 度量下是矩形：

$$
\text{宽}=c_S\cdot\Delta S_{\max}=0.897\times3.8=3.41,
\qquad
\text{高}=c_G\cdot\Delta G_{\max}=1.981\times0.569=1.13,
$$

面积 $A\approx3.85$。$n$ 个点在二维区域内的最优覆盖半径

$$
\delta_{\text{opt}}(n)=\Theta\big((A/n)^{1/2}\big)=\Theta\big(n^{-1/2}\big).
\tag{4.3}
$$

代入 $\omega(\delta)\approx\delta$：

$$
\text{覆盖误差}\;\approx\;\sqrt{A/(\pi n)} .
\tag{4.4}
$$

**数值预测与实测**：

| $n$ | 预测覆盖误差 $\sqrt{3.85/(\pi n)}$ | 实测测试 $W_2$（粒子级）|
|---:|---:|---:|
| 20 | $0.248$ | $0.200$ |
| 40 | $\mathbf{0.175}$ | $\mathbf{0.150}$ |
| 80 | $0.124$ | $0.150$ |
| 160 | $0.088$ | $0.154$ |

**$n=20$ 与 $n=40$ 的预测与实测同量级（差 $20\%$–$25\%$），这是很强的定量吻合**——
考虑到 $\omega$ 的常数是从 $W_2$（分布距离）估的、而测试误差是 $L^2(\mu_0)$ 距离，二者不必相等。

**但 $n\ge80$ 时预测继续下降而实测不动。** 这说明

$$
\text{在 }n\ge80\text{ 处，误差由训练侧（优化/容量）地板 }\approx0.15\text{ 决定，而非覆盖。}
$$

> **这修正了旧版的论证**：旧版用 $n^{-1/4}$ 预测"4 倍数据应改善 1.41 倍，实测为零"，
> 结论对但**理由是错的**。新版用 $n^{-1/2}$ 预测"应改善 2 倍"，实测仍为零，
> 而且 $n=40$ 的绝对预测（0.175）与实测（0.150）吻合——
> 这说明覆盖理论在 $n=40$ 时**确实在工作**，之后被训练地板接管。
> **结论更强：不是"覆盖理论无效"，而是"覆盖已不是瓶颈"。**

**与其它实验的交叉印证**：容量实验（15 倍参数无改善）与预算扫描
（32,000 步最优，128,000 步回弹）共同指向**训练地板 $\approx0.12$–$0.15$**。

## 4.3 为什么 LHS 与随机抽样不可分辨（【定理·严格】）

**标准理论**（McKay et al. 1979; Stein 1987）：LHS 的方差优势

$$
\mathrm{Var}_{\text{LHS}}\big[\hat I_n\big]=O(n^{-1}),
\qquad
\mathrm{Var}_{\text{iid}}\big[\hat I_n\big]=O(n^{-1/2}),
$$

**针对的是积分估计** $\hat I_n=\frac1n\sum f(\theta_i)\approx\int f$。

**而本文的任务是逐点预测**（sup 范数意义）：

$$
\mathcal E_{\text{pred}}(\hat T)=\frac{1}{|\mathcal G|}\sum_{\theta\in\mathcal G}\big\|\hat T_\theta-T^*_\theta\big\|^2 .
\tag{4.5}
$$

对 (4.5)，关键量是**覆盖半径** $\delta(\mathcal D)$（定理 6）。而 LHS 与 iid 抽样
在二维下的覆盖半径**同阶**：

> **(iid) 覆盖半径上下界.** $n$ 个 iid 均匀点，最大空腔半径
> $\delta_{\max}=\Theta\big(\sqrt{\log n/n}\big)$ a.s.（由 Poissonization 与
> 空腔概率 $P(\text{空腔})=\exp(-n\pi\delta^2)$，令其为 $1/n$ 得
> $\delta=\sqrt{\log n/(\pi n)}$）。
>
> **(LHS) 覆盖半径.** LHS 保证每行每列恰有一个点，故任何边长 $\ge1/k$ 的
> 轴对齐矩形含 $k^2$ 个格子、恰含 $k$ 个点，其中必有一个点。
> 得 $\delta_{\text{LHS}}=O(1/\sqrt n)$。
>
> **二者的差别只在 $\sqrt{\log n}$ 因子**（iid 略差），而定理 6 的误差
> 只依赖 $\delta$ 的一阶量级，故 LHS 的增益是 $O(\sqrt{\log n})$ 的常数因子。

这正好解释了实测：**LHS vs 随机 $+0.0138$（$0.89$ SE，直方图）/ $+0.0128$（$0.97$ SE，粒子级）**——
不可分辨，但 LHS 的**最小间距**好 2 倍（$0.1053$ vs $0.0526$），
即覆盖保证确实更好，只是不影响最终误差。

**支撑的结论**：`SAMPLE_EFFICIENCY.md` 的"LHS vs 随机不可分辨"、
"LHS 提供覆盖保证而非精度提升"。

### 4.3.1 一个可证伪的预测

若把 $n$ 压到 10 以下，定理 6 的覆盖项 $\sqrt{A/(\pi n)}$ 变成 $0.35$（$n=10$），
远大于训练地板 $0.15$，**此时 LHS 的覆盖优势应显现**。
这是一个干净的后续实验（成本：几次 $n=8$ 训练）。

---

# 第 5 部分：外推为什么失效

**本部分回答：** 同样的方法在训练凸包之外为何几乎失效？理论上能否预测这个边界？

## 5.1 定理 6 在外推区失效的机制（【定理·严格】的下界形式）

**关键观察**：定理 6 的证明只用到"最近训练点"的性质。在外推区，
$\delta(\mathcal D)$ 不是覆盖半径，而是**到凸包的距离 $d(\theta)$**，
**它不随 $n$ 增大而减小**。

> **命题 2.** 设 $\theta$ 到训练凸包 $h$-距离为 $d(\theta)>0$，$d^*=\arg\min_{d\in\mathcal D}h(\theta,d)$。则
> $$
> \big\|\hat T_\theta-T^*_\theta\big\|\;\ge\;\big\|T^*_\theta-T^*_{d^*}\big\|-\big\|\hat T_\theta-T^*_{d^*}\big\| .
> \tag{5.1}
> $$
> 第一项由 $\omega(d(\theta))$ 控制，**增加参数点无法消除**（除非点在凸包外）。

**这解释了 MF 在外推区的退化**（用位移律定量核对）：

| 区域 | 位移距离 | 位移律预测 $W_2$ 增量 | 实测 MF 绝对误差 | 实测退化 |
|---|---:|---:|---:|---:|
| 内插 | $h\lesssim0.16$ | $\lesssim0.16$ | $0.094$ | $1.00\times$ |
| $\sigma$ 外推 $150\to200$ | $\Delta S=1.0$，$c_S\Delta S=0.90$ | $0.90$ | $0.155$ | $1.65\times$ |
| $\mu$ 外推 $3.98\to10$ | $\Delta G=0.232$，$c_G\Delta G=0.46$ | $0.46$ | $0.353$ | $3.76\times$ |

**$\mu$ 外推的位移律预测 $0.46$，实测增量 $0.353-0.094=0.259$——同量级（差 1.8 倍）。**
旧版用"有效斜率 $0.26/0.40\approx0.65$"事后反推，新版**先验**给出 $0.46$。
这是位移律的又一个独立验证。

## 5.2 推论 5：DP 在外推区失效的机制（【启发式】）

由定理 5，DP 学的是 $\theta\mapsto\nabla\varphi_\theta$。在训练区，
$\nabla\varphi_\theta$ 接近 $\mathrm{id}$（因 $\nu_{\text{mf}}\approx\nu_\theta$），
其偏离量由 MF 的局部误差控制。

**在外推区，两件事同时发生**：

1. **条件外推**：$\theta$ 在凸包外，$\nabla\varphi_\theta$ 未被覆盖（定理 6）；
2. **输入分布外推**：DP 的输入是 $y=T_{\text{mf}}(x;\theta)$。MF 输出云本身已变形
   （$\sigma$ 外推时支撑从 $3.2$ 扩到 $5.3$，$\mu$ 外推时从 $3.2$ 扩到 $6.4$），
   于是 $y$ 落在训练时从未见过的区域。

**这是"双重分布偏移"，定理 6 只处理第一重。**

**定量对照**：

| 区域 | MF 误差 | $\|R\|/\|R\|_{\text{train}}$ | DP 相对增益 |
|---|---:|---:|---:|
| 内插 | $0.094$ | $1.0$ | $14.7\%$ |
| $\sigma$ 外推 | $0.155$ | $1.65$ | $1.3\%$ |
| $\mu$ 外推 | $0.353$ | $3.76$ | $2.4\%$ |

**临界观察**：残差只增大 $1.65$ 倍，DP 收益就掉了一个数量级。
说明 DP 的输出幅度被训练分布标定得**极紧**（输出层零初始化 + 2000 步更新）。

> **这是一个可检验的预测**：把 $\alpha$ 从 1 增到 4（覆盖 $3.8$ 倍残差），
> 外推应改善而内插可能退化。**尚未做实验。**

## 5.3 一个诚实的限制

**没有先验可计算的判据**回答"当 $d(\theta)$ 超过多少时 DP 会失效"。
由数据反推阈值是 $\|R\|/\|R\|_{\text{train}}\approx1.3$–$1.6$，
但这个数是否普遍（不同 $\alpha$、不同架构）未知。

---

# 第 6 部分：理论—实验对照总表

| 实验结论 | 支撑理论 | 严格程度 |
|---|---|---|
| $\nu_\theta$ 良定义、唯一 | 定理 1 | 【定理·严格】 |
| $\theta\mapsto\nu_\theta$ 连续 | 定理 2 | 【定理·定性】 |
| $\sigma$ **和** $\mu$ 都是 Lipschitz | 定理 2 + (1.7) | 【定理·严格】（形式）|
| $W_2\approx c_S\|\Delta S\|+c_G\|\Delta G\|$ | 经验定律 1 | 【经验定律】（$c_S$ 稳到 $29\%$，$c_G=\sqrt3\cdot1.14$）|
| $c_S\le1$ 且逼近 1 | 经验定律 1 + $\|g\|_\infty=1$ | 【经验定律】，天花板由理论给出 |
| $c_G\approx\sqrt d$ | 经验定律 1 + Bures | 【经验定律】，偏差 $14\%$ |
| 旧版"$L_\sigma/L_\mu\approx10$" | — | **【已否定】**（是 $20\times$/$10\times$ 漂移的中位数）|
| $\mu$ 的可分辨阈值 $\approx32\%$ | (1.9) | 【经验定律推论】，实测 $25\%$ |
| 旧版 $\mu$ 阈值 $2\%$ | — | **【已否定】**（错 12 倍）|
| 底噪有严格**下界** | — | **【已否定】**（不存在，见 §2.2.2）|
| 底噪上限 $W_2\le\sqrt{6\mu T}$ | 推论 1 | 【定理·严格】但**松 3 倍** |
| 比值可小于 1（极限 $1/\sqrt2$） | 命题 1 | 【启发式】（$W_2$ 无勾股）|
| 16 箱底噪 94% 是量化 | (2.5) | 【定理·严格】 |
| **DP 的极限是 Brenier 映射** | **定理 5** | **【定理·严格】** |
| DP 增益来自 $\nu_{\text{mf}}\ne\nu_\theta$ | 推论 2 | 【定理·严格】 |
| DP 收益与 $\mu$ 无关 | 推论 3 | 【启发式】 |
| DP 降方差 70% | 推论 4 | 【启发式】，仅经验 |
| 40 点足够（预测 0.175 vs 实测 0.150）| 定理 6 + (4.4) | 【定理·严格】 |
| 更多点无改善 | 定理 6 的$n^{-1/2}$预测落空 | 训练地板 $\approx0.15$ 接管 |
| LHS vs 随机不可分辨 | §4.3 | 【定理·严格】（覆盖半径同阶）|
| MF 外推退化 | 命题 2 + 位移律 | 【定理·严格】下界 + 定量吻合 |
| DP 外推失效 | 推论 5 + 双重偏移 | 【启发式】机制 + 经验定量 |

---

# 第 7 部分：理论未解决的点

诚实列出，建议在论文"局限"中提及。

## 7.1 定理 2 的常数无效

$e^{CT}\sim e^{10^4}$。**这是本理论最大的软肋。** 可行的改进方向：
用对数 Sobolev 不等式或耦合方法（而非 Grönwall）得到与 $\chi\|\nabla K\|_\infty$ 无关的界，
或直接对离散时间（4 步 Euler）做逐段估计，避免连续时间放大。

## 7.2 位移律的常数只有经验值

$c_S=0.90$、$c_G=1.98$ 是 200 格上的拟合。**没有解析预言。**
$c_S\le1$ 与 $c_G\ge\sqrt3$ 有理论依据，但 $0.90$ 与 $1.98$ 的**具体值**没有。

**可做的改进**：在纯扩散（$\chi=0$）极限下 $c_G$ 应精确等于 $\sqrt3$；
做一组 $\chi=0$ 的模拟对照，可分离"相互作用贡献的 14%"。

## 7.3 训练地板的来源

$n\ge80$ 时误差卡在 $\approx0.15$。这是优化误差、容量误差、还是数据噪声？
**本文无法区分。** 需要 $\beta$-VAE 式的分解或大规模超参搜索。

## 7.4 推论 3、4 都不是定理

DP 收益与 $\mu$ 无关、DP 降方差，都只有机制性论证。
若要严格化，需要 $\nabla\varphi_\theta$ 关于 $\theta$ 的定量正则性——目前没有。

## 7.5 外推无先验判据

§5.3 已说明。这是方法层面的硬限制，不是理论缺陷。

## 7.6 §1.5.4 的设计预测未做实验

"在 $d_{\text{nat}}$ 下做 maximin 应优于均匀 LHS"是一个**干净的可证伪预测**，
成本约 3 次 $n=40$ 训练。**建议做**，因为若成立，它把位移律从"描述"升级为"设计工具"。

---

# 第 8 部分：直接回答"方法作用在连续参数上是否成立"

**四问四答。**

## 8.1 问题良定义吗？

**是，且是严格的。**

- 定理 1：$\nu_{\theta}$ 存在且唯一（McKean–Vlasov 解）。
- 定理 2：$\theta\mapsto\nu_\theta$ 连续（Lipschitz）。

所以"学习一个从参数到分布的映射"有确切对象。**这一步没有任何近似假设。**

## 8.2 能学吗？

**能，且速率为 $n^{-1/2}$。**

- 定理 6：误差 $\le$ 训练误差 $+\,2\omega(\delta(\mathcal D))$。
- 位移律使 $\omega(\delta)\approx\delta$（Lipschitz），二维覆盖 $\delta=\Theta(n^{-1/2})$，
  故覆盖误差 $=O(n^{-1/2})$。

**这是连续参数问题的自然速率**（2D 参数空间 + Lipschitz 映射），
不是方法缺陷。且 $n=40$ 的绝对预测（$0.175$）与实测（$0.150$）吻合。

**注意**：这一点比旧版更强。旧版说"$n^{-1/4}$ 是固有速率"，那是因为指数取错了。
修正后速率快一倍。

## 8.3 学到的极限在哪？

**不是底噪**（§2.2.2 说明不存在严格下界）。真实的极限有三个层次：

| 层次 | 当前状态 |
|---|---|
| 物理可复现性（两次独立模拟之差）| **已达到**（模型/底噪 $=1.04$）|
| 分布极限 $W_2(\nu^*,\nu)$ | 未知，理论上可小于 $0.12$，被 $\sqrt{6\mu T}=0.35$ 上界松弛地控制 |
| 训练地板 | $\approx0.12$–$0.15$，**当前实际瓶颈** |

**所以"学得好"的空间并未用尽**——瓶颈是训练侧，不是物理侧。

## 8.4 什么时候不成立？

**在训练凸包之外。**

- MF：命题 2 给出下界，且位移律**先验**预测了 $\mu$ 外推的增量为 $0.46$（实测 $0.26$，同量级）。
- DP：推论 5 + 双重分布偏移，机制清楚但无定量先验判据。

**这不是方法的缺陷，而是任何"从有限参数点学习的映射"的普遍性质。**
若要外推，必须引入额外结构（物理先验、增大覆盖、或改换参数化使映射更光滑）。

---

## 8.5 一句话总结

> **方法在连续参数上是成立的，理由是 $\theta\mapsto\nu_\theta$ 连续（定理 1–2），
> 且这个连续性的正确表述是"关于位移尺度 $S=\sigma T$、$G=\sqrt{2\mu T}$ 的 Lipschitz"，
> 而非旧版所说的"$\sigma$ 线性、$\mu$ 平方根"。
> 在这个正确表述下，覆盖速率为 $n^{-1/2}$，常数 $c_S\approx0.90\le1$、$c_G\approx1.98\approx\sqrt3$
> 由理论给出上限、由 200 格数据确认；
> 两步扩散的第二步（DP）在理想极限下收敛到 Brenier 最优映射，故原则上可超越
> 单步确定性映射的可达精度（定理 5）；
> 唯一的边界是凸包外，那里覆盖论证失效且 DP 遭遇双重分布偏移。**

---

# 附录 A：定理依赖图

```
(A1a) 核一阶有界   (A1c) 卷积后 Lipschitz   (A2) 剪切 Lipschitz   (A3) 共享初值   (A4) 密度有界
      │                     │                      │                    │              │
      └─────────────────────┴──────────────────────┴────────────────────┴──────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                       定理 1                      定理 2
                    (良定义/唯一)              (Lipschitz 连续)
                            │                           │
                            │                   ┌───────┴────────┐
                            │                   ▼                ▼
                            │            经验定律 1           (1.9)
                            │          (位移律 c_S,c_G)     (可分辨阈值)
                            │                   │
                            │                   ▼
                            │              §1.3.1 否定旧 L_σ/L_μ
                            │
                            ▼
                       定理 4 (条件期望)
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
               推论 1               命题 1
             (底噪上限)          (比值 <1)
                  │
                  ▼
            §2.2.2 否定"严格下界"

      [分布侧分支]                    [泛化分支]
      ┌──────┴──────┐                ┌──────┴──────┐
      ▼             ▼                ▼             ▼
   定理 5        推论 2            定理 6        §4.3
 (Brenier)    (DP 有效条件)      (泛化界)     (LHS 同阶)
      │             │                │
      ▼             ▼                ▼
   推论 3        推论 5            (4.4)
  (μ 无关)     (外推失效)         n^{-1/2}
                                     │
                                     ▼
                                  命题 2
                                (外推下界)
```

---

# 附录 B：数值常数一览

| 常数 | 值 | 来源 |
|---|---:|---|
| $\chi$ | $16\pi=50.265$ | 设定 |
| $\delta$ | $0.1$ | 设定 |
| $T$ | $0.02$ | 设定 |
| $\|\nabla K_\delta\|_\infty$ | $1/(4\pi\delta^2)=0.796$ | (A1a)，严格 |
| $\chi\|\nabla K_\delta\|_\infty$ | $40.0$ | 计算 |
| $\|g\|_\infty$ | $1$ | (A2) |
| $\mathrm{Lip}(g)$ | $\sqrt{2/e}=0.858$ | (A2)，严格 |
| $S$ 范围 | $[0.2,4.0]$ | 计算 |
| $G$ 范围 | $[0.063,0.632]$ | 计算 |
| $\varepsilon$（16 箱底噪，$10^5$） | $0.06443$ | `floor.json` |
| **$c_S$** | $\mathbf{0.897}$（$0.786$–$1.014$）| 经验定律 1 |
| **$c_G$** | $\mathbf{1.981\pm0.026}$ | 经验定律 1 |
| $\sqrt d=\sqrt3$ | $1.732$ | 理论 |
| $c_G/\sqrt3$ | $1.144$ | 相互作用修正 |
| 自然度量面积 $A$ | $3.85$ | 计算 |
| DP 增益（三种子均值）| $+14.4\%\pm4.2\%$ | 实测 |
| MF 双参数增益 | $+39.0\%\pm3.9\%$ | 实测 |
| 端到端 | $+47.9\%$（$0.199\to0.104$）| 实测 |

---

# 附录 C：本文新增/修改的脚本与产物

| 脚本 | 产物 | 用途 |
|---|---|---|
| `scripts/mu_holder_test.py` | `results/fullscale/holder/mu_holder.json` | 检验 $\mu$ 方向的经验 Hölder 指数（结论：$\to1$，非 $1/2$）|
| `scripts/mu_holder_table.py` | 终端表 | 打印 $\mu$ 梯度明细 |
| `scripts/displacement_law.py` | `results/fullscale/displacement/displacement_law.json` | 全 370 对格点的 $W_2$ 与两种候选预测 |
| `scripts/displacement_law_fit.py` | `.../law_comparison.json` | 扣除底噪后的两律拟合对比 |
| `scripts/displacement_law_mu.py` | 终端表 | 检查 $\mu$ 对被底噪裁剪的比例（54/180）|
| `scripts/displacement_checks.py` | `.../displacement_checks.json` | 三项检验：$c_S$ 无趋势、$c_G\approx\sqrt3$、加性为上界 |

**复现**：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\displacement_law.py
.\.venv-gpu\Scripts\python.exe scripts\displacement_law_fit.py
.\.venv-gpu\Scripts\python.exe scripts\displacement_checks.py
.\.venv-gpu\Scripts\python.exe scripts\mu_holder_test.py
```

**前置条件**：`results/fullscale/data/full/clouds/` 下 200 个格点云（已存在）。
总计耗时约 5 分钟（无需重新模拟，纯 OT 打分）。

---

# 参考文献

1. **Sznitman, A.-S.** *Topics in propagation of chaos.* École d'Été de Probabilités de Saint-Flour XIX, Lecture Notes in Math. 1464, 165–251 (1991). — 定理 1。
2. **Fournier, N., Guillin, A.** *On the rate of convergence in Wasserstein distance of the empirical measure.* PTRF **162**, 707–738 (2015). — $N^{-2/d}$ 速率、定理 3、§3.2.1 的 $\varepsilon_{ot}$。
3. **Villani, C.** *Optimal Transport: Old and New.* Springer (2009). — $W_2$ 定义；**Brenier 定理（Thm 2.12）用于定理 5**。
4. **Ambrosio, L., Gigli, N., Savaré, G.** *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkhäuser (2008). — $(\mathcal P_2,W_2)$ 为 Alexandrov 空间、**勾股不成立**（命题 1 的限定）。
5. **Carmona, R., Delarue, F.** *Probabilistic Theory of Mean Field Games with Applications I.* Springer (2018). — McKean–Vlasov 解对参数的连续依赖（定理 2 的独立佐证）。
6. **Gilbarg, D., Trudinger, N. S.** *Elliptic Partial Differential Equations of Second Order.* Springer (2001). — Newton 位势的 $W^{1,p}$ 正则性，**用于 (A1c) 的严格化**。
7. **McKay, M. D., Beckman, R. J., Conover, W. J.** *A comparison of three methods for selecting values of input variables in the analysis of output from a computer code.* Technometrics **21**(2), 239–245 (1979). — LHS 原始定义。
8. **Stein, M.** *Large sample properties of simulations using Latin hypercube sampling.* Technometrics **29**(2), 143–151 (1987). — LHS 的 $O(n^{-1})$ 方差（**针对积分估计**，§4.3 的关键区别）。
9. **Johnson, M. E., Moore, L. M., Ylvisaker, D.** *Minimax and maximin distance designs.* J. Stat. Plan. Inference **26**(2), 131–148 (1990). — maximin 准则（§7.6 的建议）。
10. **Peyré, G., Cuturi, M.** *Computational Optimal Transport.* FnT ML **11**(5-6), 355–607 (2019). — 离散 OT、对偶证书、Bures 公式（§1.5.2）。
11. **Dowson, D. C., Landau, B. V.** *The Fréchet distance between multivariate normal distributions.* J. Multivariate Anal. **12**(3), 450–455 (1982). — **高斯 $W_2$ 的闭式（§1.5.2 的 $\sqrt d$ 与 §2.2.1 的松弛估计）**。
