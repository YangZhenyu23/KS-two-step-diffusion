# Harry 需求追踪文档（持续更新）

> **本文件的用途**：Harry 的邮件是本次工作的唯一验收标准。这份文档把邮件原文、
> 逐条拆解、当前状态、以及**每一轮对话中新增的细化要求**全部记录下来，
> 供后续每次修改时对照，避免遗漏或前后不一致。
>
> **维护规则**
> 1. 邮件原文（§1）**不得改动**，它是权威来源
> 2. §2 是逐条编号的需求清单（R1, R2, …），每条有固定 ID，后续引用用 ID
> 3. §3 是**更新日志**，每次用户提出新的细化要求，追加一条，写明日期、原话、影响的 R 编号
> 4. §4 记录当前交付物与状态的对应关系
> 5. 每次修改报告/文档后，回来更新 §2 的"状态"列

---

## 1. Harry 邮件原文（权威来源，不得改动）

**发件人**：Harry
**时间**：2026-11（具体日期未记录）
**主题**：对 particle-transport 项目的回复与研究契合度评估

---

Dear Zhenyu,

Thank you for writing and for sending your CV. I read the description of your particle-transport project with interest. The combination of a one-step global transport map and a near-identity corrector gives us a concrete starting point for assessing research fit. Your email's account of flow-matching acceleration for action chunks at Autel is also relevant, although I would need a technical summary to understand your specific contribution and results.

I would focus your next research note on one question: when does a learned local correction improve a fast transport model enough to justify its additional training and inference cost? Please start with the PDE setting you already understand, rather than immediately expanding across image generation, robotics, and 3D reconstruction.

The positioning needs to be precise. MeanFlow already learns an average-velocity field for one-step generation (https://arxiv.org/abs/2505.13447), while conditional flow matching with minibatch optimal transport studies transport-based couplings and their computational trade-offs (https://arxiv.org/abs/2302.00482). Combining a fast predictor with a second network would need evidence of a particular benefit beyond adding capacity or another evaluation.

A useful bounded comparison would include the global map alone, the map plus your corrector, a capacity-matched predictor, and a few-step flow baseline under comparable measured inference cost. Keep training data, test particle sets, optimization budget, and evaluation horizons matched. Report both the cost of constructing transport targets during training and deployment cost; avoiding global OT is different from eliminating all OT computation, particularly when the corrector still uses minibatch W2 or warm-start OT.

Please separate one-step distributional error from long-horizon dynamics. Alongside a prespecified distributional metric, report trajectory or task-relevant physical quantities and stability over repeated application. For held-out initial conditions or parameter regimes, show whether the corrector improves typical cases but introduces a small number of unstable rollouts. Keep those failures visible. Choose any correction-strength or stopping settings on development cases, not the final evaluation set.

An especially informative ablation would ask whether the corrector repairs an identifiable error in the first-stage map or merely compensates for its training distribution. A negative result—no gain at matched cost, or improved one-step fit with worse rollout stability—would still help us choose the next direction.

Could you send a short technical note with the exact equations, your own contribution, a small existing results table, the strongest failure case, and the estimated cost of this comparison? Please distinguish completed experiments from proposed ones. For the Autel work, a non-confidential summary of the bottleneck, your intervention, and measured latency-quality trade-off would be enough.

I cannot confirm a Fall 2027 place or funding at this stage, but these materials would let me give more specific written feedback on fit and a possible research direction.

Best regards,
Harry

---

## 2. 需求逐条拆解

### 2.0 总纲：研究笔记只能回答一个问题

**R0**（核心问题，一切内容都要服务于它）

> *when does a learned local correction improve a fast transport model enough to justify its additional training and inference cost?*

拆成三个可判定的子问题：

| 子问题 | 判定标准 |
|---|---|
| R0-a 什么条件下修正有效？ | 给出条件，且条件可检验 |
| R0-b 收益是否值得额外训练成本？ | 量化训练成本增量 |
| R0-c 收益是否值得额外推理成本？ | 量化部署延迟增量，并与收益同尺度比较 |

**重要约束**：*"Please start with the PDE setting you already understand, rather than immediately expanding across image generation, robotics, and 3D reconstruction."*
→ **笔记中不得扩展到图像/机器人/3D 重建**，只能谈 KS（或 KPP）PDE 设定。

---

### 2.1 定位要求（Positioning）

**R1** 定位必须精确。必须显式区分：
- **R1-a** MeanFlow（arXiv:2505.13447）已经做到什么：*"already learns an average-velocity field for one-step generation"*
- **R1-b** OT-CFM（arXiv:2302.00482）已经做到什么：*"studies transport-based couplings and their computational trade-offs"*
- **R1-c** 我们的贡献**不是**这两者

**R2** 必须证明收益**超出**以下两种平凡解释：
- **R2-a** *"beyond adding capacity"* —— 不是参数更多
- **R2-b** *"another evaluation"* —— 不是换了个指标显得更好

> 状态：✅ 已处理。R2-a 用容量匹配（宽度 112 = 26,547 参数 vs 26,502）；
> R2-b 用"成本匹配下两步 MF 无收益"。

---

### 2.2 有界对照（Bounded comparison）

**R3** 必须包含**四**个配置：

| ID | 配置 | 邮件原文 |
|---|---|---|
| **R3-a** | 全局映射单独 | *"the global map alone"* |
| **R3-b** | 映射 + 修正器 | *"the map plus your corrector"* |
| **R3-c** | **容量匹配**的预测器 | *"a capacity-matched predictor"* |
| **R3-d** | **少步 flow 基线** | *"a few-step flow baseline"* |

**R4** 必须**在可比的实测推理成本下**比较：*"under comparable measured inference cost"*
→ 不能只报"参数量相同"，要报**实测延迟**

**R5** 四项必须**同时匹配**（四个"matched"）：
- **R5-a** *"training data matched"* —— 训练数据相同
- **R5-b** *"test particle sets matched"* —— 测试粒子集相同
- **R5-c** *"optimization budget matched"* —— 优化预算相同
- **R5-d** *"evaluation horizons matched"* —— 评测时域相同

> ⚠️ **R5 是最容易被审稿人攻击的一条，也是用户特别强调的一条。**
> 详见 §3 日志中 2026-11-16 的记录（评测集泄露事件）。

---

### 2.3 成本报告（Cost）

**R6** 必须报告**两类**成本：

| ID | 成本类型 | 邮件原文 |
|---|---|---|
| **R6-a** | **训练时**构造传输目标的成本 | *"the cost of constructing transport targets during training"* |
| **R6-b** | **部署**成本 | *"deployment cost"* |

**R7** 必须明确一个区分：*"avoiding global OT is different from eliminating all OT computation"*
- **R7-a** 不能宣称"我们不需要 OT"
- **R7-b** 必须说明修正器仍在用 minibatch W2 或 warm-start OT
- **R7-c** 应给出 OT 在总训练时间中的**占比**

---

### 2.4 一步误差 vs 长时程（Separate one-step from long-horizon）

**R8** 必须**分开**报告两类误差：
- **R8-a** *"one-step distributional error"*
- **R8-b** *"long-horizon dynamics"*

**R9** 必须有一个**预先声明的**分布度量：*"a prespecified distributional metric"*
→ 不能事后挑指标

**R10** 必须报告**任务相关的物理量**：*"trajectory or task-relevant physical quantities"*
→ 不只是 $W_2$

**R11** 必须报告**重复应用下的稳定性**：*"stability over repeated application"*

**R12** 必须针对**留出的初始条件或参数区**，检查是否存在：
*"improves typical cases but introduces a small number of unstable rollouts"*
→ 即：**典型改善 + 少数失败**这种混合情况

**R13** 必须**保留失败案例可见**：*"Keep those failures visible."*
→ 不许只报平均值、不许删掉最差案例

---

### 2.5 超参纪律（Dev/test discipline）

**R14** 任何**修正强度**或**停止准则**的设置，必须在**开发集**上选定：
*"Choose any correction-strength or stopping settings on development cases, not the final evaluation set."*
- **R14-a** 适用于 $\alpha$（修正强度）
- **R14-b** 适用于任何 stopping criterion
- **R14-c** 最终评测集**不得参与**任何选择

---

### 2.6 关键消融（The decisive ablation）

**R15** 必须回答：修正器是在**修复第一阶段映射中的可识别误差**，还是**仅仅在补偿其训练分布**？
*"whether the corrector repairs an identifiable error in the first-stage map or merely compensates for its training distribution"*

判定逻辑：
- 若是**修复可识别误差** → 训练分布外也应有效
- 若是**补偿训练分布** → 训练分布外应失效

**R16** 明确接受**负面结果**：*"A negative result ... would still help us choose the next direction."*
两类负面结果都被明确认可：
- **R16-a** *"no gain at matched cost"*
- **R16-b** *"improved one-step fit with worse rollout stability"*

> 这条非常重要：**Harry 明确说负面结果也有价值**。
> 因此不要为了"好看"而淡化失败案例，那反而违背他的要求。

---

### 2.7 交付物清单（Deliverables）

**R17** 技术笔记必须包含**五项**：

| ID | 内容 | 邮件原文 |
|---|---|---|
| **R17-a** | 确切的方程 | *"the exact equations"* |
| **R17-b** | 你自己的贡献 | *"your own contribution"* |
| **R17-c** | 一张现成的小结果表 | *"a small existing results table"* |
| **R17-d** | **最强的失败案例** | *"the strongest failure case"* |
| **R17-e** | 这项对比的**预估成本** | *"the estimated cost of this comparison"* |

**R18** 必须**区分已完成与提议中的实验**：*"distinguish completed experiments from proposed ones"*

**R19** Autel 部分只需**非机密摘要**，含三项：
- **R19-a** 瓶颈（bottleneck）
- **R19-b** 你的介入（intervention）
- **R19-c** 实测的延迟-质量权衡（measured latency-quality trade-off）

> ⚠️ 状态：**未完成**，需要用户提供。

---

### 2.8 明确不要求的（避免过度工作）

Harry 没有要求，因此**不必做**：

| 不要求 | 依据 |
|---|---|
| 图像生成 / 机器人 / 3D 重建实验 | R0 明确说 *"rather than immediately expanding"* |
| 4D、KPP、cubic 等其他 PDE | 只要求 *"the PDE setting you already understand"* |
| 新的生成模型架构 | 只要求对比，不要求创新架构 |
| 承诺 Fall 2027 名额 | Harry 自己说 *"I cannot confirm"* |

---

## 3. 更新日志（每轮新增的细化要求追加在此）

> 格式：`### YYYY-MM-DD 第 N 轮 — 主题`
> 内容：用户原话要点 + 影响的 R 编号 + 处理方式

### 2026-11-16 第 1 轮 — 建立需求追踪

- **触发**：用户要求"单独创建一个新的记忆文档，把要求、每次要求细分的要求加在并记录到里边"
- **动作**：建立本文件，完成 R0–R19 的初始拆解
- **状态**：初始版本

### 2026-11-16 第 2 轮 — 强调"统一测试集/训练集"

- **用户原话要点**：
  > "一定要注意是要统一的测试集和训练集"
  > "目前当中我那个仓库中可能是混有了单参数和双参数的两个，你要区分两个两套东西"
- **影响的 R**：R5-a、R5-b（训练数据与测试粒子集匹配）
- **发现的实质问题**：
  1. 仓库中 `stratified_subset(pool, 40, seed=20261107)` 在开发过程中被改过，
     今天调用产生的 40 格与主 checkpoint 实际训练的 40 格**只有 6 格重叠**。
     导致若干脚本的"144 格留出集"里有 **32 格（22%）是训练格**。
     → 已修：改为从 checkpoint 的 `payload["cells"]` 读取，正确留出集为 **143 格**。
     → 修复后数值变化 **0.5%**，定性结论不变。
  2. 仓库存在 **30 套不同的 LHS-40 设计**（S1–S30），不同组之间的留出数字不可直接比较。
     主结论锚定 S1（双参数训练）+ S2（单参数 $\mu=1$ 整行）。
  3. **单参数与双参数是两套独立体系**，见 §5 的对照表。
- **处理方式**：写入 `docs/REPO_AUDIT_AND_TASKS.md`，并在报告 §3 中说明

### 2026-11-16 第 3 轮 — 要求生成 Word 报告，留出未完成实验的位置

- **用户原话要点**：
  > "写一份根据老师的那个任务报告，要有图文并茂"
  > "因为还有实验没做完，请你把实验目标，就是实验正在做的那一部分空出来，待会儿好往里填"
- **影响的 R**：R17（交付物）、R18（区分已完成/提议中）
- **处理方式**：
  - 生成 `docs/report_for_harry.docx`（8 图、14 表、2515 词）
  - §7 "Experiments in progress" 留三个占位区块（P0/P1/P2），
    每个标注 `RUNNING`/`QUEUED` 状态横幅 + 数据来源路径 + 空表格
  - 占位区块**自动检测**：重新运行 `scripts/build_report_docx.py` 时，
    若对应 JSON 已存在则自动填真实数字

### 2026-11-17 第 4 轮 — （待追加）

---

## 4. 交付物与需求对应表

| 交付物 | 覆盖的 R | 位置 |
|---|---|---|
| `docs/report_for_harry.docx` | R17-a…e, R18 | 主交付物 |
| `docs/NOTE_FOR_HARRY.md` | 同上的 markdown 底稿 | 便于版本管理 |
| `docs/THEORY.md` | R17-a（方程与证明）、R15（Brenier 论证）| 数学基础 |
| `docs/HARRY_EXPERIMENTS.md` | R8–R13（rollout、物理量、α）| 实验报告 |
| `docs/ONE_VS_TWO_PARAM.md` | R5-a（数据匹配）、R2-a（容量）| 对比协议 |
| `docs/REPO_AUDIT_AND_TASKS.md` | R5（数据匹配审计）、R3–R5 状态| 仓库审计 |
| `docs/PROJECTION_FIGURES.md` | 定性对比图 | 图 |
| `results/report_figures/` | 定量图 6 张 | 图 |

---

## 5. 两套体系的严格区分（用户特别强调）

> 这是 R5 的具体落实，也是"仓库混有两套东西"问题的解答。

### 5.1 单参数体系

| 项 | 内容 |
|---|---|
| 网络 | `OneParamMeanflow`，**6 输入**（$z,t,r,\log\sigma$）→ 3 输出 |
| 训练集 | **$\mu=1$ 整行，20 格**（设计 S2）|
| 结果目录 | `results/ks3d/` |
| 用途 | **仅作 baseline**，代表"上游方法原样扩展" |

### 5.2 双参数体系

| 项 | 内容 |
|---|---|
| 网络 | `TwoParamMeanflow`，**7 输入**（$z,t,r,\log\sigma,\log\mu$）→ 3 输出 |
| 训练集 | **LHS-40，覆盖 10 个 $\mu$**（设计 S1）|
| 结果目录 | `results/fullscale/` |
| 用途 | **主结果** |

### 5.3 最易混淆的一点

**"单参数 vs 双参数"指网络输入维度，不是训练集大小。**

| 对照 | 网络 | 训练数据的 $\mu$ | 用途 |
|---|---|---|---|
| `one_32000` | **单参数** | 只有 $\mu=1$（20 格）| 主 baseline |
| `fair_control/same-cells` | **单参数** | **全部 10 个 $\mu$**（40 格）| **诊断工具**，测信息下界 |
| `two_32000` | 双参数 | 全部 10 个 $\mu$（40 格）| 主结果 |

`same-cells` 用单参数网络看混合 $\mu$ 数据，**必然欠拟合**——这是设计目的
（`train_one_param` 里 `allow_mixed_mu=True` 是显式传入的），
它衡量的正是"缺少 $\mu$ 输入"造成的信息损失，**不是一个可用的模型**。

### 5.4 主对照的评测集（已验证）

```
双参数训练格：40 格（其中 3 格 μ=1：s11_m05, s15_m05, s19_m05）
单参数训练格：20 格（μ=1 整行）

主结果评测集：143 格
  ├─ 与双参数训练格重叠：0 格  ✅
  ├─ 与单参数训练格重叠：0 格  ✅
  └─ 是否等于 pool − 双参数训练 − μ=1整行？ 是  ✅
```

**结构性事实**：评测集里 $\mu=1$ **一格都没有**（整行是单参数训练集）。
所以单参数模型被评测的全部是它没见过的 $\mu$ → **含外推劣势**。
→ 这正是 `same-cells` 诊断（R5-a）存在的理由。

---

## 6. 当前状态总览

| R | 要求 | 状态 | 备注 |
|---|---|---|---|
| R0 | 只回答"修正值不值" | ✅ | 报告全篇围绕此问 |
| R0 约束 | 不扩展到图像/机器人/3D | ✅ | 只谈 KS |
| R1-a/b/c | 精确定位 vs MeanFlow / OT-CFM | ✅ | 报告 §1 |
| R2-a | 超出容量解释 | ✅ | 宽度 112 对照（**单种子**）|
| R2-b | 超出"另一种评估" | ✅ | 成本匹配下两步 MF 无收益 |
| R3-a | 全局映射单独 | ✅ | 0.12277 |
| R3-b | 映射 + 修正器 | ✅ | 0.10429 / 0.09566 |
| R3-c | 容量匹配预测器 | ✅ | 0.11604（**单种子**）|
| R3-d | 少步 flow 基线 | ⚠️ | 用"两步 MF"近似，非标准少步 ODE |
| R4 | 实测推理成本 | ✅ | 4.5 ms / 7.7 ms = 1.71× |
| R5-a | 训练数据匹配 | ✅ | 已修泄露；`same-cells` 预算待对齐（P0）|
| R5-b | 测试粒子集匹配 | ✅ | 共用同一初值云 |
| R5-c | 优化预算匹配 | ✅ | 32,000/格（双 vs 单）|
| R5-d | 评测时域匹配 | ⚠️ | 只有 $t=T$（P2 补长时程）|
| R6-a | 训练时传输目标成本 | ✅ | OT 8,000 次，约 11 min，占 7% |
| R6-b | 部署成本 | ✅ | 1.71× |
| R7-a/b/c | 不宣称"无需 OT" | ✅ | 报告 §5 明确 |
| R8-a/b | 一步 vs 长时程分开 | ✅ | 报告 §4.1–4.5 vs §4.6 |
| R9 | 预设分布度量 | ✅ | $W_2$（16 箱精确 OT）|
| R10 | 任务相关物理量 | ✅ | 质心/RMS/各向异性 |
| R11 | 重复应用稳定性 | ✅ | rollout $k=1,2,4,8$ |
| R12 | 典型改善 + 少数失败 | ✅ | $k\ge2$ 发散；$s19\_m09$ 各向异性 DP 更差 |
| R13 | 失败可见 | ✅ | 完整列表，不删最差 |
| R14 | dev 选超参 | ✅ | α 在 50 格 dev 选，93 格 test 报 |
| R15 | 可识别误差 vs 分布补偿 | ✅ | 内插 +14.7% / 外推 +1.3%, +2.4% |
| R16-a/b | 接受负面结果 | ✅ | 主动报告，不淡化 |
| R17-a | 确切方程 | ✅ | 报告 §2 |
| R17-b | 自己的贡献 | ✅ | 报告 §1 结尾 + §2.3 |
| R17-c | 小结果表 | ✅ | 报告 §4.1 |
| R17-d | 最强失败案例 | ✅ | rollout，$k=8$ 达 4 万倍底噪 |
| R17-e | 预估成本 | ✅ | 报告 §5 |
| R18 | 区分已完成/提议中 | ✅ | 报告 §7、§8.2 |
| R19-a/b/c | Autel 三项 | ❌ | **需用户提供** |

**统计**：R0–R18 共 32 个可判定项，**29 项完成**，2 项部分（R3-d、R5-d），1 项待用户提供（R19）。

---

## 7. 待办（阻塞项优先）

| 优先级 | 事项 | 影响的 R | 成本 |
|---|---|---|---|
| **P0** | 同格同预算的信息缺口对照（32,000/格）| R5-a | 2 h（运行中）|
| **P1** | 容量匹配 seeds 89/90 | R2-a, R3-c | 4.3 h |
| **P2** | 长时程模型 $T'=4T$ | R5-d, R8-b | 2.5 h |
| — | Autel 非机密摘要 | R19 | 用户提供 |
| 可选 | 标准少步 ODE 基线（替代两步 MF）| R3-d | 1 h |
| 可选 | 容量阶梯 96/104/120 | R2-a | 6 h |
