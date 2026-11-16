# Technical Note for Harry — Draft

> **Status markers**: [DONE] = completed experiment with numbers in this repo;
> [RUNNING] = in progress; [PROPOSED] = not yet run.
>
> Everything marked [DONE] can be reproduced with the commands in
> `docs/HARRY_EXPERIMENTS.md` §8.

---

## 1. What the project is, positioned against MeanFlow and OT-CFM

MeanFlow (arXiv:2505.13447) learns an **average-velocity field** so that a single network
evaluated once moves a sample from the noise end to the data end. It solves the
*one-step* problem. OT-CFM (arXiv:2302.00482) shows that the **coupling** used to define
the flow matters, and that a minibatch-OT coupling is a practical improvement over the
independent coupling, at a measurable computational cost.

Neither of those is the problem addressed here. The setting is a **parametric family of
distributions**: the Keller–Segel particle system

$$
dX_j=-\frac{\chi}{N}\sum_{i\ne j}\nabla K_\delta(X_j-X_i)\,dt+\sigma\,g(X_j)\,dt+\sqrt{2\mu}\,dW_j
$$

with two physical parameters $\theta=(\sigma,\mu)$, and the task is to generate the
terminal cloud at an **unseen** $\theta$ in one forward pass. So the object being learned
is a map

$$
T_\theta:\ \mathbb{R}^3\to\mathbb{R}^3,\qquad (T_\theta)_\#\mu_0\approx\nu_\theta,
$$

conditioned on a continuous parameter, and the question is what a second network buys
there.

**The claim under test** is deliberately narrow:

> H: *A near-identity corrector applied after a one-step parametric transport map improves
> the one-step distributional accuracy by more than the same parameter budget spent on a
> single wider network, at matched inference cost.*

H is falsifiable in three ways: (i) a capacity-matched single network does as well;
(ii) the corrector only helps inside the training parameter box; (iii) the corrector
trades one-step accuracy for worse stability under iteration. All three are tested below.

---

## 2. Equations

### 2.1 Stage 1 — parametric MeanFlow [DONE, reproduced from upstream]

Input vector is $(z,t,r,\log_{10}\sigma,\log_{10}\mu)$, output is a velocity. Training is
the MeanFlow identity with a JVP:

$$
u_\theta(z,r,t)=v-(t-r)\frac{d u_\theta}{dt},
\qquad v=x_T-x_0,\quad z=(1-t)x_T+t\,x_0,
$$

$$
\mathcal L_{\text{MF}}=\mathbb{E}\big\|u_\theta(z,r,t)-\mathrm{sg}\big[v-(t-r)\partial_t u_\theta\big]\big\|^2 .
$$

Generation is one evaluation: $x_T=x_0-u_\theta(x_0,0,1)$.

### 2.2 Stage 2 — the corrector [DONE]

With the MF frozen, a residual MLP with zero-initialised output head:

$$
T_{\text{dp}}(y)=y+\alpha\,\mathrm{MLP}_\phi(y,\log\sigma,\log\mu).
$$

The target is the **Brenier map** between the MF output cloud and the reference cloud,
realised through a minibatch exact-OT assignment $\pi$:

$$
\mathcal L_{\text{DP}}=\mathbb{E}_{(x,y)\sim\pi}\big\|x+\alpha\,\mathrm{MLP}_\phi(x)-y\big\|^2 .
$$

**Why this is not "just another network"**: at the population limit and with exact OT, the
minimiser of $\mathcal L_{\text{DP}}$ is $\nabla\varphi_\theta$, the gradient of the
Brenier potential transporting $\nu_{\text{mf},\theta}$ to $\nu_\theta$. MF's minimiser is
the conditional mean $\mathbb{E}[X_T\mid X_0]$. These are **different mathematical
objects**, and the former pushes forward *exactly* onto the target while the latter does
not in general. That is the structural reason a second stage can help at all; it is not a
capacity argument. (Proof: `docs/THEORY.md` Theorem 5.)

### 2.3 What the corrector does *not* fix [DONE]

The ideal-limit statement above assumes the corrector generalises in $\theta$. It does not
outside the training box, because the corrector's smallness comes from
$\partial_\theta T_{\text{mf}}\approx\partial_\theta T^*$, which fails off-distribution.
Measured (see §5): gain $+14.7\%$ inside, $+1.3\%$ and $+2.4\%$ in the two extrapolation
directions.

---

## 3. Protocols, all matched [DONE]

| Dimension | Setting |
|---|---|
| Training cells | the same 40 (stratified maximin LHS over $20\sigma\times10\mu$) |
| Updates per cell | 32,000 for MF; 2,000 for the corrector (both fixed) |
| Training particles | $10^5$ per cell |
| Held-out cells | 143 (or 144 where the split differs, stated per table) |
| Initial cloud | **one shared cloud** for every cell, so differences are parameter-driven |
| Metric | exact histogram $W_2$, POT network simplex, dual certificate checked |
| Seeds | 88, 89, 90 for the headline table |

---

## 4. Completed results

### 4.1 Headline (three seeds, 143 held-out cells) [DONE]

| | one-step MF | MF + DP | relative |
|---|---:|---:|---:|
| mean $W_2$ | $0.12152\pm0.00892$ | $\mathbf{0.10373\pm0.00267}$ | $\mathbf{+14.4\%\pm4.2\%}$ |
| per-cell wins | — | 415 / 429 | — |

Paired $t=6.0$ ($df=2$); the per-cell sign test gives $z\approx19.4$.
The corrector also **reduces across-seed variance by 70%** ($0.0089\to0.0027$).

### 4.2 The four-way comparison, at measured inference cost [DONE except (c)]

| configuration | $W_2$ | forwards | parameters |
|---|---:|---:|---:|
| (a) MF one step | 0.12219 | **1** | 9,027 |
| (b) MF + DP, $\alpha{=}1$ | 0.10376 | 2 | 26,502 |
| (b') MF + DP, $\alpha{=}2$ | **0.09515** | 2 | 26,502 |
| (d) MF two steps, same weights | 0.12332 | 2 | 9,027 |
| (c) capacity-matched single net, width 112 | 0.11604 | 1 | **26,547** |

**The informative rows**:

**(c) does not reproduce the gain at matched capacity.** Width 112 is within 0.17% of
MF+DP's parameter count (26,547 vs 26,502), on the same 40 cells with the same budget, and
it reaches 0.11604 vs MF+DP's 0.10376 — 11.8% worse. So the gain is not explained by
parameter count in this seed; splitting the map into a global transport plus a local
correction appears structurally different, consistent with the Brenier argument in §2.2.

**Caveat, stated plainly**: this is a **single seed**. The gap is $0.0123$, about 1.4×
MF's across-seed sd ($0.0089$), so there is no error bar on it. The honest claim is
"a capacity-matched network does not reproduce the gain in this run", not "the capacity
hypothesis is refuted". A three-seed version is listed as proposed work below.

**(d) removes the "more computation" explanation.** At the *same* two-forward cost as (b),
spending the second forward on another MeanFlow step does nothing ($0.12332$ vs $0.12219$
one-step). Spending it on the corrector gains 22%.

### 4.3 $\alpha$ chosen on a development split, reported on a disjoint test split [DONE]

Fixed split (seed 424242, saved to disk): 40 train / **50 dev** / **94 test**.

| $\alpha$ | dev mean $W_2$ |
|---:|---:|
| 0 (MF only) | 0.12338 |
| 1.0 (upstream default) | 0.10505 |
| **2.0 (selected)** | **0.09689** |
| 4.0 | 0.12969 |

Test split, untouched during selection:

| | test mean $W_2$ | vs MF |
|---|---:|---:|
| MF only | 0.12156 | — |
| $\alpha=1$ | 0.10308 | +15.2% |
| **$\alpha=2$** | **0.09422** | **+22.5%** |

Caveat stated plainly: the sweep scales $\alpha$ **at inference** rather than retraining
per $\alpha$. That is the deployment knob for this architecture (zero-initialised linear
head), so it is the operationally meaningful sweep, but it is not the same as retraining.
The dev curve is also flat between 1.5 and 2.0, so the exact optimum is uncertain; the
claim is "the upstream default is conservative", not "$\alpha=2$ is optimal".

### 4.4 Physical quantities [DONE]

Mean absolute relative error over six cells spanning the box:

| quantity | MF | MF+DP |
|---|---:|---:|
| centroid drift along shear, $x$ | 3.94% | **3.02%** |
| rms radius | 0.82% | 0.87% |
| **anisotropy $\lambda_1/\lambda_2$** | 1.04% | **0.88%** |
| principal std $\lambda_1$ | 1.15% | **1.04%** |
| principal std $\lambda_2$ | 0.94% | **0.65%** |

The anisotropy is the quantity $\sigma$ physically controls, and it is reproduced to
**below 1%**, which is a much stronger statement than $W_2$ alone: the model has the
shear–diffusion balance right, not merely the mass in roughly the right place.

### 4.5 Training-time OT cost [DONE]

| item | value |
|---|---|
| OT solver | POT network simplex (exact, not Sinkhorn) |
| minibatch | 2,000 |
| refresh interval | every 10 steps |
| OT solves per cell | 200 |
| OT solves total (40 cells) | 8,000 |
| wall clock | **~11 min** |
| DP training total | 157 min |
| **share** | **~7%** |

So: **no global OT at deployment**, and ~7% of DP training time is OT. Avoiding global OT
is not the same as avoiding all OT, as you noted.

---

## 5. The strongest failure case [DONE]

### 5.1 Rollout is not supported, and it fails loudly

Applying the one-step generator $k$ times and comparing to the physical solution at $k\cdot T$:

| $k$ | $W_2$ vs physics (six cells) | RMS radius |
|---:|---|---:|
| 1 | 0.092 – 0.241 (**0.59–1.55× floor**) | 0.84 – 1.57 |
| 2 | 0.20 – 4.62 | 0.99 – 5.40 |
| 4 | 4.5 – 82.5 | 5.3 – 60.7 |
| 8 | **1,968 – 41,678** | up to 27,262 |

Step amplification (model displacement per application ÷ physical displacement per $kT$)
rises from 1.5 to ~4,000.

**Mechanism, stated honestly**: the map is trained on inputs drawn from $\mu_0$. One
application moves the centroid by 1.3–2.3 and the rms radius by 0.84–1.57, so the second
application receives inputs far outside the training distribution. **The generator is not a
time stepper and cannot be iterated.** Long-horizon behaviour requires training a separate
model at the target horizon, which we have not done.

### 5.2 Does the corrector make stability worse? No — two-step MF does

Three maps, each at exactly two forward passes per application:

| cell | map | $k{=}1$ | $k{=}4$ | amplification at $k{=}4$ |
|---|---|---:|---:|---:|
| s03_m04 | MF | 0.1167 | 16.92 | 17.1 |
| | **MF+DP** | **0.1119** | **15.68** | **15.8** |
| | MF, two-step | 0.1168 | **44.66** | **45.6** |
| s09_m06 | MF | 0.1396 | 35.96 | 15.5 |
| | **MF+DP** | **0.1340** | **33.36** | **14.4** |
| | MF, two-step | 0.1413 | **118.16** | **51.9** |
| s19_m09 | MF | 0.2480 | 82.52 | 30.5 |
| | **MF+DP** | **0.2439** | **77.21** | **28.5** |
| | MF, two-step | 0.2380 | **317.69** | **118.4** |

The corrector is **not** the source of instability; it slightly improves both the one-step
error and the amplification factor. Spending the same extra forward on a second MeanFlow
step is an order of magnitude worse. This is the cleanest evidence that the second stage
should be a corrector rather than another solver step.

### 5.3 A case where the corrector is worse [DONE]

At the worst corner $(\sigma,\mu)=(200,10)$, the anisotropy error goes from $-2.5\%$ (MF)
to $-4.2\%$ (MF+DP). Small, but it is a real regression and is reported as such.

### 5.4 The ablation you asked for: identifiable error, or distribution compensation? [DONE]

| region | MF error | corrector gain |
|---|---:|---:|
| interpolation (inside the box) | 0.094 | **+14.7%** |
| $\sigma$ extrapolation (150→200) | 0.155 | **+1.3%** |
| $\mu$ extrapolation (3.98→10) | 0.353 | **+2.4%** |

**This is the negative result.** The corrector does not repair an identifiable error in
the first-stage map; it compensates for the training distribution. Outside the box it
collapses to a 1–2% effect while the one-step error itself grows 1.7–3.8×.

---

## 6. Cost of the comparison (as requested)

| item | measured |
|---|---|
| One simulation, $10^5$ particles, $T=0.02$ | **83 s** (RTX 4060 Laptop) |
| Data generation, full $20\times10$ grid | **191 min** |
| MF training, 40 cells × 32,000 updates | **147 min** per seed |
| Corrector training, 40 cells × 2,000 updates | **157 min** per seed |
| OT share of corrector training | **~7%** (11 min) |
| Inference, $10^5$ particles | **4.5 ms** (1 forward) / **7.7 ms** (2 forwards) |
| Full four-way comparison (§4.2, 144 cells) | **4.6 min** |
| Rollout study (§5.1, 6 cells × 4 horizons) | **8 min** |
| $\alpha$ sweep + test report | **5 min** |
| Capacity-matched training, width 112 | **~128 min** (measured) |

Everything except the capacity-matched run is on disk and reproducible from
`docs/HARRY_EXPERIMENTS.md` §8.

---

## 7. Completed vs proposed

**Completed**: reproduction of the upstream KS3D result (180 metrics, max deviation
0.0066 from the published values); the two-parameter extension; three-seed headline;
four-way comparison rows (a)(b)(d); $\alpha$ selected on dev and reported on test;
rollout divergence; physical quantities; OT cost accounting; the extrapolation ablation.

**Proposed, not run**:

1. **Capacity-matched seeds 89/90** — not run. The single-seed row (c) is in the table
   without an error bar; two more seeds (~4.3 h) would make it a claim with one.
2. **Long-horizon model** — retrain MeanFlow at $T'=4T$ or $8T$ and ask whether a
   corrector still helps. This is the honest way to answer the long-horizon question,
   since iteration is not available. Estimated 6–8 h.
3. **Corrector trained with a stability penalty** — add $\|T_{\text{dp}}(T_{\text{dp}}(y))-T_{\text{dp}}(y)\|^2$
   to the loss and see whether iteration becomes possible at a small one-step cost.
4. **$\alpha$ retrained per value** rather than scaled at inference, on the dev split.
5. **Narrower capacity ladder** (widths 96, 104, 120) to check that the capacity response
   is not accidentally non-monotone around 112 — the repository has already observed
   non-monotone width behaviour in the single-parameter setting, so one width is thin
   evidence for a capacity claim.

---

## 8. On the Autel work

[TO BE SUPPLIED BY THE AUTHOR — non-confidential summary of the bottleneck, the
intervention, and the measured latency–quality trade-off.]
