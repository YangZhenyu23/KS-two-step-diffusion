---
title: When does a learned local correction pay for itself?
date: 2026-11-17
tags:
  - two-step-diffusion
  - keller-segel
  - meanflow
  - optimal-transport
  - negative-result
status: answered
---

# When does a learned local correction pay for itself?

> **The question.** *When does a learned local correction improve a fast transport model enough to justify its additional training and inference cost?* — answered on a 3D Keller–Segel particle system with two physical parameters, at a fixed horizon.

---

## The answer in one paragraph

**It pays inside the training parameter range, for the specific model it was fitted to, and nowhere else.** On held-out cells drawn from the same parameter box, a near-identity corrector layered on a one-step MeanFlow map reduces the distributional error by **14.4 % ± 4.2 %** (three seeds, 415/429 cells) for **2.97×** the inference FLOPs and 8,000 extra transport solves during training. That gain survives neither of the two perturbations one would naturally test next: leaving the parameter box collapses it to **1.8 %**, and applying the same corrector to a stage-1 model that differs *only by random seed* turns it into a **1.4–3.2 % loss**. So the honest characterisation is a **local fit to one model's residual**, not a repair of an identifiable error class.

---

## What was compared, and why those four rows

Four configurations, matched on training cells (40), optimisation budget (32,000 updates per cell for stage 1), held-out cells (the same 112 for all four) and measured inference cost:

| configuration | parameters | passes | held-out $W_2$ | vs (a) |
|---|---:|---:|---:|---:|
| (a) global map alone | 9,027 | 1 | 0.12271 | — |
| (d) few-step baseline, same weights twice | 9,027 | 2 | 0.12435 | **−1.3 %** |
| (c) capacity-matched predictor | 26,547 | 1 | 0.11667 | +4.9 % |
| (b) map + corrector | 26,502 | 2 | **0.10422** | **+15.1 %** |

**Row (d) rules out "more computation."** Spending the second forward pass on another step of the same map makes the result *worse*. The gain is not the extra pass.

**Row (c) rules out "more parameters" — and this is stronger than it looks once repeated.** Its parameter count differs from the two-stage total by 45, or 0.17 %. Repeating it at three seeds:

| model | seed 88 | seed 89 | seed 90 | mean ± sd |
|---|---:|---:|---:|---:|
| width-112, capacity-matched | 0.11667 (+4.9 %) | 0.12747 (−3.9 %) | 0.12461 (−1.5 %) | **−0.2 % ± 4.6 %** |
| stage 1 + corrector | 0.10422 | 0.10422 | 0.10422 | **+15.1 %** |

A network with the same parameter count gains **nothing on average**. The single-seed +4.9 % was a favourable draw. Independently, a single-parameter network widened **15×** (9,027 → 134,403 parameters) shows no improvement either (0.1486 → 0.1456) — capacity is not the binding constraint on this task, which is why the corrector's gain cannot be a capacity effect.

---

## Where the benefit stops

### Leaving the parameter box

Training box $\sigma \le 150$, $\mu \le 3.98$. Three regions scored separately — the 45 training cells (memorisation), the 90 held-out interior cells, and the 65 cells outside:

| region | $n$ | stage 1 | + corrector | gain |
|---|---:|---:|---:|---:|
| training cells | 45 | 0.09388 | 0.08014 | +14.6 % |
| **interpolation (held out)** | **90** | **0.09404** | **0.08070** | **+14.2 %** |
| **extrapolation** | **65** | **0.21805** | 0.21407 | **+1.8 %** |

Training cells and held-out interior cells agree to 0.2 %, so the model interpolates rather than memorises. Outside the box the first-stage error grows **2.32×** and the corrector's gain falls by a **factor of eight**.

The degradation is strongly anisotropic: leaving in $\sigma$ costs 2.40× the floor, leaving in $\mu$ costs **5.48×**, the corner **5.96×**.

### Changing nothing but the random seed

This is the sharpest test. Take the seed-88 corrector and apply it to stage-89's and stage-90's outputs. Architecture, training data, parameter range, metric — all identical; only the base model's initialisation differs.

| base model | cells | without | with | gain | cells improved |
|---|---:|---:|---:|---:|---:|
| seed 88 (the one it was fitted to) | 143 | 0.12277 | 0.10429 | **+15.0 %** | 141/143 |
| seed 89 | 143 | 0.12975 | 0.13162 | **−1.4 %** | 57/143 |
| seed 90 | 143 | 0.11203 | 0.11560 | **−3.2 %** | 33/143 |
| width-112 base | 112 | 0.11667 | 0.11407 | +2.2 % | 68/112 |

**Changing nothing but the seed turns a 15 % gain into a small loss.** This rules out extrapolation, data mismatch and capacity as explanations: the corrector learned *this* base model's residual.

### Three further tests, all pointing the same way

**No decline-to-act behaviour.** Applied to the *reference cloud itself*, it still displaces particles (mean 0.0165 per particle, about 1 % of the cloud scale) and the resulting $W_2$ is **0.0717 — 1.1× the reproducibility floor**. A mechanism that detected "already correct" would leave the cloud alone.

**It needs coverage, and has not saturated.** Retraining on 4 / 10 / 20 / 40 cells, all scored on the same 143 held-out cells:

| training cells | held-out $W_2$ | vs stage 1 |
|---:|---:|---:|
| 4 | 0.14681 | **−19.6 %** |
| 10 | 0.11423 | +7.0 % |
| 20 | 0.10404 | +15.3 % |
| 40 | 0.09527 | +22.4 % |

Monotone and **still rising at 40**. With 4 cells it is actively harmful. A transferable rule would work from few cells.

**Only partially a gradient field.** The structural argument says the target is a Brenier gradient, whose Jacobian must be symmetric. Measured antisymmetry $\lVert J-J^\top\rVert_F/\lVert J\rVert_F$ by central differences:

| map | antisymmetry |
|---|---:|
| corrector | **0.51** |
| stage-1 displacement | 0.68 |

Closer to a gradient field than stage 1 — but 0.51 is nowhere near the 0 of an actual gradient field.

---

## The cost side

**Inference.** One corrector pass costs **1.97×** the FLOPs of one stage-1 pass (each residual block has two $64\times64$ projections plus a LayerNorm), so the two-stage model is **2.97×** the FLOPs of the one-stage model at $10^5$ particles — 5.25 against 1.77 GFLOP. The honest headline is **~3× FLOPs, not "two passes"**. At that price 15–23 % lower error is a good trade against stage-1-only; it is simply not a free one.

**Training.** The corrector's target is rebuilt every ten updates by solving an exact OT problem on a 2,000-particle minibatch out of 100,000: **8,000 solves**, `ot.emd` network simplex, **not Sinkhorn and not warm-started**. At leading order those solves contribute 0.064 PFLOP against 0.011 PFLOP of network arithmetic, so **~85 % of the corrector's training compute goes into transport rather than into the network** (in wall-clock terms the share is much smaller, because the solver runs on CPU).

**OT, by role — and only one role is free:**

| role | OT solves | who pays | avoidable |
|---|---:|---|---|
| training targets | 8,000 | one-time training cost | only by changing the objective |
| **deployment** | **0** | nobody | **yes, genuinely** |
| evaluation | 2,793 | the experimenter | no — $W_2$ *is* an OT problem |

Avoiding a *global* OT at inference is not the same as eliminating OT from the method. The minibatch is also an approximation to the global OT it stands in for, not the same object.

---

## Why it fails off-distribution, and why that is consistent

The corrector's smallness depends on $\partial_\theta T_{\text{mf}} \approx \partial_\theta T^*$ — it works when the stage-1 map and the target move *together* as the parameter varies. Outside the box the residual is no longer small, and the corrector's input (the stage-1 cloud) is itself deformed, so it faces a **double distribution shift**.

A one-parameter control supports this reading: with training $\sigma \in [15, 150]$ and evaluation out to 200, the first-stage error rises only **1.19×** and the corrector retains **+13.3 %** (against +15.7 % inside). A one-dimensional family can be straightened by a scalar coordinate; a two-dimensional surface cannot.

But the seed experiment shows the condition is even tighter than "same parameter regime." The corrector is tied to *one specific map*, not to a regime.

---

## What it is not

**Not a time integrator.** Applying the map $k$ times diverges: at $k{=}1$ it sits at 0.6–1.5× the reproducibility floor; by $k{=}8$ the error is **12,610–267,049×** the floor. This is expected rather than pathological — one application moves the centroid by 1.3–2.3 and the rms radius by 0.84–1.57, so the second application receives a cloud no training sample resembles. Cost is not the reason this is out of scope (eight applications take 36 ms against 83 s for one simulation); the reason is that the map has no dynamical semantics to iterate.

The single-parameter model diverges too, by 3,745–4,015× at $k{=}8$ — roughly 50–70× less, but the same behaviour. Non-iterability is a property of any fixed-horizon map.

**Retraining at a longer horizon is the honest alternative.** At $T' = 4T = 0.08$:

| quantity | $T=0.02$ | $T'=0.08$ | ratio |
|---|---:|---:|---:|
| true-cloud rms radius | 0.941 | 1.918 | 2.04× |
| raw held-out $W_2$ | 0.12277 | 0.33766 | 2.75× |
| **$W_2$ / rms radius** | 0.1304 | **0.1760** | **1.35×** |

Most of the raw growth is scale, not degradation. A longer horizon is harder, but not qualitatively different at this distance.

*(A reversal worth noting: at $T'$ a two-step application **helps** (+5.5 %), whereas at $T$ it hurts (−1.1 %). By $T'$ the transport has accumulated more shear, so the flow is more curved in time and subdividing pays. This is a different curvature from the parameter-direction one.)*

---

## The short version

> **Yes, when:** the target parameter lies inside the training box, *and* the stage-1 map is the one the corrector was fitted to. Then ~15 % error reduction for ~3× inference FLOPs is a defensible trade.
>
> **No, when:** either condition is relaxed. Off the parameter box the gain falls to 1.8 %; on a different seed of the *same* architecture it becomes a loss.
>
> **And it is not a general repair.** It has partial Brenier structure (Jacobian antisymmetry 0.51 vs 0.68), needs broad coverage without saturating, and cannot decline to act on a cloud that is already correct. It is best described as a **local structural fit to one model's residual**.

That negative answer is arguably more useful than a positive one: it says where to aim next — make the residual *identifiable* (shared trunk across base models, an explicit no-op gate) rather than merely learnable.

---

## Caveats on the numbers

- The few-shot sweep omits the residual-regularisation term used in headline training, so its 40-cell level is **not comparable** with the +15.0 %; only the trend is.
- The Jacobian test uses two cells × 24 probe points at $h = 10^{-3}$; treat it as qualitative.
- The $T'$ floor is estimated by scaling the $T$ floor with the cloud radius, not measured.
- Single-parameter rollout cells are the model's own training cells (it saw the whole $\mu{=}1$ row), so that study concerns stability, not held-out accuracy.
- Headline results use a fixed learning rate; a schedule study in the same repository found warm-up-plus-cosine **14.2 %** better at equal budget, so the reported figures are conservative by roughly that margin.

---

*Full note, equations and cost accounting: `docs/TECHNICAL_NOTE.md` / `.pdf`. Detailed experiment log: `docs/EXPERIMENTS_E1_E9.md`.*
