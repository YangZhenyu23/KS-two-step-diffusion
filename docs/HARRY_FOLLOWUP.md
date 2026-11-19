# After a better stage 1, the corrector is no longer the step that pays

**Follow-up to the note of 18 November, on the same two-parameter Keller–Segel system**

Zhenyu Yang · 20 November 2026

Numbers are read from `results/fullscale/harry_followup/`. The earlier note is [`HARRY_NOTE.md`](HARRY_NOTE.md). Its fixed-learning-rate result is left as it was: on 143 held-out cells and three seeds, a corrector paired with its own stage-1 map reduces 16-bin histogram \(W_2\) by 14.4% ± 4.2% (415 of 429 comparisons). This note asks whether that reduction is the thing that pays.

Completed runs are marked as such. The gated corrector was not run.

---

## The answer

On the fixed-learning-rate map, the corrector reduces histogram \(W_2\) by about 15%. On a stage-1 map trained with the same cells, the same update count, and a warm-up plus cosine schedule, the reduction falls to **7.3%**. A second evaluation of that better map, with no second network, reduces \(W_2\) by **15.2%**. A width-112 network with the same schedule and the same parameter count as the two-stage model is **3.6% worse** than the narrow one-step map.

The 15% was not a binning artefact, and it was not bought by extra stage-1 updates or by putting the minibatch OT loss on the stage-1 network. It was a fit to the residual of an under-scheduled map. Once that map is trained with the schedule, the residual is smaller, and the free second evaluation of the same field is the better use of a second forward pass. The absolute error is lower than before: the seed-88 pair moves from 0.10429 to **0.09153**.

One corrector trained on the residuals of seeds 88, 89, and 90 together does not recover the paired result. The three seed-wise reductions are 1.8%, 11.2%, and 5.8%. The earlier seed-transfer test, which applied the seed-88 corrector unchanged, was negative. The shared corrector is positive and smaller.

---

## What was held fixed

Cells are read from the checkpoint that was trained, not recomputed with `stratified_subset`. The warm-up run uses the 40 cells stored in the seed-88 headline checkpoint. Held-out cells are the \(20\times 10\) grid minus those cells minus the remaining \(\mu=1\) row: 143 cells. The four-row table uses the 112 cells held out by the headline design and by all three width-112 designs. The metric is the exact 16-bin histogram \(W_2\), \(10^5\) particles. The corrector protocol is unchanged: \(\alpha=1\), 2,000 updates per cell, batch 2,000, an exact assignment refreshed every 10 steps, residual penalty \(\lambda=1\). The assignment used for the new fine-tune and for the shared corrector is SciPy's Hungarian algorithm. On a 2,000-point batch it matches the optimal cost of the network simplex used in the earlier corrector.

The schedule is a 2% linear warm-up and then a cosine from \(10^{-3}\) to \(10^{-5}\). Stage 1 is 32,000 updates per cell. The stage-1 step is replayed from a CUDA graph. The batch stays 4,096.

---

## 1. The schedule, on the headline cells and on the four-row table

**Completed.** Seed 88 only for the paired corrector. Three seeds for the width-112 row.

On the 143 held-out cells the warm-up stage 1 has \(W_2=0.09874\). The corrector brings it to \(0.09153\), a reduction of 7.3%, on 128 of 143 cells. The earlier seed-88 pair, at fixed learning rate \(10^{-3}\), was 0.12277 then 0.10429, a reduction of 15.0%, on 141 of 143 cells. The new stage 1 is already below the old corrected value. The new corrected value is lower still. The fraction removed by the corrector is about half of what it was.

The four-row table, 112 cells, same schedule:

| row | what is evaluated | \(W_2\) | vs row (a) |
|---|---|---:|---:|
| (a) | warm-up stage 1, one step, seed 88 | 0.09859 | — |
| (b) | (a), then its corrector | 0.09150 | lower by 7.2%, 99 of 112 |
| (c) | width 112, three seeds, same schedule | 0.10215 ± 0.00133 | higher by 3.6% |
| (d) | (a), two sub-steps of the same weights | 0.08358 | lower by 15.2% |

The three width-112 seeds are 0.10287, 0.10062, and 0.10297. All three are above row (a).

Row (d) is the result that changes the earlier conclusion. In the fixed-learning-rate note, a second evaluation of the same field raised the error. Here it lowers the error by more than the corrector does. The corrector is still a reduction against one step of its own map. It is not the best use of the second forward pass, and a matched parameter count does not reproduce it.

The reason is the size of the residual. The corrector is initialised at the identity and penalised for leaving the stage-1 cloud. It can remove a residual that is large and systematic. The fixed-learning-rate map stops at 0.12277, and 15% of that is available. The scheduled map stops at 0.09874, and 7% of that is available. Extra updates at the fixed learning rate do not produce this drop. Section 2 is that control.

---

## 2. Two budget controls, same network, no second stage

**Completed.** Seed 88, fixed learning rate, 143 held-out cells. Both controls add 80,000 updates, which is 2,000 per training cell, the corrector's update count.

| control | \(W_2\) | vs the headline stage 1 (0.12277) |
|---|---:|---:|
| headline stage 1 | 0.12277 | — |
| 80,000 further MeanFlow updates | 0.12465 | higher by 1.5% |
| 80,000 updates against the same minibatch OT targets, no second network | 0.11911 | lower by 3.0% |

Further MeanFlow updates at \(10^{-3}\) do not help. Putting the corrector's coupling on the stage-1 network removes about 3%, against the 15% removed by the second network on this same map. The coupling matters, and it does not account for the corrector. On the scheduled map the corrector itself is only a 7% reduction, so most of what the second network was buying was a stage 1 that had not been trained with a decaying step size.

---

## 3. Where \((r,t)\) is sampled, and whether one step equals two

**Completed.** The sampler is unchanged. Two sigmoid samples of \(\mathcal N(-0.4,1)\) are ordered so that \(t\) is the larger and \(r\) the smaller. Half the batch is set to \(r=t\). One million draws give a fraction 0.9575 with \(t-r<0.5\), and a fraction 0.5000 on the diagonal.

One-step evaluation is \(x-u(x,0,1)\). Two-step evaluation is \([1,0.5]\) then \([0.5,0]\), each scaled by the interval length. On the 143 held-out cells of the fixed-learning-rate seed-88 map, the two procedures do not return the same cloud. The mean histogram \(W_2\) between them is **0.0814**. The mean particle distance is **0.0303**. This is the gap between the two outputs. It is not either output's distance to the reference.

The sampling fact matches the training code: most intervals are short, and half the batch never leaves the diagonal. The two-step schedule uses two intervals of length 0.5, which the sampler rarely exceeds and does not concentrate on. That is consistent with a composed evaluation disagreeing with the one-step evaluation. It does not by itself say that sub-steps must increase the error. On the fixed-learning-rate map they did. On the scheduled map, section 1, they decrease it. The degradation was a property of that checkpoint, not of the two-parameter family.

---

## 4. \(\alpha=2\) on the other two seeds, and a particle-level check

**Completed.** \(\alpha=2\) stays the value chosen on the seed-88 development split. It was not chosen again.

On the 93 test cells of each held-out split:

| seed | \(\alpha=0\) | \(\alpha=1\) | \(\alpha=2\) | \(\alpha=2\) vs \(\alpha=0\) |
|---|---:|---:|---:|---:|
| 89 | 0.12967 | 0.10619 | 0.10301 | lower by 20.6% |
| 90 | 0.11294 | 0.10180 | 0.10417 | lower by 7.8% |

On seed 90, \(\alpha=2\) is higher than \(\alpha=1\). The selected strength transfers as a reduction against no corrector. It is not the best of the two positive values on every seed.

Sliced \(W_2\), 64 projections, seed 88, all 143 held-out cells, fixed learning rate, \(\alpha=1\): stage 1 is 0.02226, stage 1 plus the corrector is 0.01545. This estimator is a lower bound on \(W_2\) from one-dimensional projections. It is not the histogram number, and the reduction on it is larger. The direction agrees with the histogram. Energy distance was not computed. It is quadratic in the particle count.

Mean absolute relative error of three summaries, same 143 cells:

| summary | stage 1 | stage 1 + corrector |
|---|---:|---:|
| centroid, \(x\) | 0.0275 | 0.0186 |
| rms radius | 0.00756 | 0.00590 |
| anisotropy | 0.0115 | 0.00735 |

The histogram reduction is not an artefact of the 16-bin grid, and it appears in the physical summaries as well.

---

## 5. One corrector for three stage-1 seeds

**Completed.** This is the run that was deferred until the four items above had an answer. One corrector, \(\alpha=1\), is trained on the stage-1 residuals of seeds 88, 89, and 90. Each seed keeps its own 40 training cells. The budget is 2,000 updates per seed per cell, 120 groups, 240,000 steps. Each seed is then scored on its own 143 held-out cells. The stage-1 maps are the fixed-learning-rate checkpoints. The scheduled map of section 1 is a different experiment and is not an input here.

| seed | stage 1 | shared corrector | reduction | cells lower |
|---|---:|---:|---:|---:|
| 88 | 0.12277 | 0.12058 | 1.8% | 74 / 143 |
| 89 | 0.12975 | 0.11527 | 11.2% | 143 / 143 |
| 90 | 0.11203 | 0.10551 | 5.8% | 110 / 143 |

The mean of the three reductions is 6.3%. The paired correctors in the earlier note, each fitted to its own map, reduce the same three stage-1 means by 14.4% and win 415 of 429 comparisons. Wins here are 327 of 429. Applying the seed-88 corrector unchanged to seeds 89 and 90 was a change of −1.4% and −3.2%. The shared corrector does not repeat that negative result.

The pattern inside the table is the reason. Seed 89 has the worst stage 1 and receives the largest reduction, on every cell. Seed 88, the map whose own corrector removed 15.0%, receives 1.8%, and loses almost half the cells. The residual fields of the three checkpoints are not the same direction. A corrector that is trained on all three learns a compromise. It no longer moves another seed's cloud the wrong way. It also no longer matches the residual of any one of them as closely as a paired corrector does.

---

## 6. Not run

Harry asked that iteration not be pursued. It was not. The gated corrector was left for later. The warm-up schedule was not repeated for the paired corrector on seeds 89 and 90. Row (c) of section 1 is the only warm-up result with three seeds.
