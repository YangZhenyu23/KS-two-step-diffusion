# The corrector after a scheduled stage 1

**Follow-up to the note of 18 November**

Zhenyu Yang · 20 November 2026

Chinese version: [`REPORT_TO_HARRY.zh.md`](REPORT_TO_HARRY.zh.md).

A warm-up and a cosine, on the same 40 cells and the same 32,000 updates per cell, cut the corrector's reduction from 15.0% to **7.3%**. Scheduled stage 1 reaches 0.09874 on the 143 held-out cells. That is already 5.3% below 0.10429, where the old corrector stopped. The schedule removes more error (0.0240) than the second network did (0.0185), and it costs nothing when the map is used. The corrector still lowers the scheduled map, from 0.09874 to 0.09153, on 128 of the 143 cells.

The four checks below say where the old 15% came from. Every \(W_2\) in this note is the 16-bin histogram.

---

## 1. The schedule accounts for the missing half

The corrector's gain shrinks because stage 1 is closer to the reference, and the corrector can only remove what is left. The learning rate rises linearly over the first 2% of the updates to \(10^{-3}\), then follows a cosine to \(10^{-5}\). The corrector was retrained on that new stage 1: \(\alpha=1\), 2,000 updates per cell. It is not the old corrector applied to the new map.

![Stage 1 and its corrector, at a fixed learning rate and with the warm-up.](../results/report_figures/fig_reply_schedule.png)

*Figure 1. Histogram \(W_2\) on the 143 held-out cells, seed 88. Left: stage 1, and the same map after its corrector. Right: the corrector's reduction.*

On the 112 cells held out by the headline design and by all three width-112 designs, the same schedule gives four rows.

| row | what is scored | passes | \(W_2\) | against (a) |
|---|---|---:|---:|---:|
| (a) | scheduled stage 1, one step, seed 88 | 1 | 0.09859 | — |
| (b) | (a), then its corrector | 1 + corrector | 0.09150 | −7.2%, lower on 99 of 112 |
| (c) | width 112, same schedule, seeds 88 / 89 / 90 | 1 | 0.10287 / 0.10062 / 0.10297 | each above (a) |
| (c) | mean of those three seeds | 1 | 0.10215 ± 0.00133 | +3.6% |
| (d) | the weights of (a), two sub-steps | 2 | 0.08358 | −15.2% |

Two sub-steps are the lowest row, at 2.0 times the cost of one pass. The corrector costs 2.97 times and removes less. On the fixed-rate map the same split had raised the error, by 1.3% on that note's 112 cells. Width does not substitute for either row: all three wide seeds finish above the narrow one-step map. The 15% in the earlier note is the corrector's gain against a fixed learning rate. The gain against a scheduled stage 1 is 7.3%.

## 2. Extra updates recover about a fifth of the drop

Spending the corrector's extra 80,000 updates on the same network does not reproduce its drop. These controls are on the fixed-rate network, which is the network whose 15% was in question.

| what receives the 80,000 updates | \(W_2\) | against stage 1 (0.12277) |
|---|---:|---:|
| stage 1, no extra updates | 0.12277 | — |
| further MeanFlow updates at \(10^{-3}\) | 0.12465 | +1.5% |
| the same updates, on the minibatch OT targets | 0.11911 | −3.0% |
| a second network, the paired corrector | 0.10429 | −15.0% |

The transport targets recover 0.00366 of the corrector's drop of 0.01848, which is 19.8%. That is not most of it. The 15% is the second network. The average hides a split by \(\mu\).

| \(\mu\) | \(n\) | extra MeanFlow | same network, OT | corrector | share recovered by OT |
|---|---:|---:|---:|---:|---:|
| 0.10 | 15 | −10.6% | +6.2% | −19.1% | the wrong way |
| 0.63 | 16 | −15.0% | −8.5% | −16.9% | 50% |
| 2.51 | 17 | +5.0% | −10.4% | −13.9% | 74% |
| 3.98 | 17 | +14.3% | −11.3% | −14.4% | 79% |
| 10 | 15 | +20.6% | +8.2% | −11.6% | the wrong way |

A minus sign is a lower error. At \(\mu=2.51\) and \(3.98\) the OT fine-tune recovers most of the corrector's drop. At \(\mu=10\) it raises the error, while the corrector still lowers it. On the scheduled network the corrector's remaining reduction is 7.3%, and two passes of the same weights go lower than the corrector does.

## 3. The half-intervals are rare, and upweighting them made the map worse

Two-step evaluation uses \([1,0.5]\) and then \([0.5,0]\). The one-parameter model and the two-parameter model share one sampler: two draws from a normal with mean \(-0.4\) and unit variance pass through a sigmoid, the larger becomes \(t\) and the smaller \(r\), and half of each batch is set to \(r=t\). In one million draws, 50.0% lie on \(r=t\) and 95.8% are shorter than 0.5, so 4.2% are at least as long as one sub-step. That step length was not chosen by a search.

![Where training samples \((r,t)\), and how long the intervals are.](../results/report_figures/fig_reply_intervals.png)

*Figure 2. One million draws from the training sampler. Left: density of \((r,t)\). The dashed line has length 0.5. The star is one step, \((0,1)\); the squares are the two sub-steps. Right: the cumulative share of interval length.*

On the fixed-rate two-parameter map the two procedures do not return the same cloud. Over 143 held-out cells the histogram \(W_2\) between them is 0.0814, and particles that share a source end 0.0303 apart. Against the reference, two steps raise the error from 0.12277 to 0.12418. The same split on the one-parameter KS3D runs lowers the error at 18 of 20 values of \(\sigma\), by about 2% to 9%, and raises it at \(\sigma=10\) and \(\sigma=20\). The sampler is the same in both models.

I take the fixed-rate rise to come from those half-intervals being rare in training. I tested that with one retraining of seed 88. The learning rate stayed at \(10^{-3}\), the 40 cells stayed, and the update count stayed at 32,000 per cell. Only the draw of \((r,t)\) changed. Each batch is 40% the original sampler, 20% on the diagonal, 20% on \((0,1)\), 10% on \((0,0.5)\) and 10% on \((0.5,1)\). The share of intervals of length at least 0.5 went from 4.2% to 43.4%. The corrector was then trained with the original protocol. The run is one seed.

| | original sampler | half-intervals upweighted |
|---|---:|---:|
| one step | 0.12277 | 0.13980 |
| two steps | 0.12418 | 0.12327 |
| one step, then the corrector | 0.10429 | 0.11465 |

One step is higher by 13.9%. Two steps are 0.7% below the old two-step number, and 11.8% below the new one-step number. The corrected value is higher than 0.10429. Raising the weight on \([0,0.5]\) and \([0.5,1]\) made the map worse. It did not remove the gap between a good one-step map and its two-step evaluation.

One such run takes about three hours, so I did not repeat the seed and I did not search the mixture. MeanFlow reports that the fraction of training pairs with \(r\neq t\) changes one-step FID on ImageNet: 0% gives 328.91, 25% gives 61.06, 50% gives 63.14, and 100% gives 67.32. Both of our models use 50%. That comparison is one-step image generation. It does say that the sampling rule moves the result. A search over the mixture, and a second seed, can be run if you want them.

On the scheduled map the original sampler was left unchanged, and two steps are the lowest row in Section 1, 15.2% below one step. The fixed-rate rise and that drop do not share a schedule. They do share the sampler.

## 4. \(\alpha=2\) keeps its sign on the other two seeds

\(\alpha=2\) was chosen on the seed-88 development split and was not chosen again. On the 93 test cells of seeds 89 and 90 it is still below \(\alpha=0\).

| seed | \(\alpha=0\) | \(\alpha=1\) | \(\alpha=2\) | \(\alpha=2\) against \(\alpha=0\) |
|---|---:|---:|---:|---:|
| 88, where it was chosen | 0.12353 | 0.10477 | 0.09558 | −22.6% |
| 89 | 0.12967 | 0.10619 | 0.10301 | −20.6% |
| 90 | 0.11294 | 0.10180 | 0.10417 | −7.8% |

On seed 89 most of the drop is already present at \(\alpha=1\). On seed 90, \(\alpha=1\) is lower than \(\alpha=2\). The selected value transfers as a reduction against no corrector. It is not the lower of the two positive values on every seed.

The physical summaries are now on all 143 held-out cells of the fixed-rate pair, not on six. The metric here is mean absolute relative error.

| summary | stage 1 | stage 1 + corrector |
|---|---:|---:|
| centroid, \(x\) | 0.0275 | 0.0186 |
| rms radius | 0.00756 | 0.00590 |
| anisotropy | 0.0115 | 0.00735 |

The centroid moves the most, which is what a residual trained toward the reference should do.

## 5. One corrector for three seeds

You asked for this run ahead of the gated corrector. One corrector, \(\alpha=1\), is trained on the stage-1 residuals of seeds 88, 89 and 90. Each seed keeps its own 40 cells. The budget is 2,000 updates per cell per seed. The stage-1 maps are the fixed-rate checkpoints. Each seed is scored on its own 143 held-out cells.

| seed | stage 1 | shared corrector | fitted to that seed | shared / fitted | cells lower |
|---|---:|---:|---:|---:|---:|
| 88 | 0.12277 | 0.12058 | 0.10429 | 1.8% / 15.0% | 74 / 143 |
| 89 | 0.12975 | 0.11527 | 0.10607 | 11.2% / 18.2% | 143 / 143 |
| 90 | 0.11203 | 0.10551 | 0.10082 | 5.8% / 10.0% | 110 / 143 |

The mean of the three shared reductions is 6.3%, against 14.4% for the three fitted correctors. The shared corrector is lower on 327 of 429 comparisons, against 415 of 429 for the fitted ones. Applying the seed-88 corrector unchanged to seeds 89 and 90 had changed those errors by −1.4% and −3.2%. The shared corrector does not repeat that. It does so by giving up most of the gain on seed 88, the map a fitted corrector had reduced by 15.0%.

## What follows from the four checks

Inside the training box, on the map it was fitted to, and with stage 1 trained under the schedule, the corrector reduces one-step histogram \(W_2\) by 7.3% at 2.97 times the cost of one pass. Two passes of the scheduled network reduce it by 15.2% at 2.0 times that cost. The 15% reported in November was the corrector's gain against a fixed learning rate. The schedule, free at inference, accounts for the larger share of that gain.

The gain still stops at a change of seed. One corrector for three seeds improves the two seeds a transferred corrector used to hurt, and it keeps little of the gain on the seed it fits best. Paired correctors for seeds 89 and 90 were not retrained under the schedule, so Section 5 is on fixed-rate maps. The map was not iterated. It remains a map over a fixed horizon.
