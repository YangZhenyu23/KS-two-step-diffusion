# The corrector after a scheduled stage 1

**Follow-up to the note of 18 November**

Zhenyu Yang · 20 November 2026

Chinese version: [`REPORT_TO_HARRY.zh.md`](REPORT_TO_HARRY.zh.md).

A warm-up and a cosine, on the same 40 cells and the same 32,000 updates per cell, cut the corrector's reduction from 15.0% to **7.3%**. Scheduled stage 1 reaches 0.09874 on the 143 held-out cells. That is already 5.3% below 0.10429, where the old corrector stopped, so the schedule removes more error (0.0240) than the second network did (0.0185), and it costs nothing when the map is used. The corrector still lowers the scheduled map, from 0.09874 to 0.09153, on 128 of the 143 cells.

The four checks below say where the old 15% came from. Distances are 16-bin histogram \(W_2\) unless a paragraph names another estimator.

---

## 1. The schedule accounts for the missing half

The corrector's gain shrinks because stage 1 is closer to the reference, and the corrector can only remove what is left. The learning rate rises linearly over the first 2% of the updates to \(10^{-3}\), then follows a cosine to \(10^{-5}\). The corrector is the same one as before: \(\alpha=1\), 2,000 updates per cell.

![Stage 1 and its corrector, at a fixed learning rate and with the warm-up.](../results/report_figures/fig_reply_schedule.png)

*Figure 1. Histogram \(W_2\) on the 143 held-out cells, seed 88. Left: stage 1, and the same map after its corrector. Right: the corrector's reduction. The axis starts at zero.*

On the 112 cells held out by the headline design and by all three width-112 designs, the same schedule gives four rows.

| row | what is scored | \(W_2\) | against (a) |
|---|---|---:|---:|
| (a) | scheduled stage 1, one step, seed 88 | 0.09859 | — |
| (b) | (a), then its corrector | 0.09150 | −7.2%, lower on 99 of 112 |
| (c) | width 112, same schedule, three seeds | 0.10215 ± 0.00133 | +3.6% |
| (d) | the weights of (a), two sub-steps | 0.08358 | −15.2% |

The three width-112 seeds are 0.10287, 0.10062 and 0.10297. Each sits above (a).

![Four rows on the 112 common cells.](../results/report_figures/fig_reply_four.png)

*Figure 2. Left: the four rows. Open circles on (c) are the three seeds. Right: change against one step. A negative bar is a higher error.*

Two sub-steps are the lowest row. On the fixed-rate map they had raised the error by 1.3%, because that map's velocity was a poor average over \([0,1]\) and splitting the interval exposed the error. Here the scheduled field is accurate enough that the same split helps, and it does so at 2.0 times the cost of one pass, against 2.97 times for the corrector. Width does not substitute for either: a network of 112 units, trained with the same schedule, finishes above the narrow one-step map. The 15% in the earlier note is the corrector's gain against a fixed learning rate. The gain against a scheduled stage 1 is 7.3%.

## 2. Extra updates recover about a fifth of the drop

Spending the corrector's extra 80,000 updates on the same network does not reproduce its drop. Further MeanFlow updates at \(10^{-3}\) move the fixed-rate stage 1 from 0.12277 to 0.12465, which is 1.5% worse. The same number of updates, aimed at the minibatch transport targets, reach 0.11911, which is 3.0% better. The separate corrector reaches 0.10429, which is 15.0% better.

![The same 80,000 updates, spent three ways.](../results/report_figures/fig_reply_budget.png)

*Figure 3. Seed 88, fixed learning rate, 143 held-out cells. Percentages are against stage 1. A positive sign is a higher error.*

The MeanFlow loss had already flattened, so more of it at the same learning rate adds noise. The transport targets do move the cloud, because they are the assignment the corrector is trained to match, but they move one network that was trained to be an average velocity. About one fifth of the corrector's drop appears there (0.0037 of 0.0185). The other four fifths appear only when the correction is a second map, started at the identity and penalised for leaving the cloud it corrects. If the second control had recovered most of the gain, the contribution would have been the coupling. It recovers a fifth, so the contribution is the second map.

These two controls are on the fixed-rate network, which is the network whose 15% was in question. On the scheduled network the corrector's remaining reduction is already 7.3%, and two passes of the same weights go lower than the corrector does.

## 3. Sub-steps are evaluated off the training measure

Intervals of length 0.5, the ones used by two steps, are rarely drawn in training. The one-parameter model and the two-parameter model share one sampler: two draws from a normal with mean \(-0.4\) and unit variance pass through a sigmoid, the larger becomes \(t\) and the smaller \(r\), and half of each batch is then set to \(r=t\). In one million draws, 50.0% lie on \(r=t\) and 95.8% are shorter than 0.5, so 4.2% are at least as long as one sub-step. Two-step inference uses the same fixed split in both models, \([1,0.5]\) and then \([0.5,0]\). That step length was not chosen by a search.

![Where training samples \((r,t)\), and how long the intervals are.](../results/report_figures/fig_reply_intervals.png)

*Figure 4. One million draws. Left: density of \((r,t)\). The dashed line has length 0.5. The star is one step, \((0,1)\); the squares are the two sub-steps. Right: the cumulative share of interval length.*

On the fixed-rate two-parameter map the two procedures do not produce the same cloud. Over 143 held-out cells the histogram \(W_2\) between the one-step cloud and the two-step cloud is 0.0814, and particles that share a source end 0.0303 apart. Against the reference, on the 112 common cells, the error rises from 0.12271 to 0.12435, by 1.3%. The same split on the one-parameter KS3D runs lowers the error at 18 of 20 values of \(\sigma\), by about 2% to 9%, and raises it at \(\sigma=10\) and \(\sigma=20\). The sampler and the split are the same, so the difference in sign is not a difference in how time was sampled.

MeanFlow reports that the fraction of training pairs with \(r\neq t\) changes one-step FID: 0% gives 328.91, 25% gives 61.06, 50% gives 63.14, and 100% gives 67.32. Both of our models use 50%. That comparison is one-step ImageNet FID and does not measure two-step error, so I would not say that changing the fraction is sure to improve two-step evaluation. On the scheduled two-parameter model the sampler is unchanged, and two steps fall from 0.09859 to 0.08358, 15.2% below one step. The rise at the fixed learning rate is what these rarely drawn intervals produce on a velocity field that has not been trained firmly. It is not a property of the two-parameter family.

The feedforward step integrates the flow time \(t\). One generation holds \((\sigma,\mu)\) fixed, and those two numbers only select the velocity. The update is the one the one-parameter model uses: \(x-u(x,0,1)\) for one step, and the same rule on \([1,0.5]\) and then \([0.5,0]\) for two steps. A second physical parameter does not change that rule. What was not searched is how often each pair \((r,t)\) is drawn. The sampler in this section is the upstream one, used as it stood. A run that keeps the fixed learning rate, the same 40 cells and the same 32,000 updates per cell, and changes only that distribution, is in progress. If one step stays where it is or improves, and two steps improve, the earlier comparison had not searched the sampling frequency of the time intervals. That result is not in this note.

## 4. The chosen \(\alpha\) keeps its sign, and the bins hide a few points

\(\alpha=2\), chosen on the seed-88 development split and not chosen again, still beats \(\alpha=0\) on the 93 test cells of seeds 89 and 90. The reductions are 20.6% and 7.8%. On seed 89 most of that drop is already present at \(\alpha=1\) (0.10619 against 0.10301 at \(\alpha=2\)). On seed 90, \(\alpha=1\) is lower than \(\alpha=2\), 0.10180 against 0.10417. The selected value transfers in sign and loses size, which is what a setting chosen on one seed should be expected to do.

| seed | \(\alpha=0\) | \(\alpha=1\) | \(\alpha=2\) | \(\alpha=2\) against \(\alpha=0\) |
|---|---:|---:|---:|---:|
| 88, where it was chosen | 0.12353 | 0.10477 | 0.09558 | −22.6% |
| 89 | 0.12967 | 0.10619 | 0.10301 | −20.6% |
| 90 | 0.11294 | 0.10180 | 0.10417 | −7.8% |

The same direction appears in the physical summaries, now on all 143 held-out cells of the fixed-rate pair rather than on six. Mean absolute relative error of the centroid falls from 0.0275 to 0.0186, the rms radius from 0.00756 to 0.00590, and the anisotropy from 0.0115 to 0.00735. A residual map trained to carry particles toward the reference should move the centroid most, and it does.

![Alpha on seeds 89 and 90, and the three summaries on 143 cells.](../results/report_figures/fig_reply_scope.png)

*Figure 5. Left: mean histogram \(W_2\) on 93 test cells. \(\alpha=2\) was chosen on seed 88. Right: mean absolute relative error on the fixed-rate pair.*

The histogram is the smaller of the readings we have. Sliced \(W_2\) along 64 directions, on these same 143 cells, falls from 0.02226 to 0.01545, a 30.6% reduction against the histogram's 15.0%. Sliced \(W_2\) averages one-dimensional projections, so the two percentages are not the same distance.

An exact coupling in the original dimension exists for one cell of the earlier single-parameter run, \(\sigma=100\), seed 88, with the same 30,000 particles used for both methods. The 16-bin histogram gives a 16.0% reduction, from 0.18311 to 0.15374. Exact transport at \(m=10{,}000\) gives 21.3%, and at \(m=30{,}000\) gives 23.0%, from 0.14503 to 0.11164. Leaving the bins adds about seven points. Going from 10,000 particles to 30,000 adds 1.7 of them, so the subsample size is not what the histogram is missing. A particle that moves and stays inside its bin does not change a histogram cost, and a modest correction is exactly the motion a coarse grid drops.

![One cell scored three ways, and twenty shears at ten thousand particles.](../results/report_figures/fig_reply_metric.png)

*Figure 6. Left: \(\sigma=100\), seed 88. Right: twenty shears at \(m=10{,}000\). A point above the dashed line has the larger reduction on the particles. \(\sigma=100\) is marked.*

Across those twenty shears the mean reduction goes from 39.1% on the histogram to 42.5% on the particles, and 18 of the 20 shears move the same way. A line through them has slope 1.03 and intercept 2.1, with a residual of 2.5 points. Read at the scheduled 7.3%, it gives **9.6%**, and a bootstrap over the shears gives 7.1 to 11.9. This cell is a single-parameter cloud, not a rescoring of the tables above. Scaling 7.3% by the ratio at \(\sigma=100\), 23.0 to 16.0, gives about 10%. Adding the seven points gives about 14%. The three routes sit in one range: the scheduled gain is larger than 7.3% by a few points, and the size of that correction has not been measured on the scheduled pair.

## 5. One corrector for three seeds keeps a shared piece of the residual

A single corrector trained on seeds 88, 89 and 90 together lowers all three maps, and it lowers them by less than a corrector fitted to each map. The mean reduction is 6.3%, against 14.4% for the three fitted correctors. It is lower on 327 of 429 cells, against 415 of 429 for the fitted ones. Each seed keeps its own 40 cells. The budget is 2,000 updates per cell per seed. The stage-1 maps are the fixed-rate checkpoints.

| seed | stage 1 | shared | fitted to that seed | shared / fitted |
|---|---:|---:|---:|---:|
| 88 | 0.12277 | 0.12058 | 0.10429 | 1.8% / 15.0% |
| 89 | 0.12975 | 0.11527 | 0.10607 | 11.2% / 18.2% |
| 90 | 0.11203 | 0.10551 | 0.10082 | 5.8% / 10.0% |

![Stage 1, one shared corrector, and a corrector fitted to each seed.](../results/report_figures/fig_reply_multiseed.png)

*Figure 7. Fixed learning rate, each seed on its own 143 held-out cells. Right: reduction against that seed's stage 1.*

The corrector learns the residual of the maps it is shown. The part those three residuals share is what one network can remove from all three, and that part is largest on the worst stage-1 map. Seed 89, the highest of the three, is lower on all 143 of its cells. Seed 88, where a fitted corrector removed 15.0%, keeps 1.8%. This repairs the transfer failure in the earlier note, where the seed-88 corrector applied unchanged to seeds 89 and 90 changed those errors by −1.4% and −3.2%. It does so by giving up the seed-specific residual on seed 88.

## What follows from the four checks

Inside the training box, on the map it was fitted to, and with stage 1 trained under the schedule, the corrector reduces one-step histogram \(W_2\) by 7.3% at 2.97 times the cost of one pass. Two passes of the scheduled network reduce it by 15.2% at 2.0 times that cost. The 15% reported in November was the corrector's gain against a fixed learning rate. The schedule, free at inference, accounts for the larger share of that gain.

The gain still stops at a change of seed and outside the training box, as the earlier note found. One corrector for three seeds improves the two seeds a transferred corrector used to hurt, and it keeps little of the gain on the seed it fits best. The histogram reads a modest gain low. On the one cell solved at 30,000 particles, 16.0% became 23.0%, and about five of those seven points were already present at 10,000.

Paired correctors for seeds 89 and 90 were not retrained under the schedule, so Section 5 is on fixed-rate maps. The map was not iterated: it remains a map over a fixed horizon, and conditioning the corrector on \(t\) is a different project. The budget controls, the sliced distance, the physical summaries and the one-step versus two-step comparison were measured on the fixed-rate map. If 7.3% is to replace 15% as the headline, those four measurements, and a particle-level score of the scheduled pair, are the ones still missing.
