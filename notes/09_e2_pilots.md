# 09. E2 pilots, AdamW against Dimuon

Both optimizers learn $k = 2, 3, 4$. Dimuon at learning rate $1.6 \times 10^{-2}$ passed the early-stop bar in 50 steps and AdamW at $5.4 \times 10^{-4}$ in 100. Neither puts more adapter weight in the looped layers. The looped share of squared norm is 0.45 to 0.49 for AdamW and 0.49 to 0.51 for Dimuon, where an even spread gives 0.5. With one run each, this is suggestive only. Code is in `scripts/09` and `scripts/10`, outputs and the `compare_*.png` figures in `artifacts/e2_pilot/`.

## Setup

The two runs share rank 8, seeds, batch 32 over $k = 2, 3, 4$ and the eval schedule (30 examples per condition). Training stops once $k = 2, 3, 4$ all exceed 0.95. Dimuon's whole-adapter step is 61 times smaller at equal rate (note 02), so at 30 times the rate its first step is about half of AdamW's. Dimuon learned faster, so no second rate was run.

| | Steps | Median step (s) | Mean step (s) | Wall time (s) |
|---|---|---|---|---|
| AdamW | 100 | 12.4 | 13.9 | 1815 |
| Dimuon | 50 | 11.4 | 14.4 | 1177 |

An eval takes about 45 s, an export 42 to 185 s. Neither run logged a rate limit, an API error or a queue pause.

## Accuracy

| Step | Optimizer | $k=1$ | $k=2$ | $k=3$ | $k=4$ | $k=6$ | Reverse |
|---|---|---|---|---|---|---|---|
| 0 | both | 1.00 | 0.37 | 0.13 | 0.03 | 0.00 | 0.53 |
| 10 | AdamW | 0.83 | 0.63 | 0.43 | 0.17 | 0.00 | 0.90 |
| 10 | Dimuon | 0.97 | 0.77 | 0.73 | 0.27 | 0.03 | 1.00 |
| 25 | AdamW | 0.90 | 1.00 | 0.73 | 0.00 | 0.00 | 0.97 |
| 25 | Dimuon | 0.43 | 0.97 | 0.63 | 0.70 | 0.00 | 0.90 |
| 50 | AdamW | 0.13 | 1.00 | 0.83 | 0.90 | 0.00 | 0.03 |
| 50 | Dimuon | 0.00 | 1.00 | 1.00 | 0.97 | 0.00 | 0.07 |
| 100 | AdamW | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.07 |

Each accuracy has a standard error of at most 0.09.

## Forgetting

Both final adapters apply a hop count clamped to the range 2 to 4. On $k=1$ they answer $f^2(x)$ in 28 of 30 cases. On $k=6$ they answer $f^4(x)$ in 30 of 30. Reverse lookup asks for $f^{-1}(y)$, and they answer $f^2(y)$ in 26 (AdamW) and 27 (Dimuon) of 30. Single lookups were not lost. Reverse accuracy first rose to 0.9 or more by step 10, then collapsed between steps 25 and 50 in both runs.

## Per-layer profiles

The looped share $S$ is the squared Frobenius norm summed over layers 6 to 17 divided by the sum over all 24, without `lm_head`. $R$ is the mean per-layer norm over looped layers divided by the mean over unlooped ones. "Effective" means $\frac{\alpha}{r} B A$.

| Quantity | Group | $S$ AdamW | $S$ Dimuon | $R$ AdamW | $R$ Dimuon |
|---|---|---|---|---|---|
| $B$ | all | 0.489 | 0.510 | 0.98 | 1.02 |
| $B$ | attention | 0.470 | 0.494 | 0.94 | 0.99 |
| $B$ | MLP experts | 0.489 | 0.511 | 0.98 | 1.02 |
| effective | all | 0.451 | 0.508 | 0.91 | 1.02 |
| effective | attention | 0.447 | 0.491 | 0.90 | 0.98 |
| effective | MLP experts | 0.451 | 0.509 | 0.91 | 1.02 |

By module, effective $S$ spans 0.43 to 0.46 for AdamW and 0.48 to 0.51 for Dimuon. AdamW dips in the middle of the loop. Dimuon's effective norm stays within 4% of its mean across looped layers. Neither shows a step at the loop edges. Layer 6 over layer 5 is 0.91 to 1.03 and layer 18 over layer 17 is 0.90 to 1.08, while other adjacent pairs span 0.74 to 1.14.

$\|B\|_F$ over the 24 layers is 36.2 for AdamW and 57.2 for Dimuon. The `lm_head` $B$ norm is 17.5 for AdamW and 0.85 for Dimuon, as note 02's first step predicted.

E0 put 0.95 of the step-one gradient in the looped layers (note 08). A share near 0.5 after training fits both optimizers discarding gradient size.

## Recommended main experiment

- Train on $k=2$ only, $k=4$ only and reverse only, under each optimizer, with LoRA seeds 0 and 1. Shifts of a few hundredths in $S$ need a seed spread.
- Keep AdamW at $5.4 \times 10^{-4}$ and Dimuon at $1.6 \times 10^{-2}$. Neither diverged.
- Run a fixed 75 steps with early stop off, so profiles compare equal training. Mixed $k=4$ needed 50 to 100 steps at 11 examples per batch, while single-condition batches have 32.
- A run costs $75 \times 12.4 \approx 930$ s of steps plus evals and exports, about 20 to 25 minutes, so 4 to 5 hours for 12.
- Expect each adapter to apply its training hop count to every prompt. Keep every eval condition.
- Add per-layer singular values of $B$ to `adapter_stats`.

## Not verified

- Seed noise, since there is one run per optimizer.
- Profiles at equal steps. Dimuon stopped at 50, AdamW at 100.
- The singular-value spread of $B$. The harness does not save it and the adapters are deleted.
- The effect of other Dimuon rates.
- The 75-step budget, which is extrapolated from the mixture.
