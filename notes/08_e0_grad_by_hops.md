# 08. E0, gradient by layer against hop count

Deeper pointer chasing does not put a larger share of the gradient in the looped layers. The looped share is 0.93 at $k=1$ and about 0.95 for every $k$ from 2 to 6, and the one-hop reverse control also gives 0.95. The only step separates the near-solved $k=1$ task from everything harder. Script is `scripts/08_e0_grad_by_hops.py`, outputs are in `artifacts/e0_grad_by_hops/`.

## Method

Each run is a fresh rank-2 LoRA client with LoRA seed 0, so all runs share the same $A$ matrices. It takes one AdamW step from $B=0$ with $\text{lr} = \epsilon = 10^3$ on 32 examples, supervising only the answer token (note 06). The gradient is read back as in script 03, $|g| = \epsilon u/(1-u)$ with $u = |\Delta B|/\text{lr}$. The largest $u$ in any run was 0.16. Conditions are $k \in \{1,2,3,4,6\}$ and the single-table reverse control (note 07), each with data seeds 1 and 2.

Let $n_{\ell,m}$ be the Frobenius norm of the $B$ gradient for module $m$ in layer $\ell$. The looped share, excluding `lm_head`, is

$$S = \frac{\sum_{\ell=6}^{17} \sum_m n_{\ell,m}^2}{\sum_{\ell=0}^{23} \sum_m n_{\ell,m}^2}.$$

The ratio $R$ is the mean of $\sqrt{\sum_m n_{\ell,m}^2}$ over layers 6 to 17 divided by its mean over the 12 unlooped layers.

## Results

| Condition | Seed | Loss sum | Mean $\log p$ | $S$ all | $S$ attention | $S$ MLP experts | $R$ |
|---|---|---|---|---|---|---|---|
| $k=1$ | 1 | 1.5 | -0.05 | 0.929 | 0.917 | 0.944 | 3.66 |
| $k=1$ | 2 | 1.1 | -0.04 | 0.928 | 0.914 | 0.944 | 3.72 |
| $k=2$ | 1 | 70.7 | -2.21 | 0.949 | 0.945 | 0.955 | 4.73 |
| $k=2$ | 2 | 68.7 | -2.15 | 0.951 | 0.947 | 0.957 | 4.93 |
| $k=3$ | 1 | 172.9 | -5.40 | 0.949 | 0.940 | 0.959 | 4.86 |
| $k=3$ | 2 | 154.1 | -4.82 | 0.949 | 0.940 | 0.959 | 4.89 |
| $k=4$ | 1 | 160.4 | -5.01 | 0.951 | 0.943 | 0.961 | 4.81 |
| $k=4$ | 2 | 163.2 | -5.10 | 0.953 | 0.945 | 0.962 | 5.02 |
| $k=6$ | 1 | 155.9 | -4.87 | 0.939 | 0.927 | 0.954 | 4.15 |
| $k=6$ | 2 | 155.2 | -4.85 | 0.949 | 0.939 | 0.959 | 4.65 |
| reverse | 1 | 54.0 | -1.69 | 0.951 | 0.949 | 0.956 | 5.06 |
| reverse | 2 | 39.7 | -1.24 | 0.946 | 0.945 | 0.949 | 4.95 |

From $k=1$ to $k=2$, $S$ rises by about 0.02 and $R$ by about 1.1. From $k=2$ to $k=6$ the seed means of $S$ move by at most 0.006, no more than the seed gap at $k=6$ (0.010). Elsewhere the two seeds agree within 0.005. The seed gap in $R$ reaches 0.5 at $k=6$, so $R$ is the noisier summary.

Layers 0 to 5 hold 0.04 to 0.05 of the squared norm in every run. Layers 18 to 23 hold 0.019 to 0.028 at $k=1$ and 0.004 to 0.013 elsewhere, so the $k=1$ step comes from the late unlooped layers. The per-layer profile barely changes across runs (cosine similarity of square-root profiles at least 0.97), and the largest layer is 11 or 12 in every run.

The generic chat batches of script 03 gave $S$ of 0.76 to 0.79 and $R$ near 2.0. They used different prompts and supervised whole replies, so this shows only that the task concentrates gradient in the looped block more than chat data does.

## Confounds and limits

- Loss differs about 100-fold between $k=1$ and the rest, so raw norms are not compared. The reverse control has loss and share close to $k=2$. That fits a share that tracks being wrong rather than depth, but two seeds cannot separate the readings.
- Each looped layer runs three times, so its weight gradient sums three passes. A high share is partly architectural.
- One $A$ initialisation, so the spread is data plus readout noise only. Gradient at initialisation, $B$ only, rank 2.
- Script 03 measured per-tensor readout agreement within about 6%. Shares sum many tensors, and the small seed gaps fit lower noise.

No request failed and the SDK logged no warnings or rate-limit messages over the 12 clients.
