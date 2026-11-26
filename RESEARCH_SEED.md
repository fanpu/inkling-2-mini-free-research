# Research Seed & Theoretical Framing

## Core Methodology
> Take the smallest model in which a training phenomenon still appears, solve it exactly, and then test whether the quantitative prediction holds in a real network.

Starting toy model: $y = abx$ (loss $\frac{1}{2}(p - s)^2$ where $p = ab$), which exhibits parameter redundancy $(a, b) \mapsto (\lambda a, b/\lambda)$, gradient-flow conservation laws ($a^2 - b^2$), plateaus near the origin, edge-of-stability step limits, and sharp weight-decay thresholds.

## Prior Research Threads (Through 26 Nov 2026)

1. **Thread 1 — The `abx` Lab:**
   - Scalar models of modern LLM recipes (weight decay, depth, AdamW, Muon, cautious decay) and pair interactions.
   - Open frontier: how weight-decay wiring decides which modes survive across products of layers (coupled L2 kills weak modes first; Muon + decoupled decay clips strong modes at a ceiling; AdamW caps axis-aligned modes; cautious decay preserves modes once step size decays).

2. **Thread 2 — Muon's Simplicity Bias + Weight Decay:**
   - Building on Dragutinović and Ranganath (`arXiv:2603.00742`, *"Muon's Simplicity Bias"*, two authors: Sara Dragutinović and Rajesh Ranganath) and Wu, Mansouri, and Zhu (`arXiv:2610.20112`, Newton-Schulz $\epsilon$ flipping spectral vs. Frobenius max-margin).
   - Coupled decay restores a GD-like soft-threshold fixed point; decoupled decay adds a ceiling while keeping weak-first ordering.
   - Note for Tinker: prior Tinker docs suggest `DimuonParams` exposes `learning_rate`, `beta1`, and `grad_clip_norm` without `weight_decay` or Newton-Schulz $\epsilon$ (being verified in Phase 0). Internal Muon WD-wiring remains local-PyTorch territory unless exposed.

3. **Thread 3 — The `abx` of Sequences:**
   - *(a) Delta-rule associative memory / MQAR* under superposition ($V > d_k$): rotation-symmetric forward pass vs. Adam's privileged basis.
   - *(b) 1D scalar recurrent model*: $h_t = a h_{t-1} + b x_t$, $y_t = c h_t$. Cf. Proca, Dominé, Shanahan, and Mediano (*"Learning dynamics in linear recurrent neural networks"*, ICML 2025).

## Synthesis: Learning Mechanics of a Looped Transformer (`arXiv:2604.21691`)
`arXiv:2604.21691` (*"There Will Be a Scientific Theory of Deep Learning"*, Simon, Kunin, Atanasov, Bordelon, Cohen, Jacot, Michaud, et al.) frames **Learning Mechanics** around:
1. Solvable idealized settings
2. Tractable limits
3. Simple macroscopic mathematical laws for coarse aggregate statistics
4. Theories of hyperparameters
5. Universal behaviors & bridging learning mechanics with mechanistic interpretability

**Why Inkling 2 Mini is an ideal testbed:**
Inkling 2 Mini reportedly uses a hybrid looped architecture inside a single 26B sparse MoE model (24 physical layers $\to$ 48 logical layers: layers 0–5 run 1x, layers 6–17 looped 3x with shared weights and cross-loop messages, layers 18–23 run 1x). Unlooped layers ($K=1$) and looped layers ($K=3$) coexist under the *same* training run, batch, and loss.

**Minimal Toy Model:** Unlooped LoRA $y = (w_0 + ab)x$ vs. $K$-looped LoRA $y = (w_0 + ab)^K x$ (and residual $(1 + w_0 + ab)^K x$).

## Initial Falsifiable Hypotheses
- **H1 — Optimizer Mechanics Across the Loop Boundary:** Loop multiplicity $K=3$ amplifies gradients by $O(K \rho^{K-1})$ and curvature by $O(K^2)$. Dimuon's per-layer spectral normalization forces a uniform per-layer step that compounds 3x inside layers 6–17, shifting layer profiles ($\|B_\ell A_\ell\|_F$, $\|B_\ell A_\ell\|_2$, effective rank, $\|B_\ell\|_F / \|A_\ell\|_F$) relative to AdamW across layers 0–5 vs. 6–17 vs. 18–23.
- **H2 — Stability / Edge-of-Stability & Weight-Decay Phase Boundary:** Sweeping LR (Dimuon & AdamW) and `weight_decay` (AdamW) tests whether the looped core collapses or destabilizes at quantitatively shifted thresholds relative to unlooped layers, and whether Dimuon's max stable LR is bottlenecked by layers 6–17.
- **H3 — Task-Dependent Adapter Localization:** Comparing static 1-step recall vs. multi-hop/iterative tasks tests whether adapter energy/rank localizes into the $K=3$ looped core on iterative tasks and whether Dimuon's per-layer normalization alters that localization.
