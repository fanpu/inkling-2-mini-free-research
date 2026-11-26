# 03. The minimal looped model

Checked numerically by `scripts/02_looped_abx_toy.py`. All three checks match the formulas to four digits.

## Model

A LoRA layer in scalar form is $u = w_0 + ab$, where $w_0$ is the frozen base weight and $a$, $b$ are the trainable factors. A layer applied $K$ times with shared weights gives the output

$$f = (w_0 + ab)^K x, \qquad L = \tfrac12 (f - s)^2 .$$

$K = 1$ is an ordinary layer. $K = 3$ stands in for Inkling 2 Mini's looped layers. LoRA initialisation is $b = 0$, $a \neq 0$.

## Three results

1. **Gradient amplification.** With output error $e = f - s$, the gradient on the product is $\partial L / \partial (ab) = e \, K u^{K-1}$. At init $u = w_0$, so for the same error the looped layer's gradient is $K w_0^{K-1}$ times larger. With $w_0 = 1$ and $K = 3$ the script gives $-0.45$ against $-0.15$.
2. **Sharpness.** At a minimum, $u^\ast = s^{1/K}$ and the top Hessian eigenvalue is $(K u^{\ast K-1})^2 (a^2 + b^2)$. Plain gradient descent is stable only for learning rates below $2$ divided by that value. With $s = 2$ the sharpness is 2.00 for $K = 1$ and 11.79 for $K = 3$. Runs at 0.95 of the threshold converge and runs at 1.05 diverge.
3. **Scale-free optimizers hide the amplification in weight space.** Sign descent, the first step of Adam, and Muon all discard gradient size. One sign step of size $10^{-3}$ moves $b$ by the same amount for both $K$. The output moves by $K w_0^{K-1} a \cdot 10^{-3}$, which is $5.0 \times 10^{-4}$ for $K = 1$ and $1.5 \times 10^{-3}$ for $K = 3$.

## Predictions for the real model

- **P1 (confirmed, but trivially).** Step-one update norms are the same in looped and unlooped layers under AdamW and Dimuon. Note 02 shows this. It follows from the optimizers alone.
- **P2 (open, the main target).** An equal-size weight change in a looped layer changes the model's output more than the same change in an unlooped layer. The scalar model says the factor is $K = 3$ when the three passes push the output the same way. If the per-pass effects were unrelated the factor would be nearer $\sqrt{3}$. The measured factor is a direct reading of how coherent the passes are.
- **P3 (open).** Because looped layers are sharper per unit of weight change, the largest stable learning rate should be set by layers 6 to 17. The scalar model gives a sharpness ratio of $K^2 u^{2(K-1)}$.

Not yet derived: the AdamW weight-decay equilibrium in the looped model, and anything about Dimuon's actual update rule, which is unknown (note 02).

## Update 2026-11-26: the gated-message model

Note 01 shows that Inkling's passes are not chained. Every pass restarts from the same input and reads a gated message from the previous pass. The scalar version is

$$h_1 = u x, \qquad h_k = u\,(x + g\,h_{k-1}), \qquad f = h_K,$$

where $g \ge 0$ is the gate gain. $g = 0$ is an ordinary layer. Large $g$ approaches the pure chain $u^K$.

Because the output's size also changes with $g$, the right quantity is the relative sensitivity $\frac{\partial f}{\partial u}\frac{u}{f}$, the percentage change in output per percentage change in weight. For $K = 3$ at $u = 1$ it is

$$\frac{1 + 2g + 3g^2}{1 + g + g^2},$$

which is 1 at $g = 0$, 2 at $g = 1$, and tends to 3 as $g$ grows. Check 4 in `scripts/02_looped_abx_toy.py` confirms this to four digits.

**Revised P2.** The ratio of looped to unlooped relative sensitivity lies between 1 and $K = 3$, and its value reads off how open the gates effectively are. The earlier lower reference of $\sqrt{3}$ is withdrawn. It assumed three chained passes with unrelated effects, which is not how the model is wired.

## What the literature review found (notes/lit/02)

The reviewer found no paper stating the $K u^{K-1}$ gradient factor together with sharpness scaling as its square for tied depth, and called it unpublished but unsurprising. Close relatives it lists: Pascanu et al. (1211.5063) on gradients through powers of a shared weight, Loop-μP (2606.18524) on looped updates adding coherently so that per-step output change grows with the loop count, and DeepLoop (2607.13491) on the shared update summing gradients over visits. I have not read these papers myself.
