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
