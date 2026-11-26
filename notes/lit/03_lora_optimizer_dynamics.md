# LoRA optimizer dynamics: literature (checked 2026-11-26)

**Answer.** With B = 0, only B moves at first, and theory says B should keep moving faster: LoRA+ sets η_B > η_A, and a random, untrained A does almost as well as a trained one. Per-factor Muon still grows the singular values of BA at near-equal rates (Kang et al. 2026). Riemannion instead orthogonalises the product's gradient and projects it onto the tangent space. Choi et al. (2019) say Adam approximately simulates momentum SGD as ε grows, but I found no derived equilibrium adapter norm under AdamW weight decay.

| Authors, year | Title | ID | Relevance |
|---|---|---|---|
| Hayou, Ghosh, Yu 2024 | LoRA+ | 2402.12354 | η_B = λη_A with λ > 1, so B moves more than A. |
| Hayou, Ghosh, Yu 2024 | Impact of Initialization on LoRA Finetuning Dynamics | 2406.08447 | B = 0 with random A beats the reverse and tolerates larger LR. |
| Kalajdzievski 2023 | rsLoRA | 2312.03732 | Scale by α/√r, not α/r, or learning slows at high rank. |
| Zhu et al. 2024 | Asymmetry in Low-Rank Adapters | 2402.16842 | B does most of the work. A random A is nearly as good. |
| Wang, Yu, Li 2024 | LoRA-GA | 2407.05000 | Initialises so step 1 matches the full fine-tuning gradient. Both factors move. |
| Zhang, Pilanci 2024 | Riemannian Preconditioned LoRA | 2402.02347 | An r×r preconditioner per factor stabilises SGD and AdamW. |
| Wang et al. 2024 | LoRA-Pro | 2407.18242 | LoRA is full fine-tuning with a low-rank "equivalent gradient". |
| Xu et al. 2025 | Learning Dynamics of LoRA: Gradient Flow | 2503.06982 | Exact gradient flow. Smaller initialisation gives lower final error. |
| Kim, Kim, Ryu 2025 | LoRA Training Provably Converges… | 2502.09376 | Zero init and weight decay bias toward a low-rank, small-norm minimum. |
| Castin et al. 2026 | Balanced LoRA | 2605.31484 | Balancing the factors improves conditioning and speeds convergence. |
| Kang et al. 2026 | Uniform Spectral Growth of Muon in LoRA | 2602.06385 | Per-factor Muon grows singular values at equal rates. Gradient flow learns the largest first. |
| Bogachev et al. 2025 | LoRA meets Riemannion | 2507.12142 | Per-factor Muon depends on parametrisation. Uses P_T(Ortho_r(M)) on ΔW. |
| Anson, Houghton, Milsom 2026 | Approximate Muon with low-rank adapters | 2608.14492 | Matmul-only Muon approximation. Moderate gains. |
| Zhang et al. 2023 | AdaLoRA | 2303.10512 | "Always prefers" giving rank to FFNs and top layers. |
| Pan et al. 2024 | LISA | 2403.17919 | Under LoRA, embedding and LM-head norms dwarf middle layers. |
| Awofeso, Orvieto, Geiping 2025 | LoRA Update Complexity Through Stable Rank | NeurIPS 2025 poster 133747 | Stable rank of ΔW stays narrow as r grows. |
| Choi et al. 2019 | On Empirical Comparisons of Optimizers | 1910.05446 | Section 1: Adam "can approximately simulate" momentum as ε grows. Section 4 tunes α₀/ε. |
| Su Jianlin 2024 | How does Adam's ε affect LR scaling? | kexue.fm/archives/10563 | Larger ε makes Adam behave like SGD. The step is about η/ε. |
| Thinking Machines 2025 | LoRA Without Regret | thinkingmachines.ai/blog/lora | UNVERIFIED: best LoRA LR is about 10× full fine-tuning. |

**Gaps.** I found no work that orthogonalises BA directly, apart from Riemannion's projection. Gradient flow without weight decay conserves AAᵀ − BᵀB (a standard identity). Adam and Muon steps break that conservation.

## Implications

**Step one.** The B gradient is G_B = G Aᵀ. Muon's step is −η U Vᵀ, which has r unit singular values and Frobenius norm η√r for every shape, matching the measurement. The first-order loss change per layer is −η‖G_B‖_* for Muon and −η‖G_B‖₁ for Adam's sign step. So one step already measures per-layer sensitivity. Adam with ε ≫ |g| returns η·G_B/ε, the raw gradient up to scale.

**(a) Sensitivity.** Perturb BA, not the factors, and normalise per layer, for example relative to ‖W0‖. Expect FFN and top layers to be most sensitive (AdaLoRA).

**(b) Spectra.** Track the singular values and stable rank of BA, plus ‖A‖, ‖B‖ and ‖AAᵀ − BᵀB‖.
- Muon should give flat spectra that grow uniformly (Kang).
- AdamW should give low stable rank (Awofeso).
- At balance, weight decay acts as a nuclear-norm penalty on BA, which pulls toward low rank (Kim).
- B should grow faster than A (LoRA+).
