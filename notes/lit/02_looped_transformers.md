# Looped transformers: literature (checked 2026-11-26)

**Answer.** Weight-tied depth is well studied for expressivity and reasoning, and 2026 papers (Loop-μP, DeepLoop) show shared weights add correlated updates, so per-step output change grows like N, not √N. I found nobody stating our scalar K·u^(K-1) gradient factor with sharpness scaling as its square, though it follows from the chain rule. TML has published Inkling 2 Mini's layout and gating, not its optimization; looped-LoRA work exists outside TML.

All arXiv abstracts were opened.

| Authors, year | Title | arXiv | Relevance |
|---|---|---|---|
| Dehghani+, 2018 | Universal Transformers | 1807.03819 | Recurrent-in-depth block, adaptive halting |
| Lan+, 2019 | ALBERT | 1909.11942 | Cross-layer sharing |
| Bai+, 2019 | Deep Equilibrium Models | 1909.01377 | Infinite tied depth as fixed point |
| Bai+, 2021 | Stabilizing DEQs by Jacobian Reg. | 2106.14342 | Tied depth brittle; Jacobian penalty |
| Giannou+, 2023 | Looped Transformers as Programmable Computers | 2301.13196 | Looped net emulates a computer |
| Yang+, 2023 | Looped Transformers ... Learning Algorithms | 2311.12424 | Better in-context learning |
| Fan+, 2024 | Looped Transformers for Length Generalization | 2409.15647 | Adaptive loop count |
| Bae+, 2024 | Relaxed Recursive Transformers | 2410.20672 | Per-layer LoRA on tied layers |
| Saunshi+, 2025 | Reasoning with Latent Thoughts | 2502.17416 | k layers × L loops ≈ kL layers |
| Geiping+, 2025 | Recurrent Depth (Huginn) | 2502.05171 | 3.5B recurrent-depth LM |
| Wang+, 2025 | Hierarchical Reasoning Model | 2506.21734 | Two-timescale recurrence |
| Jolicoeur-Martineau, 2025 | Tiny Recursive Model | 2510.04871 | Simpler, beats HRM |
| Zhu+, 2025 | Ouro (LoopLM) | 2510.25741 | Looped LM, 7.7T tokens |
| Fu+, 2026 | Fully Looped Transformer | 2605.18797 | More loops cause gradient oscillation, collapse |
| Wang+, 2026 | Loop-μP | 2606.18524 | 1/N residual scale makes LR N-independent |
| Li+, 2026 | DeepLoop | 2607.13491 | Shared update sums gradients over visits |
| Zhang+, 2026 | LR Scaling in Deep Scalar Linear Nets | 2607.07884 | Untied ∏w_l; optimal LR data-dependent |
| Ekbote+, 2026 | How to Loop Your Transformer | 2610.40325 | Loop gains saturate or destabilize; per-loop LoRA |
| Zhu+, 2026 | Loop Dropout | 2609.34218 | LoRA adapts mostly late loops |
| Marchenko+, 2026 | Closing the Loop | 2610.00673 | Warmup and gate regularization |
| Pascanu+, 2012 | Difficulty of Training RNNs | 1211.5063 | Gradients through repeated W |

## What is already known about our toy result

- **Powers of a shared weight.** Pascanu+ cover this for RNNs; K·u^(K-1) is the scalar d(W^K)/dW.
- **Loop-μP.** A shared layer adds correlated increments: stream grows like N, step output change like η·ε·N. Its N² is squared output change, not a Hessian.
- **DeepLoop.** The shared update sums the gradients from every visit, then every visit reads it back.
- **Linear-net theory.** 2607.07884 (and Kamber & Parhi, 2509.25783) are untied. Tied w^K with sharpness ∝ (K·u^(K-1))² is, as far as I found, unpublished but unsurprising.

## What TML has said publicly

Sources: the [news post](https://thinkingmachines.ai/news/inkling-2-mini/), the [HF blog](https://huggingface.co/blog/inkling2mini) and both model cards.

- **Layout.** Layers 6–17 run 3×, 48 visits per token, every pass starting from layer 5's output.
- **Messages.** Passes 2–3 read the previous pass via learned queries. Four rank-64 gates mix the message into q, k, v and MoE inputs (HF blog). The gate starts closed (news post). Model cards omit controllers.
- **Why loop.** More serial depth, same parameters. An unpruned looped model was >50% more compute-efficient.
- **Training.** Midtraining pruned 2/3 of experts, then 1T distillation tokens, SFT and 2.2T RL tokens. No optimizer, LR or stability details.
- **varloop.** Trained with 1–5 passes (1:2:6:2:1 in late midtraining) and post-trained at 3; may be unpredictable at other counts.
- Tinker fine-tuning supported, no loop-specific guidance; no talks found.

## Gaps

- No Hessian or sharpness analysis of tied depth; no K² edge-of-stability prediction.
- No per-layer or per-pass sensitivity study of a looped MoE; only per-loop LoRA analysis (Loop Dropout).
- Nothing on how a closed-at-init gate changes effective K during training.
- UNVERIFIED: a search snippet says Closing the Loop trains per-step LoRA banks; its abstract does not.
