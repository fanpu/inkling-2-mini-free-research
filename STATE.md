# Project State

**Last updated:** 2026-11-28
**Phase:** Phase 1. All three hypotheses have a first answer. Open work is the stability edge, Dimuon's step rule and the write-up.

## Verified facts

Details and sources are in the notes. Each line here has been checked by a script or a downloaded file in this repo.

- The free-tier model ID trains and samples at $0 with both AdamW and Dimuon, and adapters can be exported and downloaded (note 02).
- Inkling 2 Mini has 24 layers. Layers 6 to 17 run 3 times with shared weights (note 01).
- LoRA covers attention, MoE experts and the unembedding in every layer, one adapter per physical layer, so a looped layer's adapter is reused on all 3 passes. The loop controllers and cross-pass queries get no adapter (note 02).
- Tinker has no per-layer LoRA selection. Dimuon has no weight decay. AdamW has decoupled weight decay (note 02).
- At step one only B moves. AdamW takes a sign step. Dimuon takes a near-orthogonal step of the same Frobenius norm in every attention matrix. At the same learning rate the AdamW step is about 61 times larger (note 02).
- The three passes are not chained. Each restarts from layer 5's output and reads a gated message from the previous pass, and the gate was closed at the start of TML's training (note 01, read from TML's release post).
- In the gated scalar model the looped layer's relative sensitivity is between 1 and 3 times an ordinary layer's, depending on gate gain (note 03).
- Uploading external weights is disabled for this organization (HTTP 400), so edited adapters cannot be evaluated (note 05).
- `lora_alpha` is 32 at rank 2 as well as rank 8 (note 05).
- Training clients must run one at a time. An unclosed `ServiceClient` stalls the next training client for over 15 minutes with "concurrent training clients rate limit hit". Sampling ran 433 and 213 requests with no throttling (notes 04, 06, 07).
- `effort=0.0` does not stop reasoning. Prefilling `<|message_model|><|content_text|>Answer:` does (note 06).
- Untuned accuracy on pointer chasing is 0.98, 1.00, 0.35, 0.10, 0.00, 0.00 for k = 0, 1, 2, 3, 4, 6, on 60 examples each (note 06).
- Single-table reverse lookup is the hard one-hop control at 0.53 on 40 examples. Most of its errors are forward lookups (note 07).
- One AdamW step with `eps = lr = 1e3` reads raw B gradients through the exact inverse $|g| = \epsilon u/(1-u)$, $u = |\Delta B|/\text{lr}$. Per-tensor norms reproduce within about 6% (note 04).
- On small pointer-chasing batches the looped layers' B-gradient norms are about 2 times the unlooped ones after removing a linear depth trend (1.8 to 2.7 by module type). There is a sharp drop from layer 17 to 18 in every module type. The rise from layer 5 to 6 appears only in some. This suggests H1 but does not establish it (note 04).
- At initialisation the looped share of squared B-gradient norm is 0.93 at k = 1 and 0.95 for k = 2 to 6 and for the reverse control. It does not grow with hop count (note 08).
- Training on a mix of k = 2, 3, 4 at rank 8 reaches accuracy 1.00 on those in 100 steps with AdamW (lr 5.4e-4) and 50 steps with Dimuon (lr 1.6e-2). A step takes about 12 s (note 09).
- Both trained adapters clamp the hop count to between 2 and 4 on every prompt. k = 1 and reverse fall to near 0 and k = 6 stays at 0 (note 09).
- Both trained adapters put about half their squared norm in looped layers (0.45 to 0.51), the even-spread value, with no step at layers 6 or 18. One run each (note 09).
- AdamW with `eps = 1e4` acts as SGD with momentum at effective rate `lr/eps` (SGD mode). Its first step is linear in the gradient to within 0.6% (note 10).
- SGD mode at effective rate 2.9e-4 puts 0.95 of the adapter's squared norm in looped layers, with layer 6 over 5 at 2.17 and layer 18 over 17 at 0.12. It keeps k = 1 at 0.90 and reverse at 1.00 while learning k = 2, 3, 4. One run (note 10).
- SGD mode learns at effective rates up to 5e-4 and falls to a uniform guess over letters at 8.7e-4 and above. Switching the unembedding adapter off does not move this edge (note 11). Note 10's claim that 5e-4 collapses was wrong, that run was stopped on a plateau.
- AdamW at a low rate (2.5e-5, final B norm 2.82) still has a nearly flat profile (looped share 0.60). It keeps reverse at 0.87 but k = 1 falls to 0.33. So small movement explains part of the retained skills, not the profile (note 11).
- The SGD-mode result replicates with LoRA seed 1. Looped share 0.942 against 0.950, with k = 1 at 0.97 and reverse at 0.93 (note 12).
- H3 in its main form fails on two seeds. Adapters trained on k = 4 only and k = 2 only have the same looped share to within seed noise (differences +0.0008 and -0.0052, seed spread 0.005 to 0.011) (note 12).
- A weaker form holds on two seeds. A k = 4 adapter puts 0.075 to 0.088 more of its looped weight in layers 10 to 13 than a k = 2 adapter, against a seed spread of 0.011 to 0.025 (note 12).
- A single-condition adapter learns a fixed hop count. The k = 2 adapter takes two hops on every k = 3 and k = 4 prompt (note 12).
- In the pure-chain scalar model, looping multiplies the gradient by $K w_0^{K-1}$ and the sharpness by its square, and scale-free optimizers hide this in weight space (note 03).

## Unverified or unknown

- How Inkling 2 Mini was pretrained. The Muon claim from the first session has no source (note 01).
- Dimuon's update rule. We only have the step-one measurements.
- The exact cap on concurrent training clients. Storage is capped at 100 GB or 100 checkpoints per organization according to the docs review.
- Whether the gradient elevation in looped layers reflects output sensitivity. A B-gradient norm is a first-order loss sensitivity and sums over the 3 passes.
- Whether reverse lookup is truly one hop inside the model.

## Hypotheses

- **H1 (revised, see D-007 and D-012).** A weight change of fixed size in a looped layer changes the output more than in an unlooped layer, by a factor between 1 and 3 that reads off how open the cross-pass gates are. This is the revised P2 in note 03.
- **H2.** The largest stable learning rate, and the AdamW weight-decay level at which adapters collapse, are set by the looped layers.
- **H3.** As the hop count of the pointer-chasing task rises, a larger share of adapter weight lands in the looped layers, beyond what a harder-but-not-deeper control produces (D-011).

## Next actions

1. **What sets the SGD-mode edge (tests H2).** Attention-only and experts-only probes between effective rates 5e-4 and 8.7e-4. The unembedding is ruled out.
2. **Finish the k = 4 runs.** They were still learning at step 100. Rerun at 200 steps to check the middle-third shift at convergence.
3. **Dimuon step rule.** Still unknown beyond step one.
4. **Orientation page.** Add the single-condition results.

## Files

- `README.md` is the entry point and holds the working rules for agents.
- `DECISIONS.md` is the append-only decision log.
- `RESEARCH_SEED.md` is the background from earlier threads. Some of its claims were later corrected. Where it disagrees with the notes, the notes win.
- `notes/01_architecture_config.md`, `notes/02_tinker_probe.md`, `notes/03_looped_abx_theory.md`.
- `notes/04_grad_readout_probe.md`, `notes/05_upload_probe.md`, `notes/06_task_base_accuracy.md`, `notes/07_control_search.md`.
- `scripts/03` to `07` are the probes and their analysis. `scripts/task_pointer.py` is the task generator.
- `notes/08_e0_grad_by_hops.md`, `notes/09_e2_pilots.md`.
- `scripts/train_lib.py` is the training harness (AdamW and Dimuon, per-step log, fixed eval set, resume). `scripts/08` to `10` are E0 and the pilots.
- `notes/10_e2_pilot_sgd_mode.md`, `notes/11_controls.md`. `scripts/11` and `12` are the SGD-mode pilot and the controls.
- `notes/12_e2_single_condition.md` and `scripts/13_e2_single_condition.py` are the single-condition runs.
- `notes/lit/01` to `04` are sub-agent literature and documentation reviews (see D-014).
- `scripts/00_probe_free_tier.py`, `scripts/01_inspect_probe.py`, `scripts/02_looped_abx_toy.py`.
- `docs/orientation.html` is the source of the human-facing orientation page, published privately at https://claude.ai/artifact/RNJnoBSSA7htrQbjwZVWhm. Republish to that URL after editing. It follows the owner's explainer rules: one 700px column, numbered steps with one idea each, KaTeX math (MathML output), live interactives with a reset, then summary, misconceptions, blind spots and sources.
