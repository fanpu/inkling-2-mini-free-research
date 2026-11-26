# Project State

**Last updated:** 2026-11-26
**Phase:** Phase 0 and the literature review are complete. Phase 1 starts with the two feasibility probes under Next actions.

## Verified facts

Details and sources are in the notes. Each line here has been checked by a script or a downloaded file in this repo.

- The free-tier model ID trains and samples at $0 with both AdamW and Dimuon, and adapters can be exported and downloaded (note 02).
- Inkling 2 Mini has 24 layers. Layers 6 to 17 run 3 times with shared weights (note 01).
- LoRA covers attention, MoE experts and the unembedding in every layer, one adapter per physical layer, so a looped layer's adapter is reused on all 3 passes. The loop controllers and cross-pass queries get no adapter (note 02).
- Tinker has no per-layer LoRA selection. Dimuon has no weight decay. AdamW has decoupled weight decay (note 02).
- At step one only B moves. AdamW takes a sign step. Dimuon takes a near-orthogonal step of the same Frobenius norm in every attention matrix. At the same learning rate the AdamW step is about 61 times larger (note 02).
- The three passes are not chained. Each restarts from layer 5's output and reads a gated message from the previous pass, and the gate was closed at the start of TML's training (note 01, read from TML's release post).
- In the gated scalar model the looped layer's relative sensitivity is between 1 and 3 times an ordinary layer's, depending on gate gain (note 03).
- In the pure-chain scalar model, looping multiplies the gradient by $K w_0^{K-1}$ and the sharpness by its square, and scale-free optimizers hide this in weight space (note 03).

## Unverified or unknown

- How Inkling 2 Mini was pretrained. The Muon claim from the first session has no source (note 01).
- Dimuon's update rule. We only have the step-one measurements.
- Whether `external_weights_upload_*` works on the free tier. Experiment E1 depends on it.
- Free-tier rate limits. The probe never hit one and the docs give no numbers. Storage is capped at 100 GB or 100 checkpoints per organization according to the docs review.
- Whether `lora_alpha` stays 32 at other ranks.

## Hypotheses

- **H1 (revised, see D-007 and D-012).** A weight change of fixed size in a looped layer changes the output more than in an unlooped layer, by a factor between 1 and 3 that reads off how open the cross-pass gates are. This is the revised P2 in note 03.
- **H2.** The largest stable learning rate, and the AdamW weight-decay level at which adapters collapse, are set by the looped layers.
- **H3.** As the hop count of the pointer-chasing task rises, a larger share of adapter weight lands in the looped layers, beyond what a harder-but-not-deeper control produces (D-011).

## Next actions

1. **Feasibility probes (cheap, do first).**
   - Large-`eps` AdamW step to read raw per-layer gradients. Precedent is in `notes/lit/03`.
   - Upload an edited adapter and load it with `load_state` on the free tier.
   - Base accuracy of the untuned model on pointer chasing for each hop count, with a prefilled answer start and `effort=0.0`, checking that no reasoning tokens appear.
2. **E0, gradient readout (tests H1).** Per-layer gradient norms over several batches, looking for a jump at layers 6 and 18.
3. **E1, perturbation sensitivity (tests H1).** Only if the upload probe works.
4. **Dimuon probes.** Sweep learning rate, rank and seed for one to three steps to find its step rule and a matched step size.
5. **E2, training profiles and hop sweep (tests H3).** AdamW and Dimuon on the task in D-011.
6. **E3, learning-rate edge and AdamW weight-decay sweep (tests H2).** Only if E0 to E2 show a loop effect.

## Files

- `README.md` is the entry point and holds the working rules for agents.
- `DECISIONS.md` is the append-only decision log.
- `RESEARCH_SEED.md` is the background from earlier threads. Some of its claims were later corrected. Where it disagrees with the notes, the notes win.
- `notes/01_architecture_config.md`, `notes/02_tinker_probe.md`, `notes/03_looped_abx_theory.md`.
- `notes/lit/01` to `04` are sub-agent literature and documentation reviews (see D-014).
- `scripts/00_probe_free_tier.py`, `scripts/01_inspect_probe.py`, `scripts/02_looped_abx_toy.py`.
- `docs/orientation.html` is the source of the human-facing orientation page, published privately at https://claude.ai/artifact/RNJnoBSSA7htrQbjwZVWhm. Republish to that URL after editing. It follows the owner's explainer rules: one 700px column, numbered steps with one idea each, KaTeX math (MathML output), live interactives with a reset, then summary, misconceptions, blind spots and sources.
