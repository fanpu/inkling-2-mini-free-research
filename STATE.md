# Project State

**Last updated:** 2026-11-26
**Phase:** Phase 0 complete. Phase 1 not started. The next step is to pick the first experiment (see Next actions).

## Verified facts

Details and sources are in the notes. Each line here has been checked by a script or a downloaded file in this repo.

- The free-tier model ID trains and samples at $0 with both AdamW and Dimuon, and adapters can be exported and downloaded (note 02).
- Inkling 2 Mini has 24 layers. Layers 6 to 17 run 3 times with shared weights (note 01).
- LoRA covers attention, MoE experts and the unembedding in every layer, one adapter per physical layer, so a looped layer's adapter is reused on all 3 passes. The loop controllers and cross-pass queries get no adapter (note 02).
- Tinker has no per-layer LoRA selection. Dimuon has no weight decay. AdamW has decoupled weight decay (note 02).
- At step one only B moves. AdamW takes a sign step. Dimuon takes a near-orthogonal step of the same Frobenius norm in every attention matrix. At the same learning rate the AdamW step is about 61 times larger (note 02).
- In the scalar looped model, looping multiplies the gradient by $K w_0^{K-1}$ and the sharpness by its square, and scale-free optimizers hide this in weight space (note 03).

## Unverified or unknown

- How Inkling 2 Mini was pretrained. The Muon claim from the first session has no source (note 01).
- Dimuon's update rule. We only have the step-one measurements.
- Whether `external_weights_upload_*` works on the free tier. Experiment E1 depends on it.
- Free-tier rate limits. The probe never hit one.
- Whether `lora_alpha` stays 32 at other ranks.

## Hypotheses

- **H1 (revised, see D-007).** A weight change of fixed size in a looped layer changes the output more than in an unlooped layer, by a factor between $\sqrt{3}$ and 3 that measures how coherent the three passes are. This is P2 in note 03.
- **H2.** The largest stable learning rate, and the AdamW weight-decay level at which adapters collapse, are set by the looped layers.
- **H3.** Multi-step tasks put more adapter weight into the looped layers than one-step recall does.

## Next actions

1. **E1, perturbation sensitivity (tests H1).** Check that an edited adapter can be uploaded and evaluated on the free tier. If so, add a fixed-norm perturbation to one layer at a time and measure the change in loss, for all 24 layers.
2. **E2, training profiles.** Train AdamW and Dimuon on one small task for a few hundred steps with learning rates matched by step size, export every N steps, and plot per-layer adapter statistics over time.
3. **E3, AdamW weight-decay sweep (tests H2).** Only after E2 gives a working baseline.

## Files

- `README.md` is the entry point and holds the working rules for agents.
- `DECISIONS.md` is the append-only decision log.
- `RESEARCH_SEED.md` is the background from earlier threads. Some of its claims were later corrected. Where it disagrees with the notes, the notes win.
- `notes/01_architecture_config.md`, `notes/02_tinker_probe.md`, `notes/03_looped_abx_theory.md`.
- `scripts/00_probe_free_tier.py`, `scripts/01_inspect_probe.py`, `scripts/02_looped_abx_toy.py`.
- `docs/orientation.html` is the source of the human-facing orientation page (see README).
