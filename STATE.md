# Project State

**Last Updated:** 2026-11-26  
**Current Phase:** Phase 0 — Empirical & Theoretical Grounding (In Progress)

---

## Verified Facts
- Workspace initialized at `/Users/fanpu/ml/inkling-2-mini-free-research` (`.gitignore`, `README.md`, `RESEARCH_SEED.md`, `DECISIONS.md`, `STATE.md`).
- Shell environment has `TINKER_API_KEY` set; `/opt/homebrew/bin/python3.12` is available; system `python3` (`3.15.2`) has `numpy 2.5.4`.

## Open Unknowns (Being Checked in Phase 0)
- [ ] **Architecture schema:** Exact `config.json` and `model.safetensors.index.json` layout of `thinkingmachines/Inkling-2-Mini` (layer counts, loop indices, loop count $K$, cross-loop gate structure, MoE configuration, norms).
- [ ] **Tinker SDK surface:** Exact `DimuonParams`, `AdamParams`, `create_lora_training_client` arguments, and checkpoint download flow in the installed `tinker` SDK.
- [ ] **Free-tier dry run & LoRA coverage:** Confirm `thinkingmachines/Inkling-2-Mini:peft:65536:free` trains, supports `DimuonParams`, and allows downloading `adapter_model.safetensors`; verify which layers/modules (MoE experts, attention, cross-loop gates) receive adapters and whether looped layers share one adapter across passes.
- [ ] **Minimal $(w_0 + ab)^K$ theory:** Explicit quantitative predictions for GD, AdamW, and Dimuon across $K=1$ vs. $K=3$ plus numerical verification.

## Active Hypotheses
- **H1 (Optimizer Mechanics Across Loop Boundary):** Dimuon vs. AdamW produce distinct layer profiles ($\|BA\|_F$, $\|BA\|_2$, stable rank, $\|B\|_F / \|A\|_F$) across unlooped prelude (0–5), looped core (6–17), and unlooped coda (18–23).
- **H2 (Stability / EoS & Weight-Decay Phase Boundary):** Looped core shifts EoS / LR instability and AdamW `weight_decay` collapse thresholds relative to unlooped layers.
- **H3 (Task-Dependent Adapter Localization):** Iterative multi-hop tasks localize adapter energy/rank into the looped core differently than 1-step associative recall.

## File & Artifact Index
- [`README.md`](README.md) — Entrypoint and quick resume guide.
- [`DECISIONS.md`](DECISIONS.md) — Chronological decision log (`D-001`–`D-005`).
- [`RESEARCH_SEED.md`](RESEARCH_SEED.md) — Prior research threads and `arXiv:2604.21691` synthesis.
- `notes/01_architecture_config.md` — *(pending subagent 1)*
- `notes/02_tinker_probe.md` — *(pending subagent 2)*
- `notes/03_looped_abx_theory.md` — *(pending subagent 3)*

## Next Actions
1. Complete parallel Phase 0 checks (`HF Config Architecture Inspector`, `Tinker SDK LoRA Probe`, `Looped abx Theory Analyst`).
2. Reconcile empirical config/LoRA facts with the minimal $(w_0 + ab)^K$ model.
3. Update `STATE.md` and `DECISIONS.md`, then choose the first Phase 1 experiment.
