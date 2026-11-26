# Decision Log

Append-only chronological record of key architectural, experimental, and tooling decisions.

---

## 2026-11-26

- **D-001 — Free-Tier Model Enforcement:**
  All Tinker API calls must hardcode and assert `MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"` (64K context limit, \$0.00 training & sampling). Never invoke paid IDs (`thinkingmachines/Inkling-2-Mini` or `thinkingmachines/Inkling-2-Mini:peft:262144`) unless explicitly directed.

- **D-002 — Primary Research Framing:**
  Focus on **Learning Mechanics of a Looped Transformer** (`arXiv:2604.21691` + minimal $(w_0 + ab)^K$ / residual $(1 + w_0 + ab)^K$ models tested against Inkling 2 Mini LoRA adapters downloaded via Tinker).

- **D-003 — Muon Weight-Decay Scope Split:**
  Deprioritize custom internal Muon weight-decay wiring (coupled vs. decoupled vs. cautious) on Tinker pending live SDK verification of `DimuonParams`. Keep internal WD-wiring in local PyTorch unless `DimuonParams` exposes `weight_decay` or optimizer internals.

- **D-004 — Python Runtime Selection:**
  Default system `python3` is `3.15.2`, which risks missing prebuilt ML wheels. Create project `.venv` using `/opt/homebrew/bin/python3.12`.

- **D-005 — Persistent State Architecture:**
  Maintain project continuity in-repo via [`STATE.md`](STATE.md) (live dashboard), [`DECISIONS.md`](DECISIONS.md) (append-only decision log), [`RESEARCH_SEED.md`](RESEARCH_SEED.md), and numbered [`notes/`](notes/). Binary checkpoints (`*.safetensors`, `*.tar`, etc.) and `.venv/` are gitignored.
