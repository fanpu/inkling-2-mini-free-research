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

- **D-006 — Orientation page is a live artifact:**
  The human-facing summary (primer, decisions, findings) is a published Claude artifact built from `docs/orientation.html`. The user first asked for a PDF and then chose an artifact because it is easier to read and navigate.

- **D-007 — H1 revised after the probe:**
  AdamW and Dimuon both discard gradient size, so per-layer update norms cannot show loop amplification. H1 now targets output sensitivity per unit weight change (P2 in note 03). The original norm-profile version is dropped.

- **D-008 — Pretraining-optimizer claim marked unverified:**
  The model card does not say how Inkling 2 Mini was pretrained. No argument in this project may rely on "Muon-pretrained" until a source is found.

- **D-009 — Git workflow:**
  Small commits authored solely by the repo owner, pushed after each commit. No AI attribution. No force-push. The repo is public, so no agent config files and no secrets.

- **D-010 — Throwaway exports use the one-hour TTL:**
  Storage is free today, but probe and intermediate exports still set `ttl_seconds=3600`. Downloaded `.safetensors` files stay out of git.

- **D-011 — Training task is in-context pointer chasing, as a hop sweep:**
  Random permutation over 26 single-token letters, shuffled entry order, no revisited nodes, answer scored on the first token only. Hop counts k in {0, 1, 2, 3, 4, 6} instead of a single 1-versus-3 contrast, plus a control that is harder without being deeper (one hop, larger table). Base accuracy per k is measured first. Reasons are in `notes/lit/01_multihop_tasks.md`: k hops need only about log2(k) layers, pretrained models already do two hops, and harder tasks draw more adapter weight everywhere. Thinking Machines' own N-Hop Lookup result supports the task family.

- **D-012 — Toy model amended to the gated-message loop:**
  The real passes restart from the prelude output and are linked by a gated message, so the pure chain $(w_0 + ab)^K$ is an upper reference only. Predictions about sensitivity now use the gated model in note 03, with the looped-to-unlooped ratio between 1 and 3.

- **D-013 — Optimizer learning rates:**
  Tinker's docs recommend 10 to 100 times the AdamW learning rate for Dimuon, and a cookbook starting rate of 5.4e-4 for Inkling 2 Mini. Use these as starting points and match the two optimizers by measured step size, not by learning rate.

- **D-014 — Literature notes are reviewer reports:**
  The files in `notes/lit/` were written by sub-agents that opened the sources. Treat a claim from them as verified only after someone reads the cited source directly. Papers dated after mid-2026 have not been read by the lead agent.
