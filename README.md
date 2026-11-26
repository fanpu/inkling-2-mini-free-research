# inkling-2-mini-free-research

**Learning Mechanics of a Looped Transformer on Inkling 2 Mini Free Tier**

This repository investigates the training dynamics, stability boundaries, and optimizer mechanics (Dimuon vs. AdamW) of a looped transformer architecture, bridging minimal solvable scalar/matrix models ($y = (w_0 + ab)^K x$) with 26B-scale LoRA fine-tuning on Thinking Machines Lab's **Inkling 2 Mini** free tier on Tinker.

## Quick Resume Instructions

If resuming in a new session, read files in this order:
1. [`STATE.md`](STATE.md) — Current phase, verified facts vs. open unknowns, active hypotheses, and next actions.
2. [`DECISIONS.md`](DECISIONS.md) — Append-only log of settled architectural, methodological, and engineering decisions.
3. [`RESEARCH_SEED.md`](RESEARCH_SEED.md) — Full research context, prior threads, and theoretical framing.
4. `notes/` — Numbered technical findings and derivations (`01_architecture_config.md`, `02_tinker_probe.md`, `03_looped_abx_theory.md`, etc.).

## Hard Constraint: Free-Tier Model ID Only

Every Tinker API call in this repository **MUST** use the exact free-tier model ID:

```text
thinkingmachines/Inkling-2-Mini:peft:65536:free
```

- Context limit: 64K (`65536`).
- Subject to free-tier rate limits / throttling — use polite concurrency, exponential backoff/retries, and checkpointing.
- Never call paid model IDs (`thinkingmachines/Inkling-2-Mini` or `thinkingmachines/Inkling-2-Mini:peft:262144`) unless explicitly instructed.
- Never commit or log `TINKER_API_KEY`.

## Directory Layout

- `STATE.md` — Live resume dashboard (phase, verified facts, open unknowns, next steps).
- `DECISIONS.md` — Append-only dated decision log.
- `RESEARCH_SEED.md` — Condensed seed from prior threads and `arXiv:2604.21691`.
- `notes/` — Empirical probes, architecture breakdowns, and theory derivations.
- `scripts/` — Reproducible probes, toy verification scripts, and Tinker training/analysis scripts.
- `artifacts/` — Downloaded configs, probe outputs, and inventories (binary checkpoints gitignored).
