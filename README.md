# inkling-2-mini-free-research

**Learning Mechanics of a Looped Transformer on Inkling 2 Mini Free Tier**

This repository investigates the training dynamics, stability boundaries, and optimizer mechanics (Dimuon vs. AdamW) of a looped transformer architecture, bridging minimal solvable scalar/matrix models ($y = (w_0 + ab)^K x$) with 26B-scale LoRA fine-tuning on Thinking Machines Lab's **Inkling 2 Mini** free tier on Tinker.

## Quick Resume Instructions

If resuming in a new session, read files in this order:
1. [`STATE.md`](STATE.md) — Current phase, verified facts vs. open unknowns, active hypotheses, and next actions.
2. [`DECISIONS.md`](DECISIONS.md) — Append-only log of settled architectural, methodological, and engineering decisions.
3. [`RESEARCH_SEED.md`](RESEARCH_SEED.md) — Full research context, prior threads, and theoretical framing.
4. `notes/` — Numbered technical findings and derivations.

## Hard Constraint: Free-Tier Model ID Only

Every Tinker API call in this repository **MUST** use the exact free-tier model ID:

```text
thinkingmachines/Inkling-2-Mini:peft:65536:free
```

- Context limit: 64K (`65536`).
- Subject to free-tier rate limits / throttling — use polite concurrency, exponential backoff/retries, and checkpointing.
- Never call paid model IDs (`thinkingmachines/Inkling-2-Mini` or `thinkingmachines/Inkling-2-Mini:peft:262144`) unless explicitly instructed.
- Never commit or log `TINKER_API_KEY`.

## Working rules for agents

This project is run mostly by AI agents with a human reading the results. Any agent picking it up should be able to continue from this file and `STATE.md` alone.

1. **Start** by reading `STATE.md`, then `DECISIONS.md`, then the notes it points to.
2. **Separate verified from claimed.** A fact goes under "Verified facts" in `STATE.md` only if a script or downloaded file in this repo supports it. Everything else is listed as unknown. `RESEARCH_SEED.md` came from an earlier chat and contains claims that turned out to have no source.
3. **Every result has a script.** Put it in `scripts/` with a numbered name, write its outputs under `artifacts/`, and record the finding in a numbered note in `notes/`.
4. **Finish** each work session by updating `STATE.md`, appending any decision to `DECISIONS.md`, updating `docs/orientation.html` and republishing the orientation page, then committing and pushing (rule D-009).
5. **Environment.** `uv venv --python 3.12 .venv` then `uv pip install tinker tml-renderers torch numpy safetensors huggingface_hub matplotlib`. `TINKER_API_KEY` must be set in the shell.
6. **Budget.** Free-tier model ID only (D-001). The machine has 16 GB of memory and a rank-8 adapter is 371 MB, so load at most a few adapters at once.

## Directory Layout

- `STATE.md` — current phase, verified facts, unknowns, next actions.
- `DECISIONS.md` — append-only dated decision log.
- `RESEARCH_SEED.md` — background from earlier threads (partly superseded).
- `notes/` — numbered findings and derivations.
- `scripts/` — numbered, reproducible probes and analyses.
- `artifacts/` — downloaded configs and probe outputs (weights are gitignored).
- `docs/orientation.html` — source of the human-facing orientation page.
