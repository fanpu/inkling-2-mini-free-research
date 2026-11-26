# Multi-hop / pointer-chasing tasks for testing depth use

Sources checked 2026-11-26 on arXiv abstract/HTML pages unless marked UNVERIFIED.

**Answer.** k-hop pointer chasing is a standard depth probe: it is solvable in ⌊log₂k⌋+2 layers by pointer doubling (Sanford et al.), and looped models match deeper untied ones on it (Saunshi et al.). A lookup-table version has been used with a looped transformer trained from scratch, where probes show about one hop per loop iteration, but we found no work fine-tuning a *pretrained* looped model on it. Your k=1 vs k=3 pair is reasonable if you add a k sweep, shortcut-proof sampling and a difficulty-matched control.

## Papers

| Authors, year | Title | ID | Relevance |
|---|---|---|---|
| Sanford, Hsu, Telgarsky 2024 | Transformers, parallel computation, and logarithmic depth | 2402.09268 | Defines hop_k (iterated induction head). Proves log-depth bounds. In from-scratch training, each added layer roughly doubles the largest solvable k. |
| Saunshi et al. 2025 | Reasoning with Latent Thoughts: On the Power of Looped Transformers | 2502.17416 | p-hop (p=16, 32): 2 layers looped 3 times reach 99.9% vs 68.8% for 2 untied layers. Trained from scratch (implied, not stated). |
| Sesodia, Torr, Schroeder de Witt 2026 | Inside the Loop: weight-tied transformers on depth-bound tasks | NeurIPS 2026 poster 150577 (no arXiv) | Permutation traversal. In the looped model, iteration k decodes hop k; in the untied model, hops appear only in the last layers. Closest to your setup. Training details UNVERIFIED. |
| Liu et al. 2022 | Transformers Learn Shortcuts to Automata | 2210.10749 | State tracking: log-depth shortcuts always exist and are learnable. |
| Merrill, Petty, Sabharwal 2024 | The Illusion of State in State-Space Models | 2404.08819 | Permutation composition as a hard state-tracking problem. |
| Zhang et al. 2021 | Pointer Value Retrieval | 2107.12580 | Single-indirection benchmark for generalisation, not depth. |
| Wu, Geiger, Millière 2025 | How Do Transformers Learn Variable Binding in Symbolic Programs? | 2505.20896 | Assignment chains up to 4 deep. Models learn a shallow "early assignment" heuristic before real dereferencing. |
| Balesni, Korbak, Evans 2024/25 | Lessons from Studying Two-Hop Latent Reasoning (v1: "Two-Hop Curse") | 2411.16353 | Fine-tuned Llama 3 8B / GPT-4o. v4 concludes latent two-hop works and warns of spurious failures. |
| Yang et al. 2024 | Do LLMs Latently Perform Multi-Hop Reasoning? | 2402.16837 | Pretrained LLMs: first hop is strong; second hop is moderate. |
| Biran et al. 2024 | Hopping Too Late | 2406.12775 | Bridge entity is resolved early and the second hop late. Localises hops in depth. |
| Wang et al. 2024 | Grokked Transformers are Implicit Reasoners | 2405.15071 | Parametric composition is learned only via grokking. From-scratch setup UNVERIFIED. |

## Pitfalls

- **Parallel log-depth solutions.** By Sanford's bound, k=3 needs one more layer than k=1 (3 vs 2). A 24-layer model may solve both without using the loop.
- **Base model already solves small k.** Latent two-hop exists (Yang; Balesni v4), so the LoRA may learn only the format.
- **Difficulty confound.** Higher loss on k=3 means larger updates in every layer. Compare layer-normalised adapter mass, and add a harder-but-not-deeper control (k=1 with a bigger table or more distractors).
- **Positional shortcut.** A table listed in chain order puts the answer k entries ahead, so shuffle entry order (cf. Wu et al.).
- **Cycle shortcut.** Fixed points and 2-cycles make f³(x) = f(x), so reject paths that revisit a node.
- **Tokenisation.** "a→f" may merge into one token. Check with the real tokenizer.
- **Loops ≠ hops.** The staircase result is from training from scratch. In a pretrained 12-layer block, one hop need not map to one loop.

## Recommendation

Keep in-context pointer chasing: it is most directly tied to looped depth, and intermediate nodes are known for probing. Settings are our choices, not from the papers:

- **Table:** a random permutation over 26 single-token letters (≈100 prompt tokens), with entry order shuffled.
- **Sampling:** no revisits along the path, and a uniform answer distribution across k.
- **Prompt:** identical except for the digit k. Score only the first token after a fixed delimiter, so there is no hidden working.
- **k sweep:** k ∈ {0, 1, 2, 3, 4, 6}, so a dose-response curve can separate hops from difficulty.
- **Base accuracy:** measure it per k first, and train only on k values below ceiling.
- **Probes:** add per-loop linear probes for hop-j nodes.

Fallback: permutation composition (same shortcut issue, less clean intermediates).
