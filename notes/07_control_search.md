# 07. Search for a hard one-hop control

Reverse lookup in one table is the best control found. The prompt gives an output letter $y$ and asks for the $x$ with $f(x) = y$. That is one lookup, and the untuned model scores 0.53 greedy accuracy (SE 0.08) with mean $\log p(\text{correct}) = -1.50$ on 40 examples, near the $k=2$ main task (0.35, $-2.09$). Harder forward lookups stayed at 0.90 or above.

Code is `generate_control` in `scripts/task_pointer.py` and `scripts/06_control_search.py`, with outputs in `artifacts/control_search/`. Scoring matches note 06. The existing generator's seed-2026 prompts are unchanged.

## Results

Prompt tokens include template and prefill ($k=1$ main task is 173).

| Candidate | Prompt tokens | n | Acc | Mean log p |
|---|---|---|---|---|
| 8 tables | 948 | 10 | 1.00 | -0.22 |
| 16 tables | 1828 | 10 | 0.90 | -0.14 |
| 200 two-letter keys | 850-858 | 10 | 1.00 | -0.01 |
| 300 three-letter keys | 1440-1461 | 10 | 1.00 | -0.00 |
| 200 near-duplicate keys | 968-1004 | 10 | 1.00 | -0.00 |
| 1500 words of filler | 1883-1912 | 10 | 1.00 | -0.05 |
| reverse, 1 table | 154 | 40 | 0.53 | -1.50 |
| reverse, 2 tables | 269 | 40 | 0.10 | -3.68 |
| reverse, 3 tables | 379 | 40 | 0.07 | -4.44 |
| reverse, 8 tables | 929 | 10 | 0.00 | -4.18 |

Near-duplicate keys come in groups of five that differ in one position (`vtr`, `vtm`, `vtc`).

## Reverse errors

Most wrong answers are the forward image $f(y)$. With one table, 11 of 19 errors are $f(y)$, 6 repeat $y$ and 2 are other letters. With two or three tables, about 30 of 40 answers are $f(y)$. No error was the inverse from a distractor table. The difficulty is a prior toward forward lookup, which tuning could remove without adding depth.

## Recommendation

Use single-table reverse lookup as the hard one-hop control. Its log-probability is in the target band and its accuracy sits at the upper edge. Report its share of $f(y)$ answers next to the main task, since those are direction errors rather than failed composition. Two or more tables land at the $k=3$ level, harder than wanted.

The search made 213 requests. None failed or hit HTTP 429.
