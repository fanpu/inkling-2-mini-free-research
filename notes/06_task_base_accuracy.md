# 06. Pointer-chasing task and base accuracy

The untuned model solves one hop and collapses from two hops on. Greedy accuracy is 100% at $k=1$, 35% at $k=2$, 10% at $k=3$ and 0% at $k \ge 4$. Scripts are `scripts/task_pointer.py` and `scripts/05_base_accuracy.py`, with outputs in `artifacts/task_base_accuracy/`.

## Task format

Each prompt shows a random permutation $f$ of the 26 lowercase letters in shuffled order, and the answer is $f^k(\text{start})$. Paths never revisit a letter. Answers are balanced (each letter 2 or 3 times in 60). Prompts differ across $k$ only in the digit. Full example with $k=2$, answer `i` (path e, m, i):

```
Function A maps each letter to a letter:

n -> f
z -> l
g -> k
a -> t
s -> c
r -> z
k -> y
q -> b
p -> v
i -> x
d -> s
f -> j
m -> i
e -> m
v -> g
h -> e
t -> n
x -> w
o -> p
y -> a
j -> q
b -> h
l -> o
c -> r
u -> d
w -> u

Start letter: e
Number of steps: 2
Follow function A from the start letter for that number of steps (0 steps means stay on the start letter). Reply with the final letter only.
```

The control is $k=1$ with three independent tables A, B and C. The question names one, and the other two never map the start letter to the answer.

## Tokenisation

Checked on 1,400 generated prompts. Each table line is four tokens (`x`, ` ->`, ` y`, `\n`) with no merges. All 26 letters are single tokens with or without a leading space, so the answer is one token, ` x`. Rendered prompts are 173 tokens for every $k$ and 398 for the control.

## Forcing a direct answer

`Inkling2(effort=0.0)` alone does not stop reasoning. Without a prefill, all 6 probe samples opened with `<|message_model|><|content_thinking|>We need answer final letter only.` The fix appends `<|message_model|><|content_text|>Answer:` to the rendered prompt, which matches what `render_for_sft` produces for the reply "Answer: x" (asserted in code). With it, all 6 probes emitted a letter first and no thinking token. Later tokens varied (`<|end_message|>`, or visible working), so only the first token is scored, and training supervises only the answer token.

## Base accuracy (60 examples per row, temperature 0)

| Setting | Greedy acc | Mean log p(correct) | Greedy = $f(s)$ | Greedy = $f^2(s)$ |
|---|---|---|---|---|
| $k=0$ | 0.98 | -0.10 | | |
| $k=1$ | 1.00 | -0.04 | 60 | |
| $k=2$ | 0.35 | -2.09 | 39 | 21 |
| $k=3$ | 0.10 | -4.64 | 17 | 30 |
| $k=4$ | 0.00 | -5.21 | 19 | 25 |
| $k=6$ | 0.00 | -4.81 | 13 | 21 |
| control | 0.97 | -0.09 | 58 | |

Chance is 0.04 accuracy and $\ln(1/26) = -3.26$. For $k \ge 3$ the model is confidently wrong and usually stops after one or two hops. Log-probabilities are prompt logprobs on "...Answer: <correct>" from the base sampling client.

## Requests and throttling

433 requests (1 client creation, 12 probes, 420 scored), at most 4 in flight, 346 s wall time. None failed, no HTTP 429 appeared, and the SDK logged no retries. During development the server returned 400 for a 26-id logprob request, since it allows at most 20 target ids.

## Recommendation

Train on $k \in \{2,3,4\}$ with one 26-letter table. These sit below ceiling with graded signal (mean log-probability -2.1 to -5.2). Keep $k=0,1$ as ceiling checks and $k=6$ as a held-out test. Use seeds other than the eval seed 2026. A larger alphabet is unnecessary. The three-table control is at ceiling, so it does not yet separate difficulty from depth. A harder one-hop control, for example with more tables, needs testing first.
