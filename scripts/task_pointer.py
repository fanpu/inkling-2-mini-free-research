"""In-context pointer chasing task (D-011).

Each example shows a random permutation of the 26 lowercase letters as a lookup
table in shuffled order, a start letter and a step count k. The answer is
f^k(start), a single letter given with no working. The path start, f(start), ...,
f^k(start) never revisits a letter, and answers are balanced across the alphabet.

Control variant: k = 1 with three independent tables A, B and C. The question names
one of them, so the prompt is three times longer but still one hop.

The model's reply is prefilled with "Answer:" so the next token is the answer
letter (" x", one token). Only that token is supervised in training data.

Usage as a script prints a tokenisation report:
    .venv/bin/python scripts/task_pointer.py
"""
from __future__ import annotations

import random
import string
import warnings
from dataclasses import dataclass, field, asdict

warnings.simplefilter("ignore", FutureWarning)

from tml_renderers import chat, renderers  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
EFFORT = 0.0
LETTERS = string.ascii_lowercase
KS = (0, 1, 2, 3, 4, 6)
TABLE_NAMES = ("A", "B", "C")
PREFILL = "Answer:"

_RENDERER = renderers.Inkling2(effort=EFFORT)
TOK = _RENDERER.tokenizer


@dataclass
class Example:
    prompt: str
    answer: str
    k: int
    path: list[str]            # start, f(start), ..., f^k(start)
    variant: str               # "main" or "control"
    table: str                 # name of the table the question uses
    tables: dict = field(default_factory=dict)  # name -> {letter: image}

    def to_json(self) -> dict:
        return asdict(self)


def _table_lines(perm: dict, rng: random.Random) -> str:
    keys = list(LETTERS)
    rng.shuffle(keys)
    return "\n".join(f"{x} -> {perm[x]}" for x in keys)


def _format_prompt(tables: dict, rng: random.Random, use: str, start: str, k: int) -> str:
    n = len(tables)
    if n == 1:
        head = "Function A maps each letter to a letter:"
    else:
        names = ", ".join(TABLE_NAMES[: n - 1]) + f" and {TABLE_NAMES[n - 1]}"
        head = f"Functions {names} each map every letter to a letter."
    blocks = [head]
    for name, perm in tables.items():
        body = _table_lines(perm, rng)
        blocks.append(body if n == 1 else f"Function {name}:\n{body}")
    blocks.append(
        f"Start letter: {start}\nNumber of steps: {k}\n"
        f"Follow function {use} from the start letter for that number of steps "
        "(0 steps means stay on the start letter). Reply with the final letter only."
    )
    return "\n\n".join(blocks)


def _random_perm(rng: random.Random) -> dict:
    img = list(LETTERS)
    rng.shuffle(img)
    return dict(zip(LETTERS, img))


def _path(perm: dict, start: str, k: int) -> list[str]:
    p = [start]
    for _ in range(k):
        p.append(perm[p[-1]])
    return p


def generate(n: int, k: int, seed: int, variant: str = "main") -> list[Example]:
    """n examples with answers balanced over the alphabet (each letter n//26 or n//26+1 times).

    For each target answer, draw a permutation, set start = f^-k(answer), and redraw
    until the path has no repeated letter. For the control, the distractor tables
    must not send the start letter to the answer.
    """
    assert variant in ("main", "control")
    if variant == "control":
        assert k == 1, "control is one hop"
    rng = random.Random(f"{variant}-{k}-{seed}")
    targets = []
    while len(targets) < n:
        block = list(LETTERS)
        rng.shuffle(block)
        targets += block
    out = []
    for ans in targets[:n]:
        n_tables = 3 if variant == "control" else 1
        while True:
            tables = {TABLE_NAMES[i]: _random_perm(rng) for i in range(n_tables)}
            use = rng.choice(TABLE_NAMES[:n_tables])
            f = tables[use]
            inv = {v: u for u, v in f.items()}
            start = ans
            for _ in range(k):
                start = inv[start]
            path = _path(f, start, k)
            if len(set(path)) != k + 1:
                continue
            if any(t[start] == ans for name, t in tables.items() if name != use):
                continue
            break
        prompt = _format_prompt(tables, rng, use, start, k)
        out.append(Example(prompt, ans, k, path, variant, use, tables))
    return out


# ---------------------------------------------------------------- rendering

def answer_token(letter: str) -> int:
    ids = TOK.encode_ordinary(" " + letter)
    assert len(ids) == 1, (letter, ids)
    return ids[0]


LETTER_TOKENS = {L: answer_token(L) for L in LETTERS}
TOKEN_TO_LETTER = {v: k for k, v in LETTER_TOKENS.items()}


def _messages(prompt: str, reply: str | None = None):
    m = [{"role": "user", "content": prompt}]
    if reply is not None:
        m.append({"role": "assistant", "content": reply})
    return chat.OpenAIMessage.from_openai_chat_dicts(m)


def prompt_tokens(ex: Example, prefill: bool = True) -> list[int]:
    """Tokens for sampling. With prefill, the reply is opened and filled with
    'Answer:' so the next token should be the answer letter."""
    spans, _ = _RENDERER.render_for_completion(_messages(ex.prompt))
    ids = [t for s in spans for t in s.span.tokens]
    if prefill:
        ids += [TOK.encode_special("message_model"), TOK.encode_special("content_text")]
        ids += TOK.encode_ordinary(PREFILL)
    return ids


def sampling_input(ex: Example, prefill: bool = True):
    from tinker import types
    return types.ModelInput.from_ints(prompt_tokens(ex, prefill))


def sft_tokens(ex: Example) -> tuple[list[int], list[float]]:
    """Unshifted tokens and weights for prompt + 'Answer: x'. Weight 1 on the answer
    token only. Checked against render_for_sft so the layout matches the renderer."""
    pre = prompt_tokens(ex, prefill=True)
    ans = LETTER_TOKENS[ex.answer]
    ref = _RENDERER.render_for_sft(_messages(ex.prompt, f"{PREFILL} {ex.answer}"))[-1]
    ref_ids = [t for s in ref.input_token_spans for t in s.span.tokens]
    assert ref_ids[: len(pre) + 1] == pre + [ans], "prefill layout differs from renderer"
    ids = pre + [ans]
    w = [0.0] * len(pre) + [1.0]
    return ids, w


def to_datum(ex: Example):
    """tinker Datum for cross_entropy with only the answer token supervised."""
    from tinker import types
    ids, w = sft_tokens(ex)
    return types.Datum(
        model_input=types.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"target_tokens": ids[1:], "weights": w[1:]},
    )


# ---------------------------------------------------------------- checks

def check_tokenisation(ex: Example) -> dict:
    """Every letter in the prompt must be its own token: table keys as 'x', images,
    and the start letter as ' x'. Returns token counts."""
    ids = TOK.encode_ordinary(ex.prompt)
    pieces = [TOK.decode([i]) for i in ids]
    letter_pieces = [p for p in pieces if p.strip() in set(LETTERS) and len(p.strip()) == 1]
    n_tables = len(ex.tables)
    # Letters in the non-table text (start letter and the word "a") counted separately.
    rest = "\n".join(l for l in ex.prompt.split("\n") if " -> " not in l)
    rest_letters = [p for p in (TOK.decode([i]) for i in TOK.encode_ordinary(rest))
                    if p.strip() in set(LETTERS) and len(p.strip()) == 1]
    assert f" {ex.path[0]}" in rest_letters
    expected = 26 * 2 * n_tables + len(rest_letters)  # keys + images + rest
    # Also check no piece mixes a letter with an arrow or newline.
    bad = [p for p in pieces if len(p.strip()) > 1 and any(
        p.strip().startswith(c) or p.strip().endswith(c) for c in ("->",)) and p.strip() != "->"]
    assert len(letter_pieces) == expected, (len(letter_pieces), expected)
    assert not bad, bad
    return {"prompt_text_tokens": len(ids), "rendered_with_prefill": len(prompt_tokens(ex)),
            "letter_tokens": len(letter_pieces)}


if __name__ == "__main__":
    for L in LETTERS:
        assert len(TOK.encode_ordinary(L)) == 1 and len(TOK.encode_ordinary(" " + L)) == 1
    for k in KS:
        exs = generate(200, k, seed=123)
        stats = [check_tokenisation(e) for e in exs]
        for e in exs:
            assert e.path[-1] == e.answer and len(set(e.path)) == k + 1
            sft_tokens(e)
        lens = sorted({s["rendered_with_prefill"] for s in stats})
        print(f"k={k}: 200 ok, rendered length {lens}")
    exs = generate(200, 1, seed=123, variant="control")
    stats = [check_tokenisation(e) for e in exs]
    for e in exs:
        sft_tokens(e)
    print(f"control: 200 ok, rendered length {sorted({s['rendered_with_prefill'] for s in stats})}")
    ex = generate(1, 3, seed=0)[0]
    ids = prompt_tokens(ex)
    print(repr(TOK.decode(ids)))
    print([TOK.decode([i]) for i in ids][-60:])
    print("answer", ex.answer, "path", ex.path)
