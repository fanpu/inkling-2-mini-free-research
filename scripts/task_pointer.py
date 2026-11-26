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


# ---------------------------------------------------------------- one-hop controls (note 07)
# Harder one-hop controls. These use their own RNG streams ("ctl-<kind>-<seed>") and
# never touch generate(), so the main task and the 3-table control are unchanged.
# Every kind needs exactly one lookup: find the line whose left side matches the
# query and read its right side (or, for "reverse", match the right side and read
# the left). The answer is always one lowercase letter, scored as the token " x".
#
# Kinds:
#   tablesN       N independent letter permutations named A, B, ...; the question
#                 names one and uses the existing k=1 wording. No other table sends
#                 the start letter to the answer.
#   keysLxN       one table of N distinct random keys of L letters, each mapped to a
#                 random letter.
#   nearLxN       like keysLxN, but keys come in groups of 5 that share all letters
#                 except one position (near-duplicates). Members of a group map to
#                 distinct letters.
#   passageW      the one-table k=1 prompt with about W words of filler prose between
#                 the table and the question.
#   reverse       one table, the question gives an output letter and asks which input
#                 letter maps to it.
#   reverseN      reverse lookup in one of N named tables. No other table maps the
#                 answer letter to the same output letter.

CTL_NAMES = string.ascii_uppercase
_FILLER = ("the a of and to in is was for on with as by at from that this it be are "
           "river stone light market evening garden window letter morning paper table "
           "quiet old small green long early bright distant warm cold slow simple "
           "walked found carried opened noticed kept watched built moved returned "
           "village road field harbor station kitchen library forest valley bridge").split()


def _balanced_targets(rng: random.Random, n: int) -> list[str]:
    t = []
    while len(t) < n:
        block = list(LETTERS)
        rng.shuffle(block)
        t += block
    return t[:n]


def _kv_prompt(pairs: list[tuple[str, str]], query: str) -> str:
    body = "\n".join(f"{a} -> {b}" for a, b in pairs)
    return ("Table A maps each key to a letter:\n\n" + body + "\n\n"
            f"Key: {query}\nLook up the key in table A. Reply with the letter it maps to only.")


def _filler(rng: random.Random, words: int) -> str:
    sents, n = [], 0
    while n < words:
        m = rng.randint(8, 16)
        w = [rng.choice(_FILLER) for _ in range(m)]
        sents.append(" ".join(w).capitalize() + ".")
        n += m
    lines = [" ".join(sents[i:i + 5]) for i in range(0, len(sents), 5)]
    return "\n".join(lines)


def _rand_key(rng: random.Random, L: int) -> str:
    return "".join(rng.choice(LETTERS) for _ in range(L))


def generate_control(n: int, seed: int, kind: str) -> list[Example]:
    import re
    rng = random.Random(f"ctl-{kind}-{seed}")
    out = []
    for ans in _balanced_targets(rng, n):
        m = re.fullmatch(r"tables(\d+)", kind)
        if m:
            nt = int(m.group(1))
            names = CTL_NAMES[:nt]
            tables = {nm: _random_perm(rng) for nm in names}
            use = rng.choice(names)
            start = {v: u for u, v in tables[use].items()}[ans]
            for nm in names:
                while nm != use and tables[nm][start] == ans:
                    tables[nm] = _random_perm(rng)
            listed = ", ".join(names[:-1]) + f" and {names[-1]}"
            blocks = [f"Functions {listed} each map every letter to a letter."]
            for nm in names:
                blocks.append(f"Function {nm}:\n" + _table_lines(tables[nm], rng))
            blocks.append(
                f"Start letter: {start}\nNumber of steps: 1\n"
                f"Follow function {use} from the start letter for that number of steps "
                "(0 steps means stay on the start letter). Reply with the final letter only.")
            out.append(Example("\n\n".join(blocks), ans, 1, [start, ans], kind, use, tables))
            continue
        m = re.fullmatch(r"(keys|near)(\d+)x(\d+)", kind)
        if m:
            L, N = int(m.group(2)), int(m.group(3))
            keys: list[str] = []
            vals: list[str] = []
            seen = set()
            if m.group(1) == "keys":
                while len(keys) < N:
                    k_ = _rand_key(rng, L)
                    if k_ not in seen:
                        seen.add(k_)
                        keys.append(k_)
                        vals.append(rng.choice(LETTERS))
            else:
                assert N % 5 == 0
                while len(keys) < N:
                    base = _rand_key(rng, L)
                    pos = rng.randrange(L)
                    subs = rng.sample(LETTERS, 5)
                    grp = [base[:pos] + c + base[pos + 1:] for c in subs]
                    if any(g in seen for g in grp):
                        continue
                    seen.update(grp)
                    keys += grp
                    vals += rng.sample(LETTERS, 5)
            qi = rng.randrange(N)
            if m.group(1) == "near":
                # keep group members distinct: swap the answer within the query's group
                g0 = qi - qi % 5
                grp_vals = vals[g0:g0 + 5]
                if ans in grp_vals:
                    j = g0 + grp_vals.index(ans)
                    vals[j] = vals[qi]
            vals[qi] = ans
            pairs = list(zip(keys, vals))
            q = pairs[qi]
            rng.shuffle(pairs)
            out.append(Example(_kv_prompt(pairs, q[0]), ans, 1, [q[0], ans], kind, "A",
                               {"A": dict(pairs)}))
            continue
        m = re.fullmatch(r"passage(\d+)", kind)
        if m:
            f = _random_perm(rng)
            start = {v: u for u, v in f.items()}[ans]
            prompt = _format_prompt({"A": f}, rng, "A", start, 1)
            head, q = prompt.rsplit("\n\n", 1)
            prompt = (head + "\n\nThe following passage is unrelated to the question.\n\n"
                      + _filler(rng, int(m.group(1))) + "\n\n" + q)
            out.append(Example(prompt, ans, 1, [start, ans], kind, "A", {"A": f}))
            continue
        if kind == "reverse":
            f = _random_perm(rng)
            out_letter = f[ans]
            body = _table_lines(f, rng)
            prompt = ("Function A maps each letter to a letter:\n\n" + body + "\n\n"
                      f"Output letter: {out_letter}\nWhich letter does function A map to the "
                      "output letter? Reply with that input letter only.")
            out.append(Example(prompt, ans, 1, [out_letter, ans], kind, "A", {"A": f}))
            continue
        m = re.fullmatch(r"reverse(\d+)", kind)
        if m:
            nt = int(m.group(1))
            names = CTL_NAMES[:nt]
            tables = {nm: _random_perm(rng) for nm in names}
            use = rng.choice(names)
            out_letter = tables[use][ans]
            for nm in names:  # no other table maps the answer to the same output
                while nm != use and tables[nm][ans] == out_letter:
                    tables[nm] = _random_perm(rng)
            listed = ", ".join(names[:-1]) + f" and {names[-1]}"
            blocks = [f"Functions {listed} each map every letter to a letter."]
            for nm in names:
                blocks.append(f"Function {nm}:\n" + _table_lines(tables[nm], rng))
            blocks.append(f"Output letter: {out_letter}\nWhich letter does function {use} map "
                          "to the output letter? Reply with that input letter only.")
            out.append(Example("\n\n".join(blocks), ans, 1, [out_letter, ans], kind, use, tables))
            continue
        raise ValueError(kind)
    return out


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
