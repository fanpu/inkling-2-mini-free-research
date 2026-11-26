"""Base (untuned) accuracy of Inkling 2 Mini on the pointer-chasing task.

Phase 1 (probe): samples a few raw completions with and without the "Answer:"
prefill, to show what the model emits at effort 0.
Phase 2 (eval): for k in {0,1,2,3,4,6} and the 3-table control, 60 fresh examples
each. One sampling request per example scores the prompt "...Answer: <correct>"
and reads, at the answer position, the top-20 next-token logprobs (greedy token
and letter mass within the top 20) and the logprob of the correct letter. The
server allows at most 20 target ids, so all 26 letters cannot be read in one call.

Outputs in artifacts/task_base_accuracy/: probe.json, per_example.json,
summary.json, sdk_log.txt (SDK retry and error lines), errors.json.

Usage: .venv/bin/python scripts/05_base_accuracy.py [n_per_setting]
"""
import json, logging, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import task_pointer as tp  # noqa: E402

import tinker  # noqa: E402
from tinker import types  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert tp.MODEL_ID == MODEL_ID
OUT = Path("artifacts/task_base_accuracy")
EVAL_SEED = 2026  # training must use other seeds
MAX_IN_FLIGHT = 4
TOPK = 20

ERRORS = []
COUNTS = {"requests": 0, "failed_attempts": 0}


class _Filter(logging.Filter):
    KEYS = ("429", "rate", "limit", "retry", "retrying", "fail", "error", "throttl", "try_again")

    def filter(self, rec):
        msg = rec.getMessage().lower()
        return any(k in msg for k in self.KEYS)


def setup_logging():
    h = logging.FileHandler(OUT / "sdk_log.txt", mode="w")
    h.setLevel(logging.DEBUG)
    h.addFilter(_Filter())
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    lg = logging.getLogger("tinker")
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)


def call(fn, what):
    """Run fn() with polite backoff on failure. Errors are recorded verbatim."""
    delay = 5.0
    for attempt in range(6):
        COUNTS["requests"] += 1
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            COUNTS["failed_attempts"] += 1
            ERRORS.append({"what": what, "attempt": attempt, "time": time.time(),
                           "type": type(e).__name__, "error": str(e)[:2000]})
            if isinstance(e, tinker.BadRequestError):
                raise
            print(f"[error] {what} attempt {attempt}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up on {what}")


def decode_list(ids):
    return [tp.TOK.decode([int(t)]) for t in ids]


def probe(sc):
    """Raw completions at temperature 0: no prefill (40 tokens) and with prefill (8 tokens)."""
    rows = []
    exs = [e for k in (1, 3) for e in tp.generate(3, k, seed=EVAL_SEED + 1000)]
    for e in exs:
        for prefill in (False, True):
            mt = 8 if prefill else 40
            r = call(lambda: sc.sample(tp.sampling_input(e, prefill), 1,
                                       types.SamplingParams(max_tokens=mt, temperature=0.0)).result(),
                     f"probe k={e.k} prefill={prefill}")
            seq = r.sequences[0]
            rows.append({"k": e.k, "answer": e.answer, "path": e.path, "prefill": prefill,
                         "stop_reason": str(seq.stop_reason), "tokens": list(map(int, seq.tokens)),
                         "decoded": decode_list(seq.tokens)})
            print(f"[probe] k={e.k} prefill={prefill} ans={e.answer} -> {rows[-1]['decoded'][:12]}",
                  flush=True)
    return rows


def score(sc, e):
    ids = tp.prompt_tokens(e, prefill=True) + [tp.LETTER_TOKENS[e.answer]]
    L = len(ids)
    r = call(lambda: sc.sample(types.ModelInput.from_ints(ids), 1,
                               types.SamplingParams(max_tokens=1, temperature=0.0),
                               include_prompt_logprobs=True, topk_prompt_logprobs=TOPK,
                               prompt_logprobs_last_n=1).result(),
             f"score {e.variant} k={e.k}")
    lp_correct = float(r.prompt_logprobs[-1])
    top = r.topk_prompt_logprobs[-1]
    top_letters = [(tp.TOKEN_TO_LETTER[int(t)], float(lp)) for t, lp in top
                   if int(t) in tp.TOKEN_TO_LETTER]
    greedy_id, greedy_lp = top[0]
    greedy = tp.TOK.decode([int(greedy_id)])
    return {
        "variant": e.variant, "k": e.k, "answer": e.answer, "path": e.path, "table": e.table,
        "prompt_tokens": L - 1,
        "greedy_token": greedy, "greedy_logprob": float(greedy_lp),
        "greedy_correct": int(greedy_id) == tp.LETTER_TOKENS[e.answer],
        "greedy_is_letter": int(greedy_id) in tp.TOKEN_TO_LETTER,
        "logprob_correct": lp_correct,
        "best_letter": top_letters[0][0] if top_letters else None,
        "best_letter_correct": bool(top_letters) and top_letters[0][0] == e.answer,
        "letter_mass_top20": sum(math.exp(lp) for _, lp in top_letters),
        "top20_letters": top_letters,
        "top5": [(tp.TOK.decode([int(t)]), float(lp)) for t, lp in top[:5]],
        # Which path node the greedy answer equals, if any (e.g. stopping one hop early).
        "greedy_path_index": (e.path.index(greedy.strip()) if greedy.strip() in e.path else None),
        "prompt": e.prompt,
    }


def main(n):
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging()
    t0 = time.time()
    sc = tinker.ServiceClient()
    s = call(lambda: sc.create_sampling_client(base_model=MODEL_ID).result(), "create_sampling_client")
    assert s.get_base_model() == MODEL_ID or MODEL_ID.startswith(s.get_base_model())
    (OUT / "probe.json").write_text(json.dumps(probe(s), indent=1))

    settings = [("main", k) for k in tp.KS] + [("control", 1)]
    rows = []
    summary = {}
    for variant, k in settings:
        exs = tp.generate(n, k, seed=EVAL_SEED, variant=variant)
        with ThreadPoolExecutor(MAX_IN_FLIGHT) as pool:
            res = list(pool.map(lambda e: score(s, e), exs))
        rows += res
        name = f"{variant}_k{k}"
        acc = sum(r["greedy_correct"] for r in res) / len(res)
        lps = [r["logprob_correct"] for r in res]
        idx = [r["greedy_path_index"] for r in res]
        summary[name] = {
            "n": len(res), "greedy_acc": acc,
            "greedy_acc_se": math.sqrt(acc * (1 - acc) / len(res)),
            "best_letter_acc": sum(r["best_letter_correct"] for r in res) / len(res),
            "mean_logprob_correct": sum(lps) / len(lps),
            "median_logprob_correct": sorted(lps)[len(lps) // 2],
            "greedy_is_letter_frac": sum(r["greedy_is_letter"] for r in res) / len(res),
            "mean_letter_mass_top20": sum(r["letter_mass_top20"] for r in res) / len(res),
            "greedy_path_index_counts": {str(i): idx.count(i) for i in sorted(set(idx), key=str)},
            "prompt_tokens": sorted({r["prompt_tokens"] for r in res}),
        }
        print(f"[eval] {name}: acc {acc:.3f}  mean lp {summary[name]['mean_logprob_correct']:.2f}"
              f"  path idx {summary[name]['greedy_path_index_counts']}  ({time.time()-t0:.0f}s)",
              flush=True)
        (OUT / "per_example.json").write_text(json.dumps(rows, indent=1))
    summary["_meta"] = {"model": MODEL_ID, "effort": tp.EFFORT, "prefill": tp.PREFILL,
                        "eval_seed": EVAL_SEED, "max_in_flight": MAX_IN_FLIGHT,
                        "chance_acc": 1 / 26, "chance_logprob": math.log(1 / 26),
                        "requests_incl_retries": COUNTS["requests"],
                        "failed_attempts": COUNTS["failed_attempts"],
                        "wall_seconds": round(time.time() - t0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "errors.json").write_text(json.dumps(ERRORS, indent=1))
    print(json.dumps(summary["_meta"], indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
