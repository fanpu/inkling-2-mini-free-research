"""Search for a one-hop control that is hard but not deep (note 07).

Each candidate kind from task_pointer.generate_control is scored on n examples with
the same method as 05_base_accuracy.py (prefill "Answer:", one request per example,
prompt logprob of the correct letter and the top-20 greedy token at the answer slot).
Results are merged into artifacts/control_search/results.json under
results[stage][kind], so pilots (n=10) and full runs (n=40) sit side by side.

Usage: .venv/bin/python scripts/06_control_search.py <stage> <n> <kind> [<kind> ...]
  e.g. .venv/bin/python scripts/06_control_search.py pilot 10 tables8 keys3x200
"""
import importlib.util, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import task_pointer as tp  # noqa: E402

_spec = importlib.util.spec_from_file_location("ba", Path(__file__).parent / "05_base_accuracy.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

import tinker  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert tp.MODEL_ID == MODEL_ID and ba.MODEL_ID == MODEL_ID
OUT = Path("artifacts/control_search")
SEED = 2026  # eval seed, separate RNG stream from the main task ("ctl-<kind>-<seed>")


def main(stage, n, kinds):
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    t0 = time.time()
    sc = tinker.ServiceClient()
    s = ba.call(lambda: sc.create_sampling_client(base_model=MODEL_ID).result(),
                "create_sampling_client")
    assert s.get_base_model() == MODEL_ID or MODEL_ID.startswith(s.get_base_model())
    for kind in kinds:
        exs = tp.generate_control(n, SEED, kind)
        with ThreadPoolExecutor(ba.MAX_IN_FLIGHT) as pool:
            rows = list(pool.map(lambda e: ba.score(s, e), exs))
        acc = sum(r["greedy_correct"] for r in rows) / n
        lps = [r["logprob_correct"] for r in rows]
        toks = [r["prompt_tokens"] for r in rows]
        summ = {
            "n": n, "greedy_acc": acc, "greedy_acc_se": math.sqrt(acc * (1 - acc) / n),
            "mean_logprob_correct": sum(lps) / n,
            "median_logprob_correct": sorted(lps)[n // 2],
            "greedy_is_letter_frac": sum(r["greedy_is_letter"] for r in rows) / n,
            "prompt_tokens_min": min(toks), "prompt_tokens_max": max(toks),
            "prompt_tokens_mean": round(sum(toks) / n),
        }
        results.setdefault(stage, {})[kind] = summ
        (OUT / f"per_example_{stage}_{kind}.json").write_text(json.dumps(rows, indent=1))
        res_path.write_text(json.dumps(results, indent=1))
        print(f"[{stage}] {kind}: acc {acc:.2f} lp {summ['mean_logprob_correct']:.2f} "
              f"tok {summ['prompt_tokens_mean']} letter {summ['greedy_is_letter_frac']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    meta = results.setdefault("_meta", {"model": MODEL_ID, "seed": SEED, "runs": []})
    meta["runs"].append({"stage": stage, "n": n, "kinds": kinds,
                         "requests_incl_retries": ba.COUNTS["requests"],
                         "failed_attempts": ba.COUNTS["failed_attempts"],
                         "wall_seconds": round(time.time() - t0)})
    res_path.write_text(json.dumps(results, indent=1))
    if ba.ERRORS:
        with open(OUT / "errors.json", "a") as fh:
            fh.write(json.dumps(ba.ERRORS) + "\n")
        print(f"{len(ba.ERRORS)} errors recorded in errors.json")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
