"""Reusable LoRA training harness for pointer chasing on the free-tier Inkling 2 Mini.

One run = one training client (AdamW or Dimuon), fresh examples every step, train
loss logged to JSONL (flushed and fsynced every step), periodic evaluation on a fixed
seed-2026 eval set by sampling from the current weights, and at the end an adapter
export with per-layer and per-module norm statistics.

    from train_lib import Config, run
    run(Config(name="x", optimizer="adamw", lr=5.4e-4, steps=100, out_dir=Path("artifacts/x")))

Outputs in cfg.out_dir:
    train_log.jsonl     one line per step ({"type": "train"}), per eval ({"type": "eval"}),
                        per saved state ({"type": "state"}) and the run header/footer
    eval_<step>.json    per-example eval rows at that step
    adapter_stats.json  per-layer / per-module norms of the final adapter (and step-0 A)
    errors.json         every failed API attempt, verbatim
    config.json

Resume after a crash: run(cfg, resume=True) reloads the last saved state (weights and
optimizer, written at each eval point when cfg.save_state is true) and continues the step
count. Steps after the last saved state are redone with the same data, since a step's
data depends only on (data_seed, step).
"""
from __future__ import annotations

import json, logging, math, os, re, shutil, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import task_pointer as tp  # noqa: E402

import tinker  # noqa: E402
from tinker import types  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free", "free tier only (D-001)"
assert tp.MODEL_ID == MODEL_ID
EVAL_SEED = 2026
TOPK = 20
LOOP = range(6, 18)
MODS = ["attn.wq_du", "attn.wk_dv", "attn.wv_dv", "attn.wo_ud", "attn.wr_du", "attn.gate_attn",
        "mlp.experts.w1", "mlp.experts.w2", "mlp.experts.w3"]
KEY_RE = re.compile(r".*language_model\.(?:layers\.(\d+)\.)?(.*)\.lora_([AB])\.weight$")
STEP_TIMEOUT = 1800  # seconds; a step that hangs longer ends the run cleanly (resumable)

# Record every server queue-state change at once. The SDK itself only warns after a
# pause has lasted 15 minutes, which hid a 14-minute stall in the first pilot attempt.
from tinker.lib import queue_state_logger as _qsl  # noqa: E402
QUEUE_EVENTS: list = []
_orig_qs = _qsl.QueueStateLogger.on_queue_state_change


def _qs_hook(self, queue_state, queue_state_reason):
    st = str(queue_state)
    prev = getattr(self, "_e2_last", None)
    if st != prev:
        self._e2_last = st
        ev = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "what": self._message_prefix,
              "state": st, "reason": queue_state_reason}
        QUEUE_EVENTS.append(ev)
        print(f"  [queue] {ev['what']}: {st} reason={queue_state_reason!r}", flush=True)
    return _orig_qs(self, queue_state, queue_state_reason)


_qsl.QueueStateLogger.on_queue_state_change = _qs_hook


@dataclass
class Config:
    name: str
    out_dir: Path
    optimizer: str = "adamw"            # "adamw" or "dimuon"
    lr: float = 5.4e-4
    rank: int = 8
    lora_seed: int = 0
    data_seed: int = 1                  # train data seed, never 2026
    train_conds: list = field(default_factory=lambda: ["k2", "k3", "k4"])
    batch: int = 32
    steps: int = 100
    eval_steps: list = field(default_factory=lambda: [0, 10, 25, 50, 100])
    eval_conds: list = field(default_factory=lambda: ["k1", "k2", "k3", "k4", "k6", "reverse"])
    eval_n: int = 30
    early_stop_conds: list = field(default_factory=lambda: ["k2", "k3", "k4"])
    early_stop_acc: float = 0.95        # stop when every early_stop_cond is strictly above this
    adam_beta1: float = 0.9             # Tinker AdamParams defaults
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12
    weight_decay: float = 0.0
    dimuon_beta1: float = 0.9
    grad_clip_norm: float = 0.0
    max_in_flight: int = 8              # concurrent eval sampling requests
    save_state: bool = True             # save weights+optimizer at each eval point (for resume)
    state_ttl: int = 86400
    export_ttl: int = 3600
    export_step0: bool = True
    max_wall_seconds: float | None = None  # stop training (then export) after this long

    def __post_init__(self):
        assert self.optimizer in ("adamw", "dimuon"), self.optimizer
        assert self.data_seed != EVAL_SEED, "2026 is the eval seed"
        self.out_dir = Path(self.out_dir)


def eval_schedule(steps: int, every: int = 100) -> list[int]:
    """0, 10, 25, 50, 100, then every `every`, plus the final step."""
    s = {x for x in (0, 10, 25, 50) if x <= steps} | set(range(every, steps + 1, every)) | {steps}
    return sorted(s)


# ---------------------------------------------------------------- logging and retries

class RunLog:
    def __init__(self, out: Path, append: bool):
        self.out = out
        self.f = open(out / "train_log.jsonl", "a" if append else "w")
        self.errors_path = out / "errors.json"
        self.errors = json.loads(self.errors_path.read_text()) if append and self.errors_path.exists() else []
        self.warnings: list = []

    def write(self, rec: dict):
        rec = {"time": round(time.time(), 2), **rec}
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def error(self, what: str, e: Exception, attempt: int, transient: bool):
        self.errors.append({"what": what, "attempt": attempt, "transient": transient,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "type": type(e).__name__, "error": str(e)[:3000]})
        self.errors_path.write_text(json.dumps(self.errors, indent=1))

    def close(self):
        self.f.close()


class _SDKWarnings(logging.Handler):
    """Keep every distinct tinker warning (rate-limit pauses appear here) verbatim."""

    def __init__(self, sink: list):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record):
        msg = record.getMessage()[:2000]
        for w in self.sink:
            if w["msg"] == msg:
                w["count"] += 1
                w["last"] = time.strftime("%H:%M:%S")
                return
        self.sink.append({"msg": msg, "count": 1, "first": time.strftime("%H:%M:%S")})
        print(f"  [sdk warning] {msg[:300]}", flush=True)


def retry(log: RunLog, fn, what: str, tries: int = 7):
    """Retry transient errors (rate limits, timeouts, 5xx) with exponential backoff."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            transient = isinstance(e, (tinker.RateLimitError, tinker.APIConnectionError,
                                       tinker.APITimeoutError, tinker.InternalServerError)) or \
                any(s in msg.lower() for s in ("429", "rate limit", "throttl", "503", "timed out", "unavailable"))
            log.error(what, e, k, transient)
            print(f"  ! {what} attempt {k}: {msg[:300]}", flush=True)
            if not transient or k == tries - 1:
                raise
            time.sleep(min(300, 15 * 2 ** k))


# ---------------------------------------------------------------- data

def make_examples(cond: str, n: int, seed) -> list:
    if cond == "reverse":
        return tp.generate_control(n, seed, "reverse")
    return tp.generate(n, int(cond[1:]), seed)


def train_batch(cfg: Config, step: int) -> list:
    """Equal mix of cfg.train_conds. When batch is not divisible, the extra examples
    rotate across conditions by step. Seed depends only on (data_seed, step, cond)."""
    nc = len(cfg.train_conds)
    base, extra = divmod(cfg.batch, nc)
    exs = []
    for i, c in enumerate(cfg.train_conds):
        n = base + (1 if (i - step) % nc < extra else 0)
        seed = cfg.data_seed * 1_000_000 + step
        assert seed != EVAL_SEED
        exs += make_examples(c, n, seed)
    return exs


def eval_set(cfg: Config) -> dict:
    return {c: make_examples(c, cfg.eval_n, EVAL_SEED) for c in cfg.eval_conds}


# ---------------------------------------------------------------- evaluation (same scoring as script 05)

def score(log: RunLog, sc, e) -> dict:
    ids = tp.prompt_tokens(e, prefill=True) + [tp.LETTER_TOKENS[e.answer]]
    r = retry(log, lambda: sc.sample(types.ModelInput.from_ints(ids), 1,
                                     types.SamplingParams(max_tokens=1, temperature=0.0),
                                     include_prompt_logprobs=True, topk_prompt_logprobs=TOPK,
                                     prompt_logprobs_last_n=1).result(), f"score {e.variant} k={e.k}")
    top = r.topk_prompt_logprobs[-1]
    gid = int(top[0][0])
    greedy = tp.TOK.decode([gid]).strip()
    row = {"cond": "reverse" if e.variant == "reverse" else f"k{e.k}", "answer": e.answer,
           "path": e.path, "greedy": greedy, "greedy_correct": gid == tp.LETTER_TOKENS[e.answer],
           "logprob_correct": float(r.prompt_logprobs[-1]),
           "greedy_path_index": e.path.index(greedy) if greedy in e.path else None}
    if e.variant == "reverse":  # forward image f(y) of the shown output letter y
        row["forward_image"] = e.tables["A"][e.path[0]]
        row["greedy_is_forward_image"] = greedy == row["forward_image"]
    return row


def evaluate(log: RunLog, sc, evset: dict, max_in_flight: int) -> tuple[dict, list]:
    rows = []
    with ThreadPoolExecutor(max_in_flight) as pool:
        for c, exs in evset.items():
            rows += list(pool.map(lambda e: score(log, sc, e), exs))
    summ = {}
    for c in evset:
        rs = [r for r in rows if r["cond"] == c]
        acc = sum(r["greedy_correct"] for r in rs) / len(rs)
        s = {"n": len(rs), "acc": acc, "acc_se": math.sqrt(acc * (1 - acc) / len(rs)),
             "mean_logp": sum(r["logprob_correct"] for r in rs) / len(rs)}
        if c == "reverse":
            s["forward_image_share"] = sum(r["greedy_is_forward_image"] for r in rs) / len(rs)
        summ[c] = s
    return summ, rows


# ---------------------------------------------------------------- adapter statistics

def download_adapter(log: RunLog, rest, path: str, dest_dir: Path) -> tuple[Path, dict]:
    """Download adapter_model.safetensors and adapter_config.json. Returns (file, config)."""
    urls = retry(log, lambda: rest.get_external_weights_urls(path).result(), "get_urls")
    st, conf = None, {}
    for rel, url in urls.urls.items():
        dest = dest_dir / rel.replace("/", "_")
        retry(log, lambda: urllib.request.urlretrieve(url, dest), f"download {rel}")
        if rel.endswith(".safetensors"):
            assert st is None, "expected one safetensors file"
            st = dest
        elif rel.endswith("adapter_config.json"):
            conf = json.loads(dest.read_text())
            dest.unlink()
        else:
            dest.unlink()
    assert st is not None
    return st, conf


def _gram_A(A, r):
    # A: [r, in] or [E, r, in] -> [.., r, r]
    assert A.shape[-2] == r, ("A rank dim", tuple(A.shape))
    return A @ A.transpose(-1, -2)


def _gram_B(B, r):
    # B: [out, r] or [E, out, r] -> [.., r, r]
    assert B.shape[-1] == r, ("B rank dim", tuple(B.shape))
    return B.transpose(-1, -2) @ B


def adapter_stats(st_path: Path, conf: dict, A0: dict | None = None, keep_A: bool = False):
    """Per (layer, module): ||B||_F, ||A||_F, ||A - A0||_F and ||scale * B A||_F summed over
    experts. The effective-update norm uses ||BA||_F^2 = sum((B^T B) * (A A^T)) per expert,
    which broadcasts when A or B is shared across experts, so it is exact and cheap."""
    import torch
    from safetensors import safe_open
    r = int(conf["r"])
    scale = float(conf["lora_alpha"]) / r
    rows, shapes, A_out = [], {}, {}
    with safe_open(str(st_path), "pt") as f:
        keys = list(f.keys())
        bases = sorted({k.rsplit(".lora_", 1)[0] for k in keys})
        for b in bases:
            ka, kb = f"{b}.lora_A.weight", f"{b}.lora_B.weight"
            A = f.get_tensor(ka).double()
            B = f.get_tensor(kb).double()
            m = KEY_RE.match(ka)
            layer = int(m[1]) if m and m[1] is not None else None
            mod = m[2] if m else b
            if mod not in shapes:
                shapes[mod] = {"A": list(A.shape), "B": list(B.shape)}
            eff2 = (_gram_B(B, r) * _gram_A(A, r)).sum().item()
            row = {"layer": layer, "module": mod, "B_fro": B.norm().item(), "A_fro": A.norm().item(),
                   "eff_fro": scale * math.sqrt(max(eff2, 0.0))}
            if A0 is not None and ka in A0:
                row["dA_fro"] = (A - A0[ka].double()).norm().item()
            rows.append(row)
            if keep_A:
                A_out[ka] = A.float().clone()
    return {"rows": rows, "shapes": shapes, "lora_alpha": conf["lora_alpha"], "r": r,
            "scale": scale}, A_out


def summarize_stats(stats: dict) -> dict:
    """Looped share of squared norm (layers 6-17 over all 24 layers, lm_head excluded)."""
    out = {}
    for q in ("B_fro", "eff_fro", "dA_fro"):
        rows = [x for x in stats["rows"] if x["layer"] is not None and q in x]
        if not rows:
            continue
        per_layer = [0.0] * 24
        by_mod = {}
        for x in rows:
            per_layer[x["layer"]] += x[q] ** 2
            by_mod.setdefault(x["module"], [0.0] * 24)[x["layer"]] += x[q] ** 2
        groups = {"all": per_layer,
                  "attention": [sum(v[i] for m, v in by_mod.items() if m.startswith("attn")) for i in range(24)],
                  "mlp_experts": [sum(v[i] for m, v in by_mod.items() if m.startswith("mlp")) for i in range(24)]}
        groups.update(by_mod)
        out[q] = {g: {"looped_share": sum(v[i] for i in LOOP) / sum(v) if sum(v) > 0 else None,
                      "per_layer_norm": [math.sqrt(x) for x in v]} for g, v in groups.items()}
        lm = [x[q] for x in stats["rows"] if x["layer"] is None]
        out[q]["other_norm"] = lm
    return out


def export_and_stats(log: RunLog, tc, rest, name: str, ttl: int, A0=None, keep_A=False):
    exp = retry(log, lambda: tc.save_weights_external(name, ttl_seconds=ttl).result(), f"{name}:export")
    tmp = Path(tempfile.mkdtemp(prefix="e2_"))
    try:
        st, conf = download_adapter(log, rest, exp.path, tmp)
        size = st.stat().st_size
        stats, A = adapter_stats(st, conf, A0=A0, keep_A=keep_A)
        st.unlink()  # delete as soon as stats are computed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    stats["export_path"] = exp.path
    stats["file_bytes"] = size
    return stats, A


# ---------------------------------------------------------------- training loop

def _optim_params(cfg: Config):
    if cfg.optimizer == "adamw":
        return types.AdamParams(learning_rate=cfg.lr, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2,
                                eps=cfg.adam_eps, weight_decay=cfg.weight_decay,
                                grad_clip_norm=cfg.grad_clip_norm)
    return types.DimuonParams(learning_rate=cfg.lr, beta1=cfg.dimuon_beta1, grad_clip_norm=cfg.grad_clip_norm)


def _optim_config(cfg: Config):
    return types.AdamOptimizerConfig() if cfg.optimizer == "adamw" else types.DimuonOptimizerConfig()


def _tolist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _answer_logps(fb, datums):
    out = []
    for o, d in zip(fb.loss_fn_outputs, datums):
        w = _tolist(d.loss_fn_inputs["weights"])
        lp = _tolist(o["logprobs"])
        out.append(sum(a for a, b in zip(lp, w) if b > 0))
    return out


def _last_state(path: Path):
    last, step = None, -1
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "state":
                last, step = rec["path"], rec["step"]
    return last, step


def run(cfg: Config, resume: bool = False) -> dict:
    """Train, evaluate on cfg.eval_steps, export. Returns a summary dict."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state_path, state_step = _last_state(cfg.out_dir / "train_log.jsonl") if resume else (None, -1)
    log = RunLog(cfg.out_dir, append=resume)
    h = _SDKWarnings(log.warnings)
    logging.getLogger("tinker").addHandler(h)
    (cfg.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))
    evset = eval_set(cfg)
    params = _optim_params(cfg)
    t_start = time.time()
    log.write({"type": "start", "config": asdict(cfg), "resume_from": state_path, "model": MODEL_ID})
    sc, rest = tinker.ServiceClient(), tinker.RestClient()
    status, detail, summary = "errored", "", {"evals": {}}
    step_times, stopped = [], None
    try:
        meta = {"purpose": "e2", "run": cfg.name}
        if state_path:
            tc = retry(log, lambda: sc.create_training_client_from_state_with_optimizer(
                state_path, base_model=MODEL_ID, user_metadata=meta).result(), "create_from_state")
            start = state_step
            print(f"[{cfg.name}] resumed from step {start}", flush=True)
        else:
            tc = retry(log, lambda: sc.create_lora_training_client(
                base_model=MODEL_ID, rank=cfg.rank, seed=cfg.lora_seed,
                optimizer=_optim_config(cfg), user_metadata=meta).result(), "create")
            start = 0
        print(f"[{cfg.name}] client ready {time.time()-t_start:.0f}s", flush=True)

        A0 = None
        if cfg.export_step0 and start == 0:
            t0 = time.time()
            s0, A0 = export_and_stats(log, tc, rest, f"{cfg.name}-step0", cfg.export_ttl, keep_A=True)
            (cfg.out_dir / "adapter_stats_step0.json").write_text(json.dumps(s0, indent=1))
            log.write({"type": "export", "step": 0, "seconds": round(time.time() - t0, 1)})
            print(f"[{cfg.name}] step-0 export {time.time()-t0:.0f}s", flush=True)

        def do_eval(step):
            t0 = time.time()
            samp = retry(log, lambda: tc.save_weights_and_get_sampling_client().result(), f"sampler@{step}")
            t1 = time.time()
            summ, rows = evaluate(log, samp, evset, cfg.max_in_flight)
            (cfg.out_dir / f"eval_{step:05d}.json").write_text(json.dumps(rows))
            summary["evals"][step] = summ
            log.write({"type": "eval", "step": step, "summary": summ,
                       "seconds_save": round(t1 - t0, 1), "seconds_eval": round(time.time() - t1, 1)})
            print(f"[{cfg.name}] eval@{step} " + " ".join(
                f"{c}:{s['acc']:.2f}" for c, s in summ.items()) +
                (f" rev_fy:{summ['reverse']['forward_image_share']:.2f}" if "reverse" in summ else "") +
                f" ({time.time()-t0:.0f}s)", flush=True)
            if cfg.save_state and step > 0:
                st = retry(log, lambda: tc.save_state(f"{cfg.name}-s{step}", ttl_seconds=cfg.state_ttl).result(),
                           f"save_state@{step}")
                log.write({"type": "state", "step": step, "path": st.path})
            return summ

        if start in cfg.eval_steps and not (resume and start > 0):
            do_eval(start)
        for step in range(start, cfg.steps):
            exs = train_batch(cfg, step)
            datums = [tp.to_datum(e) for e in exs]
            t0 = time.time()

            def _step():
                fbf = tc.forward_backward(datums, "cross_entropy")
                opf = tc.optim_step(params)
                return fbf.result(timeout=STEP_TIMEOUT), opf.result(timeout=STEP_TIMEOUT)
            fb, op = retry(log, _step, f"step {step}")
            dt = time.time() - t0
            step_times.append(dt)
            lps = _answer_logps(fb, datums)
            per = {}
            for e, lp in zip(exs, lps):
                per.setdefault(f"k{e.k}", []).append(lp)
            rec = {"type": "train", "step": step + 1, "loss_mean": -sum(lps) / len(lps),
                   "loss_by_cond": {c: -sum(v) / len(v) for c, v in per.items()},
                   "n_by_cond": {c: len(v) for c, v in per.items()},
                   "metrics": fb.metrics, "seconds": round(dt, 2)}
            om = getattr(op, "metrics", None)
            if om:
                rec["optim_metrics"] = om
            if tl_q := len(QUEUE_EVENTS):
                rec["queue_events_total"] = tl_q
            log.write(rec)
            if (step + 1) % 5 == 0 or step < 3:
                print(f"[{cfg.name}] step {step+1} loss {rec['loss_mean']:.3f} "
                      + " ".join(f"{c}:{v:.2f}" for c, v in sorted(rec["loss_by_cond"].items()))
                      + f" {dt:.1f}s", flush=True)
            if not math.isfinite(rec["loss_mean"]):
                stopped = f"non-finite loss at step {step+1}"
                break
            if step + 1 in cfg.eval_steps:
                summ = do_eval(step + 1)
                if all(summ[c]["acc"] > cfg.early_stop_acc for c in cfg.early_stop_conds if c in summ):
                    stopped = f"early stop at step {step+1}: {cfg.early_stop_conds} all above {cfg.early_stop_acc}"
                    break
            if cfg.max_wall_seconds and time.time() - t_start > cfg.max_wall_seconds:
                stopped = f"wall-time cap at step {step+1}"
                if step + 1 not in cfg.eval_steps:
                    do_eval(step + 1)
                break
        last_step = step + 1 if cfg.steps > start else start
        if stopped:
            print(f"[{cfg.name}] {stopped}", flush=True)

        t0 = time.time()
        fin, _ = export_and_stats(log, tc, rest, f"{cfg.name}-final", cfg.export_ttl, A0=A0)
        fin["step"] = last_step
        fin["summary"] = summarize_stats(fin)
        (cfg.out_dir / "adapter_stats.json").write_text(json.dumps(fin, indent=1))
        log.write({"type": "export", "step": last_step, "seconds": round(time.time() - t0, 1)})
        print(f"[{cfg.name}] final export {time.time()-t0:.0f}s", flush=True)
        summary.update(last_step=last_step, stopped=stopped,
                       looped_share={q: v["all"]["looped_share"] for q, v in fin["summary"].items()})
        status, detail = "success", f"e2 {cfg.name}"
    except KeyboardInterrupt:
        status, detail = "interrupted", "keyboard interrupt"
        raise
    except Exception as e:  # noqa: BLE001
        detail = f"{type(e).__name__}: {e}"[:500]
        raise
    finally:
        try:  # an unclosed ServiceClient stalls the next training client for 15+ minutes
            sc.close(status, detail=detail).result()
        except Exception as e:  # noqa: BLE001
            log.error("close", e, 0, False)
        st = sorted(step_times)
        timing = {"steps_run": len(st), "wall_seconds": round(time.time() - t_start, 1),
                  "step_seconds_mean": sum(st) / len(st) if st else None,
                  "step_seconds_median": st[len(st) // 2] if st else None}
        summary["timing"] = timing
        log.write({"type": "end", "status": status, "detail": detail, "timing": timing,
                   "sdk_warnings": log.warnings, "queue_events": QUEUE_EVENTS, "n_errors": len(log.errors)})
        (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        logging.getLogger("tinker").removeHandler(h)
        log.close()
    return summary
