"""Controls for the SGD-mode pilot (note 10).

Control 1: does the unembedding adapter (lm_head) set the SGD-mode stability edge?
  30-step SGD-mode probes (AdamW, eps = 1e4, lr = lr_eff * eps, weight decay 0) with
  module switches from create_lora_training_client. Evals at 0, 5, 10, 20, 30, early stop
  off, adapter exported after step 1 and at the end.
Control 2: is AdamW's flat profile and hop clamp a matter of total movement?
  AdamW with default eps at a low rate, 150 steps, early stop off, evals at 0, 10, 25, 50,
  100, 150, export after step 1 and at the end.

Every other setting matches scripts 09 to 11 (rank 8, LoRA seed 0, data seed 1, batch 32
over k = 2, 3, 4, 30 eval examples per condition).

Usage:
  .venv/bin/python scripts/12_controls.py probe NAME LR_EFF [--no-unembed] [--no-attn] [--no-mlp]
  .venv/bin/python scripts/12_controls.py adamw NAME LR
  .venv/bin/python scripts/12_controls.py check NAME        # lm_head and module presence in exports
  .venv/bin/python scripts/12_controls.py analyze           # tables (JSON + stdout) and figure

Outputs under artifacts/e2_controls/<NAME>/, tables and figure in artifacts/e2_controls/.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_lib as tl  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert tl.MODEL_ID == MODEL_ID, "free tier only (D-001)"
OUT = Path("artifacts/e2_controls")
PILOT = Path("artifacts/e2_pilot")
EPS = 1e4
NO_STOP = 2.0  # early_stop_acc above 1 can never be passed
CONDS = ["k1", "k2", "k3", "k4", "k6", "reverse"]
CHANCE_LOSS = math.log(26)


def common(name, steps, eval_steps, **kw):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="adamw", rank=8, lora_seed=0,
                     data_seed=1, train_conds=["k2", "k3", "k4"], batch=32, steps=steps,
                     eval_steps=eval_steps, early_stop_acc=NO_STOP, adam_beta1=0.9, adam_beta2=0.95,
                     weight_decay=0.0, max_in_flight=16, export_steps=[1], **kw)


# ---------------------------------------------------------------- loading and metrics

def load(d: Path):
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    cfg = next(r["config"] for r in recs if r["type"] == "start")
    ev = {r["step"]: r["summary"] for r in recs if r["type"] == "eval"}
    tr = {r["step"]: r["loss_mean"] for r in recs if r["type"] == "train"}
    rd = lambda f: json.loads((d / f).read_text()) if (d / f).exists() else None  # noqa: E731
    return {"cfg": cfg, "ev": ev, "loss": tr, "final": rd("adapter_stats.json"),
            "s1": rd("adapter_stats_step1.json"), "s0": rd("adapter_stats_step0.json")}


def per_layer_B(st, mods=None):
    p = [0.0] * 24
    for r in st["rows"]:
        if r["layer"] is not None and (mods is None or r["module"].startswith(mods)):
            p[r["layer"]] += r["B_fro"] ** 2
    return [math.sqrt(x) for x in p]


def profile_metrics(p):
    loop = [p[i] for i in tl.LOOP]
    unl = [p[i] for i in range(24) if i not in tl.LOOP]
    tot2 = sum(x * x for x in p)
    return {"B_norm": math.sqrt(tot2), "S": sum(x * x for x in loop) / tot2,
            "R": (sum(loop) / len(loop)) / (sum(unl) / len(unl)),
            "L6/L5": p[6] / p[5], "L18/L17": p[18] / p[17]}


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def grad_profile():
    """Step-1 gradient profile: the SGD-mode 2.9e-4 step-1 update is lr_eff * g (note 10)."""
    return per_layer_B(load(PILOT / "sgd_lreff2.9e-4")["s1"])


def collapsed(run):
    """Collapsed: final-eval accuracy within 2 standard errors of chance (1/26 over 30, i.e.
    at most 0.11) on every condition and mean loss over the last 5 steps within 0.5 nats of ln 26."""
    last = max(run["ev"])
    accs = [run["ev"][last][c]["acc"] for c in CONDS]
    tail = [run["loss"][s] for s in sorted(run["loss"])[-5:]]
    return max(accs) <= 0.11 and abs(sum(tail) / len(tail) - CHANCE_LOSS) < 0.5, max(accs), sum(tail) / len(tail)


# ---------------------------------------------------------------- commands

def check(name):
    """Which modules the step-0, step-1 and final exports hold, and the lm_head B norm."""
    run = load(OUT / name)
    out = {}
    for tag in ("s0", "s1", "final"):
        st = run[tag]
        if st is None:
            continue
        mods = sorted({r["module"] for r in st["rows"]})
        lm = [r["B_fro"] for r in st["rows"] if r["layer"] is None]
        out[tag] = {"modules": mods, "n_tensors_pairs": len(st["rows"]), "lm_head_B_fro": lm,
                    "file_bytes": st.get("file_bytes")}
    (OUT / name / "module_check.json").write_text(json.dumps(out, indent=1))
    for tag, v in out.items():
        print(f"  {tag}: {len(v['modules'])} modules, lm_head B {v['lm_head_B_fro']}, bytes {v['file_bytes']}")
    return out


def step1(name):
    """Step-1 update summary. With eps = 1e4 the step is lr_eff * g, so norms read gradients."""
    run = load(OUT / name)
    cfg, s1 = run["cfg"], run["s1"]
    p = per_layer_B(s1)
    m = profile_metrics(p)
    m["max_abs_B"] = max(r["B_absmax"] for r in s1["rows"])
    top = max(s1["rows"], key=lambda r: r["B_absmax"])
    m["max_abs_B_at"] = f"{top['layer']} {top['module']}"
    m["lm_head_B"] = [r["B_fro"] for r in s1["rows"] if r["layer"] is None]
    m["cos_with_grad_profile"] = cosine(p, grad_profile())
    if cfg["adam_eps"] >= 1:
        lre = cfg["lr"] / cfg["adam_eps"]
        m["grad_norm_layers"] = m["B_norm"] / lre
    return m


def probe(name, lr_eff, flags):
    cfg = common(name, 30, [0, 5, 10, 20, 30], lr=lr_eff * EPS, adam_eps=EPS, save_state=False,
                 diverge_loss=None, train_unembed="--no-unembed" not in flags,
                 train_attn="--no-attn" not in flags, train_mlp="--no-mlp" not in flags)
    s = tl.run(cfg)
    print(json.dumps(s["timing"]), s.get("stopped"))
    check(name)
    print("  step1", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in step1(name).items()})
    print("  collapsed, max acc, tail loss", collapsed(load(OUT / name)))


def adamw(name, lr):
    cfg = common(name, 150, [0, 10, 25, 50, 100, 150], lr=lr, adam_eps=1e-12, save_state=True)
    s = tl.run(cfg)
    print(json.dumps(s["timing"]), s.get("stopped"))
    check(name)
    print("  step1", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in step1(name).items()})
    print("  final", {k: round(v, 3) for k, v in profile_metrics(per_layer_B(load(OUT / name)["final"])).items()})


def analyze():
    res = {"edge": [], "compare": {}}
    # Control 1 table: new probes plus the note-10 runs with the unembedding trained.
    old = [("sgd_lreff5e-4", "all modules (note 10)"), ("sgd_lreff1.5e-3", "all modules (note 10)"),
           ("sgd_lreff2.9e-4", "all modules (note 10)")]
    rows = [(PILOT / n, lab) for n, lab in old]
    for d in sorted(OUT.iterdir()):
        if d.is_dir() and d.name.startswith("sgd_") and (d / "summary.json").exists():
            c = load(d)["cfg"]
            off = [m for m, k in (("unembed", "train_unembed"), ("attn", "train_attn"), ("mlp", "train_mlp")) if not c[k]]
            rows.append((d, "off: " + ", ".join(off) if off else "all modules"))
    print("stability edge: run | modules | lr_eff | loss@2 | tail loss | max final acc | collapsed")
    for d, lab in rows:
        run = load(d)
        c = run["cfg"]
        col, macc, tail = collapsed(run)
        last = max(run["ev"])
        r = {"run": d.name, "modules": lab, "lr_eff": c["lr"] / c["adam_eps"], "steps": max(run["loss"]),
             "loss_step2": run["loss"].get(2), "tail_loss": tail, "final_eval_step": last,
             "final_acc": {k: run["ev"][last][k]["acc"] for k in CONDS}, "collapsed": col,
             "eval_acc": {s: {k: run["ev"][s][k]["acc"] for k in CONDS} for s in sorted(run["ev"])}}
        if run["s1"]:
            r["step1_B_norm_layers"] = profile_metrics(per_layer_B(run["s1"]))["B_norm"]
        # Plateau exit: first step from which the loss stays below 2.5 for 3 steps in a row.
        ls = [run["loss"][s] for s in sorted(run["loss"])]
        r["plateau_exit"] = next((i + 1 for i in range(len(ls) - 2) if i > 0 and max(ls[i:i + 3]) < 2.5), None)
        r["loss_first30"] = ls[:30]
        res["edge"].append(r)
        print(f"  {d.name:28s} {lab:24s} {r['lr_eff']:.1e} {r['loss_step2']:.2f} {tail:.2f} {macc:.2f} {col}"
              f" steps {r['steps']} exit {r['plateau_exit']} "
              + " ".join(f"@{s}:{max(a.values()):.2f}" for s, a in r["eval_acc"].items()))

    # Four-way comparison.
    g = grad_profile()
    four = {"AdamW 5.4e-4": PILOT / "adamw_lr5.4e-4", "Dimuon 1.6e-2": PILOT / "dimuon_lr1.6e-2",
            "SGD mode 2.9e-4": PILOT / "sgd_lreff2.9e-4"}
    low = sorted(d for d in OUT.iterdir() if d.is_dir() and d.name.startswith("adamw_lr")
                 and (d / "adapter_stats.json").exists())
    for d in low:
        four[f"AdamW {load(d)['cfg']['lr']:.1e}"] = d
    print("\nfour-way: run | steps | ||B|| | S | R | L6/L5 | L18/L17 | cos(grad) | final acc k1 k2 k3 k4 k6 rev")
    for lab, d in four.items():
        run = load(d)
        p = per_layer_B(run["final"])
        m = profile_metrics(p)
        m["cos_grad"] = cosine(p, g)
        m["steps"] = run["final"]["step"]
        last = max(run["ev"])
        m["final_acc"] = {k: run["ev"][last][k]["acc"] for k in CONDS}
        m["per_layer_B"] = p
        m["lm_head_B"] = [r["B_fro"] for r in run["final"]["rows"] if r["layer"] is None]
        if run["s1"]:
            m["step1_B_norm_layers"] = profile_metrics(per_layer_B(run["s1"]))["B_norm"]
        res["compare"][lab] = m
        print(f"  {lab:16s} {m['steps']:4d} {m['B_norm']:6.2f} {m['S']:.3f} {m['R']:.2f} {m['L6/L5']:.2f} "
              f"{m['L18/L17']:.2f} {m['cos_grad']:.3f} " + " ".join(f"{m['final_acc'][k]:.2f}" for k in CONDS))
    if low:
        # Hop count applied on the untrained conditions (reuses note 10's helper).
        import importlib
        s11 = importlib.import_module("11_e2_pilot_sgd")
        for d in low:
            s11.OUT = OUT
            run = load(d)
            hb = s11.hop_behaviour(d.name, max(run["ev"]))
            res["compare"][f"AdamW {run['cfg']['lr']:.1e}"]["hops_final"] = hb
            print(f"  hops applied ({d.name}):", json.dumps({c: hb[c] for c in ("k1", "k6", "reverse")}))
    (OUT / "controls_tables.json").write_text(json.dumps(res, indent=1, default=str))
    if low:
        plot(res, [k for k in res["compare"] if k.startswith("AdamW") and k != "AdamW 5.4e-4"][0])


def plot(res, low_lab):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    lr = float(low_lab.split()[1])
    labs = {low_lab: f"AdamW, rate {lr:.1e}".replace("e-05", "e-5"), "SGD mode 2.9e-4": "SGD mode, rate 2.9e-4"}
    col = ms.color_map(list(labs.values()))
    fig, ax = plt.subplots(figsize=figsize("blog"))
    ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
    ax.axhline(1.0, color="#bbbbc4", lw=1.0, zorder=1)
    for k, lab in labs.items():
        y = res["compare"][k]["per_layer_B"]
        mu = sum(y) / len(y)
        ax.plot(range(24), [v / mu for v in y], color=col[lab], label=lab)
        ms.dots(ax, list(range(24)), [v / mu for v in y], col[lab])
    ms.titles(ax, "Per-layer adapter norm ‖B‖ after training",
              ylabel="‖B‖ ÷ the run's mean over layers (grey line = 1)")
    ax.set_xlabel("Layer (shaded: looped layers 6 to 17)")
    ms.end_labels(ax)
    ms.save(fig, str(OUT / "profile_adamw_low_vs_sgd"))


def main():
    cmd = sys.argv[1]
    if cmd == "probe":
        probe(sys.argv[2], float(sys.argv[3]), sys.argv[4:])
    elif cmd == "adamw":
        adamw(sys.argv[2], float(sys.argv[3]))
    elif cmd == "check":
        check(sys.argv[2])
    elif cmd == "step1":
        print(step1(sys.argv[2]))
    elif cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
