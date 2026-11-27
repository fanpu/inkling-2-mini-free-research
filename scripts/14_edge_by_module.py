"""SGD-mode stability edge by module group, and the anatomy of an all-modules collapse (tests H2).

SGD mode = AdamW with eps = 1e4, lr = lr_eff * eps, weight decay 0 (note 10). Every run: rank 8,
LoRA seed 0, data seed 1, batch 32 over k = 2, 3, 4, early stop off, no divergence stop.

Part A. Edge by module group, unembedding off throughout.
  attn  = train_attn only (train_mlp=False), mlp = experts only (train_attn=False).
  40-step probes, evals at 0, 10, 20, 40 (30 examples per condition), adapter exported after
  step 1 and at the end. Rates 8.7e-4, 1.5e-3, 2.6e-3 in increasing order, stopping the group at
  the first collapse. If 8.7e-4 collapses, 5e-4 is added. If nothing collapses by 2.6e-3, 4.5e-3
  is added. After the group's first run, the step-1 export is checked for the switched-off modules
  (absent or B = 0); the group stops if they moved.
Part B. All modules (unembedding on), lr_eff 8.7e-4 and 5e-4, 10 steps, exports after steps
  1, 2, 3, 5, 10, one eval at step 10.

Classification of a probe (note 11's rule plus a plateau guard):
  at chance   = every condition at or below 0.11 at the last eval
  collapsed   = at chance, mean loss over the last 5 steps within 0.5 of ln 26, and every one of
                the last 25 steps within 0.5 of ln 26
  undecided   = at chance but not collapsed
  learning    = not at chance

Usage:
  .venv/bin/python scripts/14_edge_by_module.py group attn|mlp
  .venv/bin/python scripts/14_edge_by_module.py mid       # 1.1e-3 for both groups (added)
  .venv/bin/python scripts/14_edge_by_module.py partb
  .venv/bin/python scripts/14_edge_by_module.py check NAME
  .venv/bin/python scripts/14_edge_by_module.py analyze
Each run skips itself if its final adapter stats exist. Outputs under artifacts/e2_edge/.
"""
import json, math, sys, traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_lib as tl  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert tl.MODEL_ID == MODEL_ID, "free tier only (D-001)"
OUT = Path("artifacts/e2_edge")
CTRL = Path("artifacts/e2_controls")
EPS = 1e4
CONDS = ["k1", "k2", "k3", "k4", "k6", "reverse"]
LN26 = math.log(26)
RATES = [8.7e-4, 1.5e-3, 2.6e-3]
LOW, HIGH = 5e-4, 4.5e-3
MID = 1.1e-3  # added after both groups learned at 8.7e-4 and collapsed at 1.5e-3
PB_RATES = [8.7e-4, 5e-4]
PB_STEPS = [1, 2, 3, 5, 10]
GROUPS = {"attn": dict(train_attn=True, train_mlp=False), "mlp": dict(train_attn=False, train_mlp=True)}
OFF_PREFIX = {"attn": "mlp", "mlp": "attn"}  # module prefix that must not move


def rname(group, lr_eff):
    return f"{group}_lreff{lr_eff:.1e}".replace("e-0", "e-").replace(".0e", "e")


def pbname(lr_eff):
    return "partb_" + rname("all", lr_eff)


def done(name):
    return (OUT / name / "adapter_stats.json").exists()


def base_cfg(name, **kw):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="adamw", rank=8, lora_seed=0, data_seed=1,
                     train_conds=["k2", "k3", "k4"], batch=32, early_stop_acc=2.0, adam_eps=EPS,
                     adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.0, max_in_flight=16,
                     diverge_loss=None, save_state=False, export_step0=False, **kw)


# ---------------------------------------------------------------- loading and classification

def load(d: Path):
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    cfg = next(r["config"] for r in recs if r["type"] == "start")
    ev = {r["step"]: r["summary"] for r in recs if r["type"] == "eval"}
    loss = {r["step"]: r["loss_mean"] for r in recs if r["type"] == "train"}
    end = next((r for r in recs if r["type"] == "end"), None)
    ex = {}
    for f in d.glob("adapter_stats_step*.json"):
        ex[int(f.stem.split("step")[1])] = json.loads(f.read_text())
    if (d / "adapter_stats.json").exists():
        fin = json.loads((d / "adapter_stats.json").read_text())
        ex.setdefault(fin["step"], fin)
    errs = json.loads((d / "errors.json").read_text()) if (d / "errors.json").exists() else []
    return {"cfg": cfg, "ev": ev, "loss": loss, "exports": ex, "end": end, "errors": errs}


def classify(run):
    last = max(run["ev"])
    macc = max(run["ev"][last][c]["acc"] for c in CONDS)
    ls = [run["loss"][s] for s in sorted(run["loss"])]
    tail5 = sum(ls[-5:]) / 5
    flat25 = len(ls) >= 25 and all(abs(x - LN26) < 0.5 for x in ls[-25:])
    if macc > 0.11:
        verdict = "learning"
    elif abs(tail5 - LN26) < 0.5 and flat25:
        verdict = "collapsed"
    else:
        verdict = "undecided"
    return {"verdict": verdict, "max_acc_last": macc, "last_eval": last, "tail5": tail5, "flat25": flat25}


def check(name, group=None):
    """Modules present in the step-1 export, and B norm of the switched-off group and lm_head."""
    run = load(OUT / name)
    s1 = run["exports"][1]
    mods = sorted({r["module"] for r in s1["rows"]})
    lm = [r["B_fro"] for r in s1["rows"] if r["layer"] is None]
    out = {"modules": mods, "lm_head_B": lm, "file_bytes": s1.get("file_bytes")}
    if group:
        off = [r["B_fro"] for r in s1["rows"] if r["layer"] is not None and r["module"].startswith(OFF_PREFIX[group])]
        on = [r["B_fro"] for r in s1["rows"] if r["layer"] is not None and not r["module"].startswith(OFF_PREFIX[group])]
        out.update(off_present=len(off), off_B_max=max(off) if off else None,
                   on_B_norm=math.sqrt(sum(x * x for x in on)))
        out["ok"] = (not off or max(off) == 0.0) and (not lm or max(lm) == 0.0)
    (OUT / name / "module_check.json").write_text(json.dumps(out, indent=1))
    print(f"  check {name}: {len(mods)} modules {mods}, lm_head {lm}, "
          + (f"off-group tensors {out['off_present']} max B {out['off_B_max']}, on-group |B| {out['on_B_norm']:.3f}, ok {out['ok']}" if group else ""))
    return out


# ---------------------------------------------------------------- runs

def probe(group, lr_eff):
    name = rname(group, lr_eff)
    if not done(name):
        print(f"=== {name}", flush=True)
        cfg = base_cfg(name, lr=lr_eff * EPS, steps=40, eval_steps=[0, 10, 20, 40], export_steps=[1],
                       train_unembed=False, **GROUPS[group])
        s = tl.run(cfg)
        print(f"  timing {json.dumps(s['timing'])}", flush=True)
    c = classify(load(OUT / name))
    print(f"  {name}: {c['verdict']} max acc {c['max_acc_last']:.2f} tail5 {c['tail5']:.2f} flat25 {c['flat25']}", flush=True)
    return name, c["verdict"]


def group_chain(group):
    verdicts = {}
    for i, r in enumerate(RATES):
        name, v = probe(group, r)
        verdicts[r] = v
        if i == 0 and not check(name, group)["ok"]:
            print(f"  STOP: switched-off modules moved in {name}", flush=True)
            return
        if v == "collapsed":
            break
    if verdicts.get(RATES[0]) == "collapsed":
        probe(group, LOW)
    elif all(v != "collapsed" for v in verdicts.values()):
        probe(group, HIGH)


def partb():
    for r in PB_RATES:
        name = pbname(r)
        if done(name):
            print(f"  {name} done", flush=True)
            continue
        print(f"=== {name}", flush=True)
        cfg = base_cfg(name, lr=r * EPS, steps=10, eval_steps=[10], export_steps=PB_STEPS)
        s = tl.run(cfg)
        print(f"  timing {json.dumps(s['timing'])}", flush=True)


# ---------------------------------------------------------------- analysis

def per_layer(st, prefix=None):
    p = [0.0] * 24
    for r in st["rows"]:
        if r["layer"] is not None and (prefix is None or r["module"].startswith(prefix)):
            p[r["layer"]] += r["B_fro"] ** 2
    return [math.sqrt(x) for x in p]


def share(p):
    t = sum(x * x for x in p)
    return sum(p[i] ** 2 for i in tl.LOOP) / t if t else None


def norm(p):
    return math.sqrt(sum(x * x for x in p))


RANGES = {"0-5": range(0, 6), "6-17": range(6, 18), "18-23": range(18, 24)}


def range_norm(p, rg):
    return math.sqrt(sum(p[i] ** 2 for i in rg))


def analyze():
    res = {"partA": [], "partB": {}}
    print("Part A: run | lr_eff | loss@2 | min loss | tail5 | max acc @10/20/40 | verdict | step-1 |B| (S)")
    for d in sorted(OUT.iterdir()):
        if not (d.is_dir() and d.name[:4] in ("attn", "mlp_") and done(d.name)):
            continue
        run = load(d)
        c = classify(run)
        cfg = run["cfg"]
        p1 = per_layer(run["exports"][1])
        ls = [run["loss"][s] for s in sorted(run["loss"])]
        r = {"run": d.name, "group": d.name.split("_")[0], "lr_eff": cfg["lr"] / cfg["adam_eps"], **c,
             "loss": ls, "loss_step2": run["loss"].get(2), "min_loss_after1": min(ls[1:]),
             "max_acc": {s: max(run["ev"][s][k]["acc"] for k in CONDS) for s in sorted(run["ev"])},
             "acc_last": {k: run["ev"][c["last_eval"]][k]["acc"] for k in CONDS},
             "step1_B": norm(p1), "step1_S": share(p1), "final_B": norm(per_layer(run["exports"][max(run["exports"])])),
             "errors": run["errors"], "sdk_warnings": (run["end"] or {}).get("sdk_warnings"),
             "module_check": json.loads((d / "module_check.json").read_text()) if (d / "module_check.json").exists() else None}
        res["partA"].append(r)
        print(f"  {d.name:18s} {r['lr_eff']:.1e} {r['loss_step2']:.2f} {r['min_loss_after1']:.2f} {r['tail5']:.2f} "
              + "/".join(f"{v:.2f}" for s, v in r["max_acc"].items() if s > 0)
              + f" {c['verdict']:9s} {r['step1_B']:.3f} ({r['step1_S']:.3f}) errors {len(r['errors'])}")
    res["partA"].sort(key=lambda r: (r["group"], r["lr_eff"]))

    # Part B
    pb = {}
    for r in PB_RATES:
        d = OUT / pbname(r)
        if not done(d.name):
            continue
        run = load(d)
        ex = run["exports"]
        steps = [s for s in PB_STEPS if s in ex]
        prof = {s: {"all": per_layer(ex[s]), "attn": per_layer(ex[s], "attn"), "mlp": per_layer(ex[s], "mlp")} for s in steps}
        mods = sorted({x["module"] for x in ex[steps[0]]["rows"] if x["layer"] is not None})
        permod = {s: {m: per_layer(ex[s], m) for m in mods} for s in steps}
        lm = {s: [x["B_fro"] for x in ex[s]["rows"] if x["layer"] is None] for s in steps}
        ev = run["ev"].get(10)
        pb[r] = {"steps": steps, "loss": [run["loss"][s] for s in sorted(run["loss"])],
                 "S": {s: share(prof[s]["all"]) for s in steps},
                 "S_attn": {s: share(prof[s]["attn"]) for s in steps},
                 "S_mlp": {s: share(prof[s]["mlp"]) for s in steps},
                 "B": {s: norm(prof[s]["all"]) for s in steps},
                 "B_attn": {s: norm(prof[s]["attn"]) for s in steps},
                 "B_mlp": {s: norm(prof[s]["mlp"]) for s in steps},
                 "lm_head_B": lm,
                 "range_B": {s: {g: {k: range_norm(prof[s][g], rg) for k, rg in RANGES.items()} for g in ("all", "attn", "mlp")} for s in steps},
                 "per_layer": {s: prof[s] for s in steps}, "per_module": permod,
                 "acc10": {k: ev[k]["acc"] for k in CONDS} if ev else None,
                 "errors": run["errors"], "sdk_warnings": (run["end"] or {}).get("sdk_warnings"),
                 "step10_vs_final_same": None}
    res["partB"] = {f"{k:.1e}": v for k, v in pb.items()}
    if len(pb) == 2:
        hi, lo = pb[PB_RATES[0]], pb[PB_RATES[1]]
        steps = [s for s in PB_STEPS if s in hi["steps"] and s in lo["steps"]]
        ratio = PB_RATES[0] / PB_RATES[1]
        comp = {}
        # growth g_l(t) = |B_l(t)| / |B_l(1)|; ratio of growths = g_hi / g_lo
        for grp in ("all", "attn", "mlp"):
            comp[grp] = {}
            for s in steps:
                gh = [a / b if b else None for a, b in zip(hi["per_layer"][s][grp], hi["per_layer"][1][grp])]
                gl = [a / b if b else None for a, b in zip(lo["per_layer"][s][grp], lo["per_layer"][1][grp])]
                comp[grp][s] = {"growth_hi": gh, "growth_lo": gl,
                                "growth_ratio": [a / b if (a and b) else None for a, b in zip(gh, gl)],
                                "norm_ratio": [a / b if b else None for a, b in zip(hi["per_layer"][s][grp], lo["per_layer"][s][grp])]}
        # per-module growth ratio at step 10, by layer range
        modcomp = {}
        for m in hi["per_module"][1]:
            modcomp[m] = {}
            for k, rg in RANGES.items():
                def gr(run, s):
                    a, b = range_norm(run["per_module"][s][m], rg), range_norm(run["per_module"][1][m], rg)
                    return a / b if b else None
                gh, gl = gr(hi, steps[-1]), gr(lo, steps[-1])
                modcomp[m][k] = {"growth_hi": gh, "growth_lo": gl, "ratio": gh / gl if gh and gl else None}
        res["partB_compare"] = {"lr_ratio": ratio, "per_layer": comp, "per_module_range_step10": modcomp}
        print(f"\nPart B (lr ratio {ratio:.2f})")
        print("  step | |B| hi lo | S hi lo | S_attn hi lo | S_mlp hi lo")
        for s in steps:
            print(f"  {s:3d} | {hi['B'][s]:.3f} {lo['B'][s]:.3f} | {hi['S'][s]:.3f} {lo['S'][s]:.3f} | "
                  f"{hi['S_attn'][s]:.3f} {lo['S_attn'][s]:.3f} | {hi['S_mlp'][s]:.3f} {lo['S_mlp'][s]:.3f}")
        print("  lm_head B hi:", {s: [round(x, 3) for x in v] for s, v in hi["lm_head_B"].items()},
              " lo:", {s: [round(x, 3) for x in v] for s, v in lo["lm_head_B"].items()})
        for grp in ("all", "attn", "mlp"):
            gr = comp[grp][steps[-1]]["growth_ratio"]
            print(f"  growth ratio step1->10 ({grp}) per layer:", " ".join(f"{x:.2f}" if x else "-" for x in gr))
        for grp in ("all", "attn", "mlp"):
            for k, rg in RANGES.items():
                vals = []
                for s in steps:
                    gh = range_norm(hi["per_layer"][s][grp], rg) / range_norm(hi["per_layer"][1][grp], rg)
                    gl = range_norm(lo["per_layer"][s][grp], rg) / range_norm(lo["per_layer"][1][grp], rg)
                    vals.append(f"{s}:{gh:.2f}/{gl:.2f}={gh/gl:.2f}")
                print(f"  range growth hi/lo {grp:4s} {k:5s} " + " ".join(vals))
        print("  per-module growth ratio at step 10, layers 0-5 / 6-17 / 18-23:")
        for m, v in modcomp.items():
            print(f"    {m:16s} " + " ".join(f"{v[k]['ratio']:.2f}" if v[k]["ratio"] else "-" for k in RANGES))
        print("  loss hi:", " ".join(f"{x:.2f}" for x in hi["loss"]))
        print("  loss lo:", " ".join(f"{x:.2f}" for x in lo["loss"]))
        print("  acc@10 hi:", hi["acc10"], " lo:", lo["acc10"])
    (OUT / "edge_tables.json").write_text(json.dumps(res, indent=1, default=str))
    plot_loss(res)
    if len(pb) == 2:
        plot_partb(pb)


def plot_loss(res):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    panels = [("attn", "Attention only"), ("mlp", "Experts only")]
    allr = sorted({r["lr_eff"] for r in res["partA"]})
    # note 11's runs with both groups and no unembedding, for reference
    ref = []
    for d in sorted(CTRL.glob("sgd_noemb_lreff*")):
        if (d / "summary.json").exists():
            run = load(d)
            ref.append((run["cfg"]["lr"] / run["cfg"]["adam_eps"], [run["loss"][s] for s in sorted(run["loss"])]))
    allr = sorted(set(allr) | {r for r, _ in ref})
    shades = ms.ordered_colors(len(allr))
    col = {r: shades[i] for i, r in enumerate(allr)}
    fig, axes = plt.subplots(1, 3, figsize=figsize("blog_wide", ncols=2), sharey=True)
    for ax, (g, title) in zip(axes, panels + [("both", "Both groups, no unembedding (note 11)")]):
        ax.axhline(LN26, color="#bbbbc4", lw=1.0, zorder=1)
        rows = [(r["lr_eff"], r["loss"]) for r in res["partA"] if r["group"] == g] if g != "both" else ref
        for lr, ls in sorted(rows):
            ax.plot(range(1, len(ls) + 1), ls, color=col[lr], label=f"{lr:.1e}".replace("e-04", "e-4").replace("e-03", "e-3"))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylim(0, 12)
        ms.end_labels(ax)
    axes[0].set_ylabel("Training loss (nats), grey line = ln 26")
    ms.fig_titles(fig, "SGD-mode training loss by module group and effective learning rate")
    ms.save(fig, str(OUT / "loss_by_module_group"))


def plot_partb(pb):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    shades = ms.ordered_colors(len(PB_STEPS))
    w, h = figsize("blog_wide", ncols=2)
    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    axes[1].sharey(axes[0])
    for j, (ax, r) in enumerate(zip(axes, PB_RATES)):
        ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
        for i, s in enumerate(PB_STEPS):
            y = pb[r]["per_layer"][s]["all"]
            lab = f"Step {s}" if (j == 1 and s in (PB_STEPS[0], PB_STEPS[-1])) else None
            ax.plot(range(24), y, color=shades[i], label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("Layer (shaded: looped 6 to 17)")
        ax.set_title(f"Effective rate {r:.1e}".replace("e-04", "e-4"), fontsize=9)
    ms.end_labels(axes[1])
    axes[0].set_ylabel("‖B‖ per layer, all modules")
    ax = axes[2]
    ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
    ms.ref_line(ax, PB_RATES[0] / PB_RATES[1], "Ratio of the two rates (1.74)", side="left")
    for i, s in enumerate(PB_STEPS):
        hi, lo = pb[PB_RATES[0]]["per_layer"][s]["all"], pb[PB_RATES[1]]["per_layer"][s]["all"]
        ax.plot(range(24), [a / b for a, b in zip(hi, lo)], color=shades[i])
    ax.set_ylim(1.68, 1.82)
    ax.set_xlabel("Layer (shaded: looped 6 to 17)")
    ax.set_title("‖B‖ at 8.7e-4 ÷ ‖B‖ at 5.0e-4", fontsize=9)
    ms.fig_titles(fig, "Per-layer adapter norm grows by the same factor at both rates in the first 10 steps")
    ms.save(fig, str(OUT / "partb_B_norm_by_layer"))


def main():
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "group":
        try:
            group_chain(sys.argv[2])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            raise
    elif cmd == "mid":  # extension: split (8.7e-4, 1.5e-3] when both groups share it
        for g in ("attn", "mlp"):
            probe(g, MID)
    elif cmd == "partb":
        partb()
    elif cmd == "check":
        check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
