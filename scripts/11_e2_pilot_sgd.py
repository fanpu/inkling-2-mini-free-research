"""E2 pilot: AdamW in "SGD mode" (eps far above every |g|) on pointer chasing k in {2,3,4}.

Adam's update is lr * m_hat / (sqrt(v_hat) + eps). With eps = 1e4, far above every gradient
entry, it is (lr/eps) * m_hat to within |g|/eps, i.e. SGD with EMA momentum 0.9 at the
effective rate lr_eff = lr/eps. Same setup as scripts 09 and 10 (rank 8, LoRA seed 0,
data seed 1, batch 32, evals at 0, 10, 25, 50, 100, 150, early stop when k = 2, 3, 4 are
all above 0.95, 150-step budget). Weight decay 0.

Usage:
  .venv/bin/python scripts/11_e2_pilot_sgd.py test LR_EFF [EPS]          # 5 steps, export after step 1
  .venv/bin/python scripts/11_e2_pilot_sgd.py run NAME LR_EFF [--eps=1e4] [--resume]
  .venv/bin/python scripts/11_e2_pilot_sgd.py regime NAME               # step-1 linearity check
  .venv/bin/python scripts/11_e2_pilot_sgd.py table SGD_NAME            # 3-optimizer comparison
  .venv/bin/python scripts/11_e2_pilot_sgd.py plot SGD_NAME             # 3-optimizer overlay figures

Outputs under artifacts/e2_pilot/<NAME>/ and figures in artifacts/e2_pilot/.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_lib as tl  # noqa: E402

assert tl.MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
OUT = Path("artifacts/e2_pilot")
EPS = 1e4
ADAMW, DIMUON = "adamw_lr5.4e-4", "dimuon_lr1.6e-2"
COND_NAMES = {"k1": "k = 1", "k2": "k = 2", "k3": "k = 3", "k4": "k = 4", "k6": "k = 6 (held out)",
              "reverse": "Reverse lookup"}
DIVERGE_LOSS = 10.0  # nats per example; the untrained mean is 4.98 on the first batch


def base_cfg(name, lr_eff, steps, eps=EPS, **kw):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="adamw", lr=lr_eff * eps, rank=8,
                     lora_seed=0, data_seed=1, train_conds=["k2", "k3", "k4"], batch=32, steps=steps,
                     eval_steps=tl.eval_schedule(steps), adam_eps=eps, adam_beta1=0.9, adam_beta2=0.95,
                     weight_decay=0.0, max_in_flight=16, diverge_loss=DIVERGE_LOSS, **kw)


def load_run(name):
    d = OUT / name
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    cfg = next(r["config"] for r in recs if r["type"] == "start")
    ev = {r["step"]: r["summary"] for r in recs if r["type"] == "eval"}
    st = json.loads((d / "adapter_stats.json").read_text()) if (d / "adapter_stats.json").exists() else None
    return cfg, ev, st, recs


def label(cfg):
    if cfg["optimizer"] == "dimuon":
        return f"Dimuon (lr {cfg['lr']:.2g})"
    if cfg["adam_eps"] >= 1:
        return f"SGD mode (lr ÷ eps {cfg['lr'] / cfg['adam_eps']:.2g})"
    return f"AdamW (lr {cfg['lr']:.2g})"


def regime(name):
    """Step-1 check. At B = 0 Adam's first step is exactly |dB| = lr |g| / (|g| + eps), so
    u = |dB|/lr = |g|/(|g|+eps) and the step falls short of the linear (SGD) prediction by the
    factor 1 - u. Reads the step-1 export; B0 = 0 so dB = B1."""
    d = OUT / name
    cfg = json.loads((d / "config.json").read_text())
    s1 = json.loads((d / "adapter_stats_step1.json").read_text())
    lr, eps = cfg["lr"], cfg["adam_eps"]
    umax = max(r["B_absmax"] for r in s1["rows"]) / lr
    tot2 = sum(r["B_fro"] ** 2 for r in s1["rows"])
    lay2 = sum(r["B_fro"] ** 2 for r in s1["rows"] if r["layer"] is not None)
    per = [r["B_fro"] / (lr / eps) for r in s1["rows"] if r["layer"] is not None]
    out = {"lr": lr, "eps": eps, "lr_eff": lr / eps, "max_u": umax,
           "max_abs_g_inferred": eps * umax / (1 - umax),
           "step1_norm_all_incl_lm_head": math.sqrt(tot2), "step1_norm_layers": math.sqrt(lay2),
           "grad_norm_layers_linear": math.sqrt(lay2) / (lr / eps),
           "grad_norm_all_linear": math.sqrt(tot2) / (lr / eps),
           "per_tensor_grad_norm_min_median_max": [min(per), sorted(per)[len(per) // 2], max(per)],
           "lm_head_grad_norm_linear": [r["B_fro"] / (lr / eps) for r in s1["rows"] if r["layer"] is None]}
    (d / "regime_step1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}))
    return out


def layer_metrics(st, q):
    out = {}
    for g in ("all", "attention", "mlp_experts") + tuple(tl.MODS):
        y = st["summary"][q][g]["per_layer_norm"]
        loop = [y[i] for i in tl.LOOP]
        unl = [y[i] for i in range(24) if i not in tl.LOOP]
        out[g] = {"share": st["summary"][q][g]["looped_share"],
                  "ratio": (sum(loop) / len(loop)) / (sum(unl) / len(unl)),
                  "step6": y[6] / y[5], "step18": y[18] / y[17]}
    return out


def hop_behaviour(name, step):
    """Hop count j applied by each greedy answer: the smallest j in -1..8 with f^j(x) equal to
    the answer, where x is the start letter (main tasks) or the shown letter y (reverse,
    whose correct answer is j = -1). Eval examples are regenerated from seed 2026 and matched
    to the saved rows by order."""
    rows = json.loads((OUT / name / f"eval_{step:05d}.json").read_text())
    out = {}
    for c in ("k1", "k2", "k3", "k4", "k6", "reverse"):
        rs = [r for r in rows if r["cond"] == c]
        exs = tl.make_examples(c, len(rs), tl.EVAL_SEED)
        hist = {}
        for r, e in zip(rs, exs):
            assert r["answer"] == e.answer and r["path"] == e.path
            f = e.tables[e.table]
            x = e.path[0]
            j = None
            if c == "reverse" and {v: u for u, v in f.items()}[x] == r["greedy"]:
                j = -1
            else:
                y = x
                for jj in range(9):
                    if y == r["greedy"]:
                        j = jj
                        break
                    y = f[y]
            hist[str(j)] = hist.get(str(j), 0) + 1
        out[c] = dict(sorted(hist.items(), key=lambda t: (t[0] == "None", t[0])))
    return out


def table(sgd):
    names = [ADAMW, DIMUON, sgd]
    runs = {n: load_run(n) for n in names}
    res = {"accuracy": {}, "metrics": {}, "hops_final": {}, "sv": {}}
    print("accuracy by step")
    for n, (cfg, ev, st, recs) in runs.items():
        res["accuracy"][n] = {s: {c: ev[s][c]["acc"] for c in COND_NAMES} for s in sorted(ev)}
        for s in sorted(ev):
            print(f"  {n:22s} {s:4d} " + " ".join(f"{c}:{ev[s][c]['acc']:.2f}" for c in COND_NAMES))
    for q in ("B_fro", "eff_fro"):
        print(q, "share / ratio R / L6/L5 / L18/L17")
        res["metrics"][q] = {}
        for n, (_, _, st, _) in runs.items():
            m = layer_metrics(st, q)
            res["metrics"][q][n] = m
            for g in ("all", "attention", "mlp_experts"):
                v = m[g]
                print(f"  {n:22s} {g:12s} {v['share']:.3f} {v['ratio']:.2f} {v['step6']:.2f} {v['step18']:.2f}")
            mods = [m[x]["share"] for x in tl.MODS]
            print(f"  {n:22s} per-module share range {min(mods):.3f} to {max(mods):.3f}")
        print()
    print("final-step greedy path index on untrained conditions (index = hops applied; None = off path)")
    for n, (cfg, ev, st, _) in runs.items():
        last = max(ev)
        res["hops_final"][n] = {"step": last, **hop_behaviour(n, last)}
        print(f"  {n:22s} step {last}: " + json.dumps({c: res['hops_final'][n][c] for c in ('k1', 'k6', 'reverse')}))
    for n, (cfg, ev, st, _) in runs.items():
        print(f"  {n:22s} ||B||_F layers {math.sqrt(sum(r['B_fro']**2 for r in st['rows'] if r['layer'] is not None)):.2f}"
              f"  lm_head B {[round(r['B_fro'], 3) for r in st['rows'] if r['layer'] is None]}"
              f"  last step {st['step']}")
    st = runs[sgd][2]
    svr = [r for r in st["rows"] if "B_sv" in r]
    if svr:
        def stable_rank(sv):
            return sum(x * x for x in sv) / max(sv) ** 2 if max(sv) > 0 else None
        lo = [stable_rank(r["B_sv"]) for r in svr if r["layer"] in tl.LOOP]
        un = [stable_rank(r["B_sv"]) for r in svr if r["layer"] not in tl.LOOP]
        res["sv"] = {"stable_rank_looped_mean": sum(lo) / len(lo), "stable_rank_unlooped_mean": sum(un) / len(un)}
        print("SGD-mode attention B stable rank (sum s^2 / s_max^2), mean looped / unlooped:",
              f"{res['sv']['stable_rank_looped_mean']:.2f} / {res['sv']['stable_rank_unlooped_mean']:.2f}")
    (OUT / sgd / "comparison.json").write_text(json.dumps(res, indent=1, default=str))


def plot(sgd):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    names = [ADAMW, DIMUON, sgd]
    runs = {n: load_run(n) for n in names}
    lab = {n: label(runs[n][0]) for n in names}
    ocol = ms.color_map(list(lab.values()))

    # Figure 1: per-layer adapter norm, each optimizer's profile divided by its own mean.
    qs = {"B_fro": "‖B‖", "eff_fro": "‖(α/r)·BA‖"}
    fig, axes = plt.subplots(1, 2, figsize=figsize("blog_wide"), sharey=True)
    for ax, (q, qlab) in zip(axes, qs.items()):
        ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
        ax.axhline(1.0, color="#bbbbc4", lw=1.0, zorder=1)  # 1 = the run's own mean, named on the y axis
        for n in names:
            y = runs[n][2]["summary"][q]["all"]["per_layer_norm"]
            mu = sum(y) / len(y)
            ax.plot(range(24), [v / mu for v in y], color=ocol[lab[n]], label=lab[n])
        ms.titles(ax, f"{qlab}, all modules", ylabel="Norm ÷ its run's mean over layers (grey line = 1)" if ax is axes[0] else None,
                  small=True)
        ax.set_xlabel("Layer (shaded: looped layers 6 to 17)")
    ms.end_labels(axes[1])
    ms.fig_titles(fig, "Per-layer adapter norm under three optimizers")
    ms.save(fig, str(OUT / "compare3_adapter_norm_vs_layer"))

    # Figure 2: accuracy against step, one panel per eval condition.
    fig, axes = plt.subplots(2, 3, figsize=figsize("blog_wide", nrows=2, ncols=3), sharex=True, sharey=True)
    for ax, (c, clab) in zip(axes.flat, COND_NAMES.items()):
        for n in names:
            ev = runs[n][1]
            steps = sorted(ev)
            ys = [ev[s][c]["acc"] for s in steps]
            ax.plot(steps, ys, color=ocol[lab[n]])
            ms.dots(ax, steps, ys, ocol[lab[n]])
        ms.titles(ax, clab, ylabel="Greedy accuracy" if ax in axes[:, 0] else None, small=True,
                  key=ocol if ax is axes[0, 2] else None)
        if ax in axes[1]:
            ax.set_xlabel("Training step")
    ms.fig_titles(fig, "Eval accuracy (30 examples) while training on k = 2, 3, 4")
    ms.save(fig, str(OUT / "compare3_eval_accuracy_vs_step"))


def main():
    cmd = sys.argv[1]
    if cmd == "test":
        lr_eff = float(sys.argv[2])
        eps = float(sys.argv[3]) if len(sys.argv) > 3 else EPS
        cfg = base_cfg("sgd_test5", lr_eff, 5, eps=eps, eval_n=5, save_state=False, export_steps=[1])
        cfg.eval_steps = [0, 5]
        s = tl.run(cfg)
        print(json.dumps(s["timing"]), s.get("stopped"))
        regime("sgd_test5")
    elif cmd == "run":
        name, lr_eff = sys.argv[2], float(sys.argv[3])
        eps = EPS
        for a in sys.argv[4:]:
            if a.startswith("--eps="):
                eps = float(a.split("=")[1])
        cfg = base_cfg(name, lr_eff, 150, eps=eps, export_steps=[1])
        s = tl.run(cfg, resume="--resume" in sys.argv)
        print(json.dumps(s["timing"]), json.dumps(s.get("looped_share")), s.get("stopped"))
        regime(name)
    elif cmd == "regime":
        regime(sys.argv[2])
    elif cmd == "table":
        table(sys.argv[2])
    elif cmd == "plot":
        plot(sys.argv[2])


if __name__ == "__main__":
    main()
