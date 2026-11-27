"""E2 pilot: Dimuon LoRA training on pointer chasing k in {2,3,4}.

Same config as scripts/09_e2_pilot_adamw.py (rank 8, LoRA seed 0, data seed 1, batch 32,
150 steps, eval at 0, 10, 25, 50, 100, 150, early stop when k = 2, 3, 4 are all above 0.95).
Only the optimizer (Dimuon, which has no weight decay) and the learning rate change.

Usage:
  .venv/bin/python scripts/10_e2_pilot_dimuon.py test                   # 3 steps, small eval
  .venv/bin/python scripts/10_e2_pilot_dimuon.py run NAME LR STEPS [--resume]
  .venv/bin/python scripts/10_e2_pilot_dimuon.py plot ADAMW_NAME DIMUON_NAME
  .venv/bin/python scripts/10_e2_pilot_dimuon.py table ADAMW_NAME DIMUON_NAME

Outputs under artifacts/e2_pilot/<NAME>/ and comparison figures in artifacts/e2_pilot/.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_lib as tl  # noqa: E402

assert tl.MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
OUT = Path("artifacts/e2_pilot")
COND_NAMES = {"k1": "k = 1", "k2": "k = 2", "k3": "k = 3", "k4": "k = 4", "k6": "k = 6 (held out)",
              "reverse": "Reverse lookup"}
OPT_NAMES = {"adamw": "AdamW", "dimuon": "Dimuon"}


def base_cfg(name, lr, steps, **kw):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="dimuon", lr=lr, rank=8, lora_seed=0,
                     data_seed=1, train_conds=["k2", "k3", "k4"], batch=32, steps=steps,
                     eval_steps=tl.eval_schedule(steps), max_in_flight=16, **kw)


def load_run(name):
    d = OUT / name
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    cfg = next(r["config"] for r in recs if r["type"] == "start")
    ev = {r["step"]: r["summary"] for r in recs if r["type"] == "eval"}
    st = json.loads((d / "adapter_stats.json").read_text()) if (d / "adapter_stats.json").exists() else None
    return cfg, ev, st


def layer_metrics(st, q):
    """Looped share, looped/unlooped mean ratio of per-layer norm, and the steps at 5->6 and 17->18."""
    out = {}
    for g in ("all", "attention", "mlp_experts") + tuple(tl.MODS):
        y = st["summary"][q][g]["per_layer_norm"]
        loop = [y[i] for i in tl.LOOP]
        unl = [y[i] for i in range(24) if i not in tl.LOOP]
        out[g] = {"share": st["summary"][q][g]["looped_share"],
                  "ratio": (sum(loop) / len(loop)) / (sum(unl) / len(unl)),
                  "step6": y[6] / y[5], "step18": y[18] / y[17]}
    return out


def table(names):
    runs = {n: load_run(n) for n in names}
    print("accuracy by step")
    for n, (cfg, ev, _) in runs.items():
        for s in sorted(ev):
            print(f"  {n:22s} {s:4d} " + " ".join(f"{c}:{ev[s][c]['acc']:.2f}" for c in COND_NAMES))
    for q in ("B_fro", "eff_fro"):
        print(q, "share / loop-unloop mean ratio / L6/L5 / L18/L17")
        for n, (_, _, st) in runs.items():
            m = layer_metrics(st, q)
            for g, v in m.items():
                print(f"  {n:22s} {g:16s} {v['share']:.3f} {v['ratio']:.2f} {v['step6']:.2f} {v['step18']:.2f}")


def plot(names):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    runs = {n: load_run(n) for n in names}
    opt_lab = {n: f"{OPT_NAMES[c['optimizer']]} (lr {c['lr']:.2g})" for n, (c, _, _) in runs.items()}
    ocol = ms.color_map(list(opt_lab.values()))

    # Figure 1: accuracy against step, one panel per eval condition, optimizers overlaid.
    fig, axes = plt.subplots(2, 3, figsize=figsize("blog_wide", nrows=2, ncols=3), sharex=True, sharey=True)
    for ax, (c, clab) in zip(axes.flat, COND_NAMES.items()):
        for n, (_, ev, _) in runs.items():
            steps = sorted(ev)
            ys = [ev[s][c]["acc"] for s in steps]
            ax.plot(steps, ys, color=ocol[opt_lab[n]])
            ms.dots(ax, steps, ys, ocol[opt_lab[n]])
        ms.titles(ax, clab, ylabel="Greedy accuracy" if ax in axes[:, 0] else None, small=True,
                  key=ocol if ax is axes[0, 2] else None)
        if ax in axes[1]:
            ax.set_xlabel("Training step")
    ms.fig_titles(fig, "Eval accuracy (30 examples) while training on k = 2, 3, 4")
    ms.save(fig, str(OUT / "compare_eval_accuracy_vs_step"))

    # Figure 2: per-layer profile divided by its mean over layers, for B and the effective update.
    qs = {"B_fro": "‖B‖", "eff_fro": "‖(α/r)·BA‖"}
    groups = {"attention": "attention", "mlp_experts": "MLP experts"}
    fig, axes = plt.subplots(2, 2, figsize=figsize("blog_wide", nrows=2, ncols=2), sharex=True, sharey=True)
    for i, (q, qlab) in enumerate(qs.items()):
        for j, (g, glab) in enumerate(groups.items()):
            ax = axes[i, j]
            ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
            for n, (_, _, st) in runs.items():
                y = st["summary"][q][g]["per_layer_norm"]
                mu = sum(y) / len(y)
                ax.plot(range(24), [v / mu for v in y], color=ocol[opt_lab[n]],
                        label=opt_lab[n] if (i, j) == (0, 1) else None)
            if (i, j) == (0, 1):
                ms.ref_line(ax, 1.0, "Mean over layers", side="left")
            else:
                ax.axhline(1.0, color="#bbbbc4", lw=1.0, zorder=1)
            ms.titles(ax, f"{qlab}, {glab}", ylabel="Norm ÷ mean over layers" if j == 0 else None, small=True)
            if i == 1:
                ax.set_xlabel("Layer (shaded: looped layers 6 to 17)")
    ms.end_labels(axes[0, 1])
    ms.fig_titles(fig, "Frobenius norm of the trained adapter by layer")
    ms.save(fig, str(OUT / "compare_adapter_norm_vs_layer"))


def main():
    cmd = sys.argv[1]
    if cmd == "test":
        cfg = base_cfg("dimuon_test3", 1.6e-2, 3, eval_n=5, save_state=False, export_step0=False)
        cfg.eval_steps = [0, 3]
        s = tl.run(cfg)
        print(json.dumps(s["timing"]), json.dumps(s.get("looped_share")))
    elif cmd == "run":
        name, lr, steps = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        s = tl.run(base_cfg(name, lr, steps), resume="--resume" in sys.argv)
        print(json.dumps(s["timing"]), json.dumps(s.get("looped_share")), s.get("stopped"))
    elif cmd == "plot":
        plot(sys.argv[2:])
    elif cmd == "table":
        table(sys.argv[2:])


if __name__ == "__main__":
    main()
