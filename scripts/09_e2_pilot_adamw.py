"""E2 pilot: AdamW LoRA training on pointer chasing k in {2,3,4} (harness in train_lib.py).

Measures step throughput, whether k = 2, 3, 4 are learnable, and where the trained
adapter's weight sits across layers. Eval on 30 seed-2026 examples each of k = 1, 2,
3, 4, 6 and the reverse control at steps 0, 10, 25, 50, 100 and every 100 after.

Usage:
  .venv/bin/python scripts/09_e2_pilot_adamw.py test                 # 5 steps, small eval
  .venv/bin/python scripts/09_e2_pilot_adamw.py run NAME LR STEPS [--resume]
  .venv/bin/python scripts/09_e2_pilot_adamw.py plot NAME [NAME ...]

Outputs under artifacts/e2_pilot/<NAME>/ (see train_lib.py) and figures in artifacts/e2_pilot/.
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


def base_cfg(name, lr, steps, **kw):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="adamw", lr=lr, rank=8, lora_seed=0,
                     data_seed=1, train_conds=["k2", "k3", "k4"], batch=32, steps=steps,
                     eval_steps=tl.eval_schedule(steps), weight_decay=0.0, max_in_flight=16, **kw)


def load_run(name):
    d = OUT / name
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    return recs, json.loads((d / "adapter_stats.json").read_text()) if (d / "adapter_stats.json").exists() else None


def plot(names):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    colors = ms.color_map(list(COND_NAMES.values()))
    runs = {n: load_run(n) for n in names}

    # Figure 1: eval accuracy against step, one panel per run.
    fig, axes = plt.subplots(1, len(names), figsize=figsize("blog" if len(names) == 1 else "blog_wide"),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], names):
        recs, _ = runs[n]
        ev = {}
        for r in recs:
            if r["type"] == "eval":
                ev[r["step"]] = r["summary"]
        steps = sorted(ev)
        for c, lab in COND_NAMES.items():
            ys = [ev[s][c]["acc"] for s in steps if c in ev[s]]
            ax.plot(steps, ys, color=colors[lab], label=lab)
            ms.dots(ax, steps, ys, colors[lab])
        ax.set_xlabel("Training step")
        cfg = next(r["config"] for r in recs if r["type"] == "start")
        ms.titles(ax, f"Learning rate {cfg['lr']:.2g}" if len(names) > 1 else
                  "Eval accuracy during AdamW training on k = 2, 3, 4",
                  ylabel="Greedy accuracy (30 examples)" if ax is axes[0][0] else None)
        ms.end_labels(ax)
    if len(names) > 1:
        ms.fig_titles(fig, "Eval accuracy during AdamW training on k = 2, 3, 4")
    ms.save(fig, str(OUT / "eval_accuracy_vs_step"))

    # Figure 2: per-layer adapter norm (effective update scale * ||BA||_F), by module group.
    groups = {"Attention": "attention", "MLP experts": "mlp_experts"}
    gcol = ms.color_map(["Attention", "MLP experts"])
    fig, axes = plt.subplots(1, len(names), figsize=figsize("blog" if len(names) == 1 else "blog_wide"),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], names):
        _, st = runs[n]
        if st is None:
            continue
        ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
        for lab, g in groups.items():
            y = st["summary"]["eff_fro"][g]["per_layer_norm"]
            ax.plot(range(24), y, color=gcol[lab], label=lab)
        ax.set_xlabel("Layer (shaded: looped layers 6 to 17)")
        ms.titles(ax, f"Learning rate {st.get('lr', '')}" if False else
                  ("Trained adapter update norm by layer" if len(names) == 1 else n),
                  ylabel="Update norm ‖(α/r)·BA‖ (Frobenius)" if ax is axes[0][0] else None)
        ms.end_labels(ax)
    if len(names) > 1:
        ms.fig_titles(fig, "Trained adapter update norm by layer")
    ms.save(fig, str(OUT / "adapter_norm_vs_layer"))


def main():
    cmd = sys.argv[1]
    if cmd == "test":
        cfg = base_cfg("test5", 5.4e-4, 5, eval_n=5, save_state=False)
        cfg.eval_steps = [0, 5]
        s = tl.run(cfg)
        print(json.dumps(s["timing"]), json.dumps(s.get("looped_share")))
    elif cmd == "run":
        name, lr, steps = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        cap = None
        for a in sys.argv[5:]:
            if a.startswith("--cap="):
                cap = float(a.split("=")[1])
        cfg = base_cfg(name, lr, steps, max_wall_seconds=cap)
        s = tl.run(cfg, resume="--resume" in sys.argv)
        print(json.dumps(s["timing"]), json.dumps(s.get("looped_share")), s.get("stopped"))
    elif cmd == "plot":
        plot(sys.argv[2:])


if __name__ == "__main__":
    main()
