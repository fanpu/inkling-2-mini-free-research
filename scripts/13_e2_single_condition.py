"""E2 single-condition SGD-mode runs (tests H3) and a LoRA-seed replication of the pilot.

SGD mode = AdamW with eps = 1e4 and lr = lr_eff * eps (note 10). Every run: rank 8, batch 32,
data seed 1, weight decay 0, fixed 100 steps (early stop off, no divergence stop), evals at
0, 10, 25, 50, 100 on k1, k2, k3, k4, k6, reverse (30 examples each), adapter exported at
step 0, after step 1 and at the end.

Runs (in order, each skipped if its final stats exist, resumed if it has a saved state):
  sgd_k2_s0      train k=2 only,        LoRA seed 0
  sgd_k4_s0      train k=4 only,        LoRA seed 0
  sgd_rev_s0     train reverse only,    LoRA seed 0
  sgd_mix_s1     train k=2,3,4 mix,     LoRA seed 1 (replicates artifacts/e2_pilot/sgd_lreff2.9e-4)
  sgd_k2_s1, sgd_k4_s1  (extras, run only with --extras)
A single-condition run that stays at chance (see at_chance) is rerun once at lr_eff 1.7e-4
as <name>_lr1.7e-4, before the extras.

Usage:
  .venv/bin/python scripts/13_e2_single_condition.py chain [--extras]
  .venv/bin/python scripts/13_e2_single_condition.py analyze
Outputs under artifacts/e2_single/.
"""
import json, math, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_lib as tl  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert tl.MODEL_ID == MODEL_ID, "free tier only (D-001)"
OUT = Path("artifacts/e2_single")
PILOT = Path("artifacts/e2_pilot/sgd_lreff2.9e-4")
EPS = 1e4
LR_EFF = 2.9e-4
LR_LOW = 1.7e-4
CONDS = ["k1", "k2", "k3", "k4", "k6", "reverse"]
CHANCE_LOSS = math.log(26)

RUNS = {  # name: (train_conds, lora_seed)
    "sgd_k2_s0": (["k2"], 0),
    "sgd_k4_s0": (["k4"], 0),
    "sgd_rev_s0": (["reverse"], 0),
    "sgd_mix_s1": (["k2", "k3", "k4"], 1),
}
EXTRAS = {"sgd_k2_s1": (["k2"], 1), "sgd_k4_s1": (["k4"], 1)}


def cfg_for(name, conds, seed, lr_eff):
    return tl.Config(name=name, out_dir=OUT / name, optimizer="adamw", lr=lr_eff * EPS, rank=8,
                     lora_seed=seed, data_seed=1, train_conds=conds, batch=32, steps=100,
                     eval_steps=[0, 10, 25, 50, 100], early_stop_acc=2.0, adam_eps=EPS,
                     adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.0, max_in_flight=16,
                     diverge_loss=None, export_steps=[1], save_state=True)


def done(name):
    p = OUT / name / "adapter_stats.json"
    return p.exists() and json.loads(p.read_text()).get("step") == 100


def load(d: Path):
    recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines() if l.strip()]
    cfg = next(r["config"] for r in recs if r["type"] == "start")
    ev = {r["step"]: r["summary"] for r in recs if r["type"] == "eval"}
    tr = {r["step"]: r["loss_mean"] for r in recs if r["type"] == "train"}
    rd = lambda f: json.loads((d / f).read_text()) if (d / f).exists() else None  # noqa: E731
    return {"cfg": cfg, "ev": ev, "loss": tr, "final": rd("adapter_stats.json"),
            "s1": rd("adapter_stats_step1.json"), "s0": rd("adapter_stats_step0.json"), "recs": recs}


def at_chance(name):
    """At chance for the whole run: every eval after step 0 has accuracy at most 0.11 (chance
    1/26 plus two standard errors at n = 30) on every condition, and the mean train loss over
    steps 11 to 100 is within 0.5 nats of ln 26."""
    run = load(OUT / name)
    accs = [run["ev"][s][c]["acc"] for s in run["ev"] if s > 0 for c in CONDS]
    tail = [run["loss"][s] for s in run["loss"] if s > 10]
    return max(accs) <= 0.11 and abs(sum(tail) / len(tail) - CHANCE_LOSS) < 0.5


def run_one(name, conds, seed, lr_eff):
    if done(name):
        print(f"[chain] {name} done, skipping", flush=True)
        return True
    cfg = cfg_for(name, conds, seed, lr_eff)
    for attempt in range(2):
        resume = (OUT / name / "train_log.jsonl").exists() and tl._last_state(OUT / name / "train_log.jsonl")[0] is not None
        print(f"[chain] {name} start attempt {attempt} resume={resume} {time.strftime('%H:%M:%S')}", flush=True)
        try:
            s = tl.run(cfg, resume=resume)
            print(f"[chain] {name} finished {json.dumps(s['timing'])} S={s.get('looped_share')}", flush=True)
            return True
        except Exception:  # noqa: BLE001
            print(f"[chain] {name} failed:\n{traceback.format_exc()[-2000:]}", flush=True)
            time.sleep(60)
    return False


def chain(extras):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (conds, seed) in RUNS.items():
        ok = run_one(name, conds, seed, LR_EFF)
        if ok and conds != ["k2", "k3", "k4"] and at_chance(name):
            print(f"[chain] {name} at chance, rerunning at {LR_LOW}", flush=True)
            run_one(f"{name}_lr1.7e-4", conds, seed, LR_LOW)
    if extras:
        for name, (conds, seed) in EXTRAS.items():
            ok = run_one(name, conds, seed, LR_EFF)
            if ok and at_chance(name):
                run_one(f"{name}_lr1.7e-4", conds, seed, LR_LOW)
    print("[chain] all done", flush=True)


# ---------------------------------------------------------------- analysis

GROUPS = ("all", "attention", "mlp_experts")
BLOCKS = {"prelude 0-5": range(0, 6), "loop 6-9": range(6, 10), "loop 10-13": range(10, 14),
          "loop 14-17": range(14, 18), "coda 18-23": range(18, 24)}
LABEL = {"sgd_k2_s0": "k = 2 only, seed 0", "sgd_k4_s0": "k = 4 only, seed 0",
         "sgd_rev_s0": "Reverse only, seed 0", "sgd_mix_s1": "Mix k = 2, 3, 4, seed 1",
         "sgd_k2_s1": "k = 2 only, seed 1", "sgd_k4_s1": "k = 4 only, seed 1",
         "pilot": "Mix k = 2, 3, 4, seed 0 (pilot)"}


def profile_metrics(p):
    """p is a 24-layer norm profile (square root of summed squared Frobenius norms)."""
    sq = [x * x for x in p]
    tot = sum(sq)
    loop = [p[i] for i in tl.LOOP]
    unl = [p[i] for i in range(24) if i not in tl.LOOP]
    lsq = sum(sq[i] for i in tl.LOOP)
    return {"S": lsq / tot, "R": (sum(loop) / len(loop)) / (sum(unl) / len(unl)),
            "L6/L5": p[6] / p[5], "L18/L17": p[18] / p[17],
            "peak": max(range(24), key=lambda i: p[i]), "norm": math.sqrt(tot),
            "blocks_of_total": {b: sum(sq[i] for i in r) / tot for b, r in BLOCKS.items()},
            "loop_thirds_of_loop": {b: sum(sq[i] for i in r) / lsq for b, r in BLOCKS.items() if b.startswith("loop")},
            # weighted mean layer inside the loop, weights = squared norm
            "loop_centroid": sum(i * sq[i] for i in tl.LOOP) / lsq}


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def run_metrics(st):
    out = {}
    for q in ("B_fro", "eff_fro"):
        out[q] = {g: profile_metrics(st["summary"][q][g]["per_layer_norm"]) for g in GROUPS}
        out[q]["profile"] = st["summary"][q]["all"]["per_layer_norm"]
    out["lm_head_B"] = [r["B_fro"] for r in st["rows"] if r["layer"] is None]
    return out


def step1_profile(st):
    p = [0.0] * 24
    for r in st["rows"]:
        if r["layer"] is not None:
            p[r["layer"]] += r["B_fro"] ** 2
    return [math.sqrt(x) for x in p]


def available():
    names = [n for n in list(RUNS) + list(EXTRAS) if done(n)]
    names += [n.name for n in sorted(OUT.iterdir()) if n.is_dir() and n.name.endswith("_lr1.7e-4") and done(n.name)]
    return names


def analyze():
    import importlib
    s11 = importlib.import_module("11_e2_pilot_sgd")
    dirs = {n: OUT / n for n in available()}
    dirs["pilot"] = PILOT
    res = {"runs": {}, "cosine_B": {}, "cosine_eff": {}, "diffs": {}}
    for n, d in dirs.items():
        run = load(d)
        st = run["final"]
        m = run_metrics(st)
        m["final_step"] = st["step"]
        m["train_conds"] = run["cfg"]["train_conds"]
        m["lora_seed"] = run["cfg"]["lora_seed"]
        m["lr_eff"] = run["cfg"]["lr"] / run["cfg"]["adam_eps"]
        m["acc"] = {s: {c: run["ev"][s][c]["acc"] for c in CONDS} for s in sorted(run["ev"])}
        m["rev_forward_image_share"] = {s: run["ev"][s]["reverse"]["forward_image_share"] for s in sorted(run["ev"])}
        m["loss"] = {s: run["loss"][s] for s in (1, 2, 10, 25, 50, 100, 150) if s in run["loss"]}
        m["loss_last5"] = sum(run["loss"][s] for s in sorted(run["loss"])[-5:]) / 5
        if run["s1"]:
            p1 = step1_profile(run["s1"])
            m["step1"] = profile_metrics(p1)
            m["step1"]["profile"] = p1
            m["cos_final_vs_step1_B"] = cosine(m["B_fro"]["profile"], p1)
        s11.OUT = d.parent
        m["hops"] = {s: s11.hop_behaviour(d.name, s) for s in (0, 100, m["final_step"]) if s in run["ev"]}
        # errors and warnings recorded by the harness
        end = [r for r in run["recs"] if r["type"] == "end"]
        m["n_errors"] = end[-1]["n_errors"] if end else None
        m["sdk_warnings"] = end[-1]["sdk_warnings"] if end else None
        m["timing"] = end[-1]["timing"] if end else None
        res["runs"][n] = m
    names = list(res["runs"])
    for q, key in (("B_fro", "cosine_B"), ("eff_fro", "cosine_eff")):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                res[key][f"{a} | {b}"] = cosine(res["runs"][a][q]["profile"], res["runs"][b][q]["profile"])

    def scal(n, q="B_fro"):
        m = res["runs"][n][q]["all"]
        return {"S": m["S"], "R": m["R"], "L6/L5": m["L6/L5"], "L18/L17": m["L18/L17"],
                "loop_centroid": m["loop_centroid"], "norm": m["norm"],
                **{f"of_total {b}": v for b, v in m["blocks_of_total"].items()},
                **{f"of_loop {b}": v for b, v in m["loop_thirds_of_loop"].items()}}
    pairs = {"between: k4_s0 - k2_s0": ("sgd_k4_s0", "sgd_k2_s0"),
             "between: k2_s0 - rev_s0": ("sgd_k2_s0", "sgd_rev_s0"),
             "between: k4_s0 - rev_s0": ("sgd_k4_s0", "sgd_rev_s0"),
             "between: k4_s1 - k2_s1": ("sgd_k4_s1", "sgd_k2_s1"),
             "seed: mix_s1(100) - pilot(150)": ("sgd_mix_s1", "pilot"),
             "seed: k2_s1 - k2_s0": ("sgd_k2_s1", "sgd_k2_s0"),
             "seed: k4_s1 - k4_s0": ("sgd_k4_s1", "sgd_k4_s0")}
    for lab, (a, b) in pairs.items():
        if a in res["runs"] and b in res["runs"]:
            for q in ("B_fro", "eff_fro"):
                sa, sb = scal(a, q), scal(b, q)
                res["diffs"].setdefault(q, {})[lab] = {k: sa[k] - sb[k] for k in sa}
    (OUT / "single_tables.json").write_text(json.dumps(res, indent=1, default=str))
    report(res)
    plot(res)


def report(res):
    f = lambda x: f"{x:.3f}"  # noqa: E731
    print("run | steps | S_B S_eff | R_B R_eff | L6/5 L18/17 (B) | L6/5 L18/17 (eff) | peak | ||B|| | S_B step1")
    for n, m in res["runs"].items():
        b, e = m["B_fro"]["all"], m["eff_fro"]["all"]
        print(f"  {n:20s} {m['final_step']:3d} {f(b['S'])} {f(e['S'])} | {b['R']:.2f} {e['R']:.2f} | "
              f"{b['L6/L5']:.2f} {b['L18/L17']:.2f} | {e['L6/L5']:.2f} {e['L18/L17']:.2f} | {b['peak']:2d} | "
              f"{b['norm']:.2f} | {f(m['step1']['S']) if 'step1' in m else '-'}")
    print("S_B / S_eff by group (attention, mlp_experts)")
    for n, m in res["runs"].items():
        print(f"  {n:20s} " + " ".join(f"{m[q][g]['S']:.3f}/{m[q][g]['R']:.2f}" for q in ("B_fro", "eff_fro")
                                       for g in ("attention", "mlp_experts")))
    print("blocks of total B (prelude, loop thirds, coda) | loop thirds of loop | centroid")
    for n, m in res["runs"].items():
        b = m["B_fro"]["all"]
        print(f"  {n:20s} " + " ".join(f"{v:.3f}" for v in b["blocks_of_total"].values()) + " | "
              + " ".join(f"{v:.3f}" for v in b["loop_thirds_of_loop"].values()) + f" | {b['loop_centroid']:.2f}")
    print("accuracy (k1 k2 k3 k4 k6 rev) by step")
    for n, m in res["runs"].items():
        print(f"  {n:20s} " + "  ".join(f"@{s}:" + ",".join(f"{a[c]:.2f}" for c in CONDS) for s, a in m["acc"].items()))
    print("hops applied at final step (j: count; -1 = inverse, None = off path)")
    for n, m in res["runs"].items():
        h = m["hops"][m["final_step"]]
        print(f"  {n:20s} " + " ".join(f"{c}{h[c]}" for c in CONDS).replace(" ", "").replace("'", "")[:400])
    print("cosine of B profiles")
    for k, v in res["cosine_B"].items():
        print(f"  {k:40s} {v:.4f}  eff {res['cosine_eff'][k]:.4f}")
    print("diffs (B): S R L6/5 L18/17 centroid of_loop thirds")
    for k, d in res["diffs"].get("B_fro", {}).items():
        print(f"  {k:34s} {d['S']:+.4f} {d['R']:+.2f} {d['L6/L5']:+.2f} {d['L18/L17']:+.3f} {d['loop_centroid']:+.2f} "
              + " ".join(f"{d[x]:+.3f}" for x in d if x.startswith("of_loop")))
    for n, m in res["runs"].items():
        if m["n_errors"] or m["sdk_warnings"]:
            print("  errors/warnings", n, m["n_errors"], json.dumps(m["sdk_warnings"])[:500])


def plot(res):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    cond_lab = {"k2": "Trained on k = 2", "k4": "Trained on k = 4", "rev": "Trained on reverse",
                "mix": "Trained on k = 2, 3, 4"}
    col = ms.color_map(list(cond_lab.values()))

    def cond_of(n):
        return "mix" if n == "pilot" else n.split("_")[1]

    # Figure 1: per-layer B and effective-update profiles of the single-condition runs.
    single = [n for n in res["runs"] if cond_of(n) in ("k2", "k4", "rev") and "lr1.7e-4" not in n]
    fig, axes = plt.subplots(1, 2, figsize=figsize("blog_wide"), sharey=True)
    for ax, (q, qlab) in zip(axes, (("B_fro", "‖B‖"), ("eff_fro", "‖(α/r)·BA‖"))):
        ax.axvspan(5.5, 17.5, color="#e9e9ee", zorder=0, lw=0)
        ax.axhline(1.0, color="#bbbbc4", lw=1.0, zorder=1)
        for n in single:
            y = res["runs"][n][q]["profile"]
            mu = sum(y) / len(y)
            s1 = n.endswith("_s1")
            ax.plot(range(24), [v / mu for v in y], color=col[cond_lab[cond_of(n)]], ls="--" if s1 else "-",
                    label=(LABEL[n].replace(" only", "") if ax is axes[1] else None))
        ms.titles(ax, f"{qlab}, all modules", small=True,
                  ylabel="Norm ÷ its run's mean over layers (grey line = 1)" if ax is axes[0] else None)
        ax.set_xlabel("Layer (shaded: looped layers 6 to 17)")
    ms.end_labels(axes[1])
    ms.fig_titles(fig, "Per-layer adapter norm after 100 steps of SGD mode on one condition")
    ms.save(fig, str(OUT / "profile_single_condition"))

    # Figure 2: accuracy by step on each eval condition, every run.
    cnames = {"k1": "k = 1", "k2": "k = 2", "k3": "k = 3", "k4": "k = 4", "k6": "k = 6", "reverse": "Reverse lookup"}
    runs = [n for n in res["runs"] if "lr1.7e-4" not in n]
    fig, axes = plt.subplots(2, 3, figsize=figsize("blog_wide", nrows=2, ncols=3), sharex=True, sharey=True)
    for ax, (c, clab) in zip(axes.flat, cnames.items()):
        for n in runs:
            acc = res["runs"][n]["acc"]
            steps = [s for s in sorted(acc) if s <= 100]
            ys = [acc[s][c] for s in steps]
            ax.plot(steps, ys, color=col[cond_lab[cond_of(n)]], ls="--" if n.endswith("_s1") else "-")
        ms.titles(ax, clab, small=True, ylabel="Greedy accuracy" if ax in axes[:, 0] else None)
        if ax in axes[1]:
            ax.set_xlabel("Training step")
    hs = [plt.Line2D([], [], color=col[cond_lab[cond_of(n)]], ls="--" if n.endswith("_s1") else "-")
          for n in runs]
    axes[0, 2].legend(hs, [LABEL[n] for n in runs], loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.75,
                      borderpad=0, handlelength=1.8, labelspacing=0.55, fontsize=8)
    ms.fig_titles(fig, "Eval accuracy (30 examples per condition) in SGD mode")
    ms.save(fig, str(OUT / "accuracy_by_step"))


def main():
    cmd = sys.argv[1]
    if cmd == "chain":
        chain("--extras" in sys.argv)
    elif cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
