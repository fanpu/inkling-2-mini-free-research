"""E0: per-layer LoRA-B gradient norms on pointer chasing as a function of hop count.

Uses the gradient readout validated in script 03. A fresh rank-2 client (LoRA seed 0,
so every run starts from the same A matrices) takes one AdamW step from B = 0 with
lr = eps = 1e3. The step-0 and step-1 adapters are exported and dB = B1 - B0 is
inverted exactly, |g| = eps * u / (1 - u) with u = |dB| / lr.

Conditions are k = 1, 2, 3, 4, 6 of the main task and the single-table reverse
control (note 07), each with data seeds 1 and 2 (never the eval seed 2026). A batch
is 32 examples with only the answer token supervised (task_pointer.to_datum).

Each finished run is appended to artifacts/e0_grad_by_hops/results.json, and a rerun
skips finished runs. `--analyze` only recomputes the summary and the figure.

Usage: .venv/bin/python scripts/08_e0_grad_by_hops.py [--analyze]
"""
import importlib.util, json, math, shutil, sys, tempfile, time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import task_pointer as tp  # noqa: E402

_spec = importlib.util.spec_from_file_location("gr03", HERE / "03_grad_readout_probe.py")
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)

import tinker  # noqa: E402
from tinker import types  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free", "free tier only (D-001)"
assert gr.MODEL_ID == MODEL_ID and tp.MODEL_ID == MODEL_ID
TTL = 3600
LR = EPS = 1e3
RANK = 2
LORA_SEED = 0
N = 32
DATA_SEEDS = (1, 2)
CONDS = ["k1", "k2", "k3", "k4", "k6", "reverse"]
LOOP = range(6, 18)
OUT = Path("artifacts/e0_grad_by_hops")
RES = OUT / "results.json"

# Route script 03's retry log to this experiment, so artifacts/grad_readout is never written.
gr.LOG = OUT / "run_log.json"
gr.log.clear()
gr.log.update({"errors": [], "clients": {}, "sdk_warnings": [], "notes": []})


def examples(cond, seed):
    assert seed != 2026, "2026 is the eval seed"
    if cond == "reverse":
        return tp.generate_control(N, seed, "reverse")
    return tp.generate(N, int(cond[1:]), seed)


def load_results():
    return json.loads(RES.read_text()) if RES.exists() else []


def run_one(cond, seed):
    exs = examples(cond, seed)
    datums = [tp.to_datum(e) for e in exs]
    params = types.AdamParams(learning_rate=LR, eps=EPS)
    name = f"e0-{cond}-s{seed}"
    sc, rest = tinker.ServiceClient(), tinker.RestClient()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="e0_"))
    try:
        tc = gr.retry(lambda: sc.create_lora_training_client(
            base_model=MODEL_ID, rank=RANK, seed=LORA_SEED, optimizer=types.AdamOptimizerConfig(),
            user_metadata={"purpose": "e0-grad-by-hops", "setting": name}).result(), f"{name}:create")
        e0 = gr.retry(lambda: tc.save_weights_external(f"{name}-s0", ttl_seconds=TTL).result(), f"{name}:save0")
        w0 = gr.download_B(rest, e0.path, tmp)
        fb = gr.retry(lambda: tc.forward_backward(datums, "cross_entropy").result(), f"{name}:fb")
        gr.retry(lambda: tc.optim_step(params).result(), f"{name}:optim")
        e1 = gr.retry(lambda: tc.save_weights_external(f"{name}-s1", ttl_seconds=TTL).result(), f"{name}:save1")
        w1 = gr.download_B(rest, e1.path, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:  # an unclosed client blocks the next one for 15 minutes (script 03, run 1)
            sc.close("success", detail=f"e0 {name}").result()
        except Exception as e:  # noqa: BLE001
            gr.log["errors"].append({"what": f"{name}:close", "error": f"{type(e).__name__}: {e}"[:2000]})
            gr.save_log()
    dB, b0max, dAmax, umax = {}, 0.0, 0.0, 0.0
    for k in w0:
        d = w1[k].float() - w0[k].float()
        if ".lora_B." in k:
            dB[k] = d
            b0max = max(b0max, w0[k].float().abs().max().item())
            umax = max(umax, d.abs().max().item() / LR)
        else:
            dAmax = max(dAmax, d.abs().max().item())
    del w0, w1
    pl = gr.per_layer(dB, LR, EPS)
    # per-example log p(answer) at the supervised position
    tl = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    try:
        lp = [sum(a for a, b in zip(tl(o["logprobs"]), tl(d.loss_fn_inputs["weights"])) if b > 0)
              for o, d in zip(fb.loss_fn_outputs, datums)]
        mlp = sum(lp) / len(lp)
    except Exception as e:  # noqa: BLE001  (never lose a finished run to parsing)
        lp, mlp = None, None
        gr.log["errors"].append({"what": f"{name}:logprob_parse", "error": f"{type(e).__name__}: {e}"[:500]})
    return dict(cond=cond, seed=seed, k=1 if cond == "reverse" else int(cond[1:]),
                lr=LR, eps=EPS, rank=RANK, lora_seed=LORA_SEED, n_examples=N,
                loss_metrics=fb.metrics, loss_sum=fb.metrics.get("loss:sum"),
                mean_logp_answer=mlp, logp_answer=lp,
                per_layer=pl["per_layer"], other=pl["other"],
                max_u=umax, max_abs_g=EPS * umax / (1 - umax), B0_absmax=b0max, dA_absmax=dAmax,
                seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------- analysis

def shares(r):
    """Looped share of squared norm and looped/unlooped mean-norm ratio, per module and overall."""
    out = {}
    tot = [0.0] * 24
    for m, v in r["per_layer"].items():
        sq = [x * x for x in v]
        for i in range(24):
            tot[i] += sq[i]
        out[m] = _stats(sq)
    out["all"] = _stats(tot)
    groups = {"attention": [m for m in r["per_layer"] if m.startswith("attn")],
              "mlp_experts": [m for m in r["per_layer"] if m.startswith("mlp")]}
    for g, ms in groups.items():
        out[g] = _stats([sum(r["per_layer"][m][i] ** 2 for m in ms) for i in range(24)])
    return out


def _stats(sq):
    nrm = [math.sqrt(x) for x in sq]
    loop = sum(sq[i] for i in LOOP)
    lm = sum(nrm[i] for i in LOOP) / 12
    um = sum(nrm[i] for i in range(24) if i not in LOOP) / 12
    return dict(looped_share=loop / sum(sq), ratio_looped_over_unlooped=lm / um)


def analyze():
    rs = load_results()
    summ = {}
    for r in rs:
        s = shares(r)
        summ.setdefault(r["cond"], {})[str(r["seed"])] = dict(
            loss_sum=r["loss_sum"], mean_logp_answer=r["mean_logp_answer"],
            lm_head_norm=r["other"].get("lm_head"), max_u=r["max_u"],
            total_layer_norm=math.sqrt(sum(x * x for v in r["per_layer"].values() for x in v)),
            per_layer_norm_by_module=r["per_layer"], shares=s)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("cond    seed  loss_sum  share_all share_attn share_mlp  ratio_all  max_u")
    for c in CONDS:
        for sd, v in sorted(summ.get(c, {}).items()):
            s = v["shares"]
            print(f"{c:7s} {sd:>4s} {v['loss_sum']:9.2f}  {s['all']['looped_share']:.3f}     "
                  f"{s['attention']['looped_share']:.3f}      {s['mlp_experts']['looped_share']:.3f}      "
                  f"{s['all']['ratio_looped_over_unlooped']:.3f}   {v['max_u']:.3f}")
    if summ:
        plot(summ)
    return summ


def plot(summ):
    import matplotlib.pyplot as plt
    from mlstyle import figsize
    from mlstyle import mpl as ms
    ms.use("blog")
    series = {"All modules": "all", "Attention": "attention", "MLP experts": "mlp_experts"}
    colors = ms.color_map(list(series))
    fig, ax = plt.subplots(figsize=figsize("blog"))
    ks = [c for c in CONDS if c != "reverse" and c in summ]
    xs = [int(c[1:]) for c in ks]
    XC = 8  # x position of the one-hop reverse control, set apart from the k axis
    for name, key in series.items():
        ys = [[v["shares"][key]["looped_share"] for v in summ[c].values()] for c in ks]
        ax.plot(xs, [sum(y) / len(y) for y in ys], color=colors[name], label=name, zorder=2)
        for x, y in zip(xs, ys):
            ms.dots(ax, [x] * len(y), y, colors[name])
        if "reverse" in summ:
            y = [v["shares"][key]["looped_share"] for v in summ["reverse"].values()]
            ms.dots(ax, [XC] * len(y), y, colors[name])
    ax.axvline(7, color="#c8c8cc", lw=0.8, zorder=0)
    ax.set_xticks(xs + ([XC] if "reverse" in summ else []),
                  [str(x) for x in xs] + (["Reverse\nlookup"] if "reverse" in summ else []))
    ax.set_xlabel("Hop count k")
    ms.titles(ax, "Share of squared LoRA-B gradient norm in looped layers",
              "Lines are means of 2 data seeds, dots are single seeds",
              ylabel="Looped share (layers 6 to 17)", key=colors)
    ms.save(fig, str(OUT / "looped_share_vs_k"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--analyze" not in sys.argv:
        for cond in CONDS:
            for seed in DATA_SEEDS:
                if any(r["cond"] == cond and r["seed"] == seed for r in load_results()):
                    continue
                print(f"[{cond} s{seed}] start {time.strftime('%H:%M:%S')}", flush=True)
                r = run_one(cond, seed)
                rs = load_results() + [r]
                RES.write_text(json.dumps(rs, indent=1, default=str))
                gr.log["clients"][f"{cond}-s{seed}"] = {k: r[k] for k in ("seconds", "loss_sum", "max_u", "B0_absmax", "dA_absmax")}
                gr.save_log()
                print(f"[{cond} s{seed}] done {r['seconds']}s loss_sum {r['loss_sum']:.2f} "
                      f"mean_logp {r["mean_logp_answer"]} max_u {r['max_u']:.3f}", flush=True)
    analyze()


if __name__ == "__main__":
    main()
