"""Feasibility probe: can raw per-layer gradients be read through AdamW with a large eps?

At LoRA init B = 0, so after one step only B moves. Adam's bias-corrected first
step is dB = -lr * g / (|g| + eps). With eps >> |g| this is -(lr/eps) * g, and
the exact inverse is |g| = eps * u / (1 - u) with u = |dB| / lr.

For each setting a fresh rank-2 client (seed 0) is created, the adapter is
exported before and after one optim_step on a fixed batch, and dB = B1 - B0 is
computed per tensor. Downloaded adapters are deleted right after reading.
dB tensors are cached in the system temp dir (not in the repo) so a rerun
resumes. Outputs go to artifacts/grad_readout/ (statistics only, no weights).

Usage: .venv/bin/python scripts/03_grad_readout_probe.py
"""
import json, logging, random, re, shutil, tempfile, time, traceback, urllib.request, warnings
from pathlib import Path

import torch
from safetensors.torch import load_file
import tinker
from tinker import types
from tml_renderers import chat, renderers
from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights

warnings.simplefilter("ignore", FutureWarning)

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free", "free tier only (D-001)"
TTL = 3600
SEED = 0
RANKS = (2, 4)  # try rank 2, fall back to 4 if rejected
OUT = Path("artifacts/grad_readout")
CACHE = Path(tempfile.gettempdir()) / "inkling_grad_readout_cache"
LOG = OUT / "run_log.json"

# Each setting: candidate (lr, eps) pairs tried in order until one is accepted.
SETTINGS = {
    "a_sign": dict(cands=[(1e-3, None)], wscale=1.0, data_seed=0),
    "b_eps1": dict(cands=[(1.0, 1.0), (1e-2, 1e-2), (1e-4, 1e-4)], wscale=1.0, data_seed=0),
    "c_eps1e3": dict(cands=[(1e3, 1e3), (1e2, 1e2), (10.0, 10.0)], wscale=1.0, data_seed=0),
    "d_eps1_w2": dict(cands="same_as_b", wscale=2.0, data_seed=0),
    # Run 2 showed max |g| is about 600, so eps=1 is not "large". The batch runs use
    # the eps=1e3 setting, and e repeats data 0 at eps=1e4 to measure run-to-run noise.
    "e_eps1e4": dict(cands=[(1e4, 1e4)], wscale=1.0, data_seed=0),
    "c_eps1e3_data1": dict(cands="same_as_c", wscale=1.0, data_seed=1),
    "c_eps1e3_data2": dict(cands="same_as_c", wscale=1.0, data_seed=2),
}
VALID = ("a_sign", "b_eps1", "c_eps1e3", "d_eps1_w2", "e_eps1e4")
BATCHES = ("c_eps1e3", "c_eps1e3_data1", "c_eps1e3_data2")
MODS = ["attn.wq_du", "attn.wk_dv", "attn.wv_dv", "attn.wo_ud", "attn.wr_du", "attn.gate_attn",
        "mlp.experts.w1", "mlp.experts.w2", "mlp.experts.w3"]
KEY_RE = re.compile(r"language_model\.(?:layers\.(\d+)\.)?(.*)\.lora_([AB])\.weight")

log = {"errors": [], "clients": {}, "sdk_warnings": [],
       "notes": ["Run 1 (before ServiceClient.close was added) stalled >15 min on setting c with the SDK "
                 "warning: 'Training for <model_id>:train:0 is paused. Reason: concurrent training clients "
                 "rate limit hit.' Settings a and b had finished and were reused from cache."]}


class _WarnCapture(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if not any(w["msg"] == msg for w in log["sdk_warnings"]):
            log["sdk_warnings"].append({"msg": msg[:1000], "time": time.strftime("%H:%M:%S")})
        else:
            next(w for w in log["sdk_warnings"] if w["msg"] == msg)["last"] = time.strftime("%H:%M:%S")


logging.getLogger("tinker").addHandler(_WarnCapture(level=logging.WARNING))


def save_log():
    LOG.write_text(json.dumps(log, indent=1, default=str))


def retry(fn, what, tries=7):
    """Retry only on throttling / transient errors, with exponential backoff."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            transient = isinstance(e, (tinker.RateLimitError, tinker.APIConnectionError,
                                       tinker.APITimeoutError, tinker.InternalServerError)) or \
                any(s in msg.lower() for s in ("429", "rate limit", "throttl", "503", "timed out", "unavailable"))
            log["errors"].append({"what": what, "attempt": k, "transient": transient, "error": msg[:2000],
                                  "time": time.strftime("%H:%M:%S")})
            save_log()
            print(f"  ! {what} attempt {k}: {msg[:300]}", flush=True)
            if not transient or k == tries - 1:
                raise
            time.sleep(min(300, 20 * 2 ** k))


def make_batch(data_seed, n=8):
    """n short in-context pointer-chasing questions with a one-letter answer."""
    rng = random.Random(1000 + data_seed)
    rend = renderers.Inkling2()
    data = []
    for _ in range(n):
        letters = rng.sample("ABCDEFGHJKMNPQRSTUVWXYZ", 6)
        perm = letters[:]
        rng.shuffle(perm)
        edges = list(zip(letters, perm))
        rng.shuffle(edges)
        hops = rng.choice([1, 2, 3])
        start = rng.choice(letters)
        cur = start
        mp = dict(edges)
        for _ in range(hops):
            cur = mp[cur]
        q = ("Pointers: " + ", ".join(f"{a}->{b}" for a, b in edges)
             + f". Start at {start} and follow {hops} pointer(s). Reply with one letter.")
        ex = rend.render_for_sft(chat.OpenAIMessage.from_openai_chat_dicts([
            {"role": "user", "content": q}, {"role": "assistant", "content": cur}]))[-1]
        mi, w = training_example_to_tinker_model_input_and_weights(ex)
        ids = mi.to_ints()
        data.append((ids, list(w)))
    return data


def to_datums(data, wscale):
    return [types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                        loss_fn_inputs={"target_tokens": ids[1:], "weights": [x * wscale for x in w[1:]]})
            for ids, w in data]


def download_B(rest, path, tmp):
    """Download an exported adapter, return its tensors, delete the file."""
    urls = retry(lambda: rest.get_external_weights_urls(path).result(), "get_urls")
    out = {}
    for rel, url in urls.urls.items():
        if not rel.endswith(".safetensors"):
            continue
        dest = tmp / rel.replace("/", "_")
        retry(lambda: urllib.request.urlretrieve(url, dest), "download")
        size = dest.stat().st_size
        out.update(load_file(dest))
        dest.unlink()
        print(f"  downloaded+deleted {rel} ({size/1e6:.1f} MB)", flush=True)
    return out


def run_client(name, lr, eps, wscale, data_seed, rank):
    sc, rest = tinker.ServiceClient(), tinker.RestClient()
    t0 = time.time()
    tc = retry(lambda: sc.create_lora_training_client(
        base_model=MODEL_ID, rank=rank, seed=SEED, optimizer=types.AdamOptimizerConfig(),
        user_metadata={"purpose": "grad-readout-probe", "setting": name}).result(), f"{name}:create")
    t_create = time.time() - t0
    data = make_batch(data_seed)
    datums = to_datums(data, wscale)
    pk = dict(learning_rate=lr) if eps is None else dict(learning_rate=lr, eps=eps)
    params = types.AdamParams(**pk)  # client-side validation errors surface here
    tmp = Path(tempfile.mkdtemp(prefix="gr_"))
    try:
        e0 = retry(lambda: tc.save_weights_external(f"gr-{name}-s0", ttl_seconds=TTL).result(), f"{name}:save0")
        w0 = {k: v for k, v in download_B(rest, e0.path, tmp).items()}
        fb = retry(lambda: tc.forward_backward(datums, "cross_entropy").result(), f"{name}:fb")
        os_ = retry(lambda: tc.optim_step(params).result(), f"{name}:optim")
        e1 = retry(lambda: tc.save_weights_external(f"gr-{name}-s1", ttl_seconds=TTL).result(), f"{name}:save1")
        w1 = download_B(rest, e1.path, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        # Free tier limits concurrent training clients. An unclosed ServiceClient keeps
        # its session alive by heartbeat and blocks the next client (seen in run 1).
        try:
            sc.close("success", detail=f"grad readout {name}").result()
        except Exception as e:  # noqa: BLE001
            log["errors"].append({"what": f"{name}:close", "error": f"{type(e).__name__}: {e}"[:2000]})
    dB, dA_max, B0_max = {}, 0.0, 0.0
    for k in w0:
        d = w1[k].float() - w0[k].float()
        if ".lora_B." in k:
            dB[k] = d
            B0_max = max(B0_max, w0[k].float().abs().max().item())
        else:
            dA_max = max(dA_max, d.abs().max().item())
    meta = dict(lr=lr, eps=eps, wscale=wscale, data_seed=data_seed, rank=rank,
                loss_metrics=fb.metrics, optim=str(os_), dA_absmax=dA_max, B0_absmax=B0_max,
                n_tokens=sum(len(ids) for ids, _ in data),
                n_supervised=sum(sum(1 for x in w if x > 0) for _, w in data),
                seconds_create=round(t_create, 1), seconds_total=round(time.time() - t0, 1))
    return dB, meta


def get_setting(name, cfg, choice):
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{name}.pt"
    if cf.exists():
        obj = torch.load(cf)
        print(f"[{name}] loaded from cache", flush=True)
        return obj["dB"], obj["meta"]
    cands = [choice[cfg["cands"][-1]]] if isinstance(cfg["cands"], str) else cfg["cands"]
    last = None
    for lr, eps in cands:
        for rank in RANKS:
            print(f"[{name}] lr={lr} eps={eps} rank={rank} wscale={cfg['wscale']} data={cfg['data_seed']}", flush=True)
            try:
                dB, meta = run_client(name, lr, eps, cfg["wscale"], cfg["data_seed"], rank)
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                log["clients"].setdefault(name, []).append({"lr": lr, "eps": eps, "rank": rank, "rejected": last[:2000]})
                save_log()
                traceback.print_exc()
                if "rank" in last.lower():
                    continue  # try next rank
                break  # try next (lr, eps)
            log["clients"].setdefault(name, []).append(meta)
            save_log()
            print(f"[{name}] done in {meta['seconds_total']}s, loss {meta['loss_metrics']}", flush=True)
            torch.save({"dB": dB, "meta": meta}, cf)
            return dB, meta
    raise RuntimeError(f"[{name}] all candidates rejected; last error: {last}")


def flat(dB, keys):
    return torch.cat([dB[k].flatten() for k in keys])


def g_exact(x, lr, eps):
    """Invert dB = -lr * g / (|g| + eps) exactly: |g| = eps * u / (1 - u), u = |dB| / lr."""
    x = x.double()
    u = x.abs() / lr
    return -torch.sign(x) * eps * u / (1 - u)


def validate(D, M):
    keys = sorted(D["a_sign"])
    fl = lambda n: flat(D[n], keys).double()
    a, b, c, d, e = (fl(n) for n in VALID)
    P = {n: (M[n]["lr"], M[n]["eps"]) for n in VALID}
    gb, gc, gd, ge = (g_exact(x, *P[n]) for x, n in ((b, "b_eps1"), (c, "c_eps1e3"), (d, "d_eps1_w2"), (e, "e_eps1e4")))
    lin = lambda x, n: -x * P[n][1] / P[n][0]
    rf = lambda x, y: ((x - y).norm() / y.norm()).item()
    nz = (a != 0) & (c != 0)
    dis = nz & (torch.sign(a) != torch.sign(c))
    absb = (b.abs() / P["b_eps1"][0])[b != 0]
    v = dict(
        n_B_entries=a.numel(),
        frac_nonzero={n: (fl(n) != 0).float().mean().item() for n in VALID},
        a_abs_over_lr_median=(a.abs()[a != 0] / M["a_sign"]["lr"]).median().item(),
        sign_agree_a_c=1 - dis.sum().item() / nz.sum().item(),
        sign_agree_a_b=(torch.sign(a[(a != 0) & (b != 0)]) == torch.sign(b[(a != 0) & (b != 0)])).float().mean().item(),
        sign_disagree_g_energy_frac=((gc[dis] ** 2).sum() / (gc ** 2).sum()).item(),
        sign_disagree_median_abs_g=gc[dis].abs().median().item(),
        median_abs_g_nonzero=gc[gc != 0].abs().median().item(),
        relfro_b_vs_c_linear=rf(lin(b, "b_eps1"), lin(c, "c_eps1e3")),
        relfro_b_vs_c_exact=rf(gb, gc),
        relfro_c_vs_e_linear=rf(lin(c, "c_eps1e3"), lin(e, "e_eps1e4")),
        relfro_c_vs_e_exact=rf(gc, ge),
        relfro_e_linear_vs_exact=rf(lin(e, "e_eps1e4"), ge),
        ratio_d_over_b_fro_raw_dB=(d.norm() / b.norm()).item(),
        ratio_d_over_b_fro_exact=(gd.norm() / gb.norm()).item(),
        relfro_d_vs_2b_exact=rf(gd, 2 * gb),
        b_abs_dB_over_lr_min_median_max=[absb.min().item(), absb.median().item(), absb.max().item()],
        b_n_unique_abs_first1M=int(torch.unique(absb[:1_000_000]).numel()),
        max_abs_g=gc.abs().max().item(),
        max_abs_g_over_eps={n: (g.abs().max() / P[n][1]).item() for g, n in ((gb, "b_eps1"), (gc, "c_eps1e3"), (ge, "e_eps1e4"))},
        g_abs_quantiles_c={q: gc[gc != 0].abs()[torch.randperm(int((gc != 0).sum()), generator=torch.Generator().manual_seed(0))[:2_000_000]].quantile(q).item()
                           for q in (0.01, 0.5, 0.99)},
    )
    # per-tensor agreement (this is what per-layer norms depend on)
    pt = {"norm_ratio_b_over_c": [], "norm_ratio_e_over_c": [], "cos_c_e": [], "relfro_c_vs_e": [], "norm_ratio_d_over_b": []}
    lm = {}
    for k in keys:
        xb, xc, xd, xe = (g_exact(D[n][k], *P[n]) for n in ("b_eps1", "c_eps1e3", "d_eps1_w2", "e_eps1e4"))
        if xc.norm() == 0:
            continue
        r = dict(norm_ratio_b_over_c=(xb.norm() / xc.norm()).item(), norm_ratio_e_over_c=(xe.norm() / xc.norm()).item(),
                 cos_c_e=torch.nn.functional.cosine_similarity(xc.flatten(), xe.flatten(), dim=0).item(),
                 relfro_c_vs_e=rf(xe, xc), norm_ratio_d_over_b=(xd.norm() / xb.norm()).item())
        if "lm_head" in k:
            lm = r
            continue
        for kk in pt:
            pt[kk].append(r[kk])
    v["per_tensor_layers_min_median_max"] = {kk: [min(x), sorted(x)[len(x) // 2], max(x)] for kk, x in pt.items()}
    v["per_tensor_lm_head"] = lm
    return v


def per_layer(dB, lr, eps):
    """Frobenius norm of g_B (exact inverse) per module type and layer."""
    tab = {m: {} for m in MODS}
    other = {}
    for k, x in dB.items():
        m = KEY_RE.match(k)
        layer, mod = m[1], m[2]
        nrm = g_exact(x, lr, eps).norm().item()
        if layer is None or mod not in tab:
            other[f"{mod}" + (f".L{layer}" if layer else "")] = nrm
        else:
            tab[mod][int(layer)] = nrm
    return {"per_layer": {m: [tab[m].get(l) for l in range(24)] for m in MODS}, "other": other}


def group_summary(pl):
    out = {}
    for m, vals in pl.items():
        if any(v is None for v in vals):
            out[m] = None
            continue
        g = lambda lo, hi: sum(vals[lo:hi]) / (hi - lo)
        pre, loop, post = g(0, 6), g(6, 18), g(18, 24)
        out[m] = dict(mean_0_5=pre, mean_6_17=loop, mean_18_23=post,
                      ratio_looped_over_unlooped=loop / ((pre * 6 + post * 6) / 12),
                      jump_5_to_6=vals[6] / vals[5], jump_17_to_18=vals[18] / vals[17])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists():  # keep earlier runs' errors and timings
        prev = json.loads(LOG.read_text())
        log["previous_runs"] = prev.pop("previous_runs", []) + [prev]
    D, M, choice = {}, {}, {}
    for name, cfg in SETTINGS.items():
        dB, meta = get_setting(name, cfg, choice)
        if name in VALID or name in BATCHES:
            D[name] = dB
        M[name] = meta
        choice[name.split("_")[0]] = (meta["lr"], meta["eps"])
        if all(n in D for n in VALID) and not (OUT / "validation.json").exists():
            val = validate(D, M)
            (OUT / "validation.json").write_text(json.dumps({"validation": val, "meta": {n: M[n] for n in VALID}}, indent=1, default=str))
            print(json.dumps(val, indent=1), flush=True)
            for n in VALID:
                if n not in BATCHES:
                    del D[n]  # free memory
    layers = {}
    for name in BATCHES:
        pl = per_layer(D[name], M[name]["lr"], M[name]["eps"])
        layers[name] = dict(data_seed=M[name]["data_seed"], lr=M[name]["lr"], eps=M[name]["eps"],
                            loss_metrics=M[name]["loss_metrics"], **pl, groups=group_summary(pl["per_layer"]))
    (OUT / "per_layer_grad_norms.json").write_text(json.dumps(layers, indent=1, default=str))
    for name, r in layers.items():
        print(f"\n== {name} ==")
        for m, s_ in r["groups"].items():
            if s_:
                print(f"{m:16s} pre {s_['mean_0_5']:.3e} loop {s_['mean_6_17']:.3e} post {s_['mean_18_23']:.3e} "
                      f"ratio {s_['ratio_looped_over_unlooped']:.2f} j6 {s_['jump_5_to_6']:.2f} j18 {s_['jump_17_to_18']:.2f}")
    save_log()


if __name__ == "__main__":
    main()
