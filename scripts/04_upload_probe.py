"""Probe: can a locally edited LoRA adapter be uploaded to the Tinker free tier and
evaluated with a forward-only loss?

Steps (see notes/05_upload_probe.md):
1. Rank-2 AdamW LoRA client (seed 0). Baseline forward loss on 8 short chats.
2. Export the step-0 adapter, download it, add a random matrix of Frobenius norm 0.5
   to one lora_B tensor.
3. Upload the edited adapter (external_weights_upload_start/finish), load it with
   load_state (fallback: create_training_client_from_state), forward on the same batch.
4. Verify: re-export from the loaded client and compare the edited tensor. Control:
   upload the unedited adapter, load it into the same client, loss should equal baseline.
5. Time a second full cycle (edit -> upload -> load -> forward) on another layer.

Writes artifacts/upload_probe/result.json (no weights). Downloaded and created
.safetensors files are deleted at the end.

Usage: .venv/bin/python scripts/04_upload_probe.py
"""
import json, shutil, time, traceback, urllib.request, warnings
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import tinker
from tinker import types
from tml_renderers import chat, renderers
from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore")  # ExperimentalWarning on the upload API

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID == "thinkingmachines/Inkling-2-Mini:peft:65536:free", "free tier only (D-001)"
OUT = Path("artifacts/upload_probe")
TMP = OUT / "tmp"
TTL = 3600
SEED = 0
EDIT_NORM = 0.5
TARGET = "language_model.layers.10.attn.wo_ud.lora_B.weight"
TARGET2 = "language_model.layers.5.attn.wo_ud.lora_B.weight"  # for the timed cycle

CHATS = [
    ("Name a pangram.", "The quick brown fox jumps over the lazy dog."),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Add 17 and 25.", "17 plus 25 equals 42."),
    ("Give me a synonym for happy.", "Joyful is a synonym for happy."),
    ("What color is the sky on a clear day?", "On a clear day the sky is blue."),
    ("Translate 'thank you' into Spanish.", "In Spanish, 'thank you' is 'gracias'."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("Name a prime number between 10 and 20.", "13 is a prime number between 10 and 20."),
]

R = {"model_id": MODEL_ID, "ttl_seconds": TTL, "seed": SEED, "errors": [], "timings_s": {}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def err(stage, e):
    R["errors"].append({"stage": stage, "type": type(e).__name__, "message": str(e)})
    log(f"ERROR at {stage}: {type(e).__name__}: {e}")


def polite(fn, stage, tries=4):
    """Retry with backoff only on throttling or transient server errors."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            s = str(e).lower()
            transient = any(k in s for k in ("429", "rate limit", "throttl", "503", "502", "timeout"))
            if not transient or i == tries - 1:
                raise
            wait = 15 * 2 ** i
            R["errors"].append({"stage": stage, "type": type(e).__name__,
                                "message": str(e), "retried_after_s": wait})
            log(f"{stage}: transient error, retrying in {wait}s: {e}")
            time.sleep(wait)


def build_batch():
    rend = renderers.Inkling2()
    data, n_sup = [], 0
    for u, a in CHATS:
        ex = rend.render_for_sft(chat.OpenAIMessage.from_openai_chat_dicts([
            {"role": "user", "content": u}, {"role": "assistant", "content": a}]))[-1]
        mi, w = training_example_to_tinker_model_input_and_weights(ex)
        ids = mi.to_ints()
        w = list(w)
        n_sup += sum(x > 0 for x in w[1:])
        data.append(types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                                loss_fn_inputs={"target_tokens": ids[1:], "weights": w[1:]}))
    return data, n_sup


def fwd_loss(tc, data):
    out = polite(lambda: tc.forward(data, "cross_entropy").result(), "forward")
    total = 0.0
    for d, o in zip(data, out.loss_fn_outputs):
        tl = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
        lp, w = tl(o["logprobs"]), tl(d.loss_fn_inputs["weights"])
        total += -sum(a * b for a, b in zip(lp, w))
    return {"loss_sum_metric": out.metrics.get("loss:sum"), "loss_sum_from_logprobs": total}


def download(rest, path, dest):
    if dest.exists():
        shutil.rmtree(dest)
    urls = polite(lambda: rest.get_external_weights_urls(path).result(), "get_urls")
    for rel, url in urls.urls.items():
        f = dest / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, f)
    return sorted(urls.urls)


def upload(rest, src_dir, lora_cfg):
    st = polite(lambda: rest.external_weights_upload_start(MODEL_ID, lora_cfg).result(),
                "upload_start")
    R.setdefault("upload_file_keys", sorted(st.urls))
    for name, u in st.urls.items():
        u.upload(src_dir / name)
    fin = polite(lambda: rest.external_weights_upload_finish(st.upload_id).result(),
                 "upload_finish")
    return fin.tinker_path


def make_edit(orig_sd, name, seed, out_dir, src_dir):
    t = orig_sd[name]
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(t.shape, generator=g, dtype=torch.float32)
    d = d * (EDIT_NORM / d.norm())
    sd = dict(orig_sd)
    sd[name] = (t.float() + d).to(t.dtype).contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(sd, out_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    for f in src_dir.iterdir():
        if f.name != "adapter_model.safetensors":
            shutil.copy(f, out_dir / f.name)
    return sd[name].clone(), float((sd[name].float() - t.float()).norm())


def load_into(sc, tc, path, stage):
    """Try load_state on the given client, else a fresh client from the uploaded path."""
    try:
        polite(lambda: tc.load_state(path).result(), stage + "/load_state")
        return tc, "load_state"
    except Exception as e:
        err(stage + "/load_state", e)
    try:
        tc2 = polite(lambda: sc.create_training_client_from_state(
            path, base_model=MODEL_ID).result(), stage + "/from_state")
        return tc2, "create_training_client_from_state"
    except Exception as e:
        err(stage + "/from_state", e)
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    sc, rest = tinker.ServiceClient(), tinker.RestClient()
    data, n_sup = build_batch()
    R["batch"] = {"n_examples": len(data), "n_supervised_tokens": n_sup}

    # 1. client and baseline
    rank = None
    for rk in (2, 4):
        t0 = time.time()
        try:
            tc = sc.create_lora_training_client(
                base_model=MODEL_ID, rank=rk, seed=SEED, optimizer=types.AdamOptimizerConfig(),
                user_metadata={"purpose": "upload-probe"}).result()
            rank = rk
            R["timings_s"]["create_client"] = time.time() - t0
            break
        except Exception as e:
            err(f"create_client_rank{rk}", e)
    if rank is None:
        return
    R["rank"] = rank
    lora_cfg = types.LoraConfig(rank=rank, seed=SEED)
    log(f"client ready, rank {rank}")
    t0 = time.time()
    R["baseline"] = fwd_loss(tc, data)
    R["timings_s"]["forward"] = time.time() - t0
    log(f"baseline {R['baseline']}")

    # 2. export, download, edit
    t0 = time.time()
    ext = polite(lambda: tc.save_weights_external("upload-probe-step0", ttl_seconds=TTL).result(),
                 "save_weights_external")
    R["step0_export_path"] = ext.path
    R["export_files"] = download(rest, ext.path, TMP / "orig")
    R["timings_s"]["export_and_download"] = time.time() - t0
    orig_file = TMP / "orig" / "adapter_model.safetensors"
    R["adapter_file_mb"] = orig_file.stat().st_size / 1e6
    R["adapter_config"] = json.loads((TMP / "orig" / "adapter_config.json").read_text())
    orig_sd = load_file(orig_file)
    R["target_tensor"] = {"name": TARGET, "shape": list(orig_sd[TARGET].shape),
                          "dtype": str(orig_sd[TARGET].dtype),
                          "orig_norm": float(orig_sd[TARGET].float().norm())}
    log(f"downloaded {R['export_files']}, {R['adapter_file_mb']:.1f} MB, target {R['target_tensor']}")
    edited_t, dn = make_edit(orig_sd, TARGET, 1, TMP / "edit", TMP / "orig")
    R["edit_delta_norm_after_cast"] = dn

    # 3. upload edited and evaluate
    t0 = time.time()
    try:
        up_path = upload(rest, TMP / "edit", lora_cfg)
    except Exception as e:
        err("upload_edited", e)
        traceback.print_exc()
        return
    R["timings_s"]["upload_edited"] = time.time() - t0
    R["edited_upload_path"] = up_path
    log(f"uploaded edited: {up_path}")
    t0 = time.time()
    tc_e, route = load_into(sc, tc, up_path, "edited")
    if tc_e is None:
        try:  # last resort: sampling client from the uploaded path
            samp = sc.create_sampling_client(model_path=up_path).result()
            R["sampling_client_from_upload"] = "created"
            lp = samp.compute_logprobs(data[0].model_input).result()
            R["sampling_client_logprobs_ok"] = lp is not None
        except Exception as e:
            err("edited/sampling_client", e)
        return
    R["timings_s"]["load_edited"] = time.time() - t0
    R["load_route"] = route
    R["edited"] = fwd_loss(tc_e, data)
    log(f"edited ({route}) {R['edited']}")
    (TMP / "edit" / "adapter_model.safetensors").unlink()

    # 4a. re-export and compare
    t0 = time.time()
    ext2 = polite(lambda: tc_e.save_weights_external("upload-probe-reexport", ttl_seconds=TTL).result(),
                  "reexport")
    download(rest, ext2.path, TMP / "reexport")
    re_sd = load_file(TMP / "reexport" / "adapter_model.safetensors")
    R["reexport_check"] = {
        "target_max_abs_diff_vs_uploaded": float((re_sd[TARGET].float() - edited_t.float()).abs().max()),
        "target_max_abs_diff_vs_orig": float((re_sd[TARGET].float() - orig_sd[TARGET].float()).abs().max()),
        "other_tensors_max_abs_diff_vs_orig": max(
            float((re_sd[k].float() - orig_sd[k].float()).abs().max()) for k in orig_sd if k != TARGET),
    }
    del re_sd
    shutil.rmtree(TMP / "reexport")
    R["timings_s"]["reexport_and_download"] = time.time() - t0
    log(f"reexport {R['reexport_check']}")

    # 4b. control: unedited upload into the same client (tests reuse)
    t0 = time.time()
    try:
        ctrl_path = upload(rest, TMP / "orig", lora_cfg)
        tc_c, route_c = load_into(sc, tc_e, ctrl_path, "control")
        if tc_c is not None:
            R["control_load_route"] = route_c
            R["control"] = fwd_loss(tc_c, data)
            log(f"control ({route_c}) {R['control']}")
    except Exception as e:
        err("control", e)
    R["timings_s"]["control_cycle"] = time.time() - t0

    # 5. timed full cycle on another layer, reusing the client
    try:
        t0 = time.time()
        _, _ = make_edit(orig_sd, TARGET2, 2, TMP / "edit", TMP / "orig")
        t1 = time.time()
        p = upload(rest, TMP / "edit", lora_cfg)
        t2 = time.time()
        tc_t, route_t = load_into(sc, tc_e, p, "timed")
        t3 = time.time()
        R["timed_cycle"] = {"tensor": TARGET2, "load_route": route_t,
                            "loss": fwd_loss(tc_t, data) if tc_t else None}
        t4 = time.time()
        R["timed_cycle"]["seconds"] = {"edit_save": t1 - t0, "upload": t2 - t1, "load": t3 - t2,
                                       "forward": t4 - t3, "total": t4 - t0}
        log(f"timed cycle {R['timed_cycle']}")
    except Exception as e:
        err("timed_cycle", e)


if __name__ == "__main__":
    t_all = time.time()
    try:
        main()
    except Exception as e:
        err("main", e)
        traceback.print_exc()
    finally:
        R["timings_s"]["total"] = time.time() - t_all
        if TMP.exists():
            shutil.rmtree(TMP)
        (OUT / "result.json").write_text(json.dumps(R, indent=1, default=str))
        log(f"wrote {OUT / 'result.json'}")
