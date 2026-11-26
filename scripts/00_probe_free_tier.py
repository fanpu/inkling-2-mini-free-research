"""Phase 0 probe: does the free-tier model train with AdamW and Dimuon, and what
does the exported LoRA adapter contain?

Runs one optimizer step per optimizer, exports the adapter, downloads it, and
writes a tensor inventory (name, shape, Frobenius norm) to artifacts/probe/.

Usage: .venv/bin/python scripts/00_probe_free_tier.py [adamw|dimuon ...]
"""
import json, sys, time, urllib.request, warnings
from pathlib import Path

import tinker
from tinker import types
from tml_renderers import chat, renderers
from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights

warnings.simplefilter("ignore", FutureWarning)

MODEL_ID = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
assert MODEL_ID.endswith(":free"), "free tier only (D-001)"
OUT = Path("artifacts/probe")
RANK = 8
TTL = 3600  # shortest allowed; probe checkpoints are throwaway


def main(opts):
    sc = tinker.ServiceClient()
    rest = tinker.RestClient()
    for opt in opts:
        t0 = time.time()
        cfg = types.DimuonOptimizerConfig() if opt == "dimuon" else types.AdamOptimizerConfig()
        tc = sc.create_lora_training_client(
            base_model=MODEL_ID, rank=RANK, seed=0, optimizer=cfg,
            user_metadata={"purpose": "phase0-probe", "opt": opt},
        ).result()
        print(f"[{opt}] client ready in {time.time()-t0:.0f}s", flush=True)
        info = tc.get_info()
        (OUT / f"{opt}_info.json").write_text(json.dumps(info.model_dump(), indent=1, default=str))
        # TML models have no get_tokenizer(); render chat with tml-renderers instead.
        ex = renderers.Inkling2().render_for_sft(chat.OpenAIMessage.from_openai_chat_dicts([
            {"role": "user", "content": "Name a pangram."},
            {"role": "assistant", "content": "The quick brown fox jumps over the lazy dog."},
        ]))[-1]
        mi, w = training_example_to_tinker_model_input_and_weights(ex)
        ids = mi.to_ints()
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={"target_tokens": ids[1:], "weights": w[1:]},
        )
        print(f"[{opt}] {len(ids)} tokens, {sum(x > 0 for x in w)} supervised", flush=True)
        params = (types.DimuonParams(learning_rate=1e-3) if opt == "dimuon"
                  else types.AdamParams(learning_rate=1e-3))
        saved = {}
        for step in (0, 1):
            ext = tc.save_weights_external(f"probe-{opt}-step{step}", ttl_seconds=TTL).result()
            saved[step] = ext.path
            print(f"[{opt}] step {step} exported: {ext.path}", flush=True)
            if step == 0:
                fb = tc.forward_backward([datum], "cross_entropy").result()
                print(f"[{opt}] fwd-bwd metrics: {fb.metrics}", flush=True)
                os_ = tc.optim_step(params).result()
                print(f"[{opt}] optim_step: {os_}", flush=True)
        for step, path in saved.items():
            urls = rest.get_external_weights_urls(path).result()
            d = OUT / f"{opt}_step{step}"
            d.mkdir(parents=True, exist_ok=True)
            for rel, url in urls.urls.items():
                dest = d / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, dest)
                print(f"[{opt}] downloaded {rel} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        print(f"[{opt}] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1:] or ["adamw", "dimuon"])
