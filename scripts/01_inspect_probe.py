"""Compare step-0 vs step-1 adapters from scripts/00_probe_free_tier.py.

For each optimizer prints init statistics and, per module type, how the first
update dW = W1 - W0 looks (norms, singular values) across the 24 layers.
"""
import collections, json, re, sys
import torch
from safetensors.torch import load_file

P = "artifacts/probe"
out = {}
for opt in sys.argv[1:] or ["adamw", "dimuon"]:
    w0 = load_file(f"{P}/{opt}_step0/adapter_model.safetensors")
    w1 = load_file(f"{P}/{opt}_step1/adapter_model.safetensors")
    rows = collections.defaultdict(dict)
    for k in sorted(w0):
        a, b = w0[k].float(), w1[k].float()
        d = b - a
        m = re.match(r"language_model\.(?:layers\.(\d+)\.)?(.*)\.lora_([AB])\.weight", k)
        layer, mod, ab = (int(m[1]) if m[1] else -1), m[2], m[3]
        d2 = d.reshape(-1, *d.shape[-2:])  # (experts, out, in)
        sv = torch.linalg.svdvals(d2[0])
        rows[(mod, ab)][layer] = dict(
            dtype=str(w0[k].dtype), init_fro=a.norm().item(), init_std=a.std().item(),
            d_fro=d.norm().item(), d_absmax=d.abs().max().item(),
            d_frac_nonzero=(d != 0).float().mean().item(),
            d_abs_unique=torch.unique(d.abs().flatten()[:200000]).numel(),
            sv_max=sv.max().item(), sv_min=sv.min().item(),
        )
    out[opt] = {f"{m}.{ab}": v for (m, ab), v in rows.items()}
    print(f"\n===== {opt} =====")
    for (mod, ab), per in rows.items():
        ls = sorted(per)
        r0 = per[ls[0]]
        print(f"\n{mod}.lora_{ab} [{r0['dtype']}] init_fro={r0['init_fro']:.4g} init_std={r0['init_std']:.4g} "
              f"nonzero={r0['d_frac_nonzero']:.3f} |d|uniq={r0['d_abs_unique']}")
        for name in ("d_fro", "sv_max", "sv_min"):
            print(f"  {name:7s}", " ".join(f"{per[l][name]:.2e}" for l in ls))
json.dump(out, open(f"{P}/first_step_stats.json", "w"), indent=1)
