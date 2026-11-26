# 01. Architecture facts for Inkling 2 Mini

Sources are `artifacts/hf/inkling2mini_config.json` and `artifacts/hf/inkling2mini_README.md`, downloaded from Hugging Face on 2026-11-26, plus the tensor names in the model's `model.safetensors.index.json`.

## Verified

| Fact | Value | Source |
|---|---|---|
| Physical layers | 24 | `num_hidden_layers` |
| Looped layers | 6 to 17 (`core_start: 6`, `core_end: 18`) | `loop_config` |
| Passes through the looped layers | 3, fixed (`is_varloop: false`) | `loop_config` |
| Layer applications per token | 48 from 24 layers of weights | model card |
| Hidden size | 2304 | config |
| MoE | 128 experts per layer, 8 used per token, expert hidden size 1152 | config |
| Parameters | 26.4B total, 3.4B active, 5.4B effective active with 3 passes | model card |
| Numerics | mixed NVFP4 and MXFP8 | model card |
| Released | 2026-11-20, Apache 2.0 | model card |

The 12 looped layers carry extra per-layer tensors that the other 12 do not: `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight`. The model also has two global tensors, `middle_cycle_cross_pass_queries` and `middle_cycle_cross_pass_kv_queries`. These are the loop-specific machinery (`gate_rank: 64`). LoRA does not touch any of them (see note 02).

## Not verified

- **Pretraining optimizer.** The model card has no mention of Muon, AdamW or any optimizer. The earlier claim that Inkling was pretrained with Muon plus AdamW has no source in this repo. Treat it as unknown.
- **What the controllers compute.** Names suggest a gated low-rank message between passes. We have not read an implementation.

## A sibling worth knowing about

`thinkingmachines/TML-d24-varloop-research` has the same architecture with `is_varloop: true`. Its card says it was trained with 1 to 5 passes (sampled 1:2:6:2:1) and the pass count is chosen at inference. It would let us vary the loop count K directly. It is not in Tinker's model list today, so it is out of reach for free-tier work.
