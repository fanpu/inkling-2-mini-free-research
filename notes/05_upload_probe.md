# 05. Uploading an edited LoRA adapter

Run with `scripts/04_upload_probe.py` on 2026-11-26 (tinker 0.40.1). Output is `artifacts/upload_probe/result.json`.

## Result

Edited adapters cannot be evaluated on this account. The upload API refuses at its first call, before any file is sent. The exact error is

`BadRequestError: Error code: 400 - {'detail': 'Uploading external weights is not enabled for this organization.'}`

This matches changelog v0.39.0, which says upload support is gated by the server. The gate is per organization.

## What worked before the refusal

- A rank-2 AdamW client (seed 0) was accepted and created in 5 s.
- The baseline forward-only loss on 8 short chats (107 supervised tokens) is 242.6456 summed over tokens. B is zero at init, so this is the base model's loss.
- The first `forward` call took 101 s. Export plus download took 36 s. The rank-2 file is 92.8 MB in float32.
- The export's `adapter_config.json` says `lora_alpha: 32` with `r: 2`. Rank 8 also had alpha 32, so alpha does not scale with rank and the LoRA scale $\alpha/r$ is 16 at rank 2.
- Adding a random matrix of Frobenius norm 0.5 to `language_model.layers.10.attn.wo_ud.lora_B.weight` (2304 x 2) works locally.

Load, re-export check, control and the timed cycle were never reached.

## Other routes, checked in the SDK source

Every server call that brings weights into a client takes a `tinker://` path to a checkpoint already on the server.

- `load_weights` (used by `load_state` and `create_training_client_from_state`) accepts only `path`, plus an optional access token for another account's checkpoint.
- `copy_weights` copies an existing checkpoint.
- `create_sampling_client(model_path=...)` also takes a `tinker://` path.

Only the two gated upload endpoints create a checkpoint from local files. So the free tier offers no way to put a locally edited tensor into a training or sampling client.

## Recommendation

The exact perturbation experiment needs upload enabled for the organization. If it is enabled, the recipe is to upload the full adapter (both files, same base model and `LoraConfig(rank=2, seed=0)`), call `load_state` on one training client, and run `forward`. The script does this, with the re-export check and the unedited control, and can be rerun unchanged. The per-cycle cost is unknown. One forward on a fresh client took 101 s.

Without upload, the experiment has to be redesigned around changes the server can make itself, such as optimizer steps. Those cannot target a single layer.
