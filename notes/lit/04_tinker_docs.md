# Tinker docs review (read 2026-11-26)

`D` = https://tinker-docs.thinkingmachines.ai

## 1. Dimuon
- "Dimuon is an optimizer developed in-house at TML specifically for LoRA finetuning." (D/tinker/optimizers/)
- LR: "Dimuon tends to need a learning rate 10–100× as large as AdamW's. ... Do not reuse your AdamW learning rate." "In our SFT runs, about 15× the AdamW learning rate was best"; in RL "closer to 60×". "We recommend tuning the learning rate, keeping `beta1=0.9`, and leaving gradient clipping off." (same page)
- How it updates the LoRA factors: NOT FOUND. `DimuonOptimizerConfig` only says "Selects the Dimuon optimizer for a training model." (D/tinker/api-reference/types/dimuonoptimizerconfig/)
- Paper or blog: NOT FOUND.
- Weight decay: NOT FOUND. Documented Dimuon parameters are learning rate, beta1 and grad clipping only.

## 2. Free tier
- "Free training and sampling have a 64K context limit and are subject to rate limits. Requests may be throttled when those limits are reached." "Prepaid accounts must have a positive balance to use the free tier." (https://thinkingmachines.ai/tinker/pricing/, D/tinker/models/models_and_pricing/)
- Rate-limit numbers and concurrency limits: NOT FOUND. The only related item is the SDK v0.22.4 note: "Sampling concurrency now follows the server-provided limit." (D/changelog/sdk/)
- Storage: "free checkpoint storage up to 100 GB or 100 checkpoints per organization, whichever limit you reach first"; at the limit "your session saves one last checkpoint and stops." (D/administration/checkpoint-storage-pricing/, pricing page)
- Export limits specific to the free tier: NOT FOUND.

## 3. Uploading weights
- `external_weights_upload_start`: "Start uploading weights to become an external weights checkpoint of a new training run, which cannot be trained on." Marked experimental. "The files must be exactly what `save_weights_external` exports for `base_model` and `lora_config`." (D/tinker/api-reference/restclient/)
- `external_weights_upload_finish` registers the checkpoint; then "Load the checkpoint into a training client of the same base model and LoRA config with `load_state` (weights only)." (same page)
- Changelog v0.39.0: "upload support is gated by the server." (D/changelog/sdk/)
- Sampling from, or `forward` on, an uploaded adapter: NOT FOUND (only `load_state` is described).
- Availability for Inkling 2 Mini or the free tier: NOT FOUND.

## 4. LoRA configuration
- `lora_alpha`: NOT FOUND. `LoraConfig` documents only `rank`, `seed`, `train_unembed`, `train_mlp` and `train_attn`. (D/tinker/api-reference/types/loraconfig/)
- `train_mlp`: "Whether to add loras to the MLP layers (including MoE layers)". `train_attn`: "...the attention layers". `train_unembed`: "...the unembedding layer". (same page)
- In the cookbook, `hyperparam_utils.py` says MoE experts use "a shared-outer LoRA scheme". (https://github.com/thinking-machines-lab/tinker-cookbook)
- Per-layer control: NOT FOUND.

## 5. AdamParams
- The `AdamParams` type page lists `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay` (decoupled) and `grad_clip_norm` ("0.0 means no clipping"). (D/tinker/api-reference/types/adamparams/)
- The optimizers page gives defaults of `learning_rate` 1e-4 and `eps` 1e-12. (D/tinker/optimizers/)
- Allowed ranges for `eps` or `learning_rate`: NOT FOUND.

## 6. Inkling 2 in the cookbook
- Renderer: `model_info.py` maps `Inkling-2-Mini` to the renderer `tml_v0` and strips everything after the first `:` in a model name. Training examples come from `build_supervised_example(messages, effort=...)`. The `tml_renderers` page names `renderers.Inkling2()` for Inkling-2-Small; it does not mention Mini. (D/cookbook/inkling/tml-renderers/, cookbook repo)
- Effort is a float in [0.0, 1.0). The presets are none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9 (the default) and xhigh 0.99. The renderer inserts the "Thinking effort level" system message itself. "effort=0.0 conditions the model toward no reasoning; it does not enforce a hard no-reasoning constraint." Generic SFT dataset builders use 0.9. To render training data at another level, call `build_supervised_example(..., effort=...)` and use the same value you sample with. (D/cookbook/inkling/thinking-effort/)
- Hyperparameters: `lr_multipliers.json` has `reference_lr` 0.00075 and `Inkling-2-Mini` 0.72. `get_lr` returns their product, 0.00075 × 0.72 = 5.4e-4, described as "an AdamW LoRA SFT sweep starting point, not a tuned optimum." (cookbook repo)

## 7. Forward, sampling and per-example loss
- "Free training and sampling" (pricing footnote). "forward only passes invoked by training clients are billed at the training price"; the free model's training price is $0.00. (D/tinker/models/models_and_pricing/)
- Loss: `elementwise_loss = -target_logprobs * weights`, summed over tokens (`loss:sum`). `loss_fn_outputs` holds per-token `logprobs`. (D/tinker/losses/cross-entropy/, D/tinker/losses/)
- A documented per-example loss recipe: NOT FOUND.
