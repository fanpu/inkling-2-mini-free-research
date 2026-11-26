# 02. Tinker free-tier probe

Run with `scripts/00_probe_free_tier.py` and analysed with `scripts/01_inspect_probe.py` on 2026-11-26. SDK versions are `tinker 0.40.1` and `tml-renderers 0.5.0`. Raw statistics are in `artifacts/probe/first_step_stats.json`.

## The free tier works end to end

- `thinkingmachines/Inkling-2-Mini:peft:65536:free` is listed as trainable and sampleable with a 65,536-token context.
- One training step with AdamW and one with Dimuon both succeeded.
- `save_weights_external` followed by `RestClient.get_external_weights_urls` gives a PEFT-format `adapter_model.safetensors` we can download. At rank 8 it is 371 MB in float32.
- The API reports a checkpoint storage rate of $0.00 per GB-month for this account. Probe checkpoints still use the minimum TTL of one hour.
- One optimizer's full probe (create client, one step, two exports, two 371 MB downloads) took about 190 s.

## What the API lets us control

| Knob | Available | Notes |
|---|---|---|
| LoRA rank, seed | yes | |
| Module groups | `train_attn`, `train_mlp`, `train_unembed` | No per-layer selection |
| LoRA alpha | no | Export says `lora_alpha: 32` at rank 8. Unknown whether it changes with rank |
| AdamW | `learning_rate`, `beta1` (0.9), `beta2` (0.95), `eps` (1e-12), `weight_decay` (decoupled), `grad_clip_norm` | Set per step |
| Dimuon | `learning_rate`, `beta1` (0.9), `grad_clip_norm` | No weight decay. Optimizer family is fixed when the client is created |
| Custom loss | `forward_backward_custom` | Not tried yet |
| Upload weights | `external_weights_upload_start/finish` | Marked experimental. Not tried yet |

TML models do not support `get_tokenizer()`. Render chat with `tml_renderers.renderers.Inkling2()` as the probe script does.

## What gets a LoRA adapter

434 tensors, one A and one B per target, one copy per physical layer. A looped layer therefore has a single adapter that is reused on all three passes.

| Target (per layer) | Base shape | Adapter layout |
|---|---|---|
| `attn.wq_du`, `attn.gate_attn` | 3072 x 2304 | A 8 x 2304, B 3072 x 8 |
| `attn.wk_dv`, `attn.wv_dv` | 1024 x 2304 | A 8 x 2304, B 1024 x 8 |
| `attn.wr_du` | 384 x 2304 | A 8 x 2304, B 384 x 8 |
| `attn.wo_ud` | 2304 x 3072 | A 8 x 3072, B 2304 x 8 |
| `mlp.experts.w1`, `w3` | 128 experts, 1152 x 2304 | A shared by all experts, B per expert |
| `mlp.experts.w2` | 128 experts, 2304 x 1152 | A per expert, B shared |
| `lm_head` (once) | 201024 x 2304 | A 8 x 2304, B 201024 x 8 |

Not adapted: the loop controllers, the cross-pass queries, the MoE router (`mlp.gate`), norms and the `sconv` weights.

## The first optimizer step, measured

Setup: rank 8, seed 0, one 24-token chat example, learning rate 1e-3 for both optimizers.

- **Init.** B is exactly zero. A is random with standard deviation 0.012 (0.006 for `w2`). So the gradient on A is zero at step one and only B moves, under both optimizers.
- **AdamW.** Each entry of the B update has magnitude close to the learning rate, as expected for Adam's first step, which keeps only the sign of the gradient. The update norm is the learning rate times the square root of the entry count (0.157 for a 3072 x 8 matrix) and is identical in all 24 layers.
- **Dimuon.** Every attention B update has Frobenius norm 2.219e-3, whatever the matrix shape or layer. Its 8 singular values lie between 6.7e-4 and 9.6e-4, so the update is close to a scaled orthogonal matrix. In the two attention matrices checked, the top singular value of the product update (B update times A) is 5.00e-4, half the learning rate. We do not know the rule that produces this.
- **Same learning rate, very different step.** The whole-adapter update norm is 5.71 for AdamW and 0.093 for Dimuon, a ratio of about 61. Learning rates are not comparable across the two.
- **MoE sparsity.** In layer 10, 54 of 128 experts received an update to `w1` from the 24-token example. Experts that saw no token do not move.
- **Unembedding.** AdamW moved 200,058 of 201,024 rows of the `lm_head` B. Dimuon moved 2,079.

## What this means for the hypotheses

Both optimizers throw away the size of the gradient. A looped layer receives a gradient that is the sum over three passes, but after normalisation the update is the same size as in an unlooped layer. The flat per-layer profiles above are a property of the optimizers, and say nothing about the loop. H1 as first written (loop amplification visible in update norms) cannot be tested this way. See note 03 and DECISIONS D-007.
