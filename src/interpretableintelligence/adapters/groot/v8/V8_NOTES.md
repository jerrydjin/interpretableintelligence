# GR00T v8: pipeline diff vs lerobot 0.7.0 + fixes (2026-11-10)

## Bug fixed
**Qwen3-VL text RoPE rebuilt with theta 1e4 instead of 5e6** (low-RAM loader, used by v1–v7).
- v7 `low_ram_policy_load.py:336` calls `fn.__func__` on a staticmethod → AttributeError, swallowed →
  fallback `:348` `theta = getattr(config, "rope_theta", 10000.0)`. transformers 5.22 keeps theta in
  `config.rope_parameters` (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:384`), so the text RoPE got
  theta 1e4. Measured on box (meta-built policy): max |Δ inv_freq| = 0.1873 for
  `language_model.rotary_emb`; vision RoPE (theta 1e4) unaffected. Every one of the 16 backbone LLM layers
  saw wrong positions → VLM features off-distribution.
- Fix: `low_ram_policy_load.py` rebuilds each RoPE buffer from a real instance of its own class
  (`rebuild_rotary_inv_freq`, `repair_rotary_buffers`); `physint_inproc_v8.repair_rope()` runs in `setup()`
  and repairs an already-loaded policy in place. After rebuild or repair, each buffer is checked against
  that module's own config `rope_theta` (Qwen3-VL text 5e6, vision its own) and a mismatch raises
  `RuntimeError` (module name, expected vs actual theta, max |Δ inv_freq|). There is no silent
  `except` → theta=10000 fallback. Check: `tests/check_rope_rebuild.py`, `tests/test_rope_theta_verify.py`.

## Checked, matches lerobot 0.7.0 (no change)
| Suspect | Evidence |
| --- | --- |
| (a) image flip / keys | v7 `:478` uses lerobot `canonical_observation` (`sim/integrations/libero.py:49-58`, 180° flip). Live box render vs dataset ep10 frame0: corr 0.914 vs 180-flip −0.383 (agentview), 0.753 vs −0.149 (wrist). Keys `image`/`wrist_image` = checkpoint `input_features`; pack step order = `sorted()` (`processor_groot.py:1529`), same as the lerobot default `image2`. |
| obs size | v7 256×256 = checkpoint `image_size` + dataset (lerobot `LiberoEnvConfig` default is 360, `envs/configs.py:480`; 256 is the right choice). |
| (b) state | 8-D eef pos + axis-angle + gripper qpos (`libero.py:54-56`); live init [-0.203 0.012 1.181 3.140 0.004 -0.087 0.0387 -0.0387] vs dataset [-0.197 0.002 1.167 3.139 -0.000 -0.095 0.0388 -0.0388]; normalized state matches. |
| (c) actions | relative/delta control, min-max unnormalize, `-sign(2g-1)` gripper (`processor_groot.py:2478`); dataset gripper 1 = open → −1 = LIBERO open. Queue: `_action_queue_steps` = 16 (meta-built policy), 16 of 40 horizon executed, 1 predict per 16 steps (mock count). |
| (d) processor | All processor files in Qwen/Qwen3-VL-2B-Instruct have the same git blob ids as nvidia/Cosmos-Reason2-2B (HF tree API): chat_template.json, preprocessor_config.json, video_preprocessor_config.json, tokenizer*.json, vocab, merges. Image crop/resize come from the checkpoint's `policy_preprocessor.json`, not the repo. |
| (f) language | bddl `(:language)` = dataset task 6 "put both moka pots on the stove". |

## Not fixed / open
- (e) fp16 params + no autocast vs checkpoint `use_bf16: true, model_params_fp32: true`. The diag cell measures fp16 vs bf16 and the backbone's max |h|; the notebook switches to bf16 if fp16 fails open-loop and bf16 passes.
- Speed: on the box, env.step is 0.148 s and almost all of it is rendering (two 256² cameras every step), so ~77 s per 520 steps before any policy time. The diag cell prints the T4 breakdown.
- The checkpoint was trained with lerobot 0.6.1 and is evaluated here with 0.7.0. Processor drift between those versions is not checked.
