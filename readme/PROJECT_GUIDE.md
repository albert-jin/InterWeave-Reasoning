# InterWeave Project Guide

This note maps the paper components to the codebase.

## Paper-to-Code Map

| Paper component | Main files |
|---|---|
| Frozen LLM backbone and decoding loop | `sglang_soft_thinking_pkg/python/sglang/srt/` |
| Latent semantic token construction | `sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py`, `sglang_soft_thinking_pkg/python/sglang/srt/layers/vocab_parallel_embedding.py` |
| Mode state and trajectory hooks | `sglang_soft_thinking_pkg/python/sglang/srt/managers/schedule_batch.py`, `sglang_soft_thinking_pkg/python/sglang/srt/managers/scheduler.py` |
| Actor-critic controller | `ppo_agent_model.py` |
| PPO controller training | `train_interweave_controller.py`, `train_ppo_controller.py` |
| Math/science evaluation | `eval_interweave_controller.py`, `eval_ppo_agent.py`, `matheval.py` |
| Code evaluation | `eval_interweave_code.py`, `eval_code_ppo_agent.py`, `humanevaleval.py`, `mbppeval.py`, `local_lcb_eval.py` |
| Batch scripts | `scripts/train_interweave_controller.sh`, `scripts/eval_interweave_math.sh`, `scripts/eval_interweave_code.sh` |
| Paper figures | `sources/` |

## Naming Compatibility

Several low-level arguments and fields are still named `soft_thinking`, especially inside the patched SGLang backend. They are retained for compatibility with the inherited backend. In this repository, the user-facing interpretation is:

- `force_mode=soft`: always use latent exploration.
- `force_mode=hard`: always use symbolic grounding.
- `force_mode=ppo`: use the learned InterWeave controller.
- `enable_soft_thinking`: enable the backend hooks required for InterWeave latent-token construction and controller arbitration.

## Expected Workflow

1. Install the patched SGLang backend.
2. Download or mount the frozen LLM backbone.
3. Train the controller on `datasets/train_gsm8k.json` or a domain-specific training set.
4. Evaluate the learned checkpoint with `force_mode=ppo`.
5. Compare against `force_mode=soft` and `force_mode=hard` baselines.

The repository does not require fine-tuning the backbone model. Only the lightweight arbitration controller is optimized.
