# RobustToM v3 GRPO

This directory contains the task-specific adapter and reward integration for
training `runs/final` with the bundled verl implementation. The scorer is
imported directly from `rft.reward`; the RFT implementation is not modified.

Create the independent RL environment on the A800 host:

```bash
conda create -n robusttom-grpo python=3.10 -y
conda activate robusttom-grpo
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e . --no-deps
wandb login
```

The main configuration logs to both the terminal and Weights & Biases under
project `robust_tom_grpo_v3`. Set `WANDB_API_KEY` non-interactively on a remote
host, or use `WANDB_MODE=offline` when outbound network access is unavailable.

Build the audited parquet files:

```bash
bash grpo/run_grpo_v3.sh build
```

Run the A800 checks in order:

```bash
bash grpo/run_grpo_v3.sh validate
bash grpo/run_grpo_v3.sh smoke
bash grpo/run_grpo_v3.sh pilot
bash grpo/run_grpo_v3.sh train
```

The main run uses 8 prompts per step, 16 rollouts per prompt, 800 optimizer
steps, two epochs, a learning rate of `5e-7`, temperature `1.0`, and asymmetric
PPO clipping with ratio bounds `[0.8, 1.3]`.

Resume from an actor checkpoint without changing the frozen reference:

```bash
bash grpo/run_grpo_v3.sh train \
  trainer.resume_from_path=runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/actor/global_step_400
```
