# Standalone RobustToM-RL RFT

This directory implements rejection-sampling fine-tuning independently of
`verl`. It consumes the process-target JSONL files produced by the project and
exports a standard Hugging Face checkpoint. No canonical `process_response` is
used as a training completion.

## Pipeline

1. `prepare_data` creates the fixed train/dev/test split and validates pairs.
2. An external vLLM run (or `rft.sample`) writes raw candidates.
3. `score_candidates` accepts only strict JSON candidates with reward `1.0` and EOS.
4. `build_dataset` deduplicates by parsed JSON and keeps only complete pairs.
5. `train` performs response-only causal-LM training: prompt labels are `-100`,
   and the accepted response plus one EOS are trainable.
6. `generate` and `evaluate` report process, pair and shortcut-conflict metrics.

The dataset builder keeps every full-reward response by default and does not
require observed/hidden pairs. It never falls back to gold/canonical responses
when coverage is low. Use `--require-complete-pairs` only for the stricter
pair-balanced ablation.

## Environment

Run all commands from the repository root on the A800 machine:

```bash
conda create -n robusttom-rft python=3.10 -y
conda activate robusttom-rft

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r rft/requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

`flash-attn` is optional. If it is unavailable, `rft.train` automatically
falls back to the Transformers attention implementation.

## Prepare the fixed split

The derived files are already present in the repository. Regenerate and
validate them when the source JSONL changes:

```bash
python -m rft.prepare_data \
  --input-dir data/counterfactual_process_reward \
  --output-dir data/rft/derived \
  --seed 2026
```

## Existing candidates

If candidates have already been sampled, point `rft/run_rft.sh` at that JSONL:

```bash
CANDIDATES=/path/to/candidates.jsonl bash rft/run_rft.sh
```

Candidate rows must contain `global_sample_id`, `generation_reached_eos`, and
one of `raw_response`, `response`, or `accepted_response`. They may either
contain `process_target` and metadata or use `--data` to resolve those fields
by sample ID. Missing EOS status is rejected rather than assumed successful.

## One-time sampling

For a new one-time vLLM run:

```bash
RUN_ID=20260819-qwen25-3b-k16
MODEL=Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m rft.sample \
  --data data/rft/derived/train.jsonl \
  --model "${MODEL}" \
  --output "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --num-samples 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --gpu-memory-utilization 0.85 \
  --seed 2026
```

The sampler applies the Qwen chat template exactly once and persists raw
responses, token IDs, stop reason, prompt hashes and generation parameters.
This command is one pass over 3,080 prompts and produces 49,280 raw candidates.

## Score and build the accepted dataset

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/scored.jsonl"

python -m rft.build_dataset \
  --scored "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --output "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026
```

Only responses with process reward `1.0`, strict JSON format and a normal EOS
are accepted. By default all accepted trajectories are retained, up to
`--max-samples`; observed and hidden sides may be incomplete. Add
`--deduplicate-semantic` to collapse equivalent JSON responses, or add
`--require-complete-pairs` for the previous pair-balanced behavior.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m rft.train \
  --model "${MODEL}" \
  --train-file "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --output-dir "runs/rft_train/${RUN_ID}" \
  --logging-dir "runs/rft_train/${RUN_ID}/tensorboard" \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --logging-steps 10 \
  --seed 2026
```

Before loading the model, training writes `token_length_report.json`. If any
sample exceeds 2,048 tokens, the command fails instead of truncating the story;
rerun with `--max-seq-length 4096`. The standard Hugging Face checkpoint is
written to `runs/rft_train/${RUN_ID}/final`.

TensorBoard records `loss`, `grad_norm`, `learning_rate` and `epoch` every 10
optimizer steps. Start it in another terminal on the training machine:

```bash
conda activate robusttom-rft
tensorboard \
  --logdir "runs/rft_train/${RUN_ID}/tensorboard" \
  --host 127.0.0.1 \
  --port 6006
```

For a remote A800 server, forward the port from your local machine:

```bash
ssh -L 6006:127.0.0.1:6006 <user>@<a800-host>
```

Then open `http://127.0.0.1:6006`. Use `--logging-steps 1` for a short smoke
run where per-step visibility is more useful than lower logging overhead.

## Evaluation

```bash
python -m rft.generate \
  --data data/rft/derived/dev.jsonl \
  --model "runs/rft_train/${RUN_ID}/final" \
  --output "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl"

python -m rft.evaluate \
  --predictions "runs/rft_eval/${RUN_ID}/dev_predictions.jsonl" \
  --data data/rft/derived/dev.jsonl \
  --output "runs/rft_eval/${RUN_ID}/dev_metrics.json"
```

Sealed order-4 test data must only be evaluated after model/config selection is
complete. The metrics include strict format, full process reward, answer and
core-state accuracy, pair accuracy, intervention sensitivity, shortcut-copy
rate, last-mention-copy rate, answer-state consistency, EOS rate and p95 length.
