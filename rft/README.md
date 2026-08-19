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

The current implementation requires 1,000 to 3,000 final accepted samples by
default. It never falls back to gold/canonical responses when coverage is low.

## Existing candidates

If candidates have already been sampled, point `rft/run_rft.sh` at that JSONL:

```bash
CANDIDATES=/path/to/candidates.jsonl bash rft/run_rft.sh
```

Candidate rows must contain `global_sample_id`, `generation_reached_eos`, and
one of `raw_response`, `response`, or `accepted_response`. They may either
contain `process_target` and metadata or use `--data` to resolve those fields
by sample ID. Missing EOS status is rejected rather than assumed successful.

## New sampling

For a new one-time vLLM run:

```bash
python -m rft.sample \
  --data data/rft/derived/train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output runs/rft_sampling/candidates.jsonl \
  --num-samples 16 --seed 2026
```

The sampler applies the Qwen chat template exactly once and persists raw
responses, token IDs, stop reason, prompt hashes and generation parameters.

## Evaluation

```bash
python -m rft.generate \
  --data data/rft/derived/dev.jsonl \
  --model runs/rft_train/<run-id>/final \
  --output runs/rft_eval/dev_predictions.jsonl

python -m rft.evaluate \
  --predictions runs/rft_eval/dev_predictions.jsonl \
  --data data/rft/derived/dev.jsonl \
  --output runs/rft_eval/dev_metrics.json
```

Sealed order-4 test data must only be evaluated after model/config selection is
complete. The metrics include strict format, full process reward, answer and
core-state accuracy, pair accuracy, intervention sensitivity, shortcut-copy
rate, last-mention-copy rate, answer-state consistency, EOS rate and p95 length.
