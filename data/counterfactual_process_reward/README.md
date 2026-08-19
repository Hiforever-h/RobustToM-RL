# Counterfactual Process-Reward Dataset

This directory contains the unified Hi-ToM and ExploreToM splits augmented
with deterministic process supervision. The original dataset directories are
unchanged.

Each record adds:

- `process_target_version`: target schema version.
- `process_target`: the programmatically verified reasoning state.
- `process_prompt`: a prompt requesting only the corresponding JSON object.
- `process_response`: the compact canonical JSON completion for SFT/RFT.

Belief questions use `tom_order`, `belief_chain`, `object`, `reasoning_mode`,
`final_move_observed`, `nested_belief`, and `answer`. Order-0 factual questions
replace observation and belief fields with `world_state`.

No LLM judge or API-generated labels are used. Labels come from the original
question metadata and the deterministic observed/hidden counterfactual
construction.

## Files

- `train.jsonl`: 3,200 records.
- `val.jsonl`: 400 records.
- `test.jsonl`: 600 order-4 OOD records.
- `manifest.json`: provenance and aggregate target statistics.

## Rebuild

```bash
python3 scripts/add_process_targets.py
```

The scorer is importable as `process_reward()` from
`scripts/process_reward.py`. It returns a scalar in `[0, 1]`; the
`score_process_output()` function additionally returns per-stage diagnostics.

The current implementation scores each record independently. It does not
define a pair reward or calculate pair accuracy.

## Reward weights

Belief questions use format 0.05, order 0.05, chain 0.10, object 0.05,
visibility 0.25, nested belief 0.35, and answer 0.15. Nested-belief reward is
gated by correct visibility; answer reward is gated by both.

Order-0 questions use format 0.05, order 0.10, object 0.10, world state 0.50,
and answer 0.25. Answer reward is gated by the correct world state.
