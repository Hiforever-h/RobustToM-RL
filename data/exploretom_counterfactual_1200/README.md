# ExploreToM Counterfactual Train 1200

This directory contains 1,200 training records generated with ExploreToM's
official `FullBeliefTracker` and `QuestionGenerator`. No external LLM is used
for generation or labeling.

## Why counterfactual generation is used

The official public sample has strong procedural shortcuts:

- `Last-container`: for first/second-order container-location questions, the
  final container movement is correct about 75% of the time.
- `Only-room`: among room-location questions with only one room mentioned, that
  room is the correct answer for all 214 covered examples.

The exact audit numbers are recorded in `manifest.json`.

## Pair construction

Each of 600 base scenarios produces two variants with the same initial move,
shared anchor move, final move, people, object, containers, and rooms:

1. `observed`: queried characters see the final move and leave afterward. Their
   answer is the final container.
2. `hidden`: queried characters leave before the final move. Their answer is
   the shared anchor container.

The final container is mentioned in both variants, so Last-container accuracy
is exactly 50%. Every story mentions two rooms, making Only-room inapplicable.
Choices A-F and answer positions are exactly balanced.

## Composition

- 600 first-order records (300 pairs before expansion).
- 600 second-order records (300 pairs before expansion).
- 600 `observed` and 600 `hidden` records.
- All records belong to the `train` split.

## Files

- `train.jsonl`: the 1,200 training records.
- `ExploreToM_counterfactual_train_1200.json`: combined JSON plus metadata.
- `manifest.json`: generation, balance, provenance, and shortcut statistics.

## Rebuild

```bash
python3 scripts/generate_exploretom_counterfactual_1200.py \
  --exploretom-repo tmp/ExploreToM \
  --official-sample tmp/ExploreToM/ExploreToM-data-sample.csv \
  --output-dir data/exploretom_counterfactual_1200 \
  --seed 2026
```

ExploreToM is distributed under CC-BY-NC-4.0. Preserve its attribution and
non-commercial restriction when redistributing or using this derived data.
