# Hi-ToM Counterfactual Observation Pairs

This directory contains a deterministic counterfactual augmentation of the
official Hi-ToM JSON file. It targets two shortcuts without using an LLM to
rewrite or label examples.

## Intervention design

Every source question first receives a shared anchor event: all five characters
jointly observe the object move to a new container, making that location common
knowledge. It is then expanded into a matched pair using the same final
container:

1. `observed`: all characters named in the question jointly watch the final
   move. The answer at every ToM order changes to the final container.
2. `hidden`: those queried characters leave before the final move and receive
   no information. The reality answer (order 0) is the final container, while
   orders 1-4 retain the shared anchor location as their mental-state answer.

For higher-order questions, these paired examples invalidate or balance two
shortcuts:

- the original location when the earliest queried agent exits (invalidated in
  both variants);
- the last container mentioned in the story.

The data therefore tests whether a model conditions its answer on who observed
the causal event. It does **not** prove that every possible shortcut has been
removed.

## Files

- `Hi-ToM_counterfactual.json`: all records plus build metadata.
- `train.jsonl`: training records.
- `eval.jsonl`: held-out challenge records.
- `manifest.json`: source hash, counts, seed, and shortcut statistics.

Splitting is performed by source scenario, stratified by deception status and
story length. All five question orders and both interventions for a story stay
in the same split.

## Rebuild

Only the Python standard library is required.

```bash
python3 scripts/build_hi_tom_counterfactual.py \
  --source /Users/hiforever/Downloads/Hi-ToM_data.json \
  --output-dir data/hi_tom_counterfactual \
  --seed 42
```

Run validation tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Recommended minimal experiment

Train on `train.jsonl`, then report two numbers:

1. accuracy on the original Hi-ToM evaluation set;
2. paired accuracy on `eval.jsonl`, reported separately for `observed` and
   `hidden` and for question orders 1-4.

For an intervention pair to count as correct, both variants should be answered
correctly. This is stricter and more informative than averaging the two
variant accuracies.

The `manifest.json` file reports the known heuristic accuracies before and
after augmentation. These are dataset-level rule accuracies, not Qwen model
results.
