# Hi-ToM Counterfactual 3000

This dataset is generated from the official Hi-ToM simulator and then expanded
with the observed/hidden counterfactual intervention described in
`../hi_tom_counterfactual/README.md`.

## Exact split

| Split | ToM orders | Records |
| --- | --- | ---: |
| Train | 0-3 | 2,000 |
| Validation | 0-3 | 400 |
| OOD test | 4 only | 600 |

Every order has exactly 600 final records. Each source question produces an
`observed` and a `hidden` record, and pairs always remain in the same split.

The order-4 pool is generated from a separate set of 300 base stories. None of
its base stories occur in train or validation. Train and validation use another
300 base stories, with 250/50 story-level splitting.

Both pools are balanced over:

- story lengths 1, 2, and 3;
- deception/communication disabled and enabled;
- 50 base stories per `(pool, story length, deception)` stratum.

## Files

- `train.jsonl`: 2,000 counterfactual training records.
- `validation.jsonl`: 400 counterfactual validation records.
- `order4_ood_test.jsonl`: 600 order-4 OOD records.
- `Hi-ToM_counterfactual_3000.json`: complete combined dataset.
- `generated_source.jsonl`: 1,500 Oracle-labeled questions before pairing.
- `manifest.json`: generation and shortcut statistics.

The manifest records the exact upstream Hi-ToM commit and random seed used for
this build.

## Rebuild

The upstream checkout and NumPy are required:

```bash
python3 scripts/generate_hi_tom_counterfactual_3000.py \
  --hi-tom-repo tmp/Hi-ToM_dataset \
  --output-dir data/hi_tom_counterfactual_3000 \
  --seed 2026
```
