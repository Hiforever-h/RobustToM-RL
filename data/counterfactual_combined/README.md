# Combined Counterfactual Dataset

This directory combines the generated Hi-ToM and ExploreToM counterfactual
data into the files used by a single training/evaluation pipeline:

- `train.jsonl`: 3,200 records (2,000 Hi-ToM + 1,200 ExploreToM)
- `val.jsonl`: 400 Hi-ToM records
- `test.jsonl`: 600 Hi-ToM order-4 OOD records

The training records are deterministically shuffled with seed 2026 so that the
two source datasets are not stored as separate contiguous blocks.

The original source records are preserved. Each merged record additionally has:

- `source_dataset`: `hi-tom` or `exploretom`
- `source_split`: the original source split name
- `global_sample_id`: source-prefixed, globally unique sample ID
- `global_pair_id`: source-prefixed, globally unique counterfactual pair ID
- `split`: normalized to `train`, `val`, or `test`

The original dataset-specific directories remain unchanged. See `manifest.json`
for counts by source, split, question order, and intervention type.

## Rebuild

```bash
python3 scripts/merge_counterfactual_datasets.py
```
