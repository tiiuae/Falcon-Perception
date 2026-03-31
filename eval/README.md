# Falcon Perception — Evaluation

This folder contains benchmark evaluation scripts for Falcon Perception.

## PBench

PBench is a grounded segmentation benchmark with 6 splits of increasing difficulty
(`level_0` → `level_4`) plus a `dense` split (multi-instance scenes).

### Quick start

```bash
# Default: streams level_0 from HuggingFace, first 100 samples
python eval/pbench.py

# Full level_0
python eval/pbench.py --split level_0 --limit 0

# All 6 splits in one run → prints a final summary table
python eval/pbench.py --split all --limit 0

# Save results to disk
python eval/pbench.py --split all --limit 0 --out-dir ./results/pbench/

# Local model export (skip HF download)
python eval/pbench.py --hf-local-dir /path/to/export --split level_1

# Resolution ablation
python eval/pbench.py --split level_0 --max-dimension 768

# See all options
python eval/pbench.py --help
```

### Evaluation protocol

1. **Force-resize** — each image is scaled so its longest edge equals
   `--max-dimension` (default 1024) using LANCZOS resampling before inference.
   This is different from the soft clamp used in the demo scripts, and is
   required to reproduce published PBench numbers.

2. **Inference** — the paged inference engine generates segmentation masks
   conditioned on the expression query.

3. **Mask alignment** — predicted masks are output at the upsampled inference
   resolution.  They are resized back to the **original image resolution**
   (nearest-neighbor) before scoring.  Ground-truth masks in PBench are at
   the original resolution.

4. **NMS** — greedy area-sorted NMS at IoU=0.5 is always applied to predicted
   masks before scoring.

### Metrics

| Metric | Description |
|---|---|
| **F1** | Mean of per-sample F1 scores (computed over positive GT samples only). Each sample's F1 is the average across all IoU thresholds. |
| **IL TP/TN/FP/FN** | Image-level classification counts. IL TP = GT has objects and model predicted at least one; IL FP = GT is empty but model predicted masks; etc. |

**IoU thresholds**: `[0.5, 0.55, 0.60, …, 0.95]` (10 thresholds) for
`level_0`–`level_4`; `[0.5]` only for the `dense` split.

Per-sample F1 uses **Hungarian matching** (optimal bipartite assignment) between
predicted and GT masks at each threshold, so every GT mask is matched to at
most one prediction.

### Output

Each split produces a JSON file (when `--out-dir` is set):

```
eval_results/pbench/
├── level_0_results.json
├── level_1_results.json
├── ...
└── summary.json          ← only when --split all
```

Example `level_0_results.json`:

```json
{
  "f1": 0.612,
  "il_tp": 95,
  "il_tn": 3,
  "il_fp": 1,
  "il_fn": 1,
  "n_samples": 100,
  "split": "level_0",
  "max_dimension": 1024,
  "wall_time_s": 84.2,
  "peak_gpu_gib": 18.4
}
```

### Using the metrics module independently

`eval/metrics.py` is a standalone pure-Python module with no PyTorch dependency.
Import it directly from notebooks or scripts:

```python
import sys
sys.path.insert(0, "eval/")
import metrics

# Evaluate a single sample
result = metrics.sample_f1(pred_rles, gt_rles, metrics.IOU_THRESHOLDS)
print(f"F1: {result['f1']:.3f}")

# Aggregate over a dataset
dataset_metrics = metrics.aggregate(per_sample_results, metrics.IOU_THRESHOLDS)
print(f"F1: {dataset_metrics['f1']:.3f}")
```
