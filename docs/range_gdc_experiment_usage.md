# Range-GDC Paper Experiment Runner

The canonical entry point is:

```bash
python3 -B tools/run_range_gdc_experiment.py --config configs/r64_pipeline_example.yaml
```

`run_r64_pipeline.py` is obsolete and is not used by this workflow.

## Core pipeline

```text
sdn_depth
shared_anchor_pointcloud
shared_anchor_image_depth
gt_range
range_anchor_from_gt_range
sdn_depth_to_range
original_gdc_naive
original_gdc_naive_depth_to_range
original_gdc_optimized
original_gdc_optimized_depth_to_range
range_gdc
evaluate
```

The two correction methods intentionally use different anchor representations:

- Original GDC uses a physical sparse point cloud generated once under
  `anchor/shared_4beam_pointcloud/`, then projects it to camera z-depth under
  `anchor/image_depth/`.
- Range GDC directly copies the configured rows from the GT 64-row range image
  into `anchor/range/G64_range/`.

With the default config, Range GDC uses rows `[5, 7, 9, 11]`. Evaluation excludes
exactly these four rows and evaluates the remaining 60 rows. It does not infer
excluded rows from frame-wise projected anchor occupancy.

## Dry run and execution

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --dry-run

python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml
```

Use a fresh `output_root` after applying this patch.

## Stage selection

```bash
# Evaluation only
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --only-stage evaluate \
  --force-stage evaluate

# Rebuild the fixed-row Range-GDC anchor and every downstream stage
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --force-from range_anchor_from_gt_range

# Rebuild the Original-GDC physical anchor and all later stages
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --force-from shared_anchor_pointcloud

# Rerun only Range GDC and evaluation after the anchor is already correct
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --stages range_gdc,evaluate \
  --force
```

`shared_anchor_range` remains accepted as a compatibility alias for
`range_anchor_from_gt_range`, but new commands should use the canonical stage
name.

## Main outputs

```text
anchor/shared_4beam_pointcloud/
anchor/provenance.json
anchor/image_depth/
anchor/range/G64_range/
anchor/range/G64_mask/
anchor/range/meta/anchor_definition.json
original_gdc/naive/corrected_depth/
original_gdc/optimized/corrected_depth/
range/original_gdc_naive/G64_range/
range/original_gdc_optimized/G64_range/
range/range_gdc/G64_range/
metrics/guide_r64_summary.csv
metrics/guide_r64_distance_summary.csv
metrics/range_gdc_leakage_summary.csv
```

## Fixed-row validation

The `range_anchor_from_gt_range` stage is complete only when all of the following
hold for every frame:

- the anchor has shape `64 x 1024`;
- valid anchor cells exist only on the configured fixed rows;
- the anchor valid mask on those rows matches the GT valid mask;
- anchor values equal GT range values exactly on valid cells;
- each frame records the same configured row list;
- `anchor_definition.json` reports `mode: gt_row_mask` and the expected source,
  shape, and row indices.

This prevents an older point-cloud-projected anchor directory from being
silently reused.

## Distance-bin evaluation

Distance evaluation is enabled unless `--no-distance-eval` is supplied. To
recompute and print the distance-bin summary without rerunning correction:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --only-stage evaluate \
  --force-stage evaluate
```

The summary is written to
`<output_root>/metrics/guide_r64_distance_summary.csv`.
