# Fixed-row Range-GDC paper experiment

This experiment compares the following methods:

- `original_gdc_naive`: Pseudo-LiDAR++ GDC with point subsampling disabled.
- `original_gdc_optimized`: Original GDC with deterministic spatial
  subsampling and preserved unprocessed predictions.
- `range_gdc`: graph/direct residual transfer on a spherical range grid.

## Anchor definitions

The methods intentionally use anchors in the representation where each method
operates.

### Original GDC

`gdc/sparsify.py` selects the configured physical Velodyne bands and writes
`anchor/shared_4beam_pointcloud/*.bin`. `gdc/ptc2depthmap.py` projects these
points into camera z-depth landmarks for Original GDC. The provenance file
records the source, band indices, extraction settings, counts, and SHA-256
values.

### Range GDC

The GT Velodyne scan is first projected to the common `64 x 1024` range grid.
`range_gdc/make_anchor_from_gt_range.py` then copies only the configured fixed
rows, by default `[5, 7, 9, 11]`, into the Range-GDC anchor. No physical point
cloud reprojection is used for this anchor.

The evaluation mask is also fixed to these same rows:

```text
anchor_rows_valid = rows 5, 7, 9, 11
hidden_rows_valid = the remaining 60 rows
```

The evaluator receives the fixed row indices directly and does not use
frame-wise occupied-row metadata to redefine the hidden area.

Because the anchor construction differs by method, the paper must not claim
that Original GDC and Range GDC use exactly the same physical sparse points.
The comparison instead evaluates two correction pipelines in their native
representations.

## Projection convention

`full_360_front_centered` uses azimuth edges `[-180, 180]` degrees. Row 0 is the
highest elevation and rows proceed downward. Velodyne coordinates are `x`
forward, `y` left, and `z` up. Azimuth is `atan2(y, x)` and elevation is
`atan2(z, sqrt(x^2+y^2))`.

## Anchor rejection

`anchor_filter` controls `none`, `abs`, or `log_ratio` rejection and the final
anchor force policy. Original GDC applies the threshold to camera z-depth;
Range GDC applies it to spherical range. Therefore a common numeric threshold
does not represent the same geometric quantity in the two domains.

## Range-GDC semantics

Range GDC corrects only cells already valid in the predicted range image. It is
not a completion method. Its propagated variable is a log-range residual. With
`delta_clip: 0.3`, the multiplicative correction factor is limited to
`exp(-0.3)` through `exp(0.3)` before the configured anchor-force step.

## Evaluation

The primary area is `common_hidden_valid`: valid GT cells on the 60 non-anchor
rows intersected with valid predictions from every enabled method. Coverage is
reported separately.

The fixed-row anchor validator confirms that anchor values are copied exactly
from GT only on `[5, 7, 9, 11]`. Leakage validation fails by default if anchor
values appear on hidden rows.

## Reproduction

Use a fresh output root, for example:

```yaml
output_root: /data/kitti/pseudo_lidar_paper_fixed_rows_v1
```

Dry run:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --dry-run \
  --no-preview \
  --no-export-pointcloud
```

Full execution:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --no-preview \
  --no-export-pointcloud
```

After changing `range_anchor.selected_rows`, rebuild from the anchor stage:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_example.yaml \
  --force-from range_anchor_from_gt_range \
  --no-preview \
  --no-export-pointcloud
```
