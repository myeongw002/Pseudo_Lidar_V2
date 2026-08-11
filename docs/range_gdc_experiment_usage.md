# Canonical Range-GDC Experiment

The canonical entry point is:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml
```

## Method

```text
shared sparse physical LiDAR points
    -> canonical RGC anchor
    -> binary anchor filtering
    -> log-range residual observations
    -> graph-regularized propagation
    -> corrected range
```

Canonical projection first records each spherical cell's nearest collision
winner as an original point-cloud source index. Rows `[5, 7, 9, 11]` select
the exact original x/y/z/intensity records used by both correction baselines.
Original GDC projects those shared records to camera depth. Range-GDC derives
its range anchor from the same source-index grid and uses `norm(x,y,z)`.

Range-GDC constructs an `angular_grid8` graph over valid SDN range cells and
solves for a graph-regularized log-range residual. Binary anchor rejection,
residual clipping, and `accepted_only` anchor forcing follow the checked-in
canonical configuration. It corrects existing predicted cells only and does
not complete missing cells.

## Experimental continuous anchor reliability

The canonical configs explicitly use `anchor_reliability_mode: uniform`, so
every anchor accepted by hard rejection has `q = 1` and retains the baseline
system exactly. The train-1000 experimental config uses `quadratic`: after the
same hard rejection, each accepted anchor receives
`q = 1 - (d / tau)^2` for `d < tau`; anchors with `d >= tau` have `q = 0`
and are removed from the graph target set by hard rejection. Each accepted
anchor's graph constraint becomes `lambda_anchor * q`. Here `d` is the
unclipped absolute range discrepancy for
`abs` rejection, or the unclipped absolute log-range discrepancy for
`log_ratio`; `tau` is the corresponding rejection threshold. Quadratic mode
cannot be combined with `anchor_reject: none`. Graph edge weights, residual
clipping, accepted-target residual reinsertion, and accepted-anchor forcing
are unchanged. This mode does not add a direct residual branch or a separate
target-cell/cell-level confidence path.

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_anchor_reliability_train1000.yaml
```

## Canonical stages

```text
sdn_depth
canonical_shared_anchor
canonical_shared_anchor_image_depth
gt_range
range_anchor_from_shared_anchor
audit_shared_anchor_protocol
sdn_depth_to_range
original_gdc_naive
original_gdc_naive_depth_to_range
original_gdc_optimized
original_gdc_optimized_depth_to_range
range_gdc
evaluate
```

Stage names are exact; the runner does not translate historical names.

## Dry runs

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --dry-run

python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical_val.yaml \
  --dry-run
```

Use a fresh `output_root` for an actual run. Resume validation checks each
stage's expected artifacts before deciding whether to skip it.

## Stage selection

```bash
# Evaluation only
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --only-stage evaluate --force-stage evaluate

# Rebuild the canonical range anchor and downstream stages
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --force-from range_anchor_from_shared_anchor

# Rerun correction and evaluation
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --stages range_gdc,evaluate --force
```

## Main outputs

```text
anchor/shared_canonical_pointcloud/
anchor/shared_canonical_source_index/
anchor/shared_canonical_pointcloud_provenance.json
anchor/shared_canonical_image_depth/
anchor/shared_canonical_image_source_index/
anchor/range_shared_canonical/G64_range/
anchor/range_shared_canonical/G64_mask/
anchor/range_shared_canonical/meta/anchor_definition.json
anchor/shared_canonical_protocol_audit.csv
anchor/shared_canonical_protocol_summary.csv
original_gdc/naive/corrected_depth/
original_gdc/optimized/corrected_depth/
range/original_gdc_naive/G64_range/
range/original_gdc_optimized/G64_range/
range/range_gdc/G64_range/
range/range_gdc/meta/range_gdc_stats.csv
metrics/guide_r64_summary.csv
metrics/guide_r64_distance_summary.csv
metrics/range_gdc_leakage_summary.csv
```

## Protocol audits

The `range_anchor_from_shared_anchor` stage requires a `64 x 1024` anchor,
values only on configured rows, a valid shared source index for every anchor
cell, and a matching shared-manifest checksum.

Run the full shared-source audit with:

```bash
python3 -B tools/audit_shared_anchor_protocol.py \
  --output-root <root> \
  --split-file <split> \
  --calib-dir <kitti>/calib \
  --image-dir <kitti>/image_2
```

To compare native Original-GDC camera-depth and Range-GDC spherical-range
rejection decisions on common source IDs:

```bash
python3 -B tools/audit_anchor_rejection_consistency.py \
  --output-root <root> \
  --split-file <split> \
  --kitti-root <kitti-root> \
  --config configs/r64_pipeline_canonical.yaml
```

Evaluation excludes exactly the configured source rows and retains the
canonical hidden-row and common-mask definitions.
