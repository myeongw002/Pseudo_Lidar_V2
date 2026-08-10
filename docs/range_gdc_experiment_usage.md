# Range-GDC Paper Experiment Runner

The canonical entry point is:

```bash
python3 -B tools/run_range_gdc_experiment.py --config configs/r64_pipeline_canonical.yaml
```

`run_r64_pipeline.py` is obsolete and is not used by this workflow.
`tests/test_paper_pipeline.py` likewise targets this canonical runner and does
not import the obsolete module.

`tools/run_ablation.sh` remains a convenience utility rather than the canonical
runner. It now resolves the repository path dynamically and refuses to reuse
an existing ablation output directory; set `ROOT` and `ABLATION_ROOT` to fresh
locations when using it.

## Core pipeline

```text
sdn_depth
canonical_shared_anchor
canonical_shared_anchor_image_depth
gt_range
range_anchor_from_shared_anchor
sdn_depth_to_range
original_gdc_naive
original_gdc_naive_depth_to_range
original_gdc_optimized
original_gdc_optimized_depth_to_range
range_gdc
evaluate
```

The two correction methods use the same canonical original-LiDAR points:

- Canonical projection records the nearest collision winner's original PCD index
  for rows `[5, 7, 9, 11]`, then writes the exact original x/y/z/intensity
  records to `anchor/shared_canonical_pointcloud/` and the full source-index
  grid to `anchor/shared_canonical_source_index/`.
- Original GDC projects only that point cloud to camera z-depth with nearest
  positive camera-z collision handling.
- Range GDC creates its anchor only from that same source-index grid and point
  cloud; it does not copy rows from GT range.

With the default config, Range GDC uses rows `[5, 7, 9, 11]`. Evaluation excludes
exactly these four rows and evaluates the remaining 60 rows. It does not infer
excluded rows from frame-wise projected anchor occupancy.

## Dry run and execution

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --dry-run

python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml
```

Use a fresh `output_root` after applying this patch.

## Stage selection

```bash
# Evaluation only
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --only-stage evaluate \
  --force-stage evaluate

# Rebuild the shared canonical RGC anchor and every downstream stage
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --force-from range_anchor_from_shared_anchor

# Rebuild the Original-GDC physical anchor and all later stages
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --force-from canonical_shared_anchor

# Rerun only Range GDC and evaluation after the anchor is already correct
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --stages range_gdc,evaluate \
  --force
```

`shared_anchor_range` remains accepted as a compatibility alias for
`range_anchor_from_shared_anchor`, but new commands should use the canonical
stage name.

## Main outputs

```text
anchor/shared_canonical_pointcloud/
anchor/shared_canonical_source_index/
anchor/shared_canonical_pointcloud_provenance.json
anchor/shared_canonical_image_depth/
anchor/range_shared_canonical/G64_range/
anchor/range_shared_canonical/G64_mask/
anchor/range_shared_canonical/meta/anchor_definition.json
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

The `range_anchor_from_shared_anchor` stage is complete only when all of the following
hold for every frame:

- the anchor has shape `64 x 1024`;
- valid anchor cells exist only on the configured fixed rows;
- every valid anchor cell has a valid shared source index;
- `anchor_definition.json` references the checksum of the shared manifest;
- each frame records the same configured row list;
- `anchor_definition.json` reports `mode: shared_canonical` and the expected source,
  shape, and row indices.

Run `python3 -B tools/audit_shared_anchor_protocol.py --output-root <root>
--split-file <split> --calib-dir <kitti>/calib --image-dir <kitti>/image_2`
after the three anchor stages.  It requires zero RGC/GDC points outside the
shared source set.

## Distance-bin evaluation

Distance evaluation is enabled unless `--no-distance-eval` is supplied. To
recompute and print the distance-bin summary without rerunning correction:

```bash
python3 -B tools/run_range_gdc_experiment.py \
  --config configs/r64_pipeline_canonical.yaml \
  --only-stage evaluate \
  --force-stage evaluate
```

The summary is written to
`<output_root>/metrics/guide_r64_distance_summary.csv`.
