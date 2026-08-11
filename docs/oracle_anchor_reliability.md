# Oracle Anchor Reliability (GT-only diagnostic)

> **ORACLE / GT-ONLY / DIAGNOSTIC** — This is a train1000 headroom experiment,
> not an inference method. Canonical Range-GDC does not read GT and is not
> configured through this tool.

The standalone diagnostic consumes already-generated artifacts from a canonical
train1000 output root. It does not run SDN, rebuild anchors or GT, or run Original
GDC. The required directories are:

```text
<base>/range/raw_sdn/G64_range
<base>/anchor/range_shared_canonical/G64_range
<base>/range/gt/G64_range
```

For each canonically accepted anchor `a`, the diagnostic uses its graph-connected
one-hop hidden neighbors:

```text
N_H(a) = {i: (a,i) is a Range-GDC graph edge,
             i is not on a source row,
             GT(i) and prediction(i) are valid}

      sum_i w_ai |e_a - e_i_gt|
h_a = --------------------------
              sum_i w_ai
```

Both residuals use the canonical clipped log-range semantics. Anchors without a
valid hidden neighbor are oracle-unknown, stay at `q=1`, and are excluded from
global ranking. Known anchors are globally ranked by `h_a`; the best 90%, 75%,
50%, or 25% retain `q=1`, while the rest receive `q=0`. This weight affects only
the graph data constraint. Every accepted anchor, including one with `q=0`, is
still exactly forced to its LiDAR range after solving.

Ten-frame smoke command:

```bash
python3 -B tools/oracle_anchor_reliability.py \
  --base-output-root /data/kitti/pseudo_lidar_uniform_train1000 \
  --split-file ./split/train_1000_seed2026.txt \
  --output-root /data/kitti/rgc_oracle_train1000_smoke10 \
  --max-items 10
```

Full train1000 command (run only after reviewing the smoke outputs):

```bash
python3 -B tools/oracle_anchor_reliability.py \
  --base-output-root /data/kitti/pseudo_lidar_uniform_train1000 \
  --split-file ./split/train_1000_seed2026.txt \
  --output-root /data/kitti/rgc_oracle_train1000
```

The output root is required to be disjoint from the base root. It contains
per-anchor scores and headroom tables under `oracle/`, corrected range images
under `range/{uniform,oracle_keep90,oracle_keep75,oracle_keep50,oracle_keep25}`,
and reused canonical evaluator outputs under `metrics/`.

