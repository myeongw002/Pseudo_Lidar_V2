from pathlib import Path
import numpy as np

roots = {
    "raw_sdn": Path(
        "/media/myungw00/2TB_SSD/kitti/pseudo_lidar_train/"
        "pointcloud/raw_sdn_64ch"
    ),
    "gdc_naive": Path(
        "/media/myungw00/2TB_SSD/kitti/pseudo_lidar_train/"
        "pointcloud/original_gdc_naive_64ch"
    ),
    "rgc": Path(
        "/media/myungw00/2TB_SSD/kitti/pseudo_lidar_train/"
        "pointcloud/range_gdc_64ch"
    ),
}

expected_ids = {
    line.strip()
    for line in Path("/home/myungw00/ROS2/upsample_ws/pointpillar_ws/pointpillars_openpcdet_docker/runtime/kitti/ImageSets/trainval.txt")
    .read_text()
    .splitlines()
    if line.strip()
}

for name, root in roots.items():
    actual_ids = {p.stem for p in root.glob("*.bin")}

    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    if missing or extra:
        raise RuntimeError(
            f"{name}: missing={missing[:5]}, extra={extra[:5]}"
        )

    for path in root.glob("*.bin"):
        points = np.fromfile(path, dtype=np.float32)

        if points.size == 0 or points.size % 4 != 0:
            raise ValueError(
                f"{path}: invalid float count {points.size}"
            )

        points = points.reshape(-1, 4)

        if not np.isfinite(points).all():
            raise ValueError(f"{path}: NaN or Inf detected")

    print(f"{name}: OK, {len(actual_ids)} frames")
