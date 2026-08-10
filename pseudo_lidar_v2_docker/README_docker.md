# Pseudo_Lidar_V2 Docker training environment

This Docker setup targets `mileyan/Pseudo_Lidar_V2` training.

## Files

- `Dockerfile`: modern CUDA 12.8 + PyTorch environment, recommended for recent GPUs.
- `Dockerfile.legacy`: old CUDA 10 / PyTorch 1.0-style environment for reproduction only.
- `requirements-modern.txt`, `requirements-legacy.txt`: Python dependencies.
- `scripts/*.sh`: common data preparation, training, and inference commands.
- `docker-compose.yml`: optional compose-based launcher.

## Where to put these files

Copy all files into the root of the cloned repo:

```bash
git clone https://github.com/mileyan/Pseudo_Lidar_V2.git
cd Pseudo_Lidar_V2
# copy Dockerfile, requirements, docker-compose.yml, scripts/ here
```

Expected mounted dataset paths inside the container:

```text
/data/SceneFlow
/data/KITTI
```

KITTI should look like:

```text
/data/KITTI/training/calib
/data/KITTI/training/image_2
/data/KITTI/training/image_3
/data/KITTI/training/velodyne
/data/KITTI/testing/calib
/data/KITTI/testing/image_2
/data/KITTI/testing/image_3
```

## Build

```bash
bash scripts/build.sh
```

For a pinned PyTorch version:

```bash
docker build -t pseudo-lidar-v2:cu128 \
  --build-arg TORCH_SPEC='torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1' \
  -f Dockerfile .
```

## Run

```bash
KITTI_DIR=/path/to/KITTI \
SCENEFLOW_DIR=/path/to/SceneFlow \
RESULTS_DIR=$PWD/results \
bash scripts/run.sh
```

Inside the container:

```bash
bash scripts/check_env.sh
plv2_patch_repo
```

## Prepare data

```bash
bash scripts/prepare_data.sh
```

## Train SDNet on SceneFlow

Start small first:

```bash
BTRAIN=2 BVAL=1 WORKERS=4 bash scripts/train_sceneflow.sh
```

## Fine-tune on KITTI

```bash
BTRAIN=2 BVAL=1 WORKERS=4 bash scripts/train_kitti.sh
```

Train on train+val for test-set style output:

```bash
BTRAIN=2 BVAL=1 WORKERS=4 bash scripts/train_kitti_trainval.sh
```

## Generate KITTI depth maps

```bash
RESUME=./results/sdn_kitti_train_set/checkpoint.pth.tar \
SAVE_PATH=./results/sdn_kitti_train_set \
bash scripts/generate_depth_trainval.sh
```

## Convert depth maps to pseudo-LiDAR

```bash
DEPTH_DIR=./results/sdn_kitti_train_set/depth_maps/trainval/ \
SAVE_DIR=./results/sdn_kitti_train_set/pseudo_lidar_trainval/ \
bash scripts/depth_to_pseudo_lidar.sh
```

## Notes

- The original README command for KITTI fine-tuning appears to use `--dataset path-to-KITTI/training/`, but the code argument is `--datapath`; the scripts use `--datapath`.
- The original README has a typo `scneflow.py`; the file is `sceneflow.py`.
- For modern PyTorch, run `plv2_patch_repo` once before loading checkpoints.
- The original default batch size is large. Use `BTRAIN=1` or `BTRAIN=2` first on a single GPU.
