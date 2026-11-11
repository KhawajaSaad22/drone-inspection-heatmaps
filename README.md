# Drone inspection heatmaps

Linux-first prototype for inspecting recorded drone video with RF-DETR instance
segmentation. Analyze overlapping crops before reducing results to 64×64 summaries.
Keep original-resolution evidence for frames containing predicted defects.

**Status:** processing pipeline and synthetic demo. No trained infrastructure-defect
model or accuracy claims are included. This is supervised defect detection, not
open-ended anomaly detection. COCO weights do not recognize infrastructure defects.

## Quick start: Linux containers on Windows or Linux

On Windows, use Docker with the WSL2 Linux backend. On Linux, use Docker Engine
with the Compose plugin. Create the `outputs` folder if absent, then:

```sh
docker compose run --build --rm demo
```

This creates `outputs/demo/synthetic.avi` and a `run` directory with 20 frame
records. Five frames contain artificial magenta markers. The demo detector is a
color threshold, **not an AI defect detector**. Run directories must be new; for
another run use a different path:

```sh
docker compose run --rm demo demo --output /outputs/demo-2
```

## Inspect real footage

Place video in `data/inspection.mp4` and my fine-tuned RF-DETR Seg Nano checkpoint
in `models/defects.pth`. Supply the exact defect class IDs used in my dataset:

```sh
docker compose --profile model run --build --rm inspect inspect --video /data/inspection.mp4 --weights /models/defects.pth --output /outputs/inspection-001 --class-ids 0 1 --device cpu
```

Nano is the default; `--size small` requires a matching Small checkpoint.
For a compatible NVIDIA GPU, install/configure GPU passthrough on the host and run:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile model run --build --rm inspect inspect --video /data/inspection.mp4 --weights /models/defects.pth --output /outputs/inspection-gpu-001 --class-ids 0 1 --device cuda
```

The model target installs PyTorch through RF-DETR dependencies; check CUDA/driver
compatibility on my host. This is not a validated Jetson image. Jetson requires
a separate ARM64/JetPack-compatible build. No GPU or RF-DETR execution has yet
been verified for this project. First model initialization may download pretrained
assets; allow network access and cache them before offline deployment.

The container runs as a non-root user. On Linux, ensure the mounted `outputs`
directory is writable by that user (UID 1000 in this image), or set the container
user to my host UID/GID. Do not commit data or weights to Git.

## Local development

Python 3.11 is the initial Linux target. CPU processing can also run on Windows.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
drone-inspect demo --output outputs/local-demo
pip install -e '.[model]'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.
Dependencies currently have compatibility ranges, not a reproducible lockfile;
freeze versions after validating the model on target hardware.

## Outputs and score meaning

- `run.json`: input path, settings, backend, checkpoint path, and encoding.
- `frames.jsonl`: original frame index, timestamp, image dimensions, score,
  evidence paths and tile observations (boxes use exclusive right/bottom edges).
- `heatmaps/*.npy`: uint8 64×64 arrays. Decode with `value / 255`.
- `previews/*.png`: colored 64×64 heatmap views, for display only.
- `evidence/*.png`: lossless source-resolution suspicious frames and mask bounding-box crops.
- `summary.json`: processed/retained counts, elapsed time, and byte totals.

Each mask receives its instance confidence; overlapping masks merge by maximum.
The 64×64 grid max-pools that full-resolution score map. These are not calibrated
pixel probabilities, anomaly scores, or structural severity estimates. The summary
combines selected classes; per-tile records preserve class IDs. Observations across
overlapping tiles can duplicate a physical defect; unique defect counting is not implemented.

The 4,096-byte figure is raw grid payload only. NPY headers, previews, metadata,
and evidence add storage. PNG evidence may exceed the input video's compressed
size; no storage-reduction percentage is claimed. Source video is never deleted.
`--stride N` processes every Nth frame and can miss brief appearances. The default
is every frame. A fresh output directory prevents overwriting earlier evidence.

## Training and evaluation plan

1. Pick one surface type and define defect categories and minimum visible defect size.
2. Label segmentation masks in real drone footage; include normal surfaces, stains,
   shadows, blur, and challenging lighting. Use the same crop scale during training.
3. Split by structure/inspection flight, not adjacent frames, to avoid leakage.
4. Fine-tune `RFDETRSegNano` using RF-DETR's COCO segmentation training workflow.
5. Test held-out defect recall, false alerts/minute, and mask quality. Select the
   retention threshold on validation footage, then evaluate once on held-out flights.
6. Benchmark full video decoding, tiled inference, peak memory, power, and actual
   output bytes on the deployment computer. Tune tile size/overlap before trading
   recall for speed. Temporal tracking, geolocation and live drone integration are future work.

Linux CI checks core behavior, synthetic video integration and the demo container.
Actual model training and hardware benchmarking require data, weights and hardware.

## Upstream references and licensing

- [RF-DETR repository and model licenses](https://github.com/roboflow/rf-detr)
- [Segmentation API](https://rfdetr.roboflow.com/latest/learn/run/segmentation/)
- [Training guide](https://rfdetr.roboflow.com/latest/learn/train/)
- [Docker GPU support on Windows](https://docs.docker.com/desktop/features/gpu/)

This repository uses Apache-2.0 (see LICENSE). The adapter selects Apache-2.0
RF-DETR Seg Nano/Small models; no `rfdetr[plus]` dependency is used. Retain required
third-party notices when redistributing. Dataset and checkpoint provenance must
also permit my intended use. Docker Desktop has separate subscription terms.
