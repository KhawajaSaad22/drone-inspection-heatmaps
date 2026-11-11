import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .core import tiles, merge_detections, compact_heatmap


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def process_video(source, output, detector, *, tile_size=512, overlap=64,
                  stride=1, threshold=0.5, allowed_classes=None, synthetic=False,
                  model_info=None):
    if stride < 1 or not 0 < threshold <= 1:
        raise ValueError("Stride must be positive; threshold must be in (0, 1]")
    # Validate tiling even if the input is empty.
    list(tiles(1, 1, tile_size, overlap))
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError(f"Video not found: {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video needs a valid frame rate for timestamps")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("heatmaps", "previews", "evidence"):
            (output/name).mkdir()
        metadata = {"schema_version": 1, "source": str(source.resolve()),
                    "synthetic": synthetic, "fps": fps, "stride": stride,
                    "tile_size": tile_size, "overlap": overlap, "threshold": threshold,
                    "allowed_classes": sorted(allowed_classes) if allowed_classes is not None else None,
                    "model": model_info, "heatmap_shape": [64, 64],
                    "heatmap_encoding": "uint8; score=value/255; max-pooled instance confidence",
                    "source_deleted": False}
        (output/"run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        decoded = processed = retained = 0
        started = time.perf_counter()
        with (output/"frames.jsonl").open("w", encoding="utf-8") as manifest:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = decoded
                decoded += 1
                if index % stride:
                    continue
                h, w = frame.shape[:2]
                scores = np.zeros((h, w), dtype=np.float32)
                records = []
                for x0, y0, x1, y1 in tiles(h, w, tile_size, overlap):
                    rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                    records.extend(merge_detections(scores, detector.predict(rgb), x0, y0, allowed_classes))
                compact = compact_heatmap(scores)
                stem = f"{index:08d}"
                heatmap = f"heatmaps/{stem}.npy"
                np.save(output/heatmap, compact, allow_pickle=False)
                preview = cv2.applyColorMap(compact, cv2.COLORMAP_TURBO)
                write_image(output/"previews"/f"{stem}.png", preview)
                suspicious = bool((scores >= threshold).any())
                evidence = []
                if suspicious:
                    retained += 1
                    path = f"evidence/{stem}.png"
                    write_image(output/path, frame)
                    evidence.append(path)
                    # Tile observations can duplicate one physical defect; no unique count is claimed.
                    for number, record in enumerate(records):
                        if record["confidence"] < threshold:
                            continue
                        x0, y0, x1, y1 = record["box"]
                        path = f"evidence/{stem}_crop_{number:04d}.png"
                        write_image(output/path, frame[y0:y1, x0:x1])
                        evidence.append(path)
                manifest.write(json.dumps({"frame": index, "timestamp_s": index/fps,
                    "shape": [h, w], "max_score": float(scores.max()),
                    "suspicious": suspicious, "heatmap": heatmap,
                    "evidence": evidence, "tile_observations": records}) + "\n")
                manifest.flush()
                processed += 1
        if not processed:
            raise ValueError("Video contains no decodable frames")
        summary = {"decoded_frames": decoded, "processed_frames": processed,
                   "retained_frames": retained, "elapsed_s": time.perf_counter()-started,
                   "raw_heatmap_bytes": processed*4096,
                   "output_bytes_before_summary": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                   "source_bytes": source.stat().st_size}
        (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        cap.release()


def make_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Refusing to overwrite: {path}")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("MJPG video writer unavailable")
    try:
        for i in range(20):
            frame = np.full((240, 320, 3), 110, np.uint8)
            if 5 <= i < 10:
                cv2.line(frame, (100, 60), (140, 180), (255, 0, 255), 4)
            cv2.putText(frame, "SYNTHETIC DEMO", (10, 225), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()
