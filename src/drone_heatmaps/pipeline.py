import json
import math
from pathlib import Path
import time
import hashlib
import importlib.metadata
import platform
import cv2
import numpy as np
from .core import tiles, merge_detections, compact_heatmap


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def process_video(source, output, detector, *, tile_size=512, overlap=64,
                  stride=1, threshold=0.5, allowed_classes=None, synthetic=False,
                  model_info=None, evidence_mode='both', previews=True,
                  heatmap_format='npy', evidence_gap_s=0):
    if stride < 1 or not 0 < threshold <= 1:
        raise ValueError("Stride must be positive; threshold must be in (0, 1]")
    if evidence_mode not in ('frame','crops','both') or heatmap_format not in ('npy','packed'):
        raise ValueError('Invalid evidence mode or heatmap format')
    if not math.isfinite(evidence_gap_s) or evidence_gap_s<0:
        raise ValueError('Evidence gap must be finite and nonnegative')
    # Validate tiling even if the input is empty.
    list(tiles(1, 1, tile_size, overlap))
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError(f"Video not found: {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {source}")
    metadata=None
    packed=None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video needs a valid frame rate for timestamps")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("heatmaps", "previews", "evidence"):
            (output/name).mkdir()
        count=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames=int(count) if math.isfinite(count) and count>0 else None
        versions={}
        for package in ('numpy','opencv-python-headless','rfdetr','torch'):
            try: versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        metadata = {"schema_version": 2, "status": "running", "source": str(source.resolve()),
                    "source_sha256": sha256_file(source), "dependencies": versions,
                    "platform": platform.platform(), "python": platform.python_version(),
                    "synthetic": synthetic, "fps": fps, "stride": stride,
                    "tile_size": tile_size, "overlap": overlap, "threshold": threshold,
                    "allowed_classes": sorted(allowed_classes) if allowed_classes is not None else None,
                    "model": model_info, "heatmap_shape": [64, 64],
                    "heatmap_encoding": "uint8; score=value/255; max-pooled instance confidence",
                    "source_deleted": False, "expected_frames_advisory": expected_frames,
                    "timestamp_policy": "decoder POS_MSEC when finite and increasing; otherwise frame/fps fallback",
                    "evidence_mode": evidence_mode, "evidence_gap_s": evidence_gap_s,
                    "previews": previews, "heatmap_format": heatmap_format,
                    "completion_note": "Decoder exhaustion alone does not prove an uncorrupted source"}
        (output/"run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        decoded = processed = retained = suspicious_frames = 0
        last_timestamp=-1.; last_evidence=-math.inf; durations=[]
        if heatmap_format=='packed': packed=(output/'heatmaps/maps.bin').open('wb')
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
                frame_started=time.perf_counter()
                decoder_s=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
                timestamp_source='decoder'
                if not math.isfinite(decoder_s) or decoder_s<0 or decoder_s<=last_timestamp:
                    decoder_s=max(index/fps,last_timestamp+stride/fps)
                    timestamp_source='frame_fps_fallback'
                last_timestamp=decoder_s
                h, w = frame.shape[:2]
                scores = np.zeros((h, w), dtype=np.float32)
                records = []
                for x0, y0, x1, y1 in tiles(h, w, tile_size, overlap):
                    rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                    records.extend(merge_detections(scores, detector.predict(rgb), x0, y0, allowed_classes))
                compact = compact_heatmap(scores)
                stem = f"{index:08d}"
                heatmap = f"heatmaps/{stem}.npy" if packed is None else 'heatmaps/maps.bin'
                offset=None
                if packed is None: np.save(output/heatmap, compact, allow_pickle=False)
                else:
                    offset=packed.tell(); packed.write(compact.tobytes()); packed.flush()
                if previews:
                    preview = cv2.applyColorMap(compact, cv2.COLORMAP_TURBO)
                    write_image(output/"previews"/f"{stem}.png", preview)
                suspicious = bool((scores >= threshold).any())
                suspicious_frames+=int(suspicious)
                evidence = []
                if suspicious and decoder_s-last_evidence>=evidence_gap_s:
                    retained += 1
                    last_evidence=decoder_s
                    if evidence_mode in ('frame','both'):
                        path = f"evidence/{stem}.png"
                        write_image(output/path, frame)
                        evidence.append(path)
                    # Tile observations can duplicate one physical defect; no unique count is claimed.
                    for number, record in enumerate(records):
                        if record["confidence"] < threshold or evidence_mode=='frame':
                            continue
                        x0, y0, x1, y1 = record["box"]
                        path = f"evidence/{stem}_crop_{number:04d}.png"
                        write_image(output/path, frame[y0:y1, x0:x1])
                        evidence.append(path)
                manifest.write(json.dumps({"frame": index, "timestamp_s": decoder_s,
                    "timestamp_source": timestamp_source, "heatmap_offset_bytes": offset,
                    "shape": [h, w], "max_score": float(scores.max()),
                    "suspicious": suspicious, "heatmap": heatmap,
                    "evidence": evidence, "tile_observations": records}) + "\n")
                manifest.flush()
                processed += 1
                durations.append(time.perf_counter()-frame_started)
        if not processed:
            raise ValueError("Video contains no decodable frames")
        if packed is not None: packed.close(); packed=None
        status='incomplete' if expected_frames is not None and decoded<expected_frames else 'completed'
        metadata.update(status=status,decoded_frames=decoded,
                        decode_warning='Decoder stopped before advisory frame count; investigate truncation or inaccurate metadata' if status=='incomplete' else None)
        (output/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        summary = {"status": status, "decoded_frames": decoded, "processed_frames": processed,
                   "suspicious_frames": suspicious_frames,
                   "retained_frames": retained, "elapsed_s": time.perf_counter()-started,
                   "processing_ms_p50": float(np.percentile(durations,50)*1000),
                   "processing_ms_p95": float(np.percentile(durations,95)*1000),
                   "timing_scope": "per sampled frame: tiled inference through evidence writes; excludes video read and startup",
                   "raw_heatmap_bytes": processed*4096,
                   "output_bytes_before_summary": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                   "source_bytes": source.stat().st_size}
        (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Fixed-point iteration because the summary's own length contributes to total bytes.
        for _ in range(10):
            total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if summary.get('output_bytes_total')==total: break
            summary['output_bytes_total']=total
            summary['output_to_source_ratio']=round(total/source.stat().st_size,8) if source.stat().st_size else None
            (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        return summary
    except Exception as exc:
        if metadata is not None:
            metadata.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            (output/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        raise
    finally:
        if packed is not None: packed.close()
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
