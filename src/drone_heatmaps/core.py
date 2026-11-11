"""Pure array operations; independent of model and video dependencies."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    mask: np.ndarray
    confidence: float
    class_id: int


def tiles(height, width, size=512, overlap=64):
    if min(height, width, size) <= 0 or not 0 <= overlap < size:
        raise ValueError("Positive dimensions and 0 <= overlap < tile size required")

    def starts(length):
        last = max(0, length - size)
        return sorted(set([*range(0, last + 1, size - overlap), last]))

    for y in starts(height):
        for x in starts(width):
            yield x, y, min(width, x + size), min(height, y + size)


def merge_detections(score_map, detections, x, y, allowed_classes=None):
    """Merge overlapping masks by maximum score, without double counting."""
    records = []
    for detection in detections:
        if allowed_classes is not None and detection.class_id not in allowed_classes:
            continue
        mask = np.asarray(detection.mask, dtype=bool)
        if mask.ndim != 2 or not np.isfinite(detection.confidence) or not 0 <= detection.confidence <= 1:
            raise ValueError("Invalid mask or confidence")
        h, w = mask.shape
        region = score_map[y:y+h, x:x+w]
        if region.shape != mask.shape:
            raise ValueError("Mask does not match tile bounds")
        np.maximum(region, mask * detection.confidence, out=region)
        ys, xs = np.nonzero(mask)
        if xs.size:
            records.append({"class_id": int(detection.class_id),
                            "confidence": float(detection.confidence),
                            "box": [int(xs.min()+x), int(ys.min()+y),
                                    int(xs.max()+x+1), int(ys.max()+y+1)]})
    return records


def compact_heatmap(scores, size=64):
    """Max-reduce into a uint8 summary; small inputs may repeat source cells.

    Values encode instance confidence inside boolean masks, not calibrated
    pixel probabilities. Overlapping pooling boundaries preserve small peaks.
    """
    if scores.ndim != 2 or not scores.size or size <= 0:
        raise ValueError("A nonempty 2D score map and positive size are required")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite and between zero and one")
    h, w = scores.shape
    result = np.zeros((size, size), dtype=np.uint8)
    for row in range(size):
        y0, y1 = row*h//size, ((row+1)*h+size-1)//size
        for col in range(size):
            x0, x1 = col*w//size, ((col+1)*w+size-1)//size
            result[row, col] = round(float(scores[y0:y1, x0:x1].max())*255)
    return result
