from pathlib import Path
import numpy as np
from .core import Detection


class SyntheticDetector:
    """Find artificial magenta markers, never real defects."""
    def predict(self, rgb):
        mask = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] > 180)
        return [Detection(mask, 0.95, 0)] if mask.any() else []


class RFDETRDetector:
    def __init__(self, weights, size="nano", device="cpu", threshold=0.5):
        if size not in ('nano','small') or not 0<threshold<=1:
            raise ValueError('Use nano/small and a threshold in (0, 1]')
        if not Path(weights).is_file():
            raise ValueError("Supply an existing fine-tuned segmentation checkpoint")
        try:
            from rfdetr import RFDETRSegNano, RFDETRSegSmall
        except ImportError as exc:
            raise RuntimeError("Install the model extra: pip install '.[model]'") from exc
        cls = {"nano": RFDETRSegNano, "small": RFDETRSegSmall}[size]
        self.model = cls(pretrain_weights=str(weights), device=device)
        self.threshold = threshold

    def predict(self, rgb):
        result = self.model.predict(rgb, threshold=self.threshold)
        if len(result) == 0:
            return []
        if result.mask is None or result.confidence is None or result.class_id is None:
            raise RuntimeError("Segmentation masks, confidence and class IDs are required")
        if result.mask.shape[1:] != rgb.shape[:2]:
            raise RuntimeError("Model masks do not match input crop dimensions")
        return [Detection(mask, float(score), int(label)) for mask, score, label in
                zip(result.mask, result.confidence, result.class_id)]
