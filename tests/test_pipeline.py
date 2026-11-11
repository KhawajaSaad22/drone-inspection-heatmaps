import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from drone_heatmaps.detectors import SyntheticDetector
from drone_heatmaps.pipeline import make_demo, process_video


class PipelineTests(unittest.TestCase):
    def test_video_evidence_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"outputs") as folder:
            root = Path(folder)
            video = root/"input.avi"
            make_demo(video)
            result = process_video(video, root/"run", SyntheticDetector(),
                                   tile_size=128, overlap=32, synthetic=True)
            self.assertEqual(result["processed_frames"], 20)
            self.assertEqual(result["retained_frames"], 5)
            rows = [json.loads(line) for line in (root/"run/frames.jsonl").read_text().splitlines()]
            for row in rows:
                heatmap = np.load(root/"run"/row["heatmap"], allow_pickle=False)
                self.assertEqual(heatmap.shape, (64, 64))
                self.assertEqual(bool(heatmap.any()), row["suspicious"])
                for file in row["evidence"]:
                    self.assertTrue((root/"run"/file).is_file())
            self.assertAlmostEqual(rows[5]["timestamp_s"], .5)
            self.assertTrue(video.exists())
            with self.assertRaises(FileExistsError):
                process_video(video, root/"run", SyntheticDetector())

    def test_stride_and_class_filter(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"outputs") as folder:
            root = Path(folder)
            make_demo(root/"input.avi")
            result = process_video(root/"input.avi", root/"run", SyntheticDetector(),
                                   stride=2, allowed_classes={99})
            self.assertEqual(result["processed_frames"], 10)
            self.assertEqual(result["retained_frames"], 0)


if __name__ == "__main__":
    unittest.main()
