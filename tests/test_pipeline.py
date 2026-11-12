import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from drone_heatmaps.detectors import SyntheticDetector
from drone_heatmaps.pipeline import make_demo, process_video


class PipelineTests(unittest.TestCase):
    def test_packed_heatmaps_and_storage_modes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as folder:
            root=Path(folder); video=root/'input.avi'; make_demo(video)
            result=process_video(video,root/'run',SyntheticDetector(),previews=False,
                heatmap_format='packed',evidence_mode='frame',evidence_gap_s=.3)
            self.assertEqual((root/'run/heatmaps/maps.bin').stat().st_size,20*4096)
            self.assertEqual(len(list((root/'run/previews').iterdir())),0)
            self.assertEqual(result['suspicious_frames'],5)
            self.assertLess(result['retained_frames'],5)
            rows=[json.loads(line) for line in (root/'run/frames.jsonl').read_text().splitlines()]
            self.assertEqual(rows[5]['heatmap_offset_bytes'],5*4096)
            self.assertEqual(result['output_bytes_total'],sum(p.stat().st_size for p in (root/'run').rglob('*') if p.is_file()))
            self.assertTrue(all('_crop_' not in file for row in rows for file in row['evidence']))

    def test_failure_records_status_and_releases_video(self):
        from unittest.mock import patch
        import cv2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as folder:
            root=Path(folder); video=root/'input.avi'; make_demo(video)
            class Broken:
                def predict(self,rgb): raise RuntimeError('simulated model failure')
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                process_video(video,root/'run',Broken())
            metadata=json.loads((root/'run/run.json').read_text())
            self.assertEqual(metadata['status'],'failed')
            self.assertEqual(metadata['error_type'],'RuntimeError')
            self.assertIn('source_sha256',metadata)

    def test_decoder_metadata_mismatch_is_incomplete(self):
        from unittest.mock import patch
        import cv2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as folder:
            root=Path(folder); source=root/'fixture.avi'; source.write_bytes(b'fixture')
            class Capture:
                released=False
                def isOpened(self): return True
                def get(self,key):
                    return {cv2.CAP_PROP_FPS:10,cv2.CAP_PROP_FRAME_COUNT:5,cv2.CAP_PROP_POS_MSEC:float('nan')}.get(key,0)
                def read(self):
                    if hasattr(self,'read_once'): return False,None
                    self.read_once=True; return True,np.zeros((8,8,3),np.uint8)
                def release(self): self.released=True
            cap=Capture()
            with patch('drone_heatmaps.pipeline.cv2.VideoCapture',return_value=cap):
                result=process_video(source,root/'run',SyntheticDetector())
            self.assertEqual(result['status'],'incomplete'); self.assertTrue(cap.released)
            row=json.loads((root/'run/frames.jsonl').read_text())
            self.assertEqual(row['timestamp_source'],'frame_fps_fallback')

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
