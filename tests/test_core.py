import unittest
import numpy as np
from drone_heatmaps.core import Detection, tiles, merge_detections, compact_heatmap


class CoreTests(unittest.TestCase):
    def test_tiles_cover_irregular_frame(self):
        covered = np.zeros((729, 1027), bool)
        for x0, y0, x1, y1 in tiles(*covered.shape):
            covered[y0:y1, x0:x1] = True
            self.assertLessEqual(x1-x0, 512)
        self.assertTrue(covered.all())

    def test_small_frame(self):
        self.assertEqual(list(tiles(20, 30)), [(0, 0, 30, 20)])

    def test_thin_defect_survives_pooling(self):
        scores = np.zeros((217, 513), np.float32)
        scores[100, 301] = 0.9
        compact = compact_heatmap(scores)
        self.assertEqual(compact.shape, (64, 64))
        self.assertEqual(compact.nbytes, 4096)
        self.assertEqual(int(compact.max()), round(float(scores.max())*255))

    def test_overlap_uses_max_and_filters_classes(self):
        scores = np.zeros((10, 10), np.float32)
        mask = np.ones((4, 4), bool)
        merge_detections(scores, [Detection(mask, .8, 2)], 2, 2, {2})
        merge_detections(scores, [Detection(mask, .6, 2), Detection(mask, 1, 3)], 2, 2, {2})
        self.assertAlmostEqual(float(scores.max()), .8)
        self.assertEqual(np.count_nonzero(scores), 16)

    def test_empty_and_tiny_maps(self):
        self.assertFalse(compact_heatmap(np.zeros((5, 3))).any())
        self.assertTrue((compact_heatmap(np.ones((1, 1))) == 255).all())

    def test_invalid_inputs(self):
        for overlap in (-1, 512):
            with self.assertRaises(ValueError):
                list(tiles(100, 100, 512, overlap))
        with self.assertRaises(ValueError):
            compact_heatmap(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
