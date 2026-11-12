from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from drone_heatmaps.detectors import RFDETRDetector


class AdapterTests(unittest.TestCase):
    def test_segmentation_adapter_and_invalid_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs') as folder:
            weights=Path(folder)/'checkpoint.pth'; weights.write_bytes(b'fixture')
            result=SimpleNamespace(mask=np.ones((1,4,5),bool),confidence=[.8],class_id=[2])
            class Result:
                def __len__(self): return 1
                def __getattr__(self,key): return getattr(result,key)
            class Model:
                def __init__(self,**kw): self.kw=kw
                def predict(self,image,threshold):
                    np.testing.assert_array_equal(image,rgb)
                    return Result()
            rgb=np.zeros((4,5,3),np.uint8); rgb[:,:,0]=255
            fake=SimpleNamespace(RFDETRSegNano=Model,RFDETRSegSmall=Model)
            with patch.dict(sys.modules,{'rfdetr':fake}):
                detector=RFDETRDetector(weights)
                detection=detector.predict(rgb)[0]
                self.assertEqual(detection.class_id,2)
                self.assertEqual(detector.model.kw['pretrain_weights'],str(weights))
                result.mask=np.ones((1,2,2),bool)
                with self.assertRaisesRegex(RuntimeError,'dimensions'): detector.predict(rgb)
                result.mask=None
                with self.assertRaisesRegex(RuntimeError,'required'): detector.predict(rgb)
