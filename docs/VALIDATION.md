# Initial validation

Validated locally on Windows with Python 3.14, NumPy 2.5.4 and
opencv-python-headless 4.14.0.94. Linux container target is Python 3.11.

- Eight unittest checks passed: tiling coverage, small frames, preservation of
  tiny peaks, maximum-score overlap merging, class filtering, invalid inputs,
  end-to-end evidence/manifest creation, stride and negative retention.
- CLI synthetic demo decoded and processed 20 frames and retained exactly five
  artificial marker frames. The demo is not a trained defect detector.
- Demo raw heatmap payload: 81,920 bytes total. Actual output before the summary:
  196,697 bytes, versus 58,136 bytes for the synthetic source video. This illustrates
  why compact grids alone do not establish overall storage savings.
- `git diff --check` passed.

Not yet validated: RF-DETR checkpoint loading/prediction, real defect recall,
GPU throughput, Linux execution, container builds, or Jetson deployment. Linux CI
is configured but has not run remotely. No inspection dataset or fine-tuned weights
were supplied. Dependency versions need locking after model/hardware validation.

The first integration-test attempt hit restricted Windows temporary-folder access.
Tests now use temporary directories under the project's ignored outputs folder;
the complete suite passed after this change.
