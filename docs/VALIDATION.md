# Validation history

## Final build checks — 12 November 2026

The updated GitHub workflow passed the demo pipeline and RF-DETR model-container
build/import check: https://github.com/KhawajaSaad22/drone-inspection-heatmaps/actions/runs/43145941029 .
All six repositories' updated workflows passed. The Linux container runtime can
see my RTX 4060 through NVIDIA passthrough. This is not a trained-model GPU
accuracy or throughput benchmark. Earlier entries below describe their check dates.

The local model image also built successfully. RF-DETR Nano/Small imports and a
basic PyTorch GPU tensor computation passed with PyTorch 2.15.0+cu132 and RF-DETR
1.12.0. No trained defect checkpoint was used by this compatibility smoke check.

## Reliability update — 12 November 2026

All 12 local drone tests pass. The updated Python 3.11 Linux base container builds
locally and its synthetic video demo completes (20 processed, five suspicious/five
retained frames). Docker Compose configuration validation passes. This validates
the video pipeline, not learned defect accuracy. The RF-DETR model-target build
and remote workflow checks are tracked separately.

The initial Linux demo-container workflow succeeded on GitHub. Local regression
coverage now includes failed/incomplete run state, timestamp fallback, packed grids,
evidence options, byte accounting and a simulated RF-DETR adapter. CI also builds
the model target and checks imports without downloading model weights. Actual
checkpoint inference, GPU performance and real defect accuracy remain unverified.

The initial results below are historical, before these updates.

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
had not run remotely at the initial check; the later Linux demo workflow succeeded. No inspection dataset or fine-tuned weights
were supplied. Dependency versions need locking after model/hardware validation.

The first integration-test attempt hit restricted Windows temporary-folder access.
Tests now use temporary directories under the project's ignored outputs folder;
the complete suite passed after this change.
