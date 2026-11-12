# Project review — 12 November 2026

## Implementation follow-up

I implemented and tested the software changes after the initial review: chronological
verifier frames; approved-mask integrity and approval reset; group-overlap checks;
negative-query metrics; PDF/reindex regression tests; dataset auditing; process-memory
profiling; and drone failure states, timestamp fallback, storage options, hashes and
latency/byte reports. The latest local suites pass 114 tests in total. The Streamlit
UI test required local socket access, and passed when run with that access.

Real inspection data, trained defect weights, annotation-time studies, repeat-inspection
alignment and measured target-hardware recall/throughput remain separate work. New
standalone repositories were not created. The original review below is a historical
snapshot; its references to unimplemented fixes describe the state before this follow-up.

I reviewed the six local projects, reran their automated tests and checked the
latest GitHub Actions run for each repository. This review does not change runtime
code or establish real-world model accuracy. Model downloads were disabled for the
local test runs; desktop tests used Qt's offscreen mode.

## Checks completed

| Project | Local tests passed | Latest GitHub workflow |
|---|---:|---|
| hybrid-rag | 6 | [Success](https://github.com/KhawajaSaad22/hybrid-rag/actions/runs/43012443237) |
| video-evidence-agent | 8 | [Success](https://github.com/KhawajaSaad22/video-evidence-agent/actions/runs/43012451441) |
| defect-inspection | 7 | [Success](https://github.com/KhawajaSaad22/defect-inspection/actions/runs/43012473311) |
| annotation-assistant | 8 | [Success](https://github.com/KhawajaSaad22/annotation-assistant/actions/runs/43012433912) |
| CV2_Image_to_Script | 59 | [Success](https://github.com/KhawajaSaad22/CV2_Image_to_Script/actions/runs/43012460791) |
| drone-inspection-heatmaps | 8 | [Success](https://github.com/KhawajaSaad22/drone-inspection-heatmaps/actions/runs/43012646106) |

Total: 96 local tests passed. Existing tests primarily demonstrate behavior on
controlled inputs; passing them does not establish accuracy on production data.
GitHub workflow success does not establish local GPU compatibility or model quality.

## Specific findings and proposed updates

### 1. Video verifier frame order — reproducible correctness issue

In `video-evidence-agent/verifier.py`, `inspect()` appends additional samples to
existing samples. A six-frame reproduction produced timestamps
`[2, 4, 6, 2.5, 4.5, 6.5]`, while the verifier prompt describes them as ordered.

Proposed fix: sort paired images/timestamps before each verifier call, preserve
that order in saved results, and add a regression test for the second inspection
turn. Do not sort timestamps independently of images or assessment frame indices.

### 2. Annotation approval integrity — missing mask verification

`annotation-assistant/app.py` hashes the reviewed image but stores only a path to
the accepted mask. `import_reviews.py` verifies image content, review source and
dimensions, but does not verify that mask content is unchanged since approval.
An altered mask with identical dimensions can therefore be imported as approved.

Proposed fix: record and verify mask SHA-256, retain checkpoint identity, check
binary mask values, and test mutation after approval. Give old review records an
explicit migration/re-review path. Reset approval when the image, prediction or
uploaded correction changes; check this interaction with UI tests.

### 3. Drone video completion and timestamps

`drone-inspection-heatmaps/src/drone_heatmaps/pipeline.py` treats every failed read
as end-of-file and calculates timestamps as frame index divided by nominal FPS.
An interrupted/corrupt decode can look like a completed run; variable-frame-rate
video may have inaccurate timestamps.

Proposed updates: explicit running/completed/failed run state, recorded exceptions,
decoder timestamps where supported and tested, documented fallback behavior,
corrupt/truncated-video fixtures, and recovery checks. Frame-count metadata is only
a diagnostic and should not be treated as an authoritative completion check.

### 4. Drone storage and recall need a real benchmark

The stored grid is compact, but the pipeline also writes one NPY file, one preview
and, for suspicious frames, a full PNG plus tile crops. Tile overlaps and adjacent
video frames can repeat evidence. The original synthetic run produced more output
bytes than the compressed input video.

Proposed updates: configurable evidence modes, optional previews, temporal event
grouping, batched heatmap storage and exact total-byte accounting. Compare with
ordinary compressed video and sampled-frame baselines at equal defect recall.
Deduplication must preserve representative high-resolution evidence.

Add RF-DETR adapter tests, an actual checkpoint inference smoke test, and model
container build coverage. The existing Linux CI builds/tests the demo base image,
not the model target. Add source/checkpoint hashes and dependency versions to run
metadata. Lock dependencies after validating the model environment.

### 5. Hybrid RAG evaluation omits unanswerable questions

`hybrid-rag/evaluate.py` requires a nonempty relevant-source list for every question.
It measures source retrieval, not answer correctness or appropriate abstention.

Next checks: negative/unanswerable queries, misleading documents, citation support,
real local answer generation, PDF ingestion, and update/removal of indexed files.
Add OCR only if scanned documents are an actual input requirement. Benchmark index
size and latency before selecting a more complex storage engine.

### 6. Few-shot defect inspection needs real domain-shift evidence

The implementation has checks for exact train/calibration/test content overlap, but
exact hashes cannot identify related views from the same object or capture session.
The recorded toy results are too easy to demonstrate robustness.

Next checks: a licensed real dataset or owned capture set, splits grouped by physical
object/session, several seeds, held-out lighting sessions, and threshold stability.
Report false-positive rate and recall with uncertainty, including very small defects.
Treat preprocessing changes as controlled experiments rather than presumed improvements.

### 7. Image workbench next checks

Its 59-test suite currently covers the widest range of implemented behavior.
Next checks: large-image peak memory, cancellation during heavy operations, high-DPI
desktop layouts, installer launch/uninstall on a clean Windows account, and exported
bundles running outside the development environment. The profiling script records
warm CPU latency but explicitly does not measure memory; its docstring mentions RSS
and should be aligned with the actual output.

## Shared priorities

1. Fix the reproduced verifier ordering issue and annotation approval integrity gap.
2. Create a shared real-data evaluation protocol with group identifiers and immutable
   manifests, then select representative data for each project.
3. Validate real model paths, rather than counting synthetic tests as model validation.
4. Benchmark GPU latency and peak memory in isolated environments on my RTX 4060.
5. Refresh stale documentation. `CV_PROJECTS.md` still says projects are unpublished;
   drone validation notes predate its successful Linux CI. Preserve historical results
   while distinguishing them from current verified status.

For grouping, scikit-learn's [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
provides non-overlapping groups; defining the correct physical groups remains part of
dataset preparation. If deploying on Jetson later, validate against NVIDIA's
[PyTorch/JetPack compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
A desktop GPU container is not proof of Jetson compatibility.

## Additional project candidates

These are proposed scopes, not validated commercial opportunities.

| Candidate | Useful first deliverable | Reuses | Completion measure |
|---|---|---|---|
| Inspection dataset auditor | Dataset manifest checks for missing labels, exact/near duplicates, group overlap and class imbalance | Workbench and annotation manifests | Detect deliberately seeded dataset errors; benchmark false duplicate matches |
| Edge inference benchmark runner | Repeated runs collecting model identity, latency percentiles, memory, output bytes and task accuracy | Drone and defect pipelines | Reproduce comparisons on one fixed held-out dataset and device |
| Repeat-inspection change tracker | Align paired views of the same surface and show candidate new/expanded defects for human review | Defect maps and evidence storage | Measure false changes from viewpoint/lighting separately from actual changes |
| Inspection evidence report builder | Report with source frames, timestamps, model version and reviewer decisions | Drone outputs and retrieval components | Every reported finding resolves to its original evidence and review record |

My recommended sequence: reliability fixes, then a dataset auditor or benchmark
runner. The report builder is the smallest path toward an integrated demonstration.
Repeat-inspection change tracking is a larger research project because alignment and
viewpoint changes can resemble damage. None requires full model pretraining as its
first milestone. No new repositories or runtime fixes were created by this review.
