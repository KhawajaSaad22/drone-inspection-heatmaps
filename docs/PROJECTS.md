# Computer vision project sequence

1. `defect-inspection/`: few-shot normal-only inspection under domain shift and dataset auditing.
2. `video-evidence-agent/`: visual retrieval and bounded model verification.
3. `annotation-assistant/`: student segmentation, active selection and human-reviewed labels.
4. `hybrid-rag/`: document retrieval, reranking and optional local answer generation.
5. `CV2_Image_to_Script/`: preprocessing workbench, paired dataset generation and exports.
6. `drone-inspection-heatmaps/`: tiled defect segmentation, compact summaries and retained evidence.

All six projects are published under https://github.com/KhawajaSaad22 . Folders are
independent. My RTX 4060 has 8 GB VRAM; target GPU execution still needs validation.
Full model pretraining is outside the initial scope. See PROJECT_REVIEW_2026-11-12.md
for the review, and each repository's validation notes for checks actually completed.
