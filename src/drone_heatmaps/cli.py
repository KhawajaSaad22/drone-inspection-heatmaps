import argparse
import json
from pathlib import Path
from .detectors import RFDETRDetector, SyntheticDetector


def main():
    parser = argparse.ArgumentParser(description="Drone inspection heatmap prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run simulated marker detection, not defect AI")
    demo.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("inspect", help="Process a video with fine-tuned RF-DETR weights")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--size", choices=["nano", "small"], default="nano")
    run.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--tile-size", type=int, default=512)
    run.add_argument("--overlap", type=int, default=64)
    run.add_argument('--evidence-mode',choices=['frame','crops','both'],default='both')
    run.add_argument('--no-previews',action='store_true')
    run.add_argument('--heatmap-format',choices=['npy','packed'],default='npy')
    run.add_argument('--evidence-gap-s',type=float,default=0,
                     help='Optional minimum time between retained frames; may skip useful views')
    run.add_argument("--class-ids", type=int, nargs="+", required=True,
                     help="Defect class IDs from my training dataset")
    args = parser.parse_args()
    try:
        from .pipeline import make_demo, process_video
        if args.command == "demo":
            if args.output.exists():
                raise ValueError("Choose a new output directory")
            args.output.mkdir(parents=True)
            video = args.output/"synthetic.avi"
            make_demo(video)
            result = process_video(video, args.output/"run", SyntheticDetector(), synthetic=True,
                                   model_info={"backend": "synthetic-magenta-marker"})
        else:
            if args.output.exists():
                raise ValueError("Choose a new output directory")
            if not args.video.is_file() or args.stride < 1 or not 0 < args.threshold <= 1:
                raise ValueError("Check input video, stride, and threshold")
            if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
                raise ValueError("Require 0 <= overlap < tile size")
            from .pipeline import sha256_file
            detector = RFDETRDetector(args.weights, args.size, args.device, args.threshold)
            result = process_video(args.video, args.output, detector, tile_size=args.tile_size,
                overlap=args.overlap, stride=args.stride, threshold=args.threshold,
                allowed_classes=set(args.class_ids), model_info={"backend": "rfdetr-seg",
                    "size": args.size, "weights": str(args.weights), "weights_sha256": sha256_file(args.weights), "device": args.device},
                evidence_mode=args.evidence_mode,previews=not args.no_previews,
                heatmap_format=args.heatmap_format,evidence_gap_s=args.evidence_gap_s)
            if result['status']!='completed':
                parser.exit(3,'Video may be incomplete; inspect run.json before using the results\n')
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
