"""CLI entry point for the disease-diagnosis pipeline. See diagnosis_pipeline.py for the actual logic
(also used by api.py to serve the same pipeline over HTTP for the backend to call).

Usage:
  python scripts/full_pipeline.py --image path/to/photo.jpg --crop tomato
  (defaults point at models/rfdetr-s-synthetic-v5/; pass --severity-model to enable
  tomato severity grading, it is not bundled in this repo)
"""
import argparse
import json

from PIL import Image

from diagnosis_pipeline import CROPS, DiagnosisPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--crop", required=True, choices=CROPS,
                    help="Crop grown at this site; predictions for other crops are discarded.")
    ap.add_argument("--detector", default="models/rfdetr-s-synthetic-v5/last_ema.pth")
    ap.add_argument("--categories", default="models/rfdetr-s-synthetic-v5/categories.json")
    # checkpoints do not carry the training resolution, so it must match training (640 for this model)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--severity-model", default=None,
                    help="Optional tomato severity classifier checkpoint; not bundled in this repo.")
    ap.add_argument("--knowledge", default="data/disease_knowledge.json")
    # on unseen plants: 0.5 misses 20% of diseased images, 0.15 misses <1% with 1.4% false alarms on healthy ones
    ap.add_argument("--threshold", type=float, default=0.15)
    # 1 = close-up photos (training domain). For wide shots where a lesion is a small part of the frame use 3.
    # Simulated wide scenes (lesion ~18% of frame): 17% -> 94% correct with tiles=3, at the cost of ~5% false
    # alarms on healthy plants (vs ~1.5%). Tune on real camera footage before relying on it.
    ap.add_argument("--tiles", type=int, default=1)
    args = ap.parse_args()

    pipeline = DiagnosisPipeline(
        detector_path=args.detector,
        categories_path=args.categories,
        knowledge_path=args.knowledge,
        severity_model_path=args.severity_model,
        resolution=args.resolution,
    )
    image = Image.open(args.image)
    output = pipeline.diagnose(image, args.crop, threshold=args.threshold, tiles=args.tiles)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
