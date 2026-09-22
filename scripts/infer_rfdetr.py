"""Run inference with a fine-tuned RF-DETR checkpoint and save annotated images.

Usage:
  python scripts/infer_rfdetr.py --checkpoint D:/rf-detr-tomato/runs/rfdetr-s-tomato-v1/checkpoint_best_ema.pth \
      --dataset-dir D:/rf-detr-tomato --out-dir D:/rf-detr-inference-samples --n 8 --threshold 0.5
"""
import argparse
import json
import random
from pathlib import Path

import supervision as sv
from PIL import Image

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", type=str, default="small", choices=list(MODELS))
    ap.add_argument("--dataset-dir", default="D:/rf-detr-tomato")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out-dir", default="D:/rf-detr-inference-samples")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    # checkpoints do not carry the training resolution, so it must be passed explicitly
    ap.add_argument("--resolution", type=int, required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    split_dir = dataset_dir / args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "_annotations.coco.json", encoding="utf-8") as f:
        coco = json.load(f)

    # rfdetr trains with contiguous 0-indexed labels internally, so Detections.class_id
    # is (coco_category_id - 1) when category ids are the standard 1..N COCO convention.
    class_names = {c["id"] - 1: c["name"] for c in coco["categories"]}

    random.seed(args.seed)
    normal_cat_ids = {c["id"] for c in coco["categories"] if c["name"].endswith("_normal")}
    img_id_to_cat = {a["image_id"]: a["category_id"] for a in coco["annotations"]}
    disease_imgs = [im for im in coco["images"] if img_id_to_cat.get(im["id"]) not in normal_cat_ids]
    normal_imgs = [im for im in coco["images"] if img_id_to_cat.get(im["id"]) in normal_cat_ids]
    half = args.n // 2
    images = random.sample(disease_imgs, min(half, len(disease_imgs))) + \
        random.sample(normal_imgs, min(args.n - half, len(normal_imgs)))

    print(f"Loading {args.model} model from {args.checkpoint} ...")
    model = MODELS[args.model](pretrain_weights=args.checkpoint, resolution=args.resolution)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    coco_cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    for im_info in images:
        img_path = split_dir / im_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        gt_name = coco_cat_names.get(img_id_to_cat.get(im_info["id"]), "?")

        detections = model.predict(image, threshold=args.threshold)

        labels = [
            f"{class_names.get(cid, cid)} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        out_path = out_dir / f"pred_{img_path.name}"
        annotated.save(out_path)
        print(f"{img_path.name}: GT={gt_name} | pred={labels} | saved to {out_path}")


if __name__ == "__main__":
    main()
