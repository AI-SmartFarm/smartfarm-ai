"""End-to-end crop disease diagnosis pipeline:
  1. RF-DETR-Small detects & classifies the lesion (5 crops, 14 classes)
  2. Predictions are restricted to the crop the farm actually grows
  3. For tomato diseases, the box is cropped and graded by the severity classifier
  4. Cause/symptoms/prevention principles come from the static knowledge base

Usage:
  python scripts/full_pipeline.py --image path/to/photo.jpg --crop tomato
  (defaults point at models/rfdetr-s-synthetic-v5/; pass --severity-model to enable
  tomato severity grading, it is not bundled in this repo)
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import batched_nms

from rfdetr import RFDETRSmall

CROPS = ["pepper", "strawberry", "lettuce", "cucumber", "tomato"]
# the severity classifier was trained on tomato disease18/19 crops only
SEVERITY_DISEASES = {18, 19}
SEVERITY_LABEL = {1: "초기", 2: "중기", 3: "말기"}


def deduplicate(dets, iou_threshold=0.5):
    """RF-DETR emits several queries per lesion, so the same spot is reported repeatedly."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        x0, y0, x1, y1 = d["bbox"]
        for k in kept:
            kx0, ky0, kx1, ky1 = k["bbox"]
            iw = max(0.0, min(x1, kx1) - max(x0, kx0))
            ih = max(0.0, min(y1, ky1) - max(y0, ky0))
            inter = iw * ih
            union = (x1 - x0) * (y1 - y0) + (kx1 - kx0) * (ky1 - ky0) - inter
            if union > 0 and inter / union > iou_threshold:
                break
        else:
            kept.append(d)
    return kept


def detect(detector, image, threshold, tiles):
    """Full-frame pass, plus (when tiles > 1) overlapping tiles so that small lesions in wide shots are seen at
    a usable scale. Detections from all passes are merged with class-wise NMS. Returns (xyxy, class_id, conf) lists."""
    passes = [(image, 0, 0)]
    if tiles > 1:
        w, h = image.size
        tw, th = w / tiles * 1.25, h / tiles * 1.25          # 25% overlap between neighbouring tiles
        sx = (w - tw) / (tiles - 1)
        sy = (h - th) / (tiles - 1)
        for a in range(tiles):
            for b in range(tiles):
                ox, oy = int(a * sx), int(b * sy)
                passes.append((image.crop((ox, oy, ox + int(tw), oy + int(th))), ox, oy))
    boxes, cids, confs = [], [], []
    for img, ox, oy in passes:
        d = detector.predict(img, threshold=threshold)
        for box, cid, conf in zip(d.xyxy, d.class_id, d.confidence):
            boxes.append([box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy])
            cids.append(int(cid)); confs.append(float(conf))
    if not boxes:
        return [], [], []
    keep = batched_nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(confs), torch.tensor(cids), 0.5).tolist()
    return [boxes[k] for k in keep], [cids[k] for k in keep], [confs[k] for k in keep]


def load_severity_model(path, device):
    if path is None:
        return None, None
    ckpt = torch.load(path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"]


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.knowledge, encoding="utf-8") as f:
        knowledge = json.load(f)
    with open(args.categories, encoding="utf-8") as f:
        categories = json.load(f)
    # rfdetr returns 0-indexed class ids, i.e. (coco_category_id - 1)
    class_names = {v["id"] - 1: v["name"] for v in categories.values()}

    detector = RFDETRSmall(pretrain_weights=args.detector, resolution=args.resolution)
    severity_model, severity_classes = load_severity_model(args.severity_model, device)
    severity_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    boxes, cids, confs = detect(detector, image, args.threshold, args.tiles)

    candidates = []
    for box, cid, conf in zip(boxes, cids, confs):
        cls_name = class_names.get(int(cid))
        if cls_name is None or not cls_name.startswith(f"{args.crop}_"):
            continue
        candidates.append({"class": cls_name, "confidence": float(conf), "bbox": [float(v) for v in box]})

    results = []
    for entry in deduplicate(candidates):
        cls_name = entry["class"]
        if not cls_name.endswith("_normal"):
            code = cls_name.rsplit("_disease", 1)[1]
            if severity_model is not None and int(code) in SEVERITY_DISEASES:
                x0, y0, x1, y1 = [int(v) for v in entry["bbox"]]
                crop = image.crop((x0, y0, x1, y1))
                inp = severity_tf(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(severity_model(inp), dim=1)[0]
                    pred_idx = int(probs.argmax())
                risk_code = int(severity_classes[pred_idx].split("_")[1])  # e.g. "18_2" -> 2
                entry["severity"] = {
                    "level": SEVERITY_LABEL[risk_code],
                    "risk_code": risk_code,
                    "confidence": float(probs[pred_idx]),
                }

            info = knowledge.get(code)
            if info:
                entry["diagnosis"] = {
                    "name_kr": info["name_kr"],
                    "name_en": info["name_en"],
                    "cause": info["cause"],
                    "symptoms": info["symptoms"],
                    "prevention_principles": info["prevention_principles"],
                    "sources": info["sources"],
                }
                for k in ("evidence_level", "note"):
                    if k in info:
                        entry["diagnosis"][k] = info[k]
            else:
                entry["diagnosis"] = {"note": f"질병코드 {code}의 방제 정보가 지식베이스에 없습니다."}

        results.append(entry)

    if not results:
        print(json.dumps({"result": "no_detection", "crop": args.crop,
                          "message": "병징을 찾지 못했습니다 (threshold 미달)"},
                         ensure_ascii=False, indent=2))
        return

    print(json.dumps({
        "result": "detected",
        "crop": args.crop,
        "disclaimer": knowledge.get("_disclaimer"),
        "detections": sorted(results, key=lambda r: -r["confidence"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
