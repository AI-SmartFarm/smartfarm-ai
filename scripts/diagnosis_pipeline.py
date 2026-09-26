"""Core disease-diagnosis pipeline, shared by the CLI (full_pipeline.py) and the HTTP API (api.py):
  1. RF-DETR-Small detects & classifies the lesion (5 crops, 14 classes)
  2. Predictions are restricted to the crop the farm actually grows
  3. Each disease box is cropped and graded early/mid/late by that disease's own severity classifier
  4. Cause/symptoms/prevention principles come from the static knowledge base
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import batched_nms

from rfdetr import RFDETRSmall

CROPS = ["pepper", "strawberry", "lettuce", "cucumber", "tomato"]
SEVERITY_LABEL = {1: "초기", 2: "중기", 3: "말기"}
# severity calls from a model below this valid accuracy are flagged so the app can show them as advisory only
SEVERITY_LOW_CONFIDENCE_THRESHOLD = 0.6


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


def load_severity_models(severity_dir, device):
    """One ResNet18 per disease code (model_{code}.pt, 3 classes each). A single 27-way model across all
    9 diseases plateaued at 63.9% valid acc; per-disease models reach 69.4% weighted. Returns
    {code: (model, classes, valid_accuracy)}; diseases without a model file are simply not graded."""
    if severity_dir is None or not Path(severity_dir).is_dir():
        return {}
    summary_path = Path(severity_dir) / "_summary.json"
    accuracy = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            accuracy = {row["disease"]: row["best_acc"] for row in json.load(f)}

    loaded = {}
    for path in sorted(Path(severity_dir).glob("model_*.pt")):
        code = int(path.stem.split("_")[1])
        ckpt = torch.load(path, map_location=device)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["classes"]))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        loaded[code] = (model, ckpt["classes"], accuracy.get(code))
    return loaded


class DiagnosisPipeline:
    """Loads the detector, per-disease severity classifiers, and knowledge base once, then serves diagnose() calls.
    Instantiate a single instance and reuse it (CLI: one call; API: one instance for the process lifetime) —
    RF-DETR weight loading takes several seconds and should not happen per-request."""

    def __init__(self, detector_path, categories_path, knowledge_path,
                 severity_dir=None, resolution=640):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(knowledge_path, encoding="utf-8") as f:
            self.knowledge = json.load(f)
        with open(categories_path, encoding="utf-8") as f:
            categories = json.load(f)
        # rfdetr returns 0-indexed class ids, i.e. (coco_category_id - 1)
        self.class_names = {v["id"] - 1: v["name"] for v in categories.values()}

        self.detector = RFDETRSmall(pretrain_weights=detector_path, resolution=resolution)
        self.severity_models = load_severity_models(severity_dir, self.device)
        self.severity_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def diagnose(self, image: Image.Image, crop: str, threshold: float = 0.15, tiles: int = 1) -> dict:
        if crop not in CROPS:
            raise ValueError(f"unknown crop '{crop}', expected one of {CROPS}")

        image = image.convert("RGB")
        boxes, cids, confs = detect(self.detector, image, threshold, tiles)

        candidates = []
        for box, cid, conf in zip(boxes, cids, confs):
            cls_name = self.class_names.get(int(cid))
            if cls_name is None or not cls_name.startswith(f"{crop}_"):
                continue
            candidates.append({"class": cls_name, "confidence": float(conf), "bbox": [float(v) for v in box]})

        results = []
        for entry in deduplicate(candidates):
            cls_name = entry["class"]
            if not cls_name.endswith("_normal"):
                code = int(cls_name.rsplit("_disease", 1)[1])
                if code in self.severity_models:
                    severity_model, severity_classes, model_acc = self.severity_models[code]
                    x0, y0, x1, y1 = [int(v) for v in entry["bbox"]]
                    patch = image.crop((x0, y0, x1, y1))
                    inp = self.severity_tf(patch).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        probs = torch.softmax(severity_model(inp), dim=1)[0]
                        pred_idx = int(probs.argmax())
                    risk_code = int(severity_classes[pred_idx].split("_")[1])  # e.g. "18_2" -> 2
                    entry["severity"] = {
                        "level": SEVERITY_LABEL[risk_code],
                        "risk_code": risk_code,
                        "confidence": float(probs[pred_idx]),
                        "model_valid_accuracy": model_acc,
                        "low_confidence": model_acc is not None and model_acc < SEVERITY_LOW_CONFIDENCE_THRESHOLD,
                    }

                info = self.knowledge.get(str(code))
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
            return {"result": "no_detection", "crop": crop, "message": "병징을 찾지 못했습니다 (threshold 미달)"}

        return {
            "result": "detected",
            "crop": crop,
            "disclaimer": self.knowledge.get("_disclaimer"),
            "detections": sorted(results, key=lambda r: -r["confidence"]),
        }
