"""Fine-tune RF-DETR-Small on the converted crop-disease COCO dataset.

Usage:
  python scripts/train_rfdetr.py --dataset-dir D:/rf-detr-dataset --output-dir D:/rf-detr-dataset/runs/rfdetr-s-v1 --epochs 30
"""
import argparse

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
    ap.add_argument("--dataset-dir", type=str, default="D:/rf-detr-dataset")
    ap.add_argument("--output-dir", type=str, default="D:/rf-detr-dataset/runs/rfdetr-s-v1")
    ap.add_argument("--model", type=str, default="small", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--cls-loss-coef", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--lr-drop", type=int, default=100)
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--init-checkpoint", type=str, default=None,
                    help="Start from a previously fine-tuned checkpoint instead of the COCO-pretrained weights.")
    args = ap.parse_args()

    model = MODELS[args.model](pretrain_weights=args.init_checkpoint) if args.init_checkpoint else MODELS[args.model]()
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        resolution=args.resolution,
        use_ema=True,
        checkpoint_interval=5,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=0.001,
        cls_loss_coef=args.cls_loss_coef,
        warmup_epochs=args.warmup_epochs,
        lr_drop=args.lr_drop,
        tensorboard=True,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
