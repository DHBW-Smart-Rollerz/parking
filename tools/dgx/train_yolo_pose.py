"""Trains the YOLO pose parking spot detector on an exported dataset, without notebook.

Made for the DGX H100 of the DHBW (SLURM job train_yolo_pose.slurm next to this file),
but runs on any machine with Ultralytics. The dataset is the export of
notebooks/train_yolo_pose.ipynb (data/yolo/<dataset>/ with data.yaml, images/ and
labels/). The training settings are the same as in the notebook.

Usage:
    python3 train_yolo_pose.py <dataset dir> [--model yolo26n-pose.pt] [--imgsz 800] \
        [--epochs 150] [--batch 32] [--device 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="Exported YOLO dataset directory")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Pretrained model")
    parser.add_argument("--imgsz", type=int, default=800, help="800: parking_bev")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index or 'cpu'")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", default=None, help="Default: <model>_<imgsz>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # data.yaml of the export holds the absolute path of the machine it was exported
    # on, so a copy with the path on this machine is used
    dataset = args.dataset.resolve()
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    data["path"] = str(dataset)
    data_yaml = dataset / "data_local.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"Dataset {dataset}: classes {data['names']}")

    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=name,
        exist_ok=True,
        cache="ram",  # The dataset is small (~100 MB)
        # Augmentation, same as notebooks/train_yolo_pose.ipynb
        fliplr=0.0,  # Never mirror (side of the road matters)
        hsv_h=0.0,  # Grayscale images, no hue/saturation
        hsv_s=0.0,
        hsv_v=0.4,  # Brightness
        degrees=15.0,  # Rotated spots during the maneuvers
        scale=0.2,  # Fixed scale of the bird's eye view
        mosaic=1.0,
        close_mosaic=10,  # No mosaic in the last epochs
    )
    print(f"Best model: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
