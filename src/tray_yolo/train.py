"""Split images/labels into dataset/{train,val}, write data.yaml and train YOLO.

  pip install ultralytics
  python3 train.py [--model yolov8n-obb.pt] [--epochs 100] [--val 0.2] [--no-train]
Only images that have a label file are used (empty file = background image).
Labels are YOLO-OBB (label.py): class x1 y1 x2 y2 x3 y3 x4 y4.
Result: runs/<--name, default tray>/weights/best.pt
"""
import argparse
import os
import random
import shutil

from common import CLASSES, DATASET, IMAGES, LABELS, ROOT


def prepare(val_ratio, seed=0):
    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith(('.jpg', '.png'))
                   and os.path.exists(os.path.join(
                       LABELS, os.path.splitext(f)[0] + '.txt')))
    if len(files) < 10:
        raise SystemExit(f'only {len(files)} labelled images (need ~100+)')
    random.Random(seed).shuffle(files)
    nval = max(1, int(len(files) * val_ratio))
    if os.path.isdir(DATASET):
        shutil.rmtree(DATASET)
    for split, fs in (('val', files[:nval]), ('train', files[nval:])):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(DATASET, sub, split))
        for f in fs:
            shutil.copy(os.path.join(IMAGES, f), os.path.join(DATASET, 'images', split, f))
            shutil.copy(os.path.join(LABELS, os.path.splitext(f)[0] + '.txt'),
                        os.path.join(DATASET, 'labels', split, os.path.splitext(f)[0] + '.txt'))
    yaml_path = os.path.join(DATASET, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {DATASET}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(CLASSES):
            f.write(f'  {i}: {n}\n')
    print(f'{len(files) - nval} train / {nval} val -> {yaml_path}')
    return yaml_path


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default='yolov8n-obb.pt', help='OBB model (yolov8s-obb.pt = bigger)')
    a.add_argument('--epochs', type=int, default=100)
    a.add_argument('--imgsz', type=int, default=640)
    a.add_argument('--val', type=float, default=0.2)
    a.add_argument('--name', default='tray', help='run name -> runs/<name>/weights/best.pt')
    a.add_argument('--no-train', action='store_true')
    args = a.parse_args()
    yaml_path = prepare(args.val)
    if args.no_train:
        return
    from ultralytics import YOLO
    YOLO(args.model).train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                           project=os.path.join(ROOT, 'runs'), name=args.name, exist_ok=True,
                           flipud=0.5, degrees=0)   # OBB polygons follow flips; tray yaw is already varied


if __name__ == '__main__':
    main()
