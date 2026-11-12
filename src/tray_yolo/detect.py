"""Live YOLO detection of the tray and the pin (camera or image files).

  python3 detect.py                                   # /dev/video2, runs/tray/weights/best.pt
  python3 detect.py --model path/best.pt --dev /dev/video2 --conf 0.4
  python3 detect.py --image images/img_x.jpg
Prints pixel centres of the best 'tray' and 'pin' box; q / ESC quits.
"""
import argparse
import os
import time

import cv2

from common import CAMERA, COLORS, ROOT, open_camera


def draw(frame, r):
    best = {}
    for (x0, y0, x1, y1), c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy()):
        c = int(c)
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[c % 2], 2)
        cv2.putText(frame, f'{r.names[c]} {p:.2f}', (int(x0), max(12, int(y0) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c % 2], 1)
        if r.names[c] not in best or p > best[r.names[c]][2]:
            best[r.names[c]] = ((x0 + x1) / 2, (y0 + y1) / 2, float(p))
    return best


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default=os.path.join(ROOT, 'runs/tray/weights/best.pt'))
    a.add_argument('--dev', default=CAMERA)
    a.add_argument('--image', default='')
    a.add_argument('--conf', type=float, default=0.4)
    a.add_argument('--imgsz', type=int, default=640)
    args = a.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('pip install ultralytics')
    model = YOLO(args.model)
    cap = None if args.image else open_camera(args.dev)
    last = 0.0
    while True:
        if cap is None:
            frame = cv2.imread(args.image)
        else:
            ok, frame = cap.read()
            if not ok:
                continue
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        best = draw(frame, r)
        if time.monotonic() - last > 0.5 and best:
            print('  '.join(f'{n}: ({u:.0f}, {v:.0f}) {c:.2f}' for n, (u, v, c) in best.items()),
                  flush=True)
            last = time.monotonic()
        cv2.imshow('detect', frame)
        if cv2.waitKey(0 if cap is None else 1) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
