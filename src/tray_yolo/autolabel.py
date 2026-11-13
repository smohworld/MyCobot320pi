"""Auto-label tray / pin OBBs: YOLO finds the rough boxes, classic CV snaps them tight.

  ~/venvs/yolo/bin/python autolabel.py                 # images/ -> labels_auto/
  ~/venvs/yolo/bin/python autolabel.py --compare       # + IoU / centre offset vs labels/ (manual)
  ~/venvs/yolo/bin/python autolabel.py --montage out.jpg   # + contact sheet of the results
  python3 label.py --labels labels_auto                # review / fix (start with _review.txt)
  cp labels_auto/*.txt labels/                         # when happy, then train.py

Refinement (both fall back to the YOLO box and are listed in _review.txt when it fails):
  tray : black body (V < --tray-v) inside the enlarged YOLO box -> largest blob -> minAreaRect
         (same definition tray_detect uses). A blob touching the image border keeps the YOLO box.
  pin  : the tan tape marker on top of the pin (darker and more saturated than the desk, not
         black, orange hue) -> blob nearest the YOLO centre -> minAreaRect. The box is the tape = pin top,
         not the white body below it, whose outline changes with the viewing angle.
Output format = label.py / train.py: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised).
"""
import argparse
import os

import cv2
import numpy as np

from common import CLASSES, COLORS, IMAGES, LABELS, ROOT

TRAY, PIN, PIN_FRONT = 0, 1, 2


def roi_mask(shape, poly, scale):
    """Mask of the YOLO polygon enlarged by `scale` around its centre."""
    c = poly.mean(0)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32((poly - c) * scale + c)], 255)
    return m


def rect_poly(rect):
    return cv2.boxPoints(rect).astype(np.float64)


def touches_border(cnt, w, h, margin=2):
    x, y, bw, bh = cv2.boundingRect(cnt)
    return x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin


def refine_tray(img, hsv, poly, args):
    h, w = img.shape[:2]
    m = cv2.inRange(hsv, (0, 0, 0), (180, 255, args.tray_v))
    m &= roi_mask(img.shape, poly, 1.15)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 'no dark blob'
    cnt = max(cnts, key=cv2.contourArea)
    if touches_border(cnt, w, h):
        return None, 'at image border'
    new = rect_poly(cv2.minAreaRect(cnt))
    if iou(new, poly, img.shape) < args.min_iou:
        return None, f'tray iou {iou(new, poly, img.shape):.2f} vs yolo'
    return new, ''


def refine_pin(img, hsv, poly, args):
    h, w = img.shape[:2]
    roi = roi_mask(img.shape, poly, 1.6)
    hue, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    desk_v = np.percentile(v[roi > 0], 90)              # brightest part of the ROI = desk / pin
    tan = (hue <= args.pin_hmax) | (hue >= 170)         # orange-brown; tray edges are bluish
    m = ((v < desk_v - args.pin_dv) & (v > args.tray_v + 30) & (s > args.pin_s) & tan)
    m = m.astype(np.uint8)
    m = m * 255 & roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= args.pin_min_area]
    if not cnts:
        return None, 'no tape blob'
    c0 = poly.mean(0)

    def dist(c):
        mu = cv2.moments(c)
        return np.hypot(mu['m10'] / mu['m00'] - c0[0], mu['m01'] / mu['m00'] - c0[1])
    cnt = min(cnts, key=dist)
    if touches_border(cnt, w, h):
        return None, 'at image border'
    rect = cv2.minAreaRect(cnt)
    new = rect_poly(rect)
    side = np.sqrt(cv2.contourArea(poly.astype(np.float32)))
    if dist(cnt) > 0.6 * side:
        return None, f'tape {dist(cnt):.0f}px from yolo centre'
    if min(rect[1]) < 0.5 * max(rect[1]):               # the tape is square seen from above
        return new, f'tape not square {rect[1][0]:.0f}x{rect[1][1]:.0f}'
    return new, ''


def iou(a, b, shape):
    ma = np.zeros(shape[:2], np.uint8)
    mb = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(ma, [np.int32(np.round(a))], 1)
    cv2.fillPoly(mb, [np.int32(np.round(b))], 1)
    u = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / u if u else 0.0


def yolo_boxes(model, img, args):
    """-> {cls: best polygon (4x2 px)}; one tray and one pin per image."""
    r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    best = {}
    if r.obb is None:
        return best
    for pts, c, p in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.cpu().numpy(),
                         r.obb.conf.cpu().numpy()):
        c = int(c)
        if c not in best or p > best[c][1]:
            best[c] = (pts.astype(np.float64), float(p))
    return best


def load_manual(path, w, h):
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            t = ln.split()
            if len(t) == 9:
                out.setdefault(int(t[0]), np.array(t[1:], float).reshape(4, 2) * [w, h])
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default=os.path.join(ROOT, 'runs/tray_auto/weights/best.pt'))
    a.add_argument('--images', default=IMAGES)
    a.add_argument('--out', default=os.path.join(ROOT, 'labels_auto'))
    a.add_argument('--conf', type=float, default=0.25)
    a.add_argument('--imgsz', type=int, default=640)
    a.add_argument('--tray-v', type=int, default=60, help='tray body: HSV V below this')
    a.add_argument('--pin-dv', type=int, default=25, help='tape: this much darker than the desk')
    a.add_argument('--pin-s', type=int, default=18, help='tape: HSV S above this')
    a.add_argument('--pin-hmax', type=int, default=35, help='tape: HSV hue at most this')
    a.add_argument('--pin-min-area', type=int, default=40)
    a.add_argument('--min-iou', type=float, default=0.6, help='tray: reject refinement below')
    a.add_argument('--compare', action='store_true', help='IoU / centre offset vs labels/')
    a.add_argument('--montage', default='', help='write a contact sheet jpg here')
    args = a.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(('.jpg', '.png')))
    review, stats, tiles = [], {TRAY: [], PIN: []}, []
    for f in files:
        img = cv2.imread(os.path.join(args.images, f))
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        boxes, notes = {}, []
        for c, (poly, p) in yolo_boxes(model, img, args).items():
            if c == PIN_FRONT:              # no refinement for the front view pin yet
                new, why = None, 'not refined'
            else:
                new, why = (refine_tray if c == TRAY else refine_pin)(img, hsv, poly, args)
            boxes[c] = new if new is not None else poly
            if why:
                notes.append(f'{CLASSES[c]}: {why}' + ('' if new is not None else ' -> yolo box'))
        for c in (TRAY, PIN):
            if c not in boxes:
                notes.append(f'{CLASSES[c]}: not detected')
        stem = os.path.splitext(f)[0]
        with open(os.path.join(args.out, stem + '.txt'), 'w') as fo:
            for c, pts in sorted(boxes.items()):
                n = (pts / [w, h]).ravel()
                fo.write(f'{c} ' + ' '.join(f'{v:.6f}' for v in n) + '\n')
        if notes:
            review.append(f'{f}  ' + '; '.join(notes))
        if args.compare:
            man = load_manual(os.path.join(LABELS, stem + '.txt'), w, h)
            for c in (TRAY, PIN):
                if c in man and c in boxes:
                    stats[c].append((iou(boxes[c], man[c], img.shape),
                                     np.linalg.norm(boxes[c].mean(0) - man[c].mean(0))))
        if args.montage:
            v = img.copy()
            for c, pts in boxes.items():
                cv2.polylines(v, [np.int32(np.round(pts))], True, COLORS[c], 1)
            if notes:
                cv2.putText(v, 'REVIEW', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tiles.append(cv2.resize(v, (320, 240)))

    with open(os.path.join(args.out, '_review.txt'), 'w') as fo:
        fo.write('\n'.join(review) + ('\n' if review else ''))
    print(f'{len(files)} images -> {args.out}  ({len(review)} to review, see _review.txt)')
    for ln in review[:15]:
        print('  ' + ln)
    if args.compare:
        for c in (TRAY, PIN):
            if stats[c]:
                s = np.array(stats[c])
                print(f'{CLASSES[c]:4s} vs manual: IoU median {np.median(s[:, 0]):.2f} '
                      f'(<0.5: {np.sum(s[:, 0] < 0.5)}/{len(s)}), centre offset median '
                      f'{np.median(s[:, 1]):.1f}px, 90% {np.percentile(s[:, 1], 90):.1f}px')
    if args.montage and tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        cv2.imwrite(args.montage, np.vstack(rows))
        print('montage ->', args.montage)


if __name__ == '__main__':
    main()
