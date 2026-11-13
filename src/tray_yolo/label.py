"""Oriented-box (OBB) labelling tool -> YOLO-OBB txt files in labels/
(class x1 y1 x2 y2 x3 y3 x4 y4, the four corners, normalised).

  python3 label.py
  python3 label.py --labels labels_auto --review    # check autolabel.py output (flagged images)
Draw a rotated box in three clicks:
  1. left click   one corner of the box
  2. left click   the next corner (this edge sets the box direction: click along a side)
  3. move the mouse (the box follows, perpendicular to that edge) and left click to finish
Right click = cancel the box being drawn, or delete the box under the cursor.
Keys:  1 tray / 2 pin (current class)   d delete last box   c clear all boxes
       n / SPACE / -> next image   p / <- previous image   (labels are saved when you leave)
       u jump to next unlabelled image   q / ESC quit (saves)
Starts at the first unlabelled image. Images with no boxes get an empty label file only if you
press n/p on them (= background). Old 5-number (axis-aligned) label files are read as boxes.
"""
import argparse
import os

import cv2
import numpy as np

from common import CLASSES, COLORS, IMAGES, LABELS


def label_path(img_name):
    return os.path.join(LABELS, os.path.splitext(img_name)[0] + '.txt')


def load(img_name, w, h):
    """-> [[cls, 4x2 float array of corner pixels]]"""
    p, boxes = label_path(img_name), []
    if os.path.exists(p):
        for ln in open(p):
            t = ln.split()
            if len(t) == 9:
                pts = np.array(t[1:], float).reshape(4, 2) * [w, h]
            elif len(t) == 5:                     # axis-aligned box from the old tool
                cx, cy, bw, bh = map(float, t[1:])
                x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, \
                    (cx + bw / 2) * w, (cy + bh / 2) * h
                pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            else:
                continue
            boxes.append([int(t[0]), pts])
    return boxes


def save(img_name, boxes, w, h):
    os.makedirs(LABELS, exist_ok=True)
    with open(label_path(img_name), 'w') as f:
        for c, pts in boxes:
            n = np.clip(pts / [w, h], 0.0, 1.0)
            f.write(f'{c} ' + ' '.join(f'{v:.6f}' for v in n.reshape(-1)) + '\n')


def box_from(p1, p2, q):
    """Corners of the rectangle with edge p1-p2 whose opposite side passes through q."""
    p1, p2, q = (np.asarray(v, float) for v in (p1, p2, q))
    e = p2 - p1
    ln = np.linalg.norm(e)
    if ln < 1e-6:
        return None
    n = np.array([-e[1], e[0]]) / ln
    d = float(np.dot(q - p1, n))
    return np.array([p1, p2, p2 + n * d, p1 + n * d])


class Labeler:
    def __init__(self, files):
        self.files, self.i = files, 0
        self.cls = 0
        self.clicks = []          # corners clicked so far (0..2)
        self.mouse_pos = (0, 0)
        self.load_cur()

    def load_cur(self):
        self.name = self.files[self.i]
        self.img = cv2.imread(os.path.join(IMAGES, self.name))
        self.h, self.w = self.img.shape[:2]
        self.boxes = load(self.name, self.w, self.h)
        self.clicks = []

    def flush(self):
        save(self.name, self.boxes, self.w, self.h)

    def go(self, i):
        self.flush()
        self.i = max(0, min(len(self.files) - 1, i))
        self.load_cur()

    def mouse(self, ev, x, y, flags, _):
        self.mouse_pos = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicks) < 2:
                self.clicks.append((x, y))
            else:
                pts = box_from(self.clicks[0], self.clicks[1], (x, y))
                self.clicks = []
                if pts is not None and np.linalg.norm(pts[2] - pts[1]) > 4:
                    self.boxes.append([self.cls, pts])
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if self.clicks:
                self.clicks = []
                return
            for b in reversed(self.boxes):
                if cv2.pointPolygonTest(b[1].astype(np.float32), (float(x), float(y)),
                                        False) >= 0:
                    self.boxes.remove(b)
                    break

    def draw(self):
        v = self.img.copy()
        for c, pts in self.boxes:
            poly = np.int32(pts)
            cv2.polylines(v, [poly], True, COLORS[c], 2)
            cv2.circle(v, tuple(poly[0]), 4, COLORS[c], -1)      # first corner: direction mark
            cv2.putText(v, CLASSES[c], tuple(poly[0] + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLORS[c], 1)
        if len(self.clicks) == 1:
            cv2.line(v, self.clicks[0], self.mouse_pos, COLORS[self.cls], 1)
        elif len(self.clicks) == 2:
            pts = box_from(self.clicks[0], self.clicks[1], self.mouse_pos)
            if pts is not None:
                cv2.polylines(v, [np.int32(pts)], True, COLORS[self.cls], 1)
        done = sum(os.path.exists(label_path(f)) for f in self.files)
        cv2.putText(v, f'[{self.i + 1}/{len(self.files)}] labelled {done}  class: '
                    f'{self.cls + 1}={CLASSES[self.cls]}  clicks {len(self.clicks)}/3', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return v


def main():
    global LABELS
    a = argparse.ArgumentParser()
    a.add_argument('--labels', default=LABELS, help='label dir (e.g. labels_auto)')
    a.add_argument('--review', action='store_true',
                   help='only the images listed in <labels>/_review.txt (autolabel.py)')
    args = a.parse_args()
    LABELS = os.path.abspath(args.labels)
    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith(('.jpg', '.png')))
    if args.review:
        rv = os.path.join(LABELS, '_review.txt')
        keep = {ln.split()[0] for ln in open(rv) if ln.strip()} if os.path.exists(rv) else set()
        files = [f for f in files if f in keep]
    if not files:
        raise SystemExit(f'no images in {IMAGES} (run capture.py first)')
    L = Labeler(files)
    first = next((j for j, f in enumerate(files) if not os.path.exists(label_path(f))), None)
    if first:                              # start at the first unlabelled image
        L.go(first)
    cv2.namedWindow('label')
    cv2.setMouseCallback('label', L.mouse)
    while True:
        cv2.imshow('label', L.draw())
        k = cv2.waitKey(30)
        kk = k & 0xFF
        if kk in (ord('q'), 27):
            L.flush()
            break
        elif ord('1') <= kk < ord('1') + len(CLASSES):
            L.cls = kk - ord('1')
        elif kk == ord('d') and L.boxes:
            L.boxes.pop()
        elif kk == ord('c'):
            L.boxes.clear()
            L.clicks = []
        elif kk in (ord('n'), ord(' ')) or k == 65363:
            L.go(L.i + 1)
        elif kk == ord('p') or k == 65361:
            L.go(L.i - 1)
        elif kk == ord('u'):
            nxt = next((j for j in range(L.i + 1, len(files))
                        if not os.path.exists(label_path(files[j]))), None)
            if nxt is not None:
                L.go(nxt)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
