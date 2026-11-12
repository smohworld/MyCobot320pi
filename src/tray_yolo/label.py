"""Minimal box labelling tool -> YOLO txt files in labels/ (class cx cy w h, normalised).

  python3 label.py
Mouse: drag left button = draw a box (class = current class); right click = delete the box
under the cursor.
Keys:  1 tray / 2 pin (current class)   d delete last box   c clear all boxes
       n / SPACE / -> next image   p / <- previous image   (labels are saved when you leave)
       u jump to next unlabelled image   q / ESC quit (saves)
Starts at the first unlabelled image.
Images with no boxes get an empty label file only if you press n/p on them (= background).
"""
import os

import cv2

from common import CLASSES, COLORS, IMAGES, LABELS


def label_path(img_name):
    return os.path.join(LABELS, os.path.splitext(img_name)[0] + '.txt')


def load(img_name, w, h):
    p, boxes = label_path(img_name), []
    if os.path.exists(p):
        for ln in open(p):
            t = ln.split()
            if len(t) == 5:
                c, cx, cy, bw, bh = int(t[0]), *map(float, t[1:])
                boxes.append([c, int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    return boxes


def save(img_name, boxes, w, h):
    os.makedirs(LABELS, exist_ok=True)
    with open(label_path(img_name), 'w') as f:
        for c, x0, y0, x1, y1 in boxes:
            f.write(f'{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} '
                    f'{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n')


class Labeler:
    def __init__(self, files):
        self.files, self.i = files, 0
        self.cls = 0
        self.drag = None
        self.cur = None
        self.load_cur()

    def load_cur(self):
        self.name = self.files[self.i]
        self.img = cv2.imread(os.path.join(IMAGES, self.name))
        self.h, self.w = self.img.shape[:2]
        self.boxes = load(self.name, self.w, self.h)

    def flush(self):
        save(self.name, self.boxes, self.w, self.h)

    def go(self, i):
        self.flush()
        self.i = max(0, min(len(self.files) - 1, i))
        self.load_cur()

    def mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag, self.cur = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.cur = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0 = self.drag
            self.drag = None
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                self.boxes.append([self.cls, min(x0, x), min(y0, y), max(x0, x), max(y0, y)])
        elif ev == cv2.EVENT_RBUTTONDOWN:
            for b in reversed(self.boxes):
                if b[1] <= x <= b[3] and b[2] <= y <= b[4]:
                    self.boxes.remove(b)
                    break

    def draw(self):
        v = self.img.copy()
        for c, x0, y0, x1, y1 in self.boxes:
            cv2.rectangle(v, (x0, y0), (x1, y1), COLORS[c], 2)
            cv2.putText(v, CLASSES[c], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLORS[c], 1)
        if self.drag and self.cur:
            cv2.rectangle(v, self.drag, self.cur, COLORS[self.cls], 1)
        done = sum(os.path.exists(label_path(f)) for f in self.files)
        cv2.putText(v, f'[{self.i + 1}/{len(self.files)}] labelled {done}  class: '
                    f'{self.cls + 1}={CLASSES[self.cls]}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return v


def main():
    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith(('.jpg', '.png')))
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
        elif kk in (ord('1'), ord('2')):
            L.cls = kk - ord('1')
        elif kk == ord('d') and L.boxes:
            L.boxes.pop()
        elif kk == ord('c'):
            L.boxes.clear()
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
