"""YOLOv8 (ultralytics) detect or OBB model exported to ONNX, run with cv2.dnn: no torch needed.

  export (yolo venv):  yolo export model=best.pt format=onnx imgsz=640 opset=12
  y = OnnxYolo('best.onnx', ['tray', 'pin'])
  y.detect(bgr)      # [(cls_name, conf, x0, y0, x1, y1)]  axis-aligned (for an OBB model: the
                     #  bounding rectangle of the rotated box)
  y.detect_obb(bgr)  # [(cls_name, conf, cx, cy, w, h, angle_rad, corners 4x2)]  OBB model only
The model type is taken from the output shape: 4 + classes channels = detect, one more = OBB
(the last channel is the angle in radians). `names` may be longer than the model's class list
(e.g. ['tray', 'pin', 'pin_front'] with an older 2-class model): the output is read as
an OBB model with the first width - 5 names whenever that fits (OBB is preferred).
"""
import cv2
import numpy as np


class OnnxYolo:
    def __init__(self, path, names, imgsz=640, conf=0.4, iou=0.5):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.names, self.imgsz, self.conf, self.iou = list(names), imgsz, conf, iou
        self.obb = None                                  # set on the first inference
        self.nc = len(self.names)

    def _raw(self, bgr):
        h, w = bgr.shape[:2]
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        img = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        img[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz),
                                                swapRB=True))
        out = self.net.forward()[0].T                    # (anchors, 4 + nc [+ 1 angle])
        if self.obb is None:
            wd, n = out.shape[1], len(self.names)
            # OBB first: an older 2-class OBB model (7 channels) has the width of a 3-class
            # detect model; all tray models are OBB (pass names of the right length for a
            # detect model)
            if 1 <= wd - 5 <= n:
                self.obb, self.nc = True, wd - 5
            elif wd == 4 + n:
                self.obb, self.nc = False, n
            else:
                raise ValueError(f'ONNX output has {wd} channels, does not fit {n} class '
                                 f'names {self.names}')
        return out, s, px, py, w, h

    def decode_obb(self, out):
        """Raw (anchors, 4 + nc + 1) -> [(cls, conf, cx, cy, w, h, angle_rad)] in the letterboxed
        image, after the confidence filter and rotated NMS per class."""
        scores = out[:, 4:4 + self.nc]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= self.conf
        out, cls, conf = out[keep], cls[keep], conf[keep]
        res = []
        for c in np.unique(cls):
            m = cls == c
            b, sc = out[m], conf[m]
            rects = [((float(x), float(y)), (float(bw), float(bh)), float(np.degrees(a)))
                     for x, y, bw, bh, a in zip(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, -1])]
            for i in np.array(cv2.dnn.NMSBoxesRotated(rects, sc.tolist(), self.conf,
                                                      self.iou)).reshape(-1):
                res.append((int(c), float(sc[i]), *b[i, :4].tolist(), float(b[i, -1])))
        return res

    def detect_obb(self, bgr, raw=None):
        out, s, px, py, w, h = raw if raw is not None else self._raw(bgr)
        if not self.obb:
            raise ValueError('the ONNX model is not an OBB model')
        res = []
        for c, sc, x, y, bw, bh, a in self.decode_obb(out):
            cx, cy, bw, bh = (x - px) / s, (y - py) / s, bw / s, bh / s
            corners = cv2.boxPoints(((cx, cy), (bw, bh), float(np.degrees(a))))
            res.append((self.names[c], sc, cx, cy, bw, bh, a, corners))
        return res

    def detect(self, bgr):
        raw = self._raw(bgr)
        out, s, px, py, w, h = raw
        if self.obb:
            res = []
            for name, sc, cx, cy, bw, bh, a, corners in self.detect_obb(bgr, raw):
                x0, y0 = corners.min(axis=0)
                x1, y1 = corners.max(axis=0)
                res.append((name, sc, float(np.clip(x0, 0, w)), float(np.clip(y0, 0, h)),
                            float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h))))
            return res
        scores = out[:, 4:4 + self.nc]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= self.conf
        out, cls, conf = out[keep], cls[keep], conf[keep]
        res = []
        for c in np.unique(cls):
            m = cls == c
            b, sc = out[m, :4], conf[m]
            boxes = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                     b[:, 2], b[:, 3]])
            for i in np.array(cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), self.conf,
                                               self.iou)).reshape(-1):
                x, y, bw, bh = boxes[i]
                x0, y0 = (x - px) / s, (y - py) / s
                x1, y1 = (x + bw - px) / s, (y + bh - py) / s
                res.append((self.names[int(c)], float(sc[i]), float(np.clip(x0, 0, w)),
                            float(np.clip(y0, 0, h)), float(np.clip(x1, 0, w)),
                            float(np.clip(y1, 0, h))))
        return res
