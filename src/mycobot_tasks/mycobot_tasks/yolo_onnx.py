"""YOLOv8 (ultralytics) detect model exported to ONNX, run with cv2.dnn: no torch needed.

  export (yolo venv):  yolo export model=best.pt format=onnx imgsz=640 opset=12
  det = OnnxYolo('best.onnx', ['tray', 'pin']).detect(bgr)   # [(cls_name, conf, x0, y0, x1, y1)]
"""
import cv2
import numpy as np


class OnnxYolo:
    def __init__(self, path, names, imgsz=640, conf=0.4, iou=0.5):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.names, self.imgsz, self.conf, self.iou = list(names), imgsz, conf, iou

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        img = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        img[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz),
                                                swapRB=True))
        out = self.net.forward()[0].T                    # (anchors, 4 + nc)
        scores = out[:, 4:]
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
