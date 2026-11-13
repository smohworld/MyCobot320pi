"""Client of tray_yolo/yolo_server.py (GPU YOLO in the yolo venv) with the OnnxYolo interface.

  y = make_yolo(model_onnx, CLASSES, imgsz, conf)   # server if it runs, else ONNX (CPU)
  y.detect(bgr)      # [(cls_name, conf, x0, y0, x1, y1)]
  y.detect_obb(bgr)  # [(cls_name, conf, cx, cy, w, h, angle_rad, corners 4x2)]
One connection per instance (not shared between threads): make one per thread.
"""
import json
import os
import socket
import struct

import numpy as np

from mycobot_tasks.yolo_onnx import OnnxYolo

SOCK = '/tmp/tray_yolo.sock'
CLASSES = ['tray', 'pin', 'pin_front']   # = tray_yolo/common.py CLASSES (class id = index)
HEAD = struct.Struct('<IIIf')


class RemoteYolo:
    def __init__(self, names, conf=0.4, sock=SOCK, timeout=2.0):
        self.names, self.conf = list(names), conf
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(sock)
        self.obb = True

    def _recv(self, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = self.s.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError('YOLO server closed the connection')
            got += k
        return bytes(buf)

    def detect_obb(self, bgr, raw=None):
        img = np.ascontiguousarray(bgr, np.uint8)
        h, w = img.shape[:2]
        ch = 1 if img.ndim == 2 else img.shape[2]
        self.s.sendall(HEAD.pack(h, w, ch, float(self.conf)) + img.tobytes())
        n, = struct.unpack('<I', self._recv(4))
        res = []
        for name, c, cx, cy, bw, bh, ang, pts in json.loads(self._recv(n)):
            res.append((name, c, cx, cy, bw, bh, ang, np.array(pts, np.float32).reshape(4, 2)))
        return res

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        res = []
        for name, c, cx, cy, bw, bh, ang, corners in self.detect_obb(bgr):
            x0, y0 = corners.min(axis=0)
            x1, y1 = corners.max(axis=0)
            res.append((name, c, float(np.clip(x0, 0, w)), float(np.clip(y0, 0, h)),
                        float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h))))
        return res


class AutoYolo:
    """GPU server when it answers, else (and from the first server error on) ONNX on the CPU."""

    def __init__(self, model, names, imgsz=640, conf=0.4, sock=SOCK, log=print):
        self.args, self.log, self.y, self.cpu = (model, names, imgsz, conf), log, None, None
        if sock and os.path.exists(sock):
            try:
                self.y = RemoteYolo(names, conf, sock)
                log(f'YOLO: GPU server {sock}')
            except OSError as e:
                log(f'YOLO: server {sock} not answering ({e}): ONNX on the CPU')
        else:
            log('YOLO: ONNX on the CPU (start tray_yolo/yolo_server.py for the GPU)')
        if self.y is None:
            self.y = self._cpu()

    def _cpu(self):
        if self.cpu is None:
            m, names, imgsz, conf = self.args
            self.cpu = OnnxYolo(os.path.expanduser(m), names, imgsz, conf)
        return self.cpu

    def _call(self, fn, *a):
        try:
            return getattr(self.y, fn)(*a)
        except (OSError, ConnectionError, ValueError) as e:
            if self.y is self.cpu:                  # already on the CPU: a real error
                raise
            self.log(f'YOLO server failed ({e}): ONNX on the CPU from now on')
            self.y = self._cpu()
            return getattr(self.y, fn)(*a)

    def detect(self, bgr):
        return self._call('detect', bgr)

    def detect_obb(self, bgr):
        return self._call('detect_obb', bgr)


make_yolo = AutoYolo
