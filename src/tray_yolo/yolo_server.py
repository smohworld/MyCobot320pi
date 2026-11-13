"""GPU YOLO server (yolo venv, torch + CUDA): the ROS nodes (system Python, no torch) send
camera frames over a unix socket and get the detections back.

  ~/venvs/yolo/bin/python yolo_server.py                 # runs/tray_auto/weights/best.pt, cuda:0
  ~/venvs/yolo/bin/python yolo_server.py --model path/best.pt --sock /tmp/tray_yolo.sock

Protocol (one request after the other per connection):
  request : '<IIIf' height, width, channels (3, BGR uint8), conf  + the raw pixel bytes
  response: '<I' length + JSON list of [name, conf, cx, cy, w, h, angle_rad, [8 corner values]]
            (OBB model; a detect model gives angle 0 and its axis-aligned corners)
Client: mycobot_tasks/yolo_remote.py (RemoteYolo, same results as OnnxYolo).
"""
import argparse
import json
import os
import signal
import socket
import struct
import threading
import time

import numpy as np

from common import ROOT

HEAD = struct.Struct('<IIIf')


def recv_exact(conn, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError('closed')
        got += k
    return bytes(buf)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', default=os.path.join(ROOT, 'runs/tray_auto/weights/best.pt'))
    a.add_argument('--sock', default='/tmp/tray_yolo.sock')
    a.add_argument('--imgsz', type=int, default=640)
    a.add_argument('--device', default='0', help="'0' = first GPU, 'cpu' to compare")
    a.add_argument('--half', action='store_true', help='FP16 (a bit faster on the GPU)')
    args = a.parse_args()

    import torch
    from ultralytics import YOLO
    model = YOLO(args.model)
    names = model.names
    kw = dict(imgsz=args.imgsz, device=args.device, verbose=False)
    if args.half:
        kw['half'] = True
    dummy = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):                       # warm-up (CUDA init, cudnn autotune)
        model.predict(dummy, **kw)
    dev = torch.cuda.get_device_name(0) if args.device != 'cpu' else 'cpu'
    lock = threading.Lock()                  # one GPU model, several clients
    stats = {'n': 0, 't': 0.0}

    def detect(img, conf):
        with lock:
            t0 = time.perf_counter()
            r = model.predict(img, conf=conf, **kw)[0]
            stats['n'] += 1
            stats['t'] += time.perf_counter() - t0
        out = []
        if r.obb is not None:
            for xywhr, pts, c, p in zip(r.obb.xywhr.cpu().numpy(),
                                        r.obb.xyxyxyxy.cpu().numpy(),
                                        r.obb.cls.cpu().numpy(), r.obb.conf.cpu().numpy()):
                out.append([names[int(c)], float(p), *map(float, xywhr),
                            [float(v) for v in pts.ravel()]])
        elif r.boxes is not None:
            for (x0, y0, x1, y1), c, p in zip(r.boxes.xyxy.cpu().numpy(),
                                              r.boxes.cls.cpu().numpy(),
                                              r.boxes.conf.cpu().numpy()):
                out.append([names[int(c)], float(p), float((x0 + x1) / 2), float((y0 + y1) / 2),
                            float(x1 - x0), float(y1 - y0), 0.0,
                            [float(v) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]])
        return out

    def serve(conn):
        with conn:
            try:
                while True:
                    h, w, ch, conf = HEAD.unpack(recv_exact(conn, HEAD.size))
                    img = np.frombuffer(recv_exact(conn, h * w * ch), np.uint8).reshape(h, w, ch)
                    body = json.dumps(detect(img, conf)).encode()
                    conn.sendall(struct.pack('<I', len(body)) + body)
            except (ConnectionError, OSError):
                pass

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    srv.listen(8)
    print(f'YOLO server: {args.model} on {dev}, socket {args.sock}  (Ctrl+C stops)', flush=True)

    def report():
        while True:
            time.sleep(10)
            if stats['n']:
                print(f"  {stats['n']} frames, {stats['t'] / stats['n'] * 1000:.1f} ms each",
                      flush=True)
                stats['n'], stats['t'] = 0, 0.0
    threading.Thread(target=report, daemon=True).start()
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=serve, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if os.path.exists(args.sock):
            os.unlink(args.sock)       # no stale socket: the clients then use the CPU at once


if __name__ == '__main__':
    main()
