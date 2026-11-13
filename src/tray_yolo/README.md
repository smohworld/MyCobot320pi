# tray_yolo

Standalone YOLO tools (no ROS) for the tray (class 0 `tray`) and its grip pin (class 1 `pin`).

    pip install ultralytics opencv-python
    python3 capture.py     # SPACE saves frames from /dev/video2 -> images/  (100-200, vary pose/light/shadow)
    python3 label.py       # OBB: 3 clicks per box (2 corners of one edge, then drag out the width), 1=tray 2=pin 3=pin_front, n/p next/prev, u next unlabelled -> labels/ (class x1 y1 .. x4 y4)
    ~/venvs/yolo/bin/python autolabel.py --compare   # after a first model: YOLO box -> tight black-body / tan-tape box -> labels_auto/
    python3 label.py --labels labels_auto --review   # fix the flagged ones, then cp labels_auto/*.txt labels/
    python3 train.py       # split + data.yaml + train (yolov8n-obb) -> runs/tray/weights/best.pt
    python3 detect.py      # live detection, prints pixel centres
    ~/venvs/yolo/bin/python yolo_server.py   # GPU YOLO for the ROS nodes (tray_detect / tray_pick): ~8 ms vs ~100 ms
                                             # on the CPU; they use /tmp/tray_yolo.sock and fall back to the ONNX model without it

Classes (common.py, same order as mycobot_tasks yolo_remote.CLASSES): `tray` = whole black
tray body; `pin` = pin seen from above, the tan tape on its top, tight box; `pin_front` = pin
seen from the front (tray on the shelf), the white pin body in front of the black face. An
oblique view showing both tape and body gets both boxes. Front view frames:
`python3 capture.py --prefix front` (front_*.jpg).
Label tips: Include frames with the robot shadow and with the pin on different sides.
Pixel centres -> base-frame position needs the camera intrinsics + hand-eye (ray/desk-plane
intersection as in red_cube_hover); that step is not part of this folder yet.
