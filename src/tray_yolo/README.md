# tray_yolo

Standalone YOLO tools (no ROS) for the tray (class 0 `tray`) and its grip pin (class 1 `pin`).

    pip install ultralytics opencv-python
    python3 capture.py     # SPACE saves frames from /dev/video2 -> images/  (100-200, vary pose/light/shadow)
    python3 label.py       # drag boxes, 1=tray 2=pin, n/p next/prev, u next unlabelled -> labels/
    python3 train.py       # split + data.yaml + train -> runs/tray/weights/best.pt
    python3 detect.py      # live detection, prints pixel centres

Label tips: `tray` = whole tray seen from above; `pin` = only the small protruding pin,
tight box. Include frames with the robot shadow and with the pin on different sides.
Pixel centres -> base-frame position needs the camera intrinsics + hand-eye (ray/desk-plane
intersection as in red_cube_hover); that step is not part of this folder yet.
