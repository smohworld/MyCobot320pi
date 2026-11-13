"""Shared settings for the tray YOLO tools (standalone, no ROS)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(ROOT, 'images')
LABELS = os.path.join(ROOT, 'labels')
DATASET = os.path.join(ROOT, 'dataset')
# class id = index (keep the order: the ROS nodes use the same list, yolo_remote.CLASSES)
#   0 tray      whole black tray body
#   1 pin       pin seen from above: the tan tape on the pin top
#   2 pin_front pin seen from the front (tray on the shelf): the white pin body in front of the
#               black face. Oblique views showing both the tape and the body: label both.
CLASSES = ['tray', 'pin', 'pin_front']
COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255)]
CAMERA = '/dev/video2'
WIDTH, HEIGHT = 640, 480


def open_camera(dev=CAMERA, w=WIDTH, h=HEIGHT):
    import cv2
    d = str(dev)
    if d.startswith('/dev/video') and d[10:].isdigit():
        d = d[10:]                 # some OpenCV builds can't open V4L2 by path, only by index
    cap = cv2.VideoCapture(int(d) if d.isdigit() else d, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {dev}')
    return cap
