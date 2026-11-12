"""Shared settings for the tray YOLO tools (standalone, no ROS)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(ROOT, 'images')
LABELS = os.path.join(ROOT, 'labels')
DATASET = os.path.join(ROOT, 'dataset')
CLASSES = ['tray', 'pin']          # class id = index (0 tray: whole tray, 1 pin: grip pin)
COLORS = [(0, 255, 0), (0, 165, 255)]
CAMERA = '/dev/video2'
WIDTH, HEIGHT = 640, 480


def open_camera(dev=CAMERA, w=WIDTH, h=HEIGHT):
    import cv2
    cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {dev}')
    return cap
