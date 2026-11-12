"""Save frames from the ROS image topic for labelling.

Run in the ROS terminal (source /opt/ros/... and the workspace), NOT in the yolo venv:
  python3 capture.py [--topic /camera/image_raw]
  python3 capture.py --dev /dev/video2        # no ROS: open the camera directly
SPACE / c: save frame to images/   q / ESC: quit
Tip: vary tray position, yaw, lighting and the robot shadow; 100-200 frames.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import IMAGES


def decode(msg):
    """sensor_msgs/Image -> BGR (no cv_bridge)."""
    ch = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, '8UC1': 1, 'bgra8': 4, 'rgba8': 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    a = a.reshape(msg.height, msg.width, ch) if ch > 1 else a
    code = {'rgb8': cv2.COLOR_RGB2BGR, 'mono8': cv2.COLOR_GRAY2BGR, '8UC1': cv2.COLOR_GRAY2BGR,
            'bgra8': cv2.COLOR_BGRA2BGR, 'rgba8': cv2.COLOR_RGBA2BGR}.get(msg.encoding)
    return cv2.cvtColor(a, code) if code is not None else a.copy()


def ros_frames(topic):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    rclpy.init()
    node = Node('tray_capture')
    box = {}
    node.create_subscription(Image, topic, lambda m: box.update(msg=m), 2)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            m = box.pop('msg', None)
            yield None if m is None else decode(m)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cam_frames(dev):
    from common import open_camera
    cap = open_camera(dev)
    while True:
        ok, f = cap.read()
        yield f if ok else None


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--topic', default='/camera/image_raw')
    a.add_argument('--dev', default='', help='open this camera directly instead of the topic')
    args = a.parse_args()
    os.makedirs(IMAGES, exist_ok=True)
    n = len([f for f in os.listdir(IMAGES) if f.endswith('.jpg')])
    frame = None
    for f in cam_frames(args.dev) if args.dev else ros_frames(args.topic):
        if f is not None:
            frame = f
        if frame is None:
            cv2.waitKey(1)
            continue
        vis = frame.copy()
        cv2.putText(vis, f'SPACE/c save  q quit   saved: {n}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('capture', vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('c'), ord(' ')):
            fn = os.path.join(IMAGES, f'img_{time.strftime("%Y%m%d_%H%M%S")}_{n:04d}.jpg')
            cv2.imwrite(fn, frame)
            n += 1
            print('saved', fn, flush=True)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
