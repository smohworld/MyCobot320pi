#!/usr/bin/env python3
"""Capture ChArUco calibration images from a ROS 2 image topic (e.g. TurtleBot3 usb_cam over Wi-Fi).

Run on the PC (the robot only publishes the camera):
  source ~/final_project_ws/install/setup.bash
  python3 ~/final_project_ws/calibration/tb3_capture.py \
      --topic /image_raw/compressed --out ~/tb3_camera_calib/images

Then calibrate from the saved images:
  ros2 run mycobot_calibration intrinsics_calib --images ~/tb3_camera_calib/images \
      --board ~/final_project_ws/calibration/charuco_board.yaml --out ~/tb3_camera_calib

keys:  SPACE save   a auto-save on/off   u undo last   q quit
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from mycobot_calibration.charuco import CharucoTarget


class Grabber(Node):

    def __init__(self, topic):
        super().__init__('tb3_capture')
        self.frame, self.lock = None, threading.Lock()
        typ = CompressedImage if topic.endswith('compressed') else Image
        self.create_subscription(typ, topic, self._cb, qos_profile_sensor_data)
        self.get_logger().info(f'subscribed {topic} ({typ.__name__})')

    def _cb(self, msg):
        if isinstance(msg, CompressedImage):
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            ch = 1 if msg.encoding in ('mono8',) else 3
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
            img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif ch == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif msg.encoding != 'bgr8':
                self.get_logger().warn(f'unsupported encoding {msg.encoding}',
                                       throttle_duration_sec=5.0)
                return
        with self.lock:
            self.frame = img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topic', default='/image_raw/compressed')
    p.add_argument('--out', default='~/tb3_camera_calib/images')
    p.add_argument('--board', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'charuco_board.yaml'))
    p.add_argument('--min-corners', type=int, default=12)
    p.add_argument('--min-sharpness', type=float, default=60.0)
    p.add_argument('--min-move-px', type=float, default=40.0)
    p.add_argument('--interval', type=float, default=1.5, help='auto-save min seconds')
    a, _ = p.parse_known_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    target = CharucoTarget(os.path.expanduser(a.board))
    print('board:', target.describe())

    rclpy.init()
    node = Grabber(a.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    saved = sorted(f for f in os.listdir(out) if f.endswith('.png'))
    auto, last_t, last_c, size = False, 0.0, None, None
    print(__doc__.split('\n\n')[-1])
    while rclpy.ok():
        with node.lock:
            frame = None if node.frame is None else node.frame.copy()
        if frame is None:
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        if size is None:
            size = frame.shape[1::-1]
            print(f'image size {size[0]}x{size[1]}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = target.detector.detectBoard(gray)
        n = 0 if ids is None else len(ids)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        vis = frame.copy()
        if n:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 255, 0))
        c2 = None if n == 0 else corners.reshape(-1, 2)
        moved = None
        if c2 is not None and last_c is not None and len(c2) == len(last_c):
            moved = float(np.median(np.linalg.norm(c2 - last_c, axis=1)))
        if n < a.min_corners:
            reason = f'too few corners ({n})'
        elif sharp < a.min_sharpness:
            reason = f'blurred ({sharp:.0f})'
        elif time.monotonic() - last_t < a.interval:
            reason = 'cooldown'
        elif moved is not None and moved < a.min_move_px:
            reason = f'move to a new spot ({moved:.0f}px)'
        else:
            reason = None
        cv2.putText(vis, f'saved {len(saved)}  corners {n}/{target.n_corners}  '
                    f'auto {"ON" if auto else "off"}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if auto:
            cv2.putText(vis, 'AUTO: ' + (reason or 'saving'), (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if reason is None else (0, 0, 255), 2)
        cv2.imshow('tb3_capture', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or (auto and reason is None):
            if n < a.min_corners or sharp < a.min_sharpness:
                print(f'not saved: {reason}')
            else:
                name = f'{len(saved):03d}.png'
                while os.path.exists(os.path.join(out, name)):
                    name = f'{int(name[:3]) + 1:03d}.png'
                cv2.imwrite(os.path.join(out, name), frame)
                saved.append(name)
                last_t, last_c = time.monotonic(), c2
                print(f'saved {name}: {n} corners')
        elif key == ord('a'):
            auto = not auto
        elif key == ord('u') and saved:
            os.remove(os.path.join(out, saved.pop()))
            print(f'removed last, {len(saved)} left')
        elif key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
