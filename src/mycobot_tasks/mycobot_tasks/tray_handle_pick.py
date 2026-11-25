"""Tray relay over the desk: shelf D11 <-> TurtleBot box, tray grasped by its handle from above.

The handle is a bar between two ArUco markers on the tray top (calibration/
tray_aruco_4x4_a4.pdf, DICT_4X4_50, 22.5 mm, tray t = ID 10+2t left / 11+2t right). Grasp
point = midpoint of the two marker centres, handle top = the marker plane; the fingers close
ACROSS the bar (--grip-yaw-offset 90) and go --grasp-depth below the handle top.

Code: mycobot_tasks/handle_pick/ (base, detect, scan, bot, relay, app, window, cli).
Defaults: share/mycobot_tasks/config/tray_handle_pick.yaml, grouped by purpose (tray, markers,
detection, search, turtlebot, gripper, scene, moveit, relay_plan, relay_safety, relay_speed,
relay_end, files). Command-line options (key with - for _) win; --config FILE changes some.

Keys (window):
  n  D11 -> desk -> TurtleBot: camera over D11 (--relay-d11), tray measured, D11 -> desk
     spot planned and run; then the TurtleBot marker measured, desk -> box planned and run
  j  TurtleBot -> desk -> D11: tray in the box measured, the same backwards onto D11
  e  desk -> TurtleBot: tray measured at the desk spot (--relay-desk / --relay-spots)
  r  TurtleBot -> desk: tray in the box measured, put on the desk (--relay-desk, else auto)
  d  detect (any height)     s  search pose
  Enter next step   a auto / step mode   x / SPACE stop   q quit
  No plan / tray not seen: nothing moves (or the empty arm goes home), ERROR. Every run
  ends at home with the gripper closed (--relay-end, --relay-home).

  ros2 run mycobot_tasks tray_handle_pick                   # window, plans only (dry run)
  ros2 run mycobot_tasks tray_handle_pick --go --real       # moves the REAL robot
  simulation (RViz layout, no camera):
  ros2 run mycobot_tasks tray_handle_pick --layout-poses --scene-args=--layout --go
"""
from mycobot_tasks.handle_pick.cli import main

__all__ = ['main']

if __name__ == '__main__':
    main()
