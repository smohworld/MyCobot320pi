"""Pick the tray by its handle from above (keeping it level) and put it back where it was.

The handle is a bar between two ArUco markers on the tray top (calibration/
tray_aruco_4x4_a4.pdf, DICT_4X4_50, 22.5 mm, tray t = ID 10+2t left / 11+2t right, both
printed top edges toward the same side). Grasp point = midpoint of the two marker centres,
the handle top = the marker plane. The fingers close ACROSS the bar (perpendicular to the
L -> R line, --grip-yaw-offset 90) and go --grasp-depth below the handle top.

Code: mycobot_tasks/handle_pick/ (base, detect, scan, bot, plan, relay, app, window, cli).
Window, keys, step mode, pick plan check, level lift, put back:
  detect   --frames camera frames, both markers in every used frame -> median midpoint,
           L -> R yaw, handle top z (refused if the marker plane tilts > --max-tilt)
  tracking (detect, and the start of p / v) the camera moves to look straight down at the
           handle when only one marker is seen (estimate: that marker + the 54 mm spacing),
           the markers are small (< --min-marker-px), off the image centre or seen at an
           angle (> --max-view), and measures again (at most --recenter moves; --no-track
           = off). Nothing seen: it looks --scan-step around the search point first.
  heights  from the measured handle top h over the desk: approach / lift = h +
           --approach-above, above (start) = h + --servo-above, grasp = h - --grasp-depth
  pick     planned at once before anything moves (above, approach, grasp, lift and the put
           back), then those joint values are run. Grasped or not is decided by
           the finger gap after closing (>= --min-grip-gap), not by the driver's 'stalled'
  put back the joint values planned with the pick: same xy, yaw, height + --place-drop

Needs move_group, the driver and the camera (handeye.launch.py) and the arm in a look-down
pose over the tray (s = search pose at --search-x/-y, or goto_lookdown).

Planning scene: at start the place_scene objects (desk, floor; TurtleBot / shelf with
--scene-args "...", same options as ros2 run mycobot_tasks place_scene) replace the
1.5 x 1.5 m 'calibration_table' (it blocks everything below the desk). --no-scene = old table.
Tray below the desk: --table-z = z of the surface it stands on (base frame, floor -0.306).

  ros2 run mycobot_tasks tray_handle_pick                   # window, plans only (dry run)
  ros2 run mycobot_tasks tray_handle_pick --go --real       # moves the REAL robot

Pick and place (m, or --do pickplace): detect -> plan pick + carry + place + home at once ->
pick -> straight up (high enough for the place surface --place-z, the desk top) -> straight
across to --place-at (default 0,-0.18: on the desk, 18 cm toward -y from the base; or
--place-offset dx,dy from the pick spot), same tool yaw -> down -> open -> up -> home.

Check in simulation first: every detection (d / p / m) on the real robot is saved to
~/.ros/tray_handle_last.yaml; then (real launch stopped, or another ROS_DOMAIN_ID)
  ros2 launch mycobot_calibration handeye.launch.py sim:=true
  ros2 run mycobot_tasks tray_handle_pick --replay --go --table-z <same>
moves the simulated arm to the detection pose, and m / p run the same plan in RViz.

Regions (ros2 run mycobot_tasks regions, boxes dragged in RViz, ~/.ros/regions.yaml):
  shelf    (the desk top for now) d / p / m see nothing -> the camera looks down over the
           shelf region, view by view (--scan-height over its z, --scan-overlap), until a
           marker shows up; f = that scan from the start. Without --table-z the shelf region
           z is the tray surface.
  bot      t = find the TurtleBot marker (ID --bot-id 0, DICT_4X4_50, 70 mm) here or over the
           bot region, look straight down at it, measure; the TurtleBot + box go into the
           planning scene there (marker = box floor centre, box +x = marker +x).
  m with --place-at bot / --place-z auto / --place-yaw bot: the TurtleBot is found first,
  then place spot = marker + --bot-place-offset, surface z = marker z - --bot-marker-dz,
  tray L -> R = marker +x + --bot-yaw-offset:
  ros2 run mycobot_tasks tray_handle_pick --go --real --place-at bot --place-z auto

Relay over the desk (handle_pick/relay.py), shelf D11 <-> TurtleBot:
  n (--do d11bot) in two parts: camera over D11 (layout pose --relay-d11), tray measured,
    D11 with the reversed arm -> desk spot (--relay-spots, tilted <= --relay-tilt carry)
    planned (no plan / not seen: arm back to where n started, error) and run; then the
    TurtleBot marker measured, empty arm switch -> regrasp -> into the box planned and run
  j (--do botd11) tray on the TurtleBot measured -> the same backwards onto D11 (empty)
  sim: ros2 run mycobot_tasks tray_handle_pick --replay ~/.ros/layout_tray.yaml \
         --bot-save ~/.ros/layout_bot.yaml --regions ~/.ros/layout_regions.yaml \
         --scene-args=--layout --do d11bot [--go]   (j: --replay ~/.ros/layout_tray_bot.yaml)

Window: p pick (any height)   d detect (any height)   m pick + place
        k TurtleBot -> shelf   n D11 -> TurtleBot   j TurtleBot -> D11
        s search pose   b put the tray back where p picked it
        Enter next step   a auto / step mode   x / SPACE stop   q quit
  d / p: the tray is looked for from here (then the last position, the shelf region, the
  TurtleBot region); the surface it stands on = handle top - --tray-height, so no
  --table-z is needed. In the TurtleBot box: the TurtleBot goes into the scene from the tray.
"""
from mycobot_tasks.handle_pick.cli import main

__all__ = ['main']

if __name__ == '__main__':
    main()
