import math, numpy as np
import chainlib
from chainlib import chain, seg
from scanlib import L, orig, ap, _wait
from route import ik_any
from mycobot_tasks import place_scene as ps
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene
def apply_all(m=0.01):
    a = L.args(); a.shelf_margin = m; o, _ = ps.build(a)
    req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True)); req.scene.world.collision_objects = ps.collision_objects(o, 'base')
    r = _wait(ap.call_async(req), 20.0); assert r and r.success
    return o
def xy_overlap(o, a, b):
    flat = lambda b: ([b[0][0], b[0][1], 0.0], [b[1][0], b[1][1], 1.0], b[2])
    return any(ps._overlap(flat(p), flat(q)) for p in o[a] for q in o[b])
