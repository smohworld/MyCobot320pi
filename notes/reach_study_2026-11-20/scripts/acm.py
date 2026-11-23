import rclpy
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents
rclpy.init(); n = rclpy.create_node('acm'); c = n.create_client(GetPlanningScene, 'get_planning_scene'); c.wait_for_service(5)
rq = GetPlanningScene.Request(); rq.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX | PlanningSceneComponents.WORLD_OBJECT_NAMES | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
f = c.call_async(rq); rclpy.spin_until_future_complete(n, f, timeout_sec=5); s = f.result().scene
acm = s.allowed_collision_matrix; names = acm.entry_names
print('objects', [o.id for o in s.world.collision_objects])
for o in s.world.collision_objects:
    if o.id == 'desk': print('desk', [(p.position.x, p.position.y, p.position.z) for p in o.primitive_poses], [list(p.dimensions) for p in o.primitives])
print('acm has desk', 'desk' in names, 'base' in names)
if 'desk' in names and 'base' in names:
    print('base-desk allowed', acm.entry_values[names.index('base')].enabled[names.index('desk')])
print('default entries', list(zip(acm.default_entry_names, acm.default_entry_values)))
