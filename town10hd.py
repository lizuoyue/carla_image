"""
cd /opt/carla-simulator/bin
./CarlaUE4.sh

dir(image)
[
    '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
    '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__',
    '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
    '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
    'convert', 'fov', 'frame', 'frame_number', 'height', 'raw_data',
    'save_to_disk', 'timestamp', 'transform', 'width'
]

client.get_available_maps()
[
    '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town03',
    '/Game/Carla/Maps/Town01', '/Game/Carla/Maps/Town07',
    '/Game/Carla/Maps/Town02', '/Game/Carla/Maps/Town10HD',
    '/Game/Carla/Maps/Town06', '/Game/Carla/Maps/Town05'
]

waypoint
[
    '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
    '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
    '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
    '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'get_junction', 'get_landmarks',
    'get_landmarks_of_type', 'get_left_lane', 'get_right_lane', 'id', 'is_intersection', 'is_junction',
    'junction_id', 'lane_change', 'lane_id', 'lane_type', 'lane_width', 'left_lane_marking', 'next',
    'next_until_lane_end', 'previous', 'previous_until_lane_start', 'right_lane_marking', 'road_id',
    's', 'section_id', 'transform'
]

"""

import carla
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

if __name__ == '__main__':

    gap = 2.0
    stand_at = 3.0
    town = 'Town10HD'

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    world = client.load_world(town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0
    world.apply_settings(settings)

    cam_bp = {
        'rgb': world.get_blueprint_library().find('sensor.camera.rgb'),
        'dep': world.get_blueprint_library().find('sensor.camera.depth'),
        'sem': world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
    }
    for _, bp in cam_bp.items():
        bp.set_attribute('image_size_x', '512')
        bp.set_attribute('image_size_y', '512')
        bp.set_attribute('fov', '90')
        bp.set_attribute('sensor_tick', '0.0')

    world_map = world.get_map()
    # waypoints = world_map.generate_waypoints(gap)
    waypoints = world.get_map().get_spawn_points()
    # waypoints_np = np.array([
    #     (waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in waypoints
    # ])
    # kdtree = KDTree(waypoints_np, leaf_size=2)

    # nbs = []
    # for waypoint in waypoints:
    #     next_points = waypoint.next(gap)
    #     next_points_np = np.array([
    #         (waypoint.transform.location.x, waypoint.transform.location.y) for waypoint in next_points
    #     ])
    #     dist, idx = kdtree.query(next_points_np, k=1)
    #     nbs.append(list(idx.flatten()))

    if False:
        plt.figure(figsize=(8,8),dpi=300)
        for waypoint_np, nb in zip(waypoints_np, nbs):
            for idx in nb:
                next_point = waypoints_np[idx]
                plt.plot([waypoint_np[0], next_point[0]], [waypoint_np[1], next_point[1]],'-',color='C0',linewidth=1)
        plt.plot(waypoints_np[:,0], waypoints_np[:,1],'.',color='C1',markersize=1)
        plt.savefig('graph.png')
        quit()
    
    # np.savez_compressed(f'{town}.npz', waypoints=waypoints_np, nbs=nbs)
    callback = lambda li, cam_id: lambda image: li.append((cam_id, image))
    for point_idx, waypoint in enumerate(waypoints):

        cam_dict = {}
        cam, res = [], []

        # loc, rot = waypoint.transform.location, waypoint.transform.rotation
        loc, rot = waypoint.location, waypoint.rotation
        cx, cy, cz = loc.x, loc.y, loc.z

        cam_rots = [rot,
            carla.Rotation(rot.pitch+90, rot.yaw, rot.roll),
            carla.Rotation(rot.pitch-90, rot.yaw, rot.roll)
        ] + [carla.Rotation(rot.pitch, rot.yaw+r, rot.roll)for r in [90, 180, 270]]

        cam_loc = loc
        for dir_idx, cam_rot in zip('f,u,d,r,b,l'.split(','), cam_rots):
            cam_state = carla.Transform(cam_loc, cam_rot)
            for cam_type, bp in cam_bp.items():
                cam_ins = world.spawn_actor(bp, cam_state)
                cam.append(cam_ins)
                cam_dict[cam_ins.id] = (point_idx, dir_idx, cam_type)

        assert(len(cam) == 18)

        for cam_ins in cam:
            cam_ins.listen(callback(res, cam_ins.id))

        num = world.tick()
        print(f'Getting images for waypoint {point_idx+1}/{len(waypoints)} ...', end='')
        while len(res) < 18:
            continue
        print(f' Done!')

        for cam_id, image in res:
            image.save_to_disk('pinhole/%05d_%s_%s.png' % cam_dict[cam_id])

        for cam_ins in cam:
            cam_ins.destroy()
        quit()
