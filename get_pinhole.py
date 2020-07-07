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

    towns = ['Town%02d' % i for i in [1,2,3,4,5,6,7,10]]
    towns[-1] += 'HD'
    gaps = [2.0, 1.0, 4.0, 8.0, 6.0, 6.0, 2.0, 2.0]
    floating_at = 2.0
    # gaps = [2.0] * len(towns)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    for town, gap in zip(towns[2:3], gaps[2:3]):

        d = np.load(f'waypoints/{town}.npz')
        world = client.load_world(town)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1e-3
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
        callback = lambda li, cam_id: lambda image: li.append((cam_id, image))

        world_map = world.get_map()
        waypoints = d['waypoints']
        for pid, waypoint in enumerate(waypoints):

            if pid < 1850:
                continue

            cam_dict = {}
            cam, res = [], []

            cx, cy, cz, pitch, yaw, roll = waypoint

            cam_loc = carla.Location(cx, cy, cz + floating_at)
            cam_rots = [carla.Rotation(pitch+r, yaw, roll) for r in [0, 90, -90]]
            cam_rots += [carla.Rotation(pitch, yaw+r, roll)for r in [90, 180, 270]]

            for dir_idx, cam_rot in zip('f,u,d,r,b,l'.split(','), cam_rots):
                cam_state = carla.Transform(cam_loc, cam_rot)
                for cam_type, bp in cam_bp.items():
                    cam_ins = world.spawn_actor(bp, cam_state)
                    cam.append(cam_ins)
                    cam_dict[cam_ins.id] = (pid, dir_idx, cam_type)

            assert(len(cam) == 18)

            for cam_ins in cam:
                cam_ins.listen(callback(res, cam_ins.id))

            num = world.tick()
            print(f'{town}: Getting images for waypoint {pid+1}/{len(waypoints)} ...', end='')
            while len(res) < 18:
                continue
            print(f' Done!')

            for cam_id, image in res:
                image.save_to_disk(f'pinhole/{town}/%05d_%s_%s.png' % cam_dict[cam_id])

            for cam_ins in cam:
                cam_ins.destroy()
        
        quit()
