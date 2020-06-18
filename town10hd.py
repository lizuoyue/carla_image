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

"""

import carla
import numpy as np

if __name__ == '__main__':

    num_frames = 15
    beg = int((1 - num_frames) / 2)
    end = beg + num_frames
    gap = 2.0
    stand_at = 3.0

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    world = client.load_world('Town10HD')

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
        bp.set_attribute('image_size_x', '2048')
        bp.set_attribute('image_size_y', '2048')
        bp.set_attribute('fov', '90')
        bp.set_attribute('sensor_tick', '0.0')

    cam_dict = {}
    cam = {k: [] for k in cam_bp}
    res = {k: [] for k in cam_bp}
    callback = lambda li, cam_id: lambda image: li.append((cam_id, image))

    spawn_points = world.get_map().get_spawn_points()
    for spa_idx, spawn_point in enumerate(spawn_points):
        if spa_idx != 5:
            continue

        loc, rot = spawn_point.location, spawn_point.rotation
        cx, cy, cz = loc.x, loc.y, loc.z
        yaw = rot.yaw / 180.0 * np.pi
        dir_x, dir_y = np.cos(yaw), np.sin(yaw)

        cam_locs = [carla.Location(cx+dir_x*i*gap, cy+dir_y*i*gap, stand_at) for i in range(beg, end)]
        cam_rots = [rot,
            carla.Rotation(rot.pitch+90, rot.yaw, rot.roll),
            carla.Rotation(rot.pitch-90, rot.yaw, rot.roll)
        ] + [carla.Rotation(rot.pitch, rot.yaw+r, rot.roll)for r in [90, 180, 270]]

        for loc_idx, cam_loc in enumerate(cam_locs):
            for dir_idx, cam_rot in zip('f,u,d,r,b,l'.split(','), cam_rots):
                cam_state = carla.Transform(cam_loc, cam_rot)
                for cam_type, bp in cam_bp.items():
                    cam_ins = world.spawn_actor(bp, cam_state)
                    cam[cam_type].append(cam_ins)
                    cam_dict[cam_ins.id] = (spa_idx, loc_idx, dir_idx)

    print('Totally %dx3 cameras.' % len(cam['rgb']))

    for cam_type in cam_bp:
        for cam_ins in cam[cam_type]:
            cam_ins.listen(callback(res[cam_type], cam_ins.id))

    num = world.tick()
    print(f'Getting images from frame {num} ...')
    while sum([len(res[cam_type]) < len(cam[cam_type]) for cam_type in cam_bp]) > 0:
        continue
    print(f'Done!')

    for cam_type in cam_bp:
        for cam_id, image in res[cam_type]:
            print(cam_type, cam_id)
            image.save_to_disk(f'pinhole_{cam_type}/%02d_%02d_%s.png' % cam_dict[cam_id])

    for cam_type in cam_bp:
        for cam_ins in cam[cam_type]:
            cam_ins.destroy()
