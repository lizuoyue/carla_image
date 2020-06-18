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
    rgb_dict = {}

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    # print(client.get_available_maps())

    world = client.get_world()
    world = client.load_world('Town10HD')

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0
    world.apply_settings(settings)

    spawn_points = world.get_map().get_spawn_points()
    rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_cam_bp.set_attribute('image_size_x', '2048')
    rgb_cam_bp.set_attribute('image_size_y', '2048')
    rgb_cam_bp.set_attribute('fov', '90')
    rgb_cam_bp.set_attribute('sensor_tick', '0.0')

    images = []
    func = lambda cam_id: lambda image: images.append((cam_id, image))

    rgb_cams = []
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
                rgb_cam = world.spawn_actor(rgb_cam_bp, cam_state)
                rgb_cams.append(rgb_cam)
                rgb_dict[rgb_cam.id] = (spa_idx, loc_idx, dir_idx)

    print(f'Totally {len(rgb_cams)} cameras.')

    for rgb_cam in rgb_cams:
        rgb_cam.listen(func(rgb_cam.id))

    num = world.tick()
    print(f'Getting images from frame {num} ...')
    while len(images) < len(rgb_cams):
        continue
    print(f'Done!')

    for cam_id, image in images:
        image.save_to_disk('pinhole/%02d_%02d_%s.png' % rgb_dict[cam_id])

    for rgb_cam in rgb_cams:
        rgb_cam.destroy()
