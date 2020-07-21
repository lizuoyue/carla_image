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

import os
import carla
import numpy as np
import pinhole2panorama

if __name__ == '__main__':

    towns = ['Town%02d' % i for i in [1,2,3,4,5,6,7,10]]
    towns[-1] += 'HD'
    gaps = [2.0, 1.0, 4.0, 8.0, 6.0, 6.0, 2.0, 2.0]
    floating_at = 2.0
    weathers = {
        # 'ClearNoon'   : carla.WeatherParameters.ClearNoon,
        'CloudyNoon'  : carla.WeatherParameters.CloudyNoon,
        'ClearSunset' : carla.WeatherParameters.ClearSunset,
        'CloudySunset': carla.WeatherParameters.CloudySunset,
    }
    # for weather, weather_param in weathers.items():
    #     if weather.startswith('Cloudy'):
    #         print(weather_param.cloudiness)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    for weather in weathers:

        for town, gap in list(zip(towns, gaps))[2:3]:

            try:
                # Create folder for saving the images
                os.system(f'mkdir -p panorama/{town}_{weather}')
                
                # Set world
                world = client.load_world(town)
                world.set_weather(weathers[weather])
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1e-6
                world.apply_settings(settings)

                # Set camera attribution
                cam_bp = {
                    'rgb': world.get_blueprint_library().find('sensor.camera.rgb'),
                    'dep': world.get_blueprint_library().find('sensor.camera.depth'),
                    'sem': world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
                }
                for _, bp in cam_bp.items():
                    bp.set_attribute('image_size_x', '1024')
                    bp.set_attribute('image_size_y', '1024')
                    bp.set_attribute('fov', '90') # Horizontal (left-right)
                    bp.set_attribute('sensor_tick', '0.0')
                # w1, w2, h1, h2 = pinhole2panorama.get_fov_90_range((1024, 1024), 96)

                # Initialize 3x6=18 cameras
                zero_transform = carla.Transform(
                    carla.Location(0.0, 0.0, 0.0),
                    carla.Rotation(0.0, 0.0, 0.0),
                )
                cam_actor = {}
                for cam_type, bp in cam_bp.items():
                    for dir_idx in 'f,u,d,r,b,l'.split(','):
                        key = f'{cam_type}_{dir_idx}'
                        cam_actor[key] = world.spawn_actor(bp, zero_transform)
                
                # Register callback function for each camera
                pinhole_result = []
                callback = lambda buffer, key: lambda image: buffer.append((key, image))
                for key, cam in cam_actor.items():
                    cam.listen(callback(pinhole_result, key))

                #
                d = np.load(f'waypoints/{town}.npz')
                waypoints = d['waypoints']
                for pid, waypoint in list(enumerate(waypoints)):

                    # Clear the buffer
                    pinhole_result.clear()

                    #
                    cx, cy, cz, pitch, yaw, roll = waypoint
                    cam_loc = carla.Location(cx, cy, cz + floating_at)
                    cam_rots = [carla.Rotation(pitch+r, yaw, roll) for r in [0, 90, -90]]
                    cam_rots += [carla.Rotation(pitch, yaw+r, roll)for r in [90, 180, 270]]

                    # Set transformation for each camera
                    for dir_idx, cam_rot in zip('f,u,d,r,b,l'.split(','), cam_rots):
                        cam_transform = carla.Transform(cam_loc, cam_rot)
                        for cam_type in cam_bp:
                            key = f'{cam_type}_{dir_idx}'
                            cam_actor[key].set_transform(cam_transform)

                    # Start simulation
                    num = world.tick()
                    print(f'{town}_{weather}: Getting images for waypoint {pid+1}/{len(waypoints)} ...', end='')
                    while len(pinhole_result) < 18:
                        continue
                    print(f' Done!')

                    # Convert to numpy array
                    img_dict = {}
                    for key, image in pinhole_result:
                        # image.save_to_disk('temp.png')
                        img = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
                        img_dict[key] = img.reshape((image.height, image.width, 4))[..., [2,1,0,3]] # BGRA to RGBA

                    pinhole2panorama.pinhole_to_panorama(f'panorama/{town}_{weather}/%05d' % pid, img_dict)
               
                # End of all waypoints
            
            except Exception as e:
                print(f'{town}_{weather} ERROR!')
                print(e)
        
        # End of all towns
    
    # End of all weathers
