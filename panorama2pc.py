import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time, tqdm
from PIL import Image
import carla

def get_panorama_vector(
        panorama_size=[512, 256],
        min_max_lat=[-np.pi/2, np.pi/2],
    ):

    ncol, nrow = panorama_size
    min_lat, max_lat = min_max_lat

    x = (torch.arange(0, ncol, 1, dtype=torch.float32) + 0.5) / ncol
    y = (torch.arange(0, nrow, 1, dtype=torch.float32) + 0.5) / nrow
    lon = (x * 2.0 * np.pi - np.pi)
    lat = (1.0 - y) * (max_lat - min_lat) + min_lat

    sin_lon = torch.sin(lon).view(1, ncol).expand(nrow, ncol)
    cos_lon = torch.cos(lon).view(1, ncol).expand(nrow, ncol)
    sin_lat = torch.sin(lat).view(nrow, 1).expand(nrow, ncol)
    cos_lat = torch.cos(lat).view(nrow, 1).expand(nrow, ncol)

    vx = cos_lat.mul(sin_lon).numpy()
    vy = cos_lat.mul(cos_lon).numpy()
    vz = sin_lat.numpy()

    return vx, vy, vz

def decode_dep(dep_file):
    im = np.array(Image.open(dep_file).convert('RGB')).astype(np.float).transpose([2,0,1])
    im = (im[0:1] + im[1:2]*256 + im[2:3]*256*256) / (256**3-1) * 1000 * np.sqrt(3)
    return im[0]

def decode_sem(sem_file):
    im = Image.open(sem_file)
    palette = np.array(im.getpalette()).reshape((256, 3))
    im = np.array(im)
    im = palette[im.flatten()].reshape(im.shape+(3,))
    return im

def write_numpy_array(filename, arr):
    with open(filename, 'w') as f:
        for line in arr:
            coord = ('%.3f;'*3) % tuple(list(line[:3]))
            rgb = ('%d;'*3) % tuple(list(line[3:]))
            f.write(coord + rgb[:-1] + '\n')

def get_transformation_matrix(transform):
    location = transform.location
    rotation = transform.rotation

    yaw = rotation.yaw
    cy = np.cos(np.deg2rad(yaw))
    sy = np.sin(np.deg2rad(yaw))
    
    roll = rotation.roll
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))

    pitch = rotation.pitch
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))

    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
        [     sp,               -cp * sr,                 cp * cr, location.z],
        [    0.0,                    0.0,                     0.0,        1.0],
    ])

def homo(arr):
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)

if __name__ == '__main__':

    # panorama coord: right x, forward y, top z (right-handed)
    # carla coord: right y, forward x, top z (left-handed)

    random.seed(1993)
    folders = sorted(glob.glob('panorama/*'))
    vec = get_panorama_vector()
    for folder in folders:
        town = os.path.basename(folder)
        d = np.load(f'waypoints/{town}.npz', allow_pickle=True)
        nbs = d['nbs']
        waypoints = d['waypoints']
        num_points = len(nbs)
        for _ in range(10): # number of pc files
            flag = True
            while flag:
                li = random.sample(range(num_points), 1)
                for _ in range(14): # length of a pc
                    nb = nbs[li[-1]]
                    if nb:
                        li.append(random.choice(nb))
                if len(li) == 15:
                    flag = False
            pc_coord, pc_rgb, pc_sem =[], [], []
            for it in li:
                waypoint = waypoints[it]
                rgb = np.array(Image.open(f'{folder}/%05d_rgb.png' % it)).reshape((-1, 3))
                dep = decode_dep(f'{folder}/%05d_dep.png' % it).reshape((-1))
                sem = decode_sem(f'{folder}/%05d_sem.png' % it).reshape((-1, 3))
                assert(rgb.shape[0] == dep.shape[0])
                assert(rgb.shape[0] == sem.shape[0])
                coord = np.stack([v.flatten() * dep for v in vec], axis=1)
                coord = coord[:, [1, 0, 2]] # carla uses left-handed coordinates
                local2world = get_transformation_matrix(carla.Transform(
                    carla.Location(waypoint[0], waypoint[1], waypoint[2]),
                    carla.Rotation(waypoint[3], waypoint[4], waypoint[5])
                ))
                world_coord = local2world.dot(homo(coord).T)[:3].T
                pc_coord.append(world_coord)
                pc_rgb.append(rgb)
                pc_sem.append(sem)

                # write_numpy_array(f'pc/{town}_%05d_rgb.txt' % it, np.hstack([pc_coord[-1], pc_rgb[-1]]))
                # write_numpy_array(f'pc/{town}_%05d_sem.txt' % it, np.hstack([pc_coord[-1], pc_sem[-1]]))

            pc_coord = np.vstack(pc_coord)
            pc_rgb = np.vstack(pc_rgb)
            pc_sem = np.vstack(pc_sem)
        
            write_numpy_array(f'pc/{town}_%05d_rgb.txt' % li[0], np.hstack([pc_coord, pc_rgb]))
            write_numpy_array(f'pc/{town}_%05d_sem.txt' % li[0], np.hstack([pc_coord, pc_sem]))

            quit()
