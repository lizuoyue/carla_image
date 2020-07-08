import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, tqdm
from PIL import Image

palette = np.zeros((256,3), np.uint8)
palette[:14] = np.array([
    (0, 0, 0),
    (70, 70, 70),
    (190, 153, 153),
    (250, 170, 160),
    (220, 20, 60),
    (153, 153, 153),
    (157, 234, 50),
    (128, 64, 128),
    (244, 35, 232),
    (107, 142, 35),
    (0, 0, 142),
    (102, 102, 156),
    (220, 220, 0),
    (70, 130, 180)
])
one_hot = np.eye(13)

def get_distance_depth_ratio(
        image_size,
        fov=90,
    ):
    w, h = image_size
    fov_rad = fov / 180.0 * np.pi
    f = (h/2.0) / np.tan(fov_rad/2)
    x = np.arange(w) + 0.5 - w / 2
    y = np.arange(h) + 0.5 - h / 2
    x, y = np.meshgrid(x, y)
    return np.sqrt(x ** 2 + y ** 2 + f ** 2) / f

def depth_to_rgb(depth):
    max_depth = 1000 * np.sqrt(3)
    depth = np.clip(depth, 0, max_depth)
    depth /= max_depth
    depth *= (256**3-1)
    depth = depth.astype(np.int)
    r = depth % 256
    g = (depth // 256) % 256
    b = (depth // 65536) % 256
    return np.dstack([r, g, b]).astype(np.uint8)

def get_panorama(
        filenames,
        pinhole_size=[512, 512],
        panorama_size=[512, 256],
        min_max_lat=[-np.pi/2, np.pi/2],
    ):

    assert(len(filenames) == 18)
    pid = list({filename.split('_')[0] for filename in filenames})
    assert(len(pid) == 1)
    filenames.sort()
    pid = pid[0].replace('pinhole', 'panorama')

    li = 'f,u,d,r,b,l'.split(',')
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

    xy = vx / vy
    xz = vx / vz
    yx = vy / vx
    yz = vy / vz
    zx = vz / vx
    zy = vz / vy

    d = {'area': {}, 'x': {}, 'y': {}}
    d['area']['f'] = (vy > 0) & (-1 <= xy) & (xy <= 1) & (-1 <= zy) & (zy <= 1)
    d['area']['l'] = (vx < 0) & (-1 <= yx) & (yx <= 1) & (-1 <= zx) & (zx <= 1)
    d['area']['r'] = (vx > 0) & (-1 <= yx) & (yx <= 1) & (-1 <= zx) & (zx <= 1)
    d['area']['b'] = (vy < 0) & (-1 <= xy) & (xy <= 1) & (-1 <= zy) & (zy <= 1)
    d['area']['u'] = (vz > 0) & (-1 <= xz) & (xz <= 1) & (-1 <= yz) & (yz <= 1)
    d['area']['d'] = (vz < 0) & (-1 <= xz) & (xz <= 1) & (-1 <= yz) & (yz <= 1)
    d['x'] = {key: val for key, val in zip(li, [ xy, xz, -xz, -yx, xy, -yx])}
    d['y'] = {key: val for key, val in zip(li, [-zy, yz,  yz, -zx, zy,  zx])}

    ratio = get_distance_depth_ratio(image_size=pinhole_size)

    rgb = np.zeros((nrow, ncol, 3), np.uint8)
    dep = np.zeros((nrow, ncol), np.float)
    sem = np.zeros((nrow, ncol), np.uint8)
    for i, which in enumerate(sorted(li)):
        dep_file, rgb_file, sem_file = filenames[3*i], filenames[3*i+1], filenames[3*i+2]
        assert(which == dep_file.split('_')[1])
        assert(which == rgb_file.split('_')[1])
        assert(which == sem_file.split('_')[1])

        coord_x = d['x'][which]
        coord_y = d['y'][which]
        area = d['area'][which]
        coord = np.stack([coord_x[area], coord_y[area]], axis=-1)
        coord = torch.from_numpy(coord).unsqueeze(0).unsqueeze(0)

        # rgb
        im = np.array(Image.open(rgb_file).convert('RGB')).transpose([2,0,1])
        im = torch.from_numpy(im).float().unsqueeze(0)
        rgb[area] = F.grid_sample(im, coord, mode='nearest', align_corners=True).squeeze().numpy().T

        # dep
        im = np.array(Image.open(dep_file).convert('RGB')).astype(np.float).transpose([2,0,1])
        im = (im[0:1] + im[1:2]*256 + im[2:3]*256*256) / (256**3-1) * 1000
        im = im * ratio
        im = torch.from_numpy(im).float().unsqueeze(0)
        dep[area] = F.grid_sample(im, coord, mode='nearest', align_corners=True).squeeze().numpy().T

        # sem
        im = np.array(Image.open(sem_file).convert('RGB'))[..., 0:1].transpose([2,0,1])
        # im = one_hot[im.flatten()].reshape(im.shape[:2] + (13,)).transpose([2,0,1])
        # im = im.transpose([2,0,1])
        im = torch.from_numpy(im).float().unsqueeze(0)
        sem[area] = F.grid_sample(im, coord, mode='nearest', align_corners=True).squeeze().numpy().T
        # sem[area] = F.grid_sample(im, coord, mode='nearest', align_corners=True).squeeze().numpy().T

    # rgb
    Image.fromarray(rgb.astype(np.uint8)).save(pid + '_rgb.png')

    # dep
    dep[dep > 999.99] = 1000
    dep_rgb = depth_to_rgb(dep)
    Image.fromarray(dep_rgb).save(pid + '_dep.png')

    # sem
    # sem = sem.argmax(axis=-1).astype(np.uint8)
    sem[dep > 999.99] = 13
    sem = Image.fromarray(sem)
    sem.putpalette(palette.flatten())
    sem.save(pid + '_sem.png')

    # coord
    coord = np.dstack([dep * vx, dep * vy, dep * vz])
    return coord

if __name__ == '__main__':

    folders = sorted(glob.glob('pinhole/*'))
    for folder in folders:
        filenames = sorted(glob.glob(f'{folder}/*.png'))
        num_files = len(filenames)
        assert(num_files % 18 == 0)
        for i in range(num_files // 18):
            coord = get_panorama(filenames[i*18:i*18+18])
            # sem_rgb = palette[sem.flatten()].reshape(rgb.shape)
            # coord[..., 1] += bias
            # bias += gap
            # for a, b, c in zip(coord.reshape((-1, 3)), rgb.reshape((-1, 3)), sem_rgb.reshape((-1, 3))):
            #     f_rgb.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (tuple(a) + tuple(b)))
            #     f_sem.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (tuple(a) + tuple(c)))
