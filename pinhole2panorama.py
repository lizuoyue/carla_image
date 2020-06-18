import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, tqdm
from PIL import Image

def get_panorama(
        filenames,
        panorama_size=[512, 256],
        min_max_lat=[-np.pi/2, np.pi/2],
    ):

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

    # Compute the unit vector of each pixel
    vx = cos_lat.mul(sin_lon).numpy()
    vy = -cos_lat.mul(cos_lon).numpy()
    vz = sin_lat.numpy()

    xy = vx / vy
    xz = vx / vz
    yx = vy / vx
    yz = vy / vz
    zx = vz / vx
    zy = vz / vy

    d = {'area':{}, 'x':{}, 'y':{}}
    d['area']['f'] = (vy < 0) & (-1 <= xy) & (xy <= 1) & (-1 <= zy) & (zy <= 1)
    d['area']['l'] = (vx < 0) & (-1 <= yx) & (yx <= 1) & (-1 <= zx) & (zx <= 1)
    d['area']['r'] = (vx > 0) & (-1 <= yx) & (yx <= 1) & (-1 <= zx) & (zx <= 1)
    d['area']['b'] = (vy > 0) & (-1 <= xy) & (xy <= 1) & (-1 <= zy) & (zy <= 1)
    d['area']['u'] = (vz > 0) & (-1 <= xz) & (xz <= 1) & (-1 <= yz) & (yz <= 1)
    d['area']['d'] = (vz < 0) & (-1 <= xz) & (xz <= 1) & (-1 <= yz) & (yz <= 1)
    d['x'] = {key: val for key, val in zip(li, [-xy, xz, -xz, yx, -xy, yx])}
    d['y'] = {key: val for key, val in zip(li, [zy, -yz, -yz, -zx, -zy, zx])}

    res = np.zeros((nrow, ncol, 3))
    for filename in filenames:
        which = filename.split('_')[-1].replace('.png', '')
        assert(which in d['area'])
        coord_x = d['x'][which]
        coord_y = d['y'][which]
        area = d['area'][which]
        im = np.array(Image.open(filename).convert('RGB')).transpose([2,0,1])
        im = torch.from_numpy(im).float().unsqueeze(0)
        coord = np.stack([coord_x[area], coord_y[area]], axis=-1)
        coord = torch.from_numpy(coord).unsqueeze(0).unsqueeze(0)
        val = F.grid_sample(im, coord, align_corners=True).squeeze().numpy().T
        res[area] = val
    Image.fromarray(res.astype(np.uint8)).save(filename.replace('pinhole', 'panorama').replace(f'_{which}', ''))

if __name__ == '__main__':

    # num = 6
    # filenames = sorted(glob.glob('pinhole/*.png'))
    # assert(len(filenames) % num == 0)
    # for i in range(0, len(filenames), num):
    #     get_panorama(filenames[i: i+num])
    
    num = 15
    filenames = sorted(glob.glob('panorama/*.png'))
    assert(len(filenames) % num == 0)
    for i in range(0, len(filenames), num):
        ims = [Image.open(filename) for filename in filenames[i: i+num]]
        ims[0].save(filenames[i].split('_')[0] + '.gif', save_all=True, append_images=ims[1:], loop=0, duration=250)
