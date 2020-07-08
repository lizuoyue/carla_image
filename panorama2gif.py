import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time, tqdm
from PIL import Image

if __name__ == '__main__':

    random.seed(1993)
    folders = sorted(glob.glob('panorama/*'))
    for folder in folders:
        town = os.path.basename(folder)
        nbs = np.load(f'waypoints/{town}.npz', allow_pickle=True)['nbs']
        num_points = len(nbs)
        for _ in range(10): # number of gif files
            flag = True
            while flag:
                li = random.sample(range(num_points), 1)
                for _ in range(14): # length of a gif
                    nb = nbs[li[-1]]
                    if nb:
                        li.append(random.choice(nb))
                if len(li) == 15:
                    flag = False
            ims = []
            for it in li:
                filename = f'{folder}/%05d_rgb.png' % it
                ims.append(Image.open(filename))
            ims[0].save(f'gif/{town}_%05d.gif' % li[0], save_all=True, append_images=ims[1:], loop=0, duration=100)
