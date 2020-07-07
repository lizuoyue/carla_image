import carla
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import random

if __name__ == '__main__':

    towns = ['Town%02d' % i for i in [1,2,3,4,5,6,7,10]]
    towns[-1] += 'HD'

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    for town in towns:

        world = client.load_world(town)
        world_map = world.get_map()
        topology = world_map.get_topology()

        plt.figure(figsize=(8,8),dpi=300)
        for w1, w2 in topology:
            x1, y1 = w1.transform.location.x, w1.transform.location.y
            x2, y2 = w2.transform.location.x, w2.transform.location.y
            plt.plot([x1, x2], [y1, y2],'-',color='C0',linewidth=1)
            plt.plot([x1, x2], [y1, y2],'.',color='C1',markersize=1)
            # if n1.road_id == n2.road_id:
                # plt.text(x1, y1, f'{n1.road_id},{n1.lane_id}',fontsize=1)
                # plt.text((x1+x2)/2, (y1+y2)/2, f'{n1.road_id}',fontsize=1)
        plt.savefig(f'topology/{town}.pdf')
