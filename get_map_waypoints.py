import carla
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import random

if __name__ == '__main__':

    towns = ['Town%02d' % i for i in [1,2,3,4,5,6,7,10]]
    towns[-1] += 'HD'
    gaps = [2.0, 1.0, 4.0, 8.0, 6.0, 6.0, 2.0, 2.0]
    # gaps = [2.0] * len(towns)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    for town, gap in zip(towns, gaps):

        world = client.load_world(town)
        world_map = world.get_map()
        waypoints = world_map.generate_waypoints(gap)
        waypoints_np = np.array([
            (w.transform.location.x, w.transform.location.y, w.transform.location.z) + 
            (w.transform.rotation.pitch, w.transform.rotation.yaw, w.transform.rotation.roll)
        for w in waypoints])
        kdtree = KDTree(waypoints_np[:,:3], leaf_size=8)

        nbs = []
        for waypoint in waypoints:
            next_points = waypoint.next(gap)
            if next_points:
                next_points_np = np.array([
                    [nn.transform.location.x, nn.transform.location.y, nn.transform.location.z] +
                    [nn.transform.rotation.pitch, nn.transform.rotation.yaw, nn.transform.rotation.roll]
                for nn in next_points])
                dist, idx = kdtree.query(next_points_np[:,:3], k=5)
                angle = np.sin(np.abs(
                    waypoints_np[idx.flatten(),3:] - np.stack([next_points_np[:,3:]]*5,axis=1).reshape(-1,3)
                )/180*np.pi/2).mean(axis=-1).reshape(dist.shape)
                nb = []
                for i, n in enumerate((dist + angle * gap).argmin(axis=-1)):
                    nb.append(idx[i, n])
                nbs.append(nb)
            else:
                nbs.append([])

        plt.figure(figsize=(8,8),dpi=300)
        for waypoint_np, nb in zip(waypoints_np, nbs):
            for idx in nb:
                next_point = waypoints_np[idx]
                plt.plot([waypoint_np[0], next_point[0]], [waypoint_np[1], next_point[1]],'-',color='C0',linewidth=1)
        plt.plot(waypoints_np[:,0], waypoints_np[:,1],'.',color='C1',markersize=1)
        for idx, wp in enumerate(waypoints_np):
            plt.text(wp[0], wp[1], f'{idx}',fontsize=1)
        plt.savefig(f'waypoints/{town}.pdf')
        np.savez_compressed(f'waypoints/{town}.npz', waypoints=waypoints_np, nbs=nbs)
