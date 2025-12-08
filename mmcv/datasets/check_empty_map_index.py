import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from mmcv import Config
from mmcv.datasets import build_dataset
from tqdm import tqdm
import numpy as np

cfg = Config.fromfile('adzoo/orion/configs/orion_stage1_train.py')
dataset_cfg = getattr(cfg.data.train, 'dataset', cfg.data.train)
dataset = build_dataset(dataset_cfg)

def check_empty(i):
    # 轻量判定版：避免创建 LineString / torch 对象
    info = dataset.data_infos[i]
    town = info['town_name']
    mi = dataset.map_infos[town]
    world2lidar = np.array(info['sensors']['LIDAR_TOP']['world2lidar'])
    inv = np.linalg.inv(world2lidar)
    ego_xy = inv[0:2,3]
    max_distance = 50
    pcr = dataset.point_cloud_range

    # 先筛附近lane
    chosen = []
    for sp in mi['lane_sample_points']:
        if np.min(np.linalg.norm(sp[:,0:2]-ego_xy, axis=-1)) < max_distance:
            chosen.append(True)
        else:
            chosen.append(False)

    # 只要有任意一条在范围内有>=2点就非空
    for ok, pts in zip(chosen, mi['lane_points']):
        if not ok: continue
        pts_h = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=-1)
        pts_l = (world2lidar @ pts_h.T).T
        m = (pcr[0]<pts_l[:,0])&(pts_l[:,0]<pcr[3])&(pcr[1]<pts_l[:,1])&(pts_l[:,1]<pcr[4])
        if np.count_nonzero(m) > 1:
            return None  # 非空
    # 触发器（按原逻辑要求全部点在范围内才算）
    for pts in mi['trigger_volumes_points']:
        pts_h = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=-1)
        pts_l = (world2lidar @ pts_h.T).T
        m = (pcr[0]<pts_l[:,0])&(pts_l[:,0]<pcr[3])&(pcr[1]<pts_l[:,1])&(pts_l[:,1]<pcr[4])
        if m.all():
            return None  # 非空
    return i  # 空

N = len(dataset.data_infos)
max_workers = min(32, os.cpu_count() or 8)
empty_indices = []
with ProcessPoolExecutor(max_workers=max_workers) as ex, tqdm(total=N, desc='Checking empty polylines') as bar:
    futures = [ex.submit(check_empty, i) for i in range(N)]
    for f in as_completed(futures):
        r = f.result()
        if r is not None:
            empty_indices.append(r)
        bar.update(1)

empty_indices.sort()
print('empty count:', len(empty_indices))
print('total samples:', N)
print('empty ratio: {:.2%}'.format(len(empty_indices)/N))
print('first few empty indices:', empty_indices[:50])