import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from mmcv import Config
from mmcv.datasets import build_dataset

cfg = Config.fromfile('adzoo/orion/configs/orion_stage1_train.py')
dataset_cfg = getattr(cfg.data.train, 'dataset', cfg.data.train)
dataset = build_dataset(dataset_cfg)

def check_empty(i):
    gt_labels, gt_bboxes = dataset.get_map_info(i)
    return i if len(gt_bboxes.instance_list) == 0 else None

N = len(dataset.data_infos)
max_workers = min(32, os.cpu_count() or 8)  # 可按机器调整
print("max workers:", max_workers)
empty_indices = []

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(check_empty, i) for i in range(N)]
    for fut in as_completed(futures):
        idx = fut.result()
        if idx is not None:
            empty_indices.append(idx)

empty_indices.sort()
print("empty count:", len(empty_indices))
print("total samples:", N)
print("empty ratio: {:.2%}".format(len(empty_indices) / N))
print("first few empty indices:", empty_indices[:50])