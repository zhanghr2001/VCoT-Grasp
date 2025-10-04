"""
filter data that:
have a obj name "nan"
have too few mask points
bbox don't match the obj name
"""

import os
import csv
import numpy as np
import pickle
from tqdm import tqdm

def merge_names(s, names):
    """
    deal with data like:
    ('A green lime, a brown wooden cutting board, and a silver knife with a black handle', ['lime', 'cutting', 'board', 'knife'])
    merge names into: ['lime', 'cutting board', 'knife']
    """
    merged = []
    i = 0
    while i < len(names) - 1:
        cur = f"{names[i]} {names[i+1]}"
        if cur in s:
            merged.append(cur)
            i += 2
        else:
            merged.append(names[i])
            i += 1

    if i == len(names) - 1:
        merged.append(names[-1])

    return merged

csv_path = "split/grasp_anything_filter/all.csv"
ds_root = "/baishuanghao/mllm_data/grasp_anything"
image_root = os.path.join(ds_root, "image")
label_root = os.path.join(ds_root, "grasp_label_positive")
description_root = os.path.join(ds_root, "scene_description")
mask_root = os.path.join(ds_root, "mask")

mask_point_thresh = 400

data_ids = os.listdir(label_root)
data_ids = [f.removesuffix(".pt") for f in data_ids if f.endswith(".pt")]
data_ids.sort()

data = []
skip_nan = 0
skip_not_enough_points = 0
skip_index = 0
for data_id in tqdm(data_ids):
    scene_id, obj_index = data_id.split("_")
    obj_index = int(obj_index)

    description_path = os.path.join(description_root, f"{scene_id}.pkl")
    with open(description_path, "rb") as f:
        description_data = pickle.load(f)
    description = description_data[0]
    obj_names = description_data[1]
    obj_names = merge_names(description, obj_names)
    if obj_index >= len(obj_names):
        skip_index += 1
        continue
    obj_name = obj_names[obj_index]
    if obj_name in ["", "nan", "NaN"]:
        skip_nan += 1
        continue

    mask_path = os.path.join(mask_root, f"{data_id}.npy")
    mask = np.load(mask_path)
    mask_points = np.count_nonzero(mask)
    if mask_points < mask_point_thresh:
        skip_not_enough_points += 1
        continue

    data.append([data_id, obj_name, description])


with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)


print(len(data_ids))
print(data_ids[0:100])
print(f"skip nan: {skip_nan}")
print(f"skip index: {skip_index}")
print(f"skip not enough points: {skip_not_enough_points}")
