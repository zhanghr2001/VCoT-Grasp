import os
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLOWorld
from ultralytics import settings

def mask_to_bbox_position(mask):
    """Get bbox position from boolean mask array."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)

ds_root = "/baishuanghao/mllm_data/grasp_anything"
image_root = os.path.join(ds_root, "image")
mask_root = os.path.join(ds_root, "mask")

device = "cuda"
settings.update({"runs_dir": "/baishuanghao/GraspLLM/data_prepare/yolo_world/runs"})
model = YOLOWorld("data_prepare/yolo_world/yolov8s-worldv2.pt").to(device)  # or choose yolov8m/l-world.pt

load_path = "split/grasp_anything_filter/all.csv"
save_path = "split/grasp_anything_filter/all_filter.csv"
data = pd.read_csv(load_path)
iou_threshold = 0.25

class_batch_size = 100
filter_mask = []
for i in tqdm(range(data.shape[0])):
    data_id, obj_name, scene_description = data.iloc[i]
    image_id, obj_id = data_id.split("_")

    mask = np.load(os.path.join(mask_root, f"{data_id}.npy"))
    bbox_label = mask_to_bbox_position(mask)
    bbox_label = torch.tensor(bbox_label).unsqueeze(0).to(device)

    if i % class_batch_size == 0:
        classes = data.iloc[i:i + class_batch_size, 1].tolist()
        classes = list(set(classes))
        # model.set_classes([obj_name])
        model.model.set_classes(classes, cache_clip_model=False) # see https://github.com/ultralytics/ultralytics/issues/20889
        model.model.names = classes
        if model.predictor:
            model.predictor.model.names = classes

    results = model.predict(os.path.join(image_root, f"{image_id}.jpg"), save=False, verbose=False)
    results = results[0]
    bbox_pred = results.boxes.xyxy
    cls_pred = results.boxes.cls
    obj_cls = classes.index(obj_name)
    bbox_pred = bbox_pred[cls_pred == obj_cls]
    if bbox_pred.shape[0] != 1:
        filter_mask.append(False)
        continue
    iou = box_iou(bbox_pred, bbox_label).item()
    filter_mask.append(iou > iou_threshold)

filterd_data = data[filter_mask]
filterd_data.to_csv(save_path, index=False, header=False)

print(f"before: {data.shape[0]}, after: {filterd_data.shape[0]}")