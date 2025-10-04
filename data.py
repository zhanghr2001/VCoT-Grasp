import os
import re
import random
import math

import pandas as pd
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split

from constants import *


def get_dataloaders(
    arch_config,
    dataset_name,
    processor,
    train_batch_size,
    eval_batch_size,
    test_batch_size=1,
    bbox_ratio=1.0,
    num_workers=8,
    dtype=torch.bfloat16,
    data_ratio=1.0,
):
    def grasp_collate_fn(batch):
        images = []
        prompts = []
        labels = []
        grasp_labels = []
        obj_names = []
        bbox_images = []
        for img, prompt, label, grasp, obj_name, bbox_image in batch:
            images.append(img)
            prompts.append(prompt)
            labels.append(label)
            grasp_labels.append(grasp)
            obj_names.append(obj_name)
            bbox_images.append(bbox_image)

        inputs = processor(
            texts=prompts,
            images=images,
            suffix=labels,
            obj_names=obj_names,
            bbox_images=bbox_images,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,  # When using mixed precision we want round multiples of 8/16
        )
        grasp_labels = list(filter(lambda x: x is not None, grasp_labels))
        if grasp_labels:
            inputs["grasp_labels"] = torch.stack(grasp_labels).to(dtype)
        return inputs

    if dataset_name == "grasp_anything":
        ds = GraspAnythingForGraspGeneration(
            grasp_anything_planar_grasp_train_csv_path,
            grasp_anything_rgb_root,
            grasp_anything_planar_grasp_root,
            grasp_anything_mask_root,
            use_bbox=arch_config.use_bbox,
            action_head_type=arch_config.action_head,
        )
        if arch_config.use_bbox:
            ds_bbox = GraspAnythingForBbox(
                grasp_anything_planar_grasp_train_csv_path,
                grasp_anything_rgb_root,
                grasp_anything_mask_root,
            )
            bbox_num = int(len(ds_bbox) * bbox_ratio)
            ds_bbox, _ = random_split(ds_bbox, [bbox_num, len(ds_bbox) - bbox_num])
            ds = ConcatDataset([ds, ds_bbox])
        eval_size = 5000
        train_size = len(ds) - eval_size
        train_ds = Subset(ds, range(train_size))
        if data_ratio < 1:
            used_len = int(data_ratio * len(train_ds))
            discarded_len = len(train_ds) - used_len
            train_ds, _ = random_split(train_ds, [used_len, discarded_len])
        eval_ds = Subset(ds, range(train_size, train_size + eval_size))

        train_dataloader = DataLoader(
            train_ds, batch_size=train_batch_size, collate_fn=grasp_collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        eval_dataloader = DataLoader(
            eval_ds, batch_size=eval_batch_size, collate_fn=grasp_collate_fn, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataloader = None
    elif dataset_name == "real_data":
        ds = RealDataForGraspGeneration(
            grasp_xml_path=real_grasp_data_path,
            bbox_xml_path=real_bbox_data_path,
            image_root=real_image_root,
            split="train",
            use_bbox=arch_config.use_bbox,
            action_head_type=arch_config.action_head,
        )
        if arch_config.use_bbox:
            ds_bbox = RealDataForBbox(
                grasp_xml_path=real_grasp_data_path,
                bbox_xml_path=real_bbox_data_path,
                image_root=real_image_root,
                split="train",
            )
            bbox_num = int(len(ds_bbox) * bbox_ratio)
            ds_bbox, _ = random_split(ds_bbox, [bbox_num, len(ds_bbox) - bbox_num])
            ds = ConcatDataset([ds, ds_bbox])

        eval_size = 50
        train_size = len(ds) - eval_size
        train_ds, eval_ds = random_split(ds, [train_size, eval_size])

        train_dataloader = DataLoader(
            train_ds, batch_size=train_batch_size, collate_fn=grasp_collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        eval_dataloader = DataLoader(
            eval_ds, batch_size=eval_batch_size, collate_fn=grasp_collate_fn, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataloader = None
    else:
        raise ValueError("Invalid dataset name")

    return train_dataloader, eval_dataloader, test_dataloader


class GraspAnythingForGraspGeneration(Dataset):
    def __init__(self, csv_path, rgb_root, grasp_root, mask_root, use_bbox, action_head_type):
        super().__init__()
        self.data = pd.read_csv(csv_path, header=None)
        self.rgb_root = rgb_root
        self.grasp_root = grasp_root
        self.mask_root = mask_root
        self.instruction = "grasp the {}"

        self.use_bbox = use_bbox
        self.use_cropped_bbox = True
        self.action_head_type = action_head_type
        self.input_image_size = 416
        self.output_image_size = 224

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        grasp_id, obj_name, scene_description = self.data.iloc[index]
        scene_id = grasp_id.split("_")[0]

        image = Image.open(os.path.join(self.rgb_root, scene_id + ".jpg"))

        grasp = torch.load(os.path.join(self.grasp_root, grasp_id + ".pt"))
        grasp = grasp[0][1:]  # highest score

        if self.use_bbox:
            mask = np.load(os.path.join(self.mask_root, grasp_id + ".npy"))
            bbox_pil = mask_to_bbox_position(mask)
            bbox = convert_bbox(bbox_pil)
            bbox = normalize_bbox(bbox, self.input_image_size)  # normalize to 0~1
            bbox_image = crop_bbox(image, bbox_pil)
        else:
            bbox_image = None

        prompt = self.instruction.format(obj_name)
        if self.action_head_type == "MLP":
            grasp = normalize_rect_grasp_with_angle_bins(grasp, self.input_image_size, angle_bins)
            prompt += "<action>" * action_with_binned_angle_seq_len
            label = ""
        elif self.action_head_type == "Diffusion":
            grasp = normalize_rect_grasp(grasp, self.input_image_size)
            prompt += "<action>" * action_seq_len
            label = ""
        elif self.action_head_type in ["LM_pretrained", "LM_new"]:
            grasp_str = normalize_rect_grasp(grasp, self.input_image_size)
            grasp_str = rect_grasp_to_str(grasp, use_pretrained_token=(self.action_head_type == "LM_pretrained"))
            label = grasp_str
            grasp = None
        else:
            raise ValueError

        return image, prompt, label, grasp, obj_name, bbox_image


class GraspAnythingForBbox(Dataset):
    def __init__(self, csv_path, rgb_root, mask_root):
        super().__init__()
        self.data = pd.read_csv(csv_path, header=None)
        self.rgb_root = rgb_root
        self.mask_root = mask_root
        self.instruction = "detect {}"

        self.input_image_size = 416
        self.output_image_size = 224

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        grasp_id, obj_name, scene_description = self.data.iloc[index]
        scene_id = grasp_id.split("_")[0]

        image = Image.open(os.path.join(self.rgb_root, scene_id + ".jpg"))

        mask = np.load(os.path.join(self.mask_root, grasp_id + ".npy"))
        bbox_pil = mask_to_bbox_position(mask)
        bbox = convert_bbox(bbox_pil)
        bbox = normalize_bbox(bbox, self.input_image_size)  # normalize to 0~1
        bbox_str = bbox_to_str(bbox)
        label = bbox_str

        prompt = self.instruction.format(obj_name)

        grasp = None
        bbox_image = None

        return image, prompt, label, grasp, obj_name, bbox_image


class RealDataForGraspGeneration(Dataset):
    def __init__(self, grasp_xml_path, bbox_xml_path, image_root, split, use_bbox, action_head_type):
        super().__init__()
        self.grasp_data = parse_grasp_data(grasp_xml_path)
        self.bbox_data = parse_bbox_data(bbox_xml_path)
        self.image_root = image_root
        self.split = split
        self.use_bbox = use_bbox
        self.action_head_type = action_head_type

        self.input_image_size = 720
        self.output_image_size = 224
        self.unseen_objs = real_unseen_objs
        self.instruction = "grasp the {}"
        self.error_lines = [34, 160, 161, 162, 164, 165, 168, 197, 377, 379, 380, 381, 383, 384, 385, 386, 387, 470, 666]

        if self.split == "train":
            grasp_data = list(filter(lambda x: x["obj_name"] not in self.unseen_objs, self.grasp_data))
        else:
            raise ValueError

        self.grasp_data = []
        for i, d in enumerate(grasp_data):
            if i not in self.error_lines:
                self.grasp_data.append(d)

    def __len__(self):
        return len(self.grasp_data)

    def __getitem__(self, index):
        data = self.grasp_data[index]
        image_id, obj_name, grasp = data["image_id"], data["obj_name"], data["grasp_label"]
        bbox_pil = self.bbox_data[image_id][obj_name]
        image = Image.open(os.path.join(self.image_root, f"{image_id:04d}.jpg"))
        grasp = torch.tensor(grasp)

        if self.use_bbox:
            bbox = convert_bbox(bbox_pil)
            bbox = normalize_bbox(bbox, self.input_image_size)  # normalize to 0~1
            bbox_image = crop_bbox(image, bbox_pil)
        else:
            bbox_image = None

        prompt = self.instruction.format(obj_name)
        if self.action_head_type == "MLP":
            grasp = normalize_rect_grasp_with_angle_bins(grasp, self.input_image_size, angle_bins)
            prompt += "<action>" * action_with_binned_angle_seq_len
            label = ""
        elif self.action_head_type == "Diffusion":
            grasp = normalize_rect_grasp(grasp, self.input_image_size)
            prompt += "<action>" * action_seq_len
            label = ""
        elif self.action_head_type in ["LM_pretrained", "LM_new"]:
            grasp_str = normalize_rect_grasp(grasp, self.input_image_size)
            grasp_str = rect_grasp_to_str(grasp, use_pretrained_token=(self.action_head_type == "LM_pretrained"))
            label = grasp_str
            grasp = None
        else:
            raise ValueError

        return image, prompt, label, grasp, obj_name, bbox_image


class RealDataForBbox(Dataset):
    def __init__(self, grasp_xml_path, bbox_xml_path, image_root, split):
        super().__init__()
        self.grasp_data = parse_grasp_data(grasp_xml_path)
        self.bbox_data = parse_bbox_data(bbox_xml_path)
        self.image_root = image_root
        self.split = split

        self.input_image_size = 720
        self.output_image_size = 224
        self.unseen_objs = real_unseen_objs
        self.instruction = "detect {}"
        self.error_lines = [34, 160, 161, 162, 164, 165, 168, 197, 377, 379, 380, 381, 383, 384, 385, 386, 387, 470, 666]

        if self.split == "train":
            grasp_data = list(filter(lambda x: x["obj_name"] not in self.unseen_objs, self.grasp_data))
        else:
            raise ValueError

        self.grasp_data = []
        for i, d in enumerate(grasp_data):
            if i not in self.error_lines:
                self.grasp_data.append(d)

    def __len__(self):
        return len(self.grasp_data)

    def __getitem__(self, index):
        data = self.grasp_data[index]
        image_id, obj_name, grasp = data["image_id"], data["obj_name"], data["grasp_label"]
        bbox_pil = self.bbox_data[image_id][obj_name]
        image = Image.open(os.path.join(self.image_root, f"{image_id:04d}.jpg"))

        bbox = convert_bbox(bbox_pil)
        bbox = normalize_bbox(bbox, self.input_image_size)  # normalize to 0~1
        bbox_str = bbox_to_str(bbox)
        label = bbox_str

        prompt = self.instruction.format(obj_name)

        grasp = None
        bbox_image = None

        return image, prompt, label, grasp, obj_name, bbox_image


def mask_to_bbox_position(mask):
    """Get bbox position from boolean mask array. Return cv format bbox."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max, y_max)


def convert_bbox(bbox):
    """Convert bbox format between paligemma (row_min, col_min, row_max, col_max) and cv (col_min, row_min, col_max, row_max)."""
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def bbox_to_str(bbox, bins=1024):
    """Convert normalized bbox (0~1) to paligemma bbox tokens (<loc0000> to <loc1023>)."""
    result = [int(b * (bins - 1)) for b in bbox]
    result = [f"<loc{b:04d}>" for b in result]
    result = "".join(result)
    return result


def str_to_bbox(str, bins=1024):
    """
    Convert paligemma bbox tokens (`<loc0000>` to `<loc1023>`) to normalized bbox (0~1).
    Return None if token amount is not 4 or bbox invalid.
    """
    bbox = parse_token_str(str)
    if len(bbox) != 4 or not is_valid_bbox(bbox):
        return None
    bbox = [b / (bins - 1) for b in bbox]
    return bbox


def normalize_bbox(bbox, image_size=416):
    return [b / image_size for b in bbox]


def denormalize_bbox(bbox, image_size=416):
    return [int(b * image_size) for b in bbox]


def parse_token_str(str):
    tokens = re.findall(r"\d+", str)
    tokens = [int(i) for i in tokens]
    return tokens


def is_valid_bbox(bbox):
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def normalize_rect_grasp_with_angle_bins(grasp: torch.Tensor, image_size=416, angle_bins=18):
    # grasp[0:4] normalized to 0~1, grasp[4] converted to 0~17 (int)
    grasp[:4] /= image_size
    grasp[4] = math.floor(grasp[4] / (180 / angle_bins))
    if grasp[4] == 18:
        grasp[4] = 17
    return grasp


def denormalize_rect_grasp_with_angle_bins(grasp, image_size=416, angle_bins=18):
    result = [int(g * image_size) for g in grasp[:4]]
    result.append(float(grasp[4] * 180 / angle_bins))
    return result


def normalize_rect_grasp(grasp: torch.Tensor, image_size=416):
    # normalized all to 0~1
    grasp[:4] /= image_size
    grasp[4] /= 180
    return grasp


def denormalize_rect_grasp(grasp, image_size=416):
    result = [int(g * image_size) for g in grasp[:4]]
    result.append(float(grasp[4] * 180))
    return result


def rect_grasp_to_str(grasp, bins=1024, use_pretrained_token=True):
    # convert normalized grasp label to paligemma location token
    result = [int(b * (bins - 1)) for b in grasp]
    if use_pretrained_token:
        result = [f"<loc{b:04d}>" for b in result]
    else:
        result = [f"<pos{b:04d}>" for b in result]
    result = "".join(result)
    return result


def str_to_rect_grasp(grasp, bins=1024):
    result = parse_token_str(grasp)
    if len(result) != 5:
        return None
    result = [i / (bins - 1) for i in result]
    return result


def crop_bbox(pil_img, bbox_pos, min_half_size=50, edge_expand=15):
    """Visual sampler from Visual CoT."""
    width, height = pil_img.size
    x_min, y_min, x_max, y_max = bbox_pos
    if sum([x_min, y_min, x_max, y_max]) < 5:
        # if bbox pos is normalized
        x_min = x_min * max(width, height)
        y_min = y_min * max(width, height)
        x_max = x_max * max(width, height)
        y_max = y_max * max(width, height)
    if width > height:
        overlay = (width - height) // 2
        y_min = max(0, y_min - overlay)
        y_max = max(0, y_max - overlay)
    else:
        overlay = (height - width) // 2
        x_min = max(0, x_min - overlay)
        x_max = max(0, x_max - overlay)

    center_point = [(x_min + x_max) // 2, (y_min + y_max) // 2]
    half_sizes = [(x_max - x_min) // 2, (y_max - y_min) // 2]
    cropped_half_size = max(max(half_sizes) + edge_expand, min_half_size)
    upper_left_point = [center_point[0] - cropped_half_size, center_point[1] - cropped_half_size]
    if upper_left_point[0] < 0:
        center_point[0] += -upper_left_point[0]
    if upper_left_point[1] < 0:
        center_point[1] += -upper_left_point[1]
    lower_right_point = [center_point[0] + cropped_half_size, center_point[1] + cropped_half_size]
    if lower_right_point[0] > width:
        center_point[0] -= lower_right_point[0] - width
    if lower_right_point[1] > height:
        center_point[1] -= lower_right_point[1] - height

    cropped_region = [
        max(0, center_point[0] - cropped_half_size),
        max(0, center_point[1] - cropped_half_size),
        min(width, center_point[0] + cropped_half_size),
        min(height, center_point[1] + cropped_half_size),
    ]
    cropped_image = pil_img.crop(cropped_region)
    return cropped_image


def parse_grasp_data(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for image in root.findall("image"):
        image_id = int(image.get("id"))

        for obj in image.findall("*"):
            if obj.tag in ["box", "polygon"]:
                obj_name = obj.get("label")
                grasp_label = obj.get("grasp_label")
                grasp_label = [float(i) for i in grasp_label.split()]

                if grasp_label:
                    data.append(
                        {
                            "image_id": image_id,
                            "obj_name": obj_name,
                            "grasp_label": grasp_label,
                        }
                    )
    return data


def parse_bbox_data(xml_path):
    """return [{obj_name: bbox}], bbox in cv format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for image in root.findall("image"):
        image_id = int(image.get("id"))

        image_data = {}

        for box in image.findall("box"):
            obj_name = box.get("label")

            bbox = [
                float(box.get("xtl")),  # x_min (xtl)
                float(box.get("ytl")),  # y_min (ytl)
                float(box.get("xbr")),  # x_max (xbr)
                float(box.get("ybr")),  # y_max (ybr)
            ]

            image_data[obj_name] = bbox
        data.append(image_data)

    return data


real_unseen_objs = [
    "spirit",
    "banana",
    "mango",
    "corn",
    "potato",
    "red potato",
    "green pepper",
    "basketball",
    "tennis ball",
    "green cube",
    "yellow cube",
    "white bowl",
    "blue bowl",
    "green cup",
    "red cup",
]
