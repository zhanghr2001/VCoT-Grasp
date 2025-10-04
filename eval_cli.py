import os
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm

from model import VCoTGraspConfig, VCoTGraspForConditionalGeneration, VCoTGraspProcessor
from data import *
from constants import *

from inference import *
from planar_vis import draw_grasp_rectangle


def eval(load_checkpoint_dir, test_split, use_bbox, action_head, device, visualize_dir, result_dir, use_lora=False):
    inferencer = VCoTGraspInferencer(
        load_checkpoint_dir,
        use_bbox=use_bbox,
        action_head=action_head,
        device=device,
        use_lora=use_lora,
    )

    if test_split == "seen":
        csv_path = grasp_anything_planar_grasp_test_seen_csv_path
        visualize_dir = os.path.join(visualize_dir, "seen")
    elif test_split == "unseen":
        csv_path = grasp_anything_planar_grasp_test_unseen_csv_path
        visualize_dir = os.path.join(visualize_dir, "unseen")
    else:
        raise ValueError
    os.makedirs(visualize_dir, exist_ok=True)

    ds = GraspAnythingForGraspGeneration(
        csv_path,
        grasp_anything_rgb_root,
        grasp_anything_planar_grasp_root,
        grasp_anything_mask_root,
        use_bbox=use_bbox,
        action_head_type=action_head,
    )

    image_size = 416
    save_image_range = 100
    IOU_THRESHOLD = 0.25
    ANGLE_THRESHOLD = 30

    n_success = 0
    n_valid = 0
    n_all = len(ds)
    for i in tqdm(range(len(ds))):
        image, prompt, _, _, obj_name, _ = ds[i]
        grasp_id, obj_name, scene_description = ds.data.iloc[i]
        mask = np.load(os.path.join(ds.mask_root, grasp_id + ".npy"))
        bbox_gt = mask_to_bbox_position(mask)

        grasp_gts = torch.load(os.path.join(ds.grasp_root, grasp_id + ".pt"))
        grasp_gts = grasp_gts[:, 1:].tolist()

        grasp_pred, bbox_pred = inferencer.generate_postprocess(image, prompt, obj_name)

        if grasp_pred is None or (use_bbox and bbox_pred is None):
            continue

        success = eval_grasp_all_labels(grasp_pred, grasp_gts, IOU_THRESHOLD, ANGLE_THRESHOLD)
        n_success += int(success)
        n_valid += 1

        if i < save_image_range:
            if use_bbox:
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox_pred, outline="yellow", width=3)
                draw.rectangle(bbox_gt, outline="blue", width=3)
            for grasp_gt in grasp_gts:
                image = draw_grasp_rectangle(image, grasp_gt, "green")
            image = draw_grasp_rectangle(image, grasp_pred, "red")
            save_path = os.path.join(visualize_dir, f"{i:02d}_{success}_{obj_name}.jpg")
            image.save(save_path)

    res = (
        f"checkpoint: {load_checkpoint_dir}\n"
        f"test split: {test_split}\n"
        f"num success: {n_success}, num valid: {n_valid}, num all: {n_all}\n"
        f"success rate: {n_success / n_all * 100} %\n"
    )

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{test_split}.txt")
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--load-checkpoint-dir", type=str, help="finished checkpoint dir")
    parser.add_argument("--test-split", type=str, help="all, seen, unseen")

    # architecture choice
    parser.add_argument("--use-bbox", action="store_true")
    parser.add_argument("--action-head", type=str, help="MLP, Diffusion, LM_pretrained, LM_new")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize-dir", type=str)
    parser.add_argument("--result-dir", type=str)

    parser.add_argument("--use-lora", action="store_true")

    args = parser.parse_args()

    if args.test_split == "all":
        eval(args.load_checkpoint_dir, "seen", args.use_bbox, args.action_head, args.device, args.visualize_dir, args.result_dir, args.use_lora)
        eval(args.load_checkpoint_dir, "unseen", args.use_bbox, args.action_head, args.device, args.visualize_dir, args.result_dir, args.use_lora)
    else:
        eval(args.load_checkpoint_dir, args.test_split, args.use_bbox, args.action_head, args.device, args.visualize_dir, args.result_dir, args.use_lora)
