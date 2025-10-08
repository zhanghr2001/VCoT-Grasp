import os
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm

from model import VCoTGraspConfig, VCoTGraspForConditionalGeneration, VCoTGraspProcessor
from data import *
from constants import *

from inference import *
from planar_vis import draw_grasp_rectangle


image_path = "assets/demo.jpg"
obj_name = "apple"
prompt = f"grasp the {obj_name}"

visualize_path = "demo_visualize.jpg"
input_image_size = 416

load_checkpoint_dir = "checkpoints/vcot"
use_bbox = True
action_head = "MLP"
device = "cuda:0"

inferencer = VCoTGraspInferencer(
    load_checkpoint_dir,
    use_bbox=use_bbox,
    action_head=action_head,
    input_image_size=input_image_size,
    device=device,
    use_lora=False,
)

image = Image.open(image_path)
if image.size[0] != image.size[1]:
    print("Warning: image is not square, applying cropping")
    image = image.crop((0, 0, min(image.size), min(image.size)))
image = image.resize((input_image_size, input_image_size))

grasp_pred, bbox_pred = inferencer.generate_postprocess(image, prompt, obj_name)
if use_bbox:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_pred, outline="yellow", width=3)
image = draw_grasp_rectangle(image, grasp_pred, "red")
image.save(visualize_path)
