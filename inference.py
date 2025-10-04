import cv2
import torch
from peft import PeftModel

from model import VCoTGraspForConditionalGeneration, VCoTGraspProcessor, ArchConfig, VCoTGraspConfig
from data import *


class VCoTGraspInferencer:
    def __init__(
        self,
        checkpoint_dir,
        use_bbox,
        action_head,
        input_image_size=416,
        device="cuda",
        use_lora=False,
    ):
        if not use_lora:
            self.model = VCoTGraspForConditionalGeneration.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16).to(device)
        else:
            config = VCoTGraspConfig.from_json_file("model/config.json")
            config.arch_config = ArchConfig(use_bbox=use_bbox, action_head=action_head)
            # disable flash attention on v100
            config._attn_implementation = "sdpa"
            model = VCoTGraspForConditionalGeneration(config)
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            self.model = model.to(torch.bfloat16).to(device)
        arch_config = ArchConfig(use_bbox=use_bbox, action_head=action_head)
        self.processor = VCoTGraspProcessor(arch_config)
        self.device = device

        self.use_bbox = use_bbox
        self.action_head = action_head
        self.input_image_size = input_image_size

    def get_bbox_prompt(self, obj_name):
        return f"detect {obj_name}"

    def get_grasp_prompt(self, obj_name):
        return f"grasp the {obj_name}"

    @torch.inference_mode
    def generate_until_eos(self, image, prompt, obj_name, bbox_image=None):
        """generate text until <eos> and remove <eos>"""
        model_inputs = self.processor(
            texts=[prompt],
            images=[image],
            obj_names=[obj_name],
            bbox_images=[bbox_image],
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=False)

        # remove <eos>
        decoded = decoded[:-5]
        print("generated: ", decoded)
        return decoded

    @torch.inference_mode
    def generate_n_tokens(self, image, prompt, obj_name, bbox_image=None, n=4):
        """generate n tokens"""
        model_inputs = self.processor(
            texts=[prompt],
            images=[image],
            obj_names=[obj_name],
            bbox_images=[bbox_image],
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation = self.model.generate(**model_inputs, max_new_tokens=n, do_sample=False)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=False)

        return decoded

    def generate_bbox(self, image, obj_name):
        bbox_prompt = self.get_bbox_prompt(obj_name)
        generated_bbox = self.generate_n_tokens(image, bbox_prompt, obj_name, n=4)
        bbox = str_to_bbox(generated_bbox)
        if bbox is None:
            return None
        bbox = denormalize_bbox(bbox)
        bbox = convert_bbox(bbox)
        return bbox

    @torch.inference_mode
    def generate_tokens_with_bbox(self, image, obj_name):
        bbox = self.generate_bbox(image, obj_name)
        if bbox is None:
            return None, None
        bbox_image = crop_bbox(image, bbox)
        grasp_prompt = self.get_grasp_prompt(obj_name)
        generated_actions = self.generate_n_tokens(image, grasp_prompt, obj_name, bbox_image=bbox_image, n=5)
        print("generated_action:", generated_actions)

        return generated_actions, bbox

    @torch.inference_mode
    def generate_actions_with_bbox(self, image, obj_name, action_seq_len):
        bbox = self.generate_bbox(image, obj_name)
        if bbox is None:
            return None, None
        bbox_image = crop_bbox(image, bbox)
        grasp_prompt = self.get_grasp_prompt(obj_name) + "<action>" * action_seq_len
        model_inputs = self.processor(
            texts=[grasp_prompt],
            images=[image],
            obj_names=[obj_name],
            bbox_images=[bbox_image],
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        outputs = self.model(**model_inputs, use_cache=False)
        actions = outputs.actions.squeeze()

        return actions, bbox

    @torch.inference_mode
    def generate_actions_without_bbox(self, image, obj_name, action_seq_len):
        grasp_prompt = self.get_grasp_prompt(obj_name) + "<action>" * action_seq_len
        model_inputs = self.processor(
            texts=[grasp_prompt],
            images=[image],
            obj_names=[obj_name],
            bbox_images=[None],
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        outputs = self.model(**model_inputs, use_cache=False)
        actions = outputs.actions.squeeze()

        return actions

    def generate_postprocess(self, image, prompt, obj_name):
        # generate and postprocess, return normalized bboxes and actions
        if self.action_head == "MLP":
            if self.use_bbox:
                actions, bbox = self.generate_actions_with_bbox(image, obj_name, action_seq_len=action_with_binned_angle_seq_len)
                if actions is None or bbox is None:
                    return None, None
                actions = actions.clone()
                actions[4] = torch.argmax(actions[4:])
                actions = actions[:5]
                actions = denormalize_rect_grasp_with_angle_bins(actions)
            else:
                bbox = None
                actions = self.generate_actions_without_bbox(image, obj_name, action_seq_len=action_with_binned_angle_seq_len)
                actions = actions.clone()
                actions[4] = torch.argmax(actions[4:])
                actions = actions[:5]
                actions = denormalize_rect_grasp_with_angle_bins(actions)

        elif self.action_head == "Diffusion":
            if self.use_bbox:
                actions, bbox = self.generate_actions_with_bbox(image, obj_name, action_seq_len=action_seq_len)
                if actions is None or bbox is None:
                    return None, None
                actions = denormalize_rect_grasp(actions)
            else:
                bbox = None
                actions = self.generate_actions_without_bbox(image, obj_name, action_seq_len=action_seq_len)
                actions = denormalize_rect_grasp(actions)

        elif self.action_head in ["LM_pretrained", "LM_new"]:
            if self.use_bbox:
                actions, bbox = self.generate_tokens_with_bbox(image, obj_name)
                if actions is None or bbox is None:
                    return None, None
                actions = str_to_rect_grasp(actions)
                if actions is not None:
                    actions = denormalize_rect_grasp(actions)
            else:
                bbox = None
                actions = self.generate_n_tokens(image, prompt, obj_name, n=5)
                actions = str_to_rect_grasp(actions)
                if actions is not None:
                    actions = denormalize_rect_grasp(actions)
        else:
            raise ValueError("Invalid action head type")

        return actions, bbox


def is_angle_within_threshold(angle1, angle2, threshold=30):
    """
    Check if two angles are within the threshold
    """
    diff = abs(angle1 - angle2)
    diff = min(diff, 180 - diff)  # 处理角度环绕问题(如179度和1度实际上只差2度)
    return diff <= threshold


def calculate_iou(grasp1, grasp2):
    """
    Calculate IoU of two rotated rectangles
    """
    rect1 = ((grasp1[0], grasp1[1]), (grasp1[2], grasp1[3]), grasp1[4])
    rect2 = ((grasp2[0], grasp2[1]), (grasp2[2], grasp2[3]), grasp2[4])

    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)

    intersection, _ = cv2.intersectConvexConvex(box1, box2)

    area1 = grasp1[2] * grasp1[3]
    area2 = grasp2[2] * grasp2[3]

    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    return iou


def eval_grasp(pred, label, iou_threshold, angle_threshold):
    """
    Evaluate if the predicted grasp is correct based on IoU and angle difference
    """
    iou = calculate_iou(pred, label)
    angle_ok = is_angle_within_threshold(pred[4], label[4], angle_threshold)
    is_correct = (iou >= iou_threshold) and angle_ok

    return is_correct


def eval_grasp_all_labels(pred, labels, iou_threshold, angle_threshold):
    for label in labels:
        if eval_grasp(pred, label, iou_threshold, angle_threshold):
            return True

    return False
