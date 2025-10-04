from typing import List, Optional, Union
from PIL import Image
import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import (
    TextKwargs,
    ImagesKwargs,
    CommonKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from transformers.utils import logging

from transformers import PaliGemmaProcessor
import torch
from constants import paligemma_model_id, pretrained_paligemma_dir

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"
DUMMY_ACTION_TOKEN = "<action>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]
POS_TOKENS = [f"<pos{i:0>4}>" for i in range(1024)]


class VCoTGraspTextKwargs(TextKwargs):
    suffix: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    pad_to_multiple_of: Optional[int]


class VCoTGraspImagesKwargs(ImagesKwargs):
    do_convert_rgb: Optional[bool]


class VCoTGraspProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: VCoTGraspTextKwargs
    images_kwargs: VCoTGraspImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


class VCoTGraspProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, arch_config):
        paligemma_processor = PaliGemmaProcessor.from_pretrained(paligemma_model_id, cache_dir=pretrained_paligemma_dir)

        self.image_processor = paligemma_processor.image_processor
        self.tokenizer = paligemma_processor.tokenizer

        dummy_action_token = AddedToken(DUMMY_ACTION_TOKEN, normalized=False, special=True)
        special_tokens_to_add = {"additional_special_tokens": [dummy_action_token]}
        self.tokenizer.add_special_tokens(special_tokens_to_add)
        if arch_config.action_head == "LM_new":
            self.tokenizer.add_tokens(POS_TOKENS)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False

        self.image_token_id = paligemma_processor.image_token_id
        self.dummy_action_token_id = self.tokenizer.convert_tokens_to_ids(DUMMY_ACTION_TOKEN)
        self.image_seq_length = paligemma_processor.image_seq_length
        self.image_size = self.image_processor.size

        super().__init__()

    def __call__(
        self,
        texts: List[str] = None,
        images: List[Image.Image] = None,
        obj_names: List[str] = None,
        bbox_images: List[Image.Image] = None,
        **kwargs: Unpack[VCoTGraspProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            VCoTGraspProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        suffix = output_kwargs["text_kwargs"].pop("suffix", None)
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

        return_token_type_ids = True if suffix is not None else False

        # if have bbox image, image seq len is 2 * 256, else 256
        image_seq_len = [self.image_seq_length if bbox_image is None else 2 * self.image_seq_length for bbox_image in bbox_images]
        input_strings = [
            self.build_string_from_input_with_cropped_bbox(
                prompt=prompt,
                bos_token="<bos>",
                image_seq_len=seq_len,
                image_token=IMAGE_TOKEN,
            )
            for prompt, seq_len in zip(texts, image_seq_len)
        ]

        all_images = []
        for image, bbox_image in zip(images, bbox_images):
            all_images.append(image)
            if bbox_image is not None:
                all_images.append(bbox_image)

        image_pixel_values = self.image_processor(all_images, **output_kwargs["images_kwargs"])["pixel_values"]

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length + self.depth_seq_length + self.point_seq_length

        text_inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        return_data = {
            **text_inputs,
            "image_pixel_values": image_pixel_values,
        }

        if obj_names is not None:
            obj_name_ids = self.tokenizer(obj_names, padding="longest")
            obj_name_ids = torch.tensor(obj_name_ids["input_ids"], dtype=torch.long)
            return_data["obj_name_ids"] = obj_name_ids

        if return_token_type_ids:
            labels = text_inputs["input_ids"].masked_fill(
                text_inputs["token_type_ids"] == 0, -100
            )  # prefix token ids are replaced with -100, will be ignored in nn.CrossEntropyLoss()
            return_data.update({"labels": labels})

        return BatchFeature(data=return_data)

    def build_string_from_input(self, prompt, bos_token, image_seq_len, image_token):
        """
        Builds a string from the input prompt and image tokens.
        For example, for the call:
        build_string_from_input(
            prompt="Prefix str"
            bos_token="<s>",
            image_seq_len=3,
            image_token="<im>",
        )
        The output will be:
        "<im><im><im><s>Initial str"
        Args:
            prompt (`List[Union[str, ImageInput]]`): The input prompt.
            bos_token (`str`): The beginning of sentence token.
            image_seq_len (`int`): The length of the image sequence.
            image_token (`str`): The image token.
            num_images (`int`): Number of images in the prompt.
        """
        return f"{image_token * image_seq_len}{bos_token}{prompt}\n"

    def build_string_from_input_with_cropped_bbox(self, prompt, bos_token, image_seq_len, image_token):
        r"""Remove \n here and add \n in Dataset.__getitem__ for compatible with inference."""
        return f"{image_token * image_seq_len}{bos_token}{prompt}\n"

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
