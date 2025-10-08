from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

from transformers import PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoModel
from .configuration_vcot_grasp import VCoTGraspConfig, ArchConfig
from .action_head import L1RegressionActionHead, DiffusionActionHead
from constants import *

if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)


@dataclass
class VCoTGraspOutputWithPast(ModelOutput):
    """
    Base class for PaliGemmacausal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_info: Optional[dict] = None
    logits: torch.FloatTensor = None
    actions: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class VCoTGraspImageProjector(nn.Module):
    def __init__(self, config: VCoTGraspConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


class VCoTGraspPreTrainedModel(PreTrainedModel):
    config_class = VCoTGraspConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VCoTGraspImageProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def init_weights(self):
        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(paligemma_model_id, cache_dir=pretrained_paligemma_dir)
        paligemma.language_model.resize_token_embeddings(self.vocab_size)

        lanuage_model_weights = paligemma.language_model.state_dict()
        image_encoder_weights = paligemma.vision_tower.state_dict()
        image_projector_weights = paligemma.multi_modal_projector.state_dict()

        self.language_model.load_state_dict(lanuage_model_weights)
        self.image_encoder.load_state_dict(image_encoder_weights, strict=False)
        self.image_projector.load_state_dict(image_projector_weights)

        if self.action_head_type == "LM_new":
            self.resize_token_embeddings(257154 + 1024, pad_to_multiple_of=8)

        self.tie_weights()

    def _init_weights(self, module):
        # important: this ported version of PaliGemma isn't meant for training from scratch - only
        # inference and fine-tuning
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else self.config.text_config.initializer_range

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VCoTGraspForConditionalGeneration(VCoTGraspPreTrainedModel, GenerationMixin):
    def __init__(self, config: VCoTGraspConfig):
        super().__init__(config)
        language_model = AutoModelForCausalLM.from_config(config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        self.image_encoder = AutoModel.from_config(config.vision_config)
        self.image_projector = VCoTGraspImageProjector(config)

        self.action_head_type = config.arch_config.action_head
        if self.action_head_type == "MLP":
            self.action_head = L1RegressionActionHead(
                num_blocks=1,
                input_dim=config.hidden_size,
                hidden_dim=256,
                action_dim=action_with_binned_angle_seq_len,
                output_dim=1,
            )
            self.action_seq_len = action_with_binned_angle_seq_len
        elif self.action_head_type == "Diffusion":
            self.action_head = DiffusionActionHead(
                token_size=5,
                model_type="DiT-S",
                in_channels=5,
                future_action_window_size=0,
                past_action_window_size=0,
            )
            self.action_seq_len = action_seq_len
        elif self.action_head_type in ["LM_pretrained", "LM_new"]:
            self.action_head = None
            self.action_seq_len = 0
        else:
            raise NotImplementedError("Action head not implemented!")

        self.vocab_size = self.config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id
        self.image_token_index = self.config.image_token_index
        self.dummy_action_token_index = self.config.dummy_action_token_index

        self.post_init()

    def set_trainable(self, *, image_encoder=False, image_projector=True, embeddings=True, lm=True, action_head=True, print_trainable=True):
        for param in self.image_encoder.parameters():
            param.requires_grad = image_encoder

        for param in self.image_projector.parameters():
            param.requires_grad = image_projector

        for name, param in self.language_model.named_parameters():
            if "embed_tokens" in name:
                # Output head doesn't exist in named_parameters, will have the same freeze state as embed_tokens.
                param.requires_grad = embeddings
            else:
                param.requires_grad = lm

        if self.action_head is not None:
            for param in self.action_head.parameters():
                param.requires_grad = action_head

        if print_trainable:
            print(f"amount of trainable param: {self.num_parameters(only_trainable=True)}")

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights with Llava->PaliGemma
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device)
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor):
        image_outputs = self.image_encoder(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.image_projector(selected_image_feature)
        image_features = image_features / (self.config.hidden_size**0.5)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_pixel_values: torch.FloatTensor = None,
        obj_name_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        grasp_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **lm_kwargs,
    ) -> Union[Tuple, VCoTGraspOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/PaliGemma-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/PaliGemma-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and vision.
        if image_pixel_values is not None:
            image_features = self.get_image_features(image_pixel_values)
            special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)  # [batch, input_seq_len, 1]
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)  # [batch, input_seq_len, hidden_size]
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        # mask eos token
        # if labels is not None and self.action_head_type not in ["LM_pretrained", "LM_new"]:
        #     labels = torch.where(input_ids == self.config.eos_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training)
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits

        actions = None
        action_tokens_mask = input_ids == self.dummy_action_token_index
        have_action = action_tokens_mask.any()
        if have_action:
            action_hidden_states = last_hidden_states[action_tokens_mask].reshape(
                -1, self.action_seq_len, self.config.text_config.hidden_size
            )  # [*, action_seq, hidden_size]
            if self.action_head_type == "MLP":
                actions = self.action_head(action_hidden_states).reshape(-1, self.action_seq_len)
            elif self.action_head_type == "Diffusion":
                # only get action when testing
                if not is_training:
                    actions = self.action_head.get_action(action_hidden_states).reshape(-1, self.action_seq_len)

        loss = 0
        loss_info = {}
        if labels is not None:
            if self.config.arch_config.use_bbox or self.action_head_type in ["LM_pretrained", "LM_new"]:
                # Token loss for bbox prediction.
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                if attention_mask is not None:
                    # we use the input attention mask to shift the logits and labels, because it is 2D.
                    # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                    shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                    shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
                else:
                    shift_logits = shift_logits.contiguous()
                    shift_labels = shift_labels.contiguous()
                # Flatten the tokens
                token_loss_fct = nn.CrossEntropyLoss()

                flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
                flat_labels = shift_labels.view(-1).to(shift_logits.device)
                token_loss = token_loss_fct(flat_logits, flat_labels)

                # We scale each loss to balance their gradient.
                # if self.action_head_type not in ["LM_pretrained", "LM_new"]:
                #     token_loss_scale = 1 / 5
                #     token_loss *= token_loss_scale

                loss += token_loss
                loss_info["token_loss"] = token_loss.detach()

            if self.action_head_type == "MLP" and have_action:
                # Action loss
                # position_loss_fct = nn.MSELoss()
                position_loss_fct = nn.L1Loss()
                angle_loss_fct = nn.CrossEntropyLoss()
                angle_label_pos = 4
                position, angle = actions[:, :angle_label_pos], actions[:, angle_label_pos:]
                position_labels, angle_labels = grasp_labels[:, :angle_label_pos], grasp_labels[:, angle_label_pos]
                angle_labels = angle_labels.long()

                action_position_loss = position_loss_fct(position, position_labels)
                action_angle_loss = angle_loss_fct(angle, angle_labels)
                position_loss_scale = 1
                angle_loss_scale = 1 / 10
                # position_loss_scale = 10
                # angle_loss_scale = 1 / 4
                action_loss = position_loss_scale * action_position_loss + angle_loss_scale * action_angle_loss

                loss += action_loss
                loss_info["action_position_loss"] = action_position_loss.detach()
                loss_info["action_angle_loss"] = action_angle_loss.detach()

            elif self.action_head_type == "Diffusion" and have_action:
                repeated_diffusion_steps = 8
                grasp_labels_repeated = grasp_labels.unsqueeze(1).repeat(repeated_diffusion_steps, 1, 1)  # [B, 5] -> [B, 1, 5]
                action_hidden_states = action_hidden_states.permute(0, 2, 1).mean(dim=1, keepdim=True)  # [B, 5, 2304] -> [B, 1, 5]
                action_repeated = action_hidden_states.repeat(repeated_diffusion_steps, 1, 1)
                action_loss = self.action_head.loss(grasp_labels_repeated, action_repeated)
                loss += action_loss
                loss_info["action_loss"] = action_loss.detach()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VCoTGraspOutputWithPast(
            loss=loss,
            loss_info=loss_info,
            logits=logits,
            actions=actions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if image_pixel_values is not None else None,
        )

    # called by model.generate()
    def prepare_inputs_for_generation(
        self,
        input_ids,
        image_pixel_values=None,
        obj_name_ids=None,
        bbox_image_pixel_values=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["image_pixel_values"] = image_pixel_values
            model_inputs["obj_name_ids"] = obj_name_ids
            model_inputs["bbox_image_pixel_values"] = bbox_image_pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs
