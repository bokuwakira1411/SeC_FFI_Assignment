# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import time
import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_sec import SeCConfig
from .sam2.utils.misc import fill_holes_in_mask_scores
from .modeling_intern_vit import InternVisionModel, has_flash_attn

from .sam2_video_predictor import build_sam2_video_predictor, SAM2VideoPredictor
from .templates import PROMPT_TEMPLATE

import cv2
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
import copy
import random
random.seed(42)
try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class SeCModel(PreTrainedModel):
    config_class = SeCConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: SeCConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.template = self.template.replace('-', '_')
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'sdpa'  # patched: flash_attn not installed

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        apply_postprocessing = getattr(config, 'apply_postprocessing', True)
        hydra_overrides_extra = getattr(config, 'hydra_overrides_extra', [])
        grounding_maskmem_num = getattr(config, 'grounding_maskmem_num', 22)
        
        # Compatible with previous versions
        if config.grounding_encoder_config == "longsam2.1/longsam2.1_hiera_l.yaml":
            grounding_encoder_config = "sec_sam2.1_hiera_l.yaml"
        else:
            grounding_encoder_config = config.grounding_encoder_config
        
        self.grounding_encoder = build_sam2_video_predictor(
            grounding_encoder_config,
            num_maskmem=grounding_maskmem_num,
            apply_postprocessing=apply_postprocessing,
            hydra_overrides_extra=hydra_overrides_extra
        )
        self.grounding_encoder.token_attn = copy.deepcopy(self.grounding_encoder.memory_attention)

        in_dim = llm_hidden_size
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.init_prediction_config = False

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error.
        input_embeds = self.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        fast_vit_embeds = None

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape='
                    f'{input_embeds[selected].shape}, '
                    f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            if n_token > len(vit_embeds):
                print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
                expand_ratio = n_token // len(vit_embeds) + 1
                vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

            input_embeds[selected] = vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        if labels is None:  # inference: only hidden states needed, skip lm_head logits (saves ~2-3 GB)
            base = self.language_model.model(
                inputs_embeds=input_embeds, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False,
                output_hidden_states=True, return_dict=True)
            return CausalLMOutputWithPast(hidden_states=base.hidden_states)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device
        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)

                vit_embeds = self.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0

            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_size = patch_size

        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # change phi3 prepare for generation fuction
        if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_context_token_id = img_context_token_id
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        return
    
    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        init_mask=None,
        tokenizer=None,
        mllm_memory_size=7,
        exp_ctx=None,
    ):
        # exp_ctx (experiments, see CHANGES.md): None = original behaviour. Otherwise a dict with
        #   hsv_thresh (float), concept_mode ('default'|'off'|'always'|'swap'),
        #   swap_concepts (list of saved language_embd tensors, for 'swap'),
        #   rows (list, per-frame log rows are appended), concept_dir (save language_embd per trigger),
        #   video, obj (names used in rows / concept file names),
        #   bank_rule ('default': MLLM memory updated on frames where the concept was computed |
        #              'hsv': only on frames with an HSV trigger, even if the concept is computed more often),
        #   bank_extra_gap (int, 0 = off): also add other tracked frames that pass the same test
        #              (mask not empty, object score > 1) if >= gap frames after the last added frame.
        #   oracle_keyframes (list of (frame_idx, mask)): put into the MLLM memory from the start, after the
        #              init frame (e.g. a later frame with its GT mask),
        #   force_trigger (set of frame_idx): trigger the concept update on these frames whatever hsv_dist,
        #   bank_score_thresh (float, default 1): object score a frame needs to enter the MLLM memory.
        #   concept_every (int, 0 = off): also compute the concept on frames with frame_idx % N == 0.
        #   burst_after_trigger (int K, 0 = off): also compute the concept on the K tracked frames after each HSV
        #              trigger (a new trigger during a burst restarts the count),
        #   concept_every_sparse (int P, 0 = off): outside bursts, also compute the concept when the last
        #              concept call (or the start frame) is >= P frames back.
        #   trigger_mode ('hsv' | 'unc' | 'hsv_or_unc'): 'unc' ignores HSV for the concept call (HSV still drives
        #              bank_rule='hsv'); uncertainty triggers (any enabled signal, decided from frames < f):
        #   unc_logit_drop (D, None = off): object score logit of f-1 < mean of the w frames before it - D,
        #   unc_area_drop (r, None = off): mask area of f-1 < (1 - r) * mean of the w frames before it,
        #   unc_empty_n (N, 0 = off): mask empty on frames f-N..f-1,
        #   unc_window (w, default 5), unc_cooldown (C): no uncertainty call if the last concept call is <= C back.
        #   bank_gt_gap (int, 0 = off; oracle): like bank_extra_gap, but the frame is added with its GT mask
        #              (gt_mask_fn(frame_idx) -> bool mask or None), skipped if the GT is missing / empty; no
        #              mask / object-score test.
        #   bank_diverse_tau (float, None = off): replaces the bank rule. Candidate = tracked frame with mask not
        #              empty and object score > bank_score_thresh, if HSV trigger or >= bank_diverse_gap (5) frames
        #              after the last candidate. Descriptor = Hiera feature (current_vision_feats[-1]) averaged
        #              in the mask, L2-normalised. Added if max cosine to the bank < tau; when the bank holds
        #              mllm_memory_size frames, the frame (not the start frame) most similar to it is dropped.
        #   concept_only_lost (bool, E17): "lost" = mask empty or object score logit <= 0 on the last lost_n (2)
        #              tracked frames (the start frame does not count). On a concept-call frame while lost, the
        #              decoder also runs on the concept branch alone (token_attn output, no (a+b)/2); that mask is
        #              kept if not empty, pred IoU >= concept_only_iou (0.78) and object score > concept_only_logit
        #              (1), and the frame is then added to the MLLM memory (bank_reason added_concept) if not
        #              added by the bank rule; otherwise the original output is kept. gt_mask_fn (if given) is
        #              only used for logging J of both masks.
        #   e17_commit ('full' default = E17a; E18): where a kept concept mask goes. 'pixel': output and pixel
        #              memory, not added to the MLLM memory directly (the bank rule still applies); 'output': only
        #              the scored output (yielded mask), the tracking state and bank stay those of the mix output.
        #   bank_confirm_k (int k, 0 = off; E18b, with e17_commit 'pixel'): a kept concept mask's frame (if not
        #              added by the bank rule) is added to the MLLM memory only after the next k tracked frames all
        #              have a mask not empty, pred IoU >= concept_only_iou and object score > concept_only_logit;
        #              otherwise it is dropped. Logged in column bank_confirm (pend / add / drop:<frame>).
        #   fusion_w (float, default 1.0; E17c): with concept_only_lost, the decoder runs on
        #              f_w = (1 - w) a + w b (a = memory, b = concept) instead of b alone; w = 0.5 = original.
        #   fusion_sweep (list of w, None = off; E17c offline): with concept_only_lost, the decoder runs for every
        #              w on concept-call frames while lost (columns sw<100 w>_*), the output is never changed.
        #   lost_call_every (int K, 0 = off; E17b): while lost, also compute the concept when the last concept
        #              call is >= K frames back (concept_reason 'lost').
        #   log_feat_norms (bool): log the norms of a, b, (a+b)/2 and cos(a, b) on concept-call frames.
        hsv_thresh = 0.35 if exp_ctx is None else exp_ctx.get("hsv_thresh", 0.35)
        concept_mode = "default" if exp_ctx is None else exp_ctx.get("concept_mode", "default")
        exp_rows = None if exp_ctx is None else exp_ctx.get("rows")
        n_triggers = 0
        bank_rule = "default" if exp_ctx is None else exp_ctx.get("bank_rule", "default")
        bank_gap = 0 if exp_ctx is None else exp_ctx.get("bank_extra_gap", 0)
        force_trigger = set() if exp_ctx is None else set(exp_ctx.get("force_trigger") or ())
        bank_score_thresh = 1 if exp_ctx is None else exp_ctx.get("bank_score_thresh", 1)
        concept_every = 0 if exp_ctx is None else exp_ctx.get("concept_every", 0)
        burst_after = 0 if exp_ctx is None else exp_ctx.get("burst_after_trigger", 0)
        sparse_every = 0 if exp_ctx is None else exp_ctx.get("concept_every_sparse", 0)
        trigger_mode = "hsv" if exp_ctx is None else exp_ctx.get("trigger_mode", "hsv")
        unc_D = None if exp_ctx is None else exp_ctx.get("unc_logit_drop")
        unc_r = None if exp_ctx is None else exp_ctx.get("unc_area_drop")
        unc_N = 0 if exp_ctx is None else exp_ctx.get("unc_empty_n", 0)
        unc_w = 5 if exp_ctx is None else exp_ctx.get("unc_window", 5)
        unc_C = 0 if exp_ctx is None else exp_ctx.get("unc_cooldown", 0)
        unc_on = trigger_mode != "hsv"
        bank_gt_gap = 0 if exp_ctx is None else exp_ctx.get("bank_gt_gap", 0)
        div_tau = None if exp_ctx is None else exp_ctx.get("bank_diverse_tau")
        div_gap = 5 if exp_ctx is None else exp_ctx.get("bank_diverse_gap", 5)
        e17_on = False if exp_ctx is None else exp_ctx.get("concept_only_lost", False)
        lost_n = 2 if exp_ctx is None else exp_ctx.get("lost_n", 2)
        lost_every = 0 if exp_ctx is None else exp_ctx.get("lost_call_every", 0)
        log_norms = False if exp_ctx is None else exp_ctx.get("log_feat_norms", False)
        _lost_hist = []  # per tracked frame: mask empty or object score logit <= 0 (only filled when e17_on)
        # read by the grounding encoder's _track_step (None = original behaviour)
        self.grounding_encoder._exp_e17 = None if not (e17_on or log_norms) else dict(
            iou_thresh=exp_ctx.get("concept_only_iou", 0.78), logit_thresh=exp_ctx.get("concept_only_logit", 1.0),
            log_norms=log_norms, fusion_w=exp_ctx.get("fusion_w", 1.0), sweep_ws=exp_ctx.get("fusion_sweep"),
            commit=exp_ctx.get("e17_commit", "full"))
        e17_commit = "full" if exp_ctx is None else exp_ctx.get("e17_commit", "full")
        confirm_k = 0 if exp_ctx is None else exp_ctx.get("bank_confirm_k", 0)
        _pending = []  # bank_confirm_k: [frame, mask, frames left to confirm]
        _unc_logit, _unc_area = [], []  # per processed frame (incl. cond frame), only filled when unc_on
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)
        
        """Propagate the input points across frames to track in the entire video."""
        self.grounding_encoder.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        video_paths = inference_state["video_paths"]

        batch_size = self.grounding_encoder._get_obj_num(inference_state)

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.grounding_encoder.clear_non_cond_mem_around_input and (
            self.grounding_encoder.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)


        _burst_left, _last_call = 0, start_frame_idx  # burst_after_trigger / concept_every_sparse state
        mllm_memory = [(start_frame_idx, Image.open(video_paths[start_frame_idx]).convert('RGB'), init_mask)]
        for k, k_mask in ([] if exp_ctx is None else exp_ctx.get("oracle_keyframes") or []):
            mllm_memory.append((k, Image.open(video_paths[k]).convert('RGB'), k_mask))
        if div_tau is not None:
            # bank_diverse_tau: one descriptor per mllm_memory entry; the bank is kept at <= mllm_memory_size
            assert len(mllm_memory) == 1, "bank_diverse_tau does not support oracle_keyframes"
            _vf = self.grounding_encoder._get_image_feature(inference_state, start_frame_idx, batch_size)[2][-1]
            _div_desc = [mask_pooled_descriptor(_vf, init_mask)]
            _div_last_cand = start_frame_idx
        # experiments: frames used by the SAM 2 pixel memory (filled by _prepare_memory_conditioned_features)
        self.grounding_encoder._exp_mem_frames = [] if exp_rows is not None else None

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            _update_flag = False
            _hsv_trig = False
            _tracked = False
            if exp_rows is not None:
                torch.cuda.synchronize()
                _t_frame = time.perf_counter()
                _row = {"video": exp_ctx.get("video"), "obj": exp_ctx.get("obj"), "frame": frame_idx,
                        "is_cond": 1, "hsv_dist": "", "triggered": 0, "mllm_memory_len": len(mllm_memory),
                        "n_ctx_frames": 0, "predict_forward_ms": "", "ctx_frames": ""}
                self.grounding_encoder._exp_mem_frames.clear()
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self.grounding_encoder._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                # language_embd = None
                inference_params = {
                    "inference_state": inference_state,
                    "output_dict": output_dict,
                    "frame_idx": frame_idx,
                    "batch_size": batch_size,
                    "is_init_cond_frame": False,
                    "point_inputs": None,
                    "mask_inputs": None,
                    "reverse": reverse,
                    "run_mem_encoder": True,
                    "start_frame_idx": start_frame_idx,
                }
                
                current_img = Image.open(video_paths[frame_idx]).convert('RGB')
                last_img = Image.open(video_paths[frame_idx-1]).convert('RGB')
                hsv_dist = hsv_distance(current_img, last_img)
                _hsv_trig = bool(hsv_dist > hsv_thresh)
                _tracked = True
                flags = [hsv_dist > hsv_thresh]  # == is_scene_change_hsv(current_img, last_img) by default
                if trigger_mode == "unc":
                    flags = [False]
                if concept_mode == "off":
                    flags = [False]
                elif concept_mode == "always":
                    flags = [True]
                if frame_idx in force_trigger:
                    flags = [True]
                elif concept_every > 0 and frame_idx % concept_every == 0:
                    flags = [True]
                _reason = "hsv" if flags == [True] and _hsv_trig else ("other" if flags == [True] else "")
                if unc_on and flags != [True] and frame_idx - _last_call > unc_C and _unc_area:
                    _j = len(_unc_area) - 1
                    _lo = max(0, _j - unc_w)
                    _why = ""
                    if unc_N and _j - unc_N + 1 >= 0 and not any(_unc_area[_j - unc_N + 1:]):
                        _why = "unc_empty"
                    elif unc_D is not None and _j > _lo and _unc_logit[_j] < np.mean(_unc_logit[_lo:_j]) - unc_D:
                        _why = "unc_logit"
                    elif unc_r is not None and _j > _lo and np.mean(_unc_area[_lo:_j]) > 0 and \
                            _unc_area[_j] < (1 - unc_r) * np.mean(_unc_area[_lo:_j]):
                        _why = "unc_area"
                    if _why:
                        flags, _reason = [True], _why
                if _hsv_trig:
                    _burst_left = burst_after
                elif _burst_left > 0:
                    _burst_left -= 1
                    if flags != [True]:
                        flags, _reason = [True], "burst"
                elif sparse_every > 0 and frame_idx - _last_call >= sparse_every and flags != [True]:
                    flags, _reason = [True], "sparse"
                _lost = e17_on and len(_lost_hist) >= lost_n and all(_lost_hist[-lost_n:])
                if _lost and lost_every > 0 and flags != [True] and frame_idx - _last_call >= lost_every:
                    flags, _reason = [True], "lost"
                if flags == [True]:
                    _last_call = frame_idx
                if len(mllm_memory) > mllm_memory_size:
                    _mllm_memory = [mllm_memory[0]] + mllm_memory[-(mllm_memory_size-1):]
                else:
                    _mllm_memory = mllm_memory
                if exp_rows is not None:
                    _row.update(is_cond=0, hsv_dist=round(float(hsv_dist), 5))

                if False in flags:
                    _update_flag = False
                    language_embd = None
                elif concept_mode == "swap":
                    # concept taken from another object (fixed mapping); the MLLM is not called
                    _update_flag = True
                    swap = exp_ctx["swap_concepts"]
                    language_embd = swap[n_triggers % len(swap)].to(self.device)
                    n_triggers += 1
                else:
                    _update_flag = True
                    video = [label_img_with_mask(img, mask) for _, img, mask in _mllm_memory]
                    video.append(current_img)
                    text = "<image>Please segment the object in the last frame based on the object labeled in the first several images."
                    if exp_rows is not None:
                        torch.cuda.synchronize()
                        _t_pf = time.perf_counter()
                    specific_language_embd = self.predict_forward(video=video, text=text)
                    language_embd = specific_language_embd.unsqueeze(0)
                    if exp_rows is not None:
                        torch.cuda.synchronize()
                        _row["predict_forward_ms"] = round((time.perf_counter() - _t_pf) * 1000, 2)
                    if exp_ctx is not None and exp_ctx.get("concept_dir"):
                        torch.save(language_embd.detach().cpu(), os.path.join(
                            exp_ctx["concept_dir"], f"{exp_ctx['video']}_{exp_ctx['obj']}_{frame_idx:05d}.pt"))
                    n_triggers += 1
                if exp_rows is not None and _update_flag:
                    _row.update(triggered=1, n_ctx_frames=len(_mllm_memory), concept_reason=_reason,
                                ctx_frames=" ".join(str(f) for f, _, _ in _mllm_memory))


                _e17 = self.grounding_encoder._exp_e17
                if _e17 is not None:
                    _e17["lost"] = _lost
                    _e17.pop("out", None)
                    _e17.pop("norms", None)
                current_out, pred_masks = self.grounding_encoder._run_single_frame_inference(
                    **inference_params, language_embd=language_embd
                )
                # optionally offload the output to CPU memory to save GPU space
                for key, value in current_out.items():
                    if isinstance(value, torch.Tensor):
                        current_out[key] = value.to('cpu', non_blocking=True)
                pred_masks = pred_masks.to('cpu', non_blocking=True)

                output_dict[storage_key][frame_idx] = current_out
            
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self.grounding_encoder._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self.grounding_encoder._get_orig_video_res_output(
                inference_state, pred_masks
            )
            if unc_on:
                _unc_logit.append(float(current_out["object_score_logits"].float().max()))
                _unc_area.append(int((video_res_masks[0] > 0.0).sum()))
            # MLLM memory update. Original rule (bank_rule='default', bank_gap=0): a frame where the concept
            # was computed is added if its mask is not empty and its object score > 1.
            _mem_appended = 0
            _bank_reason = ""
            _div_maxcos = ""
            _e17_row = {}
            _out_masks = video_res_masks  # yielded (scored) masks; differs only with e17_commit 'output'
            if e17_on and _tracked:
                _e17_row["e17_lost"] = int(_lost)
                _e17_out = self.grounding_encoder._exp_e17.get("out")
                if _e17_out is not None:
                    torch.cuda.synchronize()
                    _t_e17 = time.perf_counter()
                    _vm = {}
                    # E17c sweep: one entry per w, columns sw<100 w>_* (sw050 = mix)
                    _ent = {"mix": _e17_out["mix"], "con": _e17_out["con"],
                            **{f"sw{round(_w * 100):03d}": _st for _w, _st in _e17_out.get("sweep", {}).items()}}
                    for _k, _st in _ent.items():
                        _lr = _st["low_res"]
                        if self.grounding_encoder.fill_hole_area > 0:
                            _lr = fill_holes_in_mask_scores(_lr, self.grounding_encoder.fill_hole_area)
                        _vm[_k] = self.grounding_encoder._get_orig_video_res_output(inference_state, _lr)[1][0, 0] > 0.0
                        _e17_row.update({f"{_k}_iou": round(_st["iou"], 5),
                                         f"{_k}_logit": round(_st["logit"], 5),
                                         f"{_k}_px": int(_vm[_k].sum())})
                        if _k.startswith("sw"):
                            _e17_row[f"{_k}_gate"] = int(_st["gate"])
                    _e17_row["e17_accept"] = int(_e17_out["accept"])
                    if _e17_out["accept"] and e17_commit == "output":
                        # same path as the tracked output: fill holes on the low-res logits, resize to video res
                        _lr = _e17_out["con"]["low_res"]
                        if self.grounding_encoder.fill_hole_area > 0:
                            _lr = fill_holes_in_mask_scores(_lr, self.grounding_encoder.fill_hole_area)
                        _out_masks = self.grounding_encoder._get_orig_video_res_output(
                            inference_state, _lr.to('cpu', non_blocking=True))[1]
                    if exp_ctx.get("gt_mask_fn") is not None:
                        _gt = exp_ctx["gt_mask_fn"](frame_idx)
                        if _gt is not None:
                            _g = torch.from_numpy(np.asarray(_gt)).to(_vm["mix"].device).reshape(_vm["mix"].shape)
                            _e17_row["gt_empty"] = int(not _g.any())
                            for _k in _vm:
                                _u = (_g | _vm[_k]).sum()
                                _e17_row[f"J_{_k}"] = round(1.0 if _u == 0 else float((_g & _vm[_k]).sum() / _u), 5)
                    if _vm["con"].any():
                        # cosine of the concept mask's descriptor to the bank frames given to the MLLM on this
                        # frame (descriptor as bank_diverse_tau)
                        _d = mask_pooled_descriptor(self.grounding_encoder._get_image_feature(
                            inference_state, frame_idx, batch_size)[2][-1], _vm["con"])
                        _cos = [float(_d @ mask_pooled_descriptor(self.grounding_encoder._get_image_feature(
                            inference_state, _bf, batch_size)[2][-1], np.asarray(_bm).squeeze()))
                                for _bf, _, _bm in _mllm_memory]
                        _e17_row["con_maxcos"] = round(max(_cos), 5)
                    torch.cuda.synchronize()
                    # logging only (J, descriptors): subtract from frame_ms for timing
                    _e17_row["e17_log_ms"] = round((time.perf_counter() - _t_e17) * 1000, 2)
            if _tracked and self.grounding_encoder._exp_e17 is not None and \
                    self.grounding_encoder._exp_e17.get("norms") is not None:
                _e17_row.update({k: round(v, 5) for k, v in self.grounding_encoder._exp_e17["norms"].items()})
            if e17_on and _tracked:
                _lost_hist.append(bool((video_res_masks[0] > 0.0).sum() == 0
                                       or current_out["object_score_logits"].item() <= 0))
            _bank_candidate = _hsv_trig if bank_rule == "hsv" else _update_flag
            _periodic = (not _bank_candidate and _tracked and bank_gap > 0
                         and frame_idx - mllm_memory[-1][0] >= bank_gap)
            _gt_periodic = (not _bank_candidate and _tracked and bank_gt_gap > 0
                            and frame_idx - mllm_memory[-1][0] >= bank_gt_gap)
            if div_tau is not None:
                _cur_mask = video_res_masks[0] > 0.0
                if (_tracked and _cur_mask.sum() > 0
                        and current_out["object_score_logits"].item() > bank_score_thresh
                        and (_hsv_trig or frame_idx - _div_last_cand >= div_gap)):
                    _div_last_cand = frame_idx
                    _vf = self.grounding_encoder._get_image_feature(inference_state, frame_idx, batch_size)[2][-1]
                    _d = mask_pooled_descriptor(_vf, _cur_mask)
                    _cos = [float(_d @ e) for e in _div_desc]
                    _div_maxcos = round(max(_cos), 5)
                    if max(_cos) >= div_tau:
                        _bank_reason = "div_redundant"
                    else:
                        _bank_reason = "added_div"
                        if len(mllm_memory) >= mllm_memory_size:
                            _drop = 1 + int(np.argmax(_cos[1:]))
                            del mllm_memory[_drop], _div_desc[_drop]
                            _bank_reason = "added_div_replace"
                        mllm_memory.append((
                            frame_idx, Image.open(video_paths[frame_idx]).convert('RGB'),
                            _cur_mask.cpu().numpy()
                        ))
                        _div_desc.append(_d)
                        _mem_appended = 1
            elif _gt_periodic:
                _gt = exp_ctx["gt_mask_fn"](frame_idx)
                if _gt is None or not _gt.any():
                    _bank_reason = "gt_empty"
                else:
                    mllm_memory.append((frame_idx, Image.open(video_paths[frame_idx]).convert('RGB'), _gt[None]))
                    _mem_appended = 1
                    _bank_reason = "added_gt"
            if div_tau is None and (_bank_candidate or _periodic):
                if (video_res_masks[0] > 0.0).sum() == 0:
                    _bank_reason = "empty_mask" if _bank_candidate else ""
                elif current_out["object_score_logits"].item() <= bank_score_thresh:
                    _bank_reason = f"obj_score<={bank_score_thresh:g}" if _bank_candidate else ""
                elif any(f == frame_idx for f, _, _ in mllm_memory):
                    # already in the bank (e.g. an oracle keyframe): no duplicate frame
                    _bank_reason = "already_in_bank" if _bank_candidate else ""
                else:
                    mllm_memory.append((
                        frame_idx, Image.open(video_paths[frame_idx]).convert('RGB'),
                        (video_res_masks[0] > 0.0).cpu().numpy()
                    ))
                    _mem_appended = 1
                    _bank_reason = "added" if _bank_candidate else "added_periodic"
            if _e17_row.get("e17_accept") and e17_commit == "full" and not _mem_appended and \
                    not any(f == frame_idx for f, _, _ in mllm_memory):
                # E17: a frame whose concept-only mask was kept goes into the MLLM memory directly
                mllm_memory.append((
                    frame_idx, Image.open(video_paths[frame_idx]).convert('RGB'),
                    (video_res_masks[0] > 0.0).cpu().numpy()
                ))
                _mem_appended = 1
                _bank_reason = "added_concept"
            if confirm_k and _tracked:
                _confirm_log = []
                _ok = bool((video_res_masks[0] > 0.0).sum() > 0) \
                    and float(current_out["ious"].float().max()) >= self.grounding_encoder._exp_e17["iou_thresh"] \
                    and current_out["object_score_logits"].item() > self.grounding_encoder._exp_e17["logit_thresh"]
                for _p in list(_pending):
                    _p[2] -= 1
                    if not _ok:
                        _pending.remove(_p)
                        _confirm_log.append(f"drop:{_p[0]}")
                    elif _p[2] == 0:
                        _pending.remove(_p)
                        if not any(f == _p[0] for f, _, _ in mllm_memory):
                            mllm_memory.append((_p[0], Image.open(video_paths[_p[0]]).convert('RGB'), _p[1]))
                            _confirm_log.append(f"add:{_p[0]}")
                if _e17_row.get("e17_accept") and not _mem_appended and \
                        not any(f == frame_idx for f, _, _ in mllm_memory):
                    _pending.append([frame_idx, (video_res_masks[0] > 0.0).cpu().numpy(), confirm_k])
                    _confirm_log.append(f"pend:{frame_idx}")
                _e17_row["bank_confirm"] = " ".join(_confirm_log)
            if exp_rows is not None:
                torch.cuda.synchronize()
                ious = current_out.get("ious")
                obj_score = current_out.get("object_score_logits")
                _row.update(
                    hsv_trigger=int(_hsv_trig),
                    mem_appended=_mem_appended,
                    bank_reason=_bank_reason,
                    div_maxcos=_div_maxcos,
                    mask_px=int((_out_masks[0] > 0.0).sum()),
                    pred_iou=round(float(ious.float().max()), 5) if ious is not None else "",
                    object_score_logit=round(float(obj_score.float().max()), 5) if obj_score is not None else "",
                    frame_ms=round((time.perf_counter() - _t_frame) * 1000, 2),
                    bank_frames=" ".join(str(f) for f, _, _ in mllm_memory),
                    sam_mem_frames=" ".join(str(f) for f in self.grounding_encoder._exp_mem_frames),
                    **_e17_row,
                )
                exp_rows.append(_row)
            yield frame_idx, obj_ids, _out_masks
        self.grounding_encoder._exp_mem_frames = None
        self.grounding_encoder._exp_e17 = None

    def predict_forward(
        self,
        image=None,
        video=None,
        text=None,
        num_seg_token=1
    ):  
        assert image is not None or video is not None

        input_dict = {}
        if video is not None:
            pixel_values = []
            ori_image_size = video[0].size
            for frame_idx, frame_image in enumerate(video):
                assert ori_image_size == frame_image.size
                img = self.transformer(frame_image)
                pixel_values.append(img)

            pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
            num_image_tokens = self.patch_token
            num_frames = len(pixel_values)
        else:
            ori_image_size = image.size
            images = dynamic_preprocess(
                image, self.min_dynamic_patch, self.max_dynamic_patch,
                self.image_size, self.use_thumbnail
            )

            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
            num_image_tokens = pixel_values.shape[0] * self.patch_token
            num_frames = 1
        
        input_dict['pixel_values'] = pixel_values
        image_token_str = f'{self.IMG_START_TOKEN}' \
                            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                            f'{self.IMG_END_TOKEN}'
        image_token_str = image_token_str + '\n'
        image_token_str = image_token_str * num_frames
        image_token_str = image_token_str.strip()

        text += "It is [SEG].".replace('[SEG]', '[SEG]' * num_seg_token)
        text = text.replace('<image>', image_token_str)
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.bot_name)

        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        data ={
            'input_ids': ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values.unsqueeze(0).to(self.device),
            'position_ids': None,
            'labels': None,
        }

        output = self.forward(data)
        seg_token_mask = ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = hidden_states[-1][seg_token_mask]
        hidden_states = self.text_hidden_fcs(hidden_states)
        _zero = hidden_states.mean() * 0.0
        pred_embeddings = hidden_states + _zero # [n, 256]

        return pred_embeddings

def label_img_with_mask(img, mask):
    frame = np.array(img)
    mask = np.uint8(mask).squeeze()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    frame = Image.fromarray(frame) 
    return frame

def mask_pooled_descriptor(vision_feat, mask):
    """Bank descriptor (bank_diverse_tau): SAM 2 Hiera feature of a frame (current_vision_feats[-1],
    [HW, B, C], B = 1) averaged inside the mask (bool, any resolution, area-resized to the feature grid),
    L2-normalised. Returns a float32 [C] tensor on the feature's device."""
    hw, _, c = vision_feat.shape
    side = int(round(hw ** 0.5))
    m = mask if torch.is_tensor(mask) else torch.from_numpy(np.asarray(mask))
    m = m.to(vision_feat.device).float().reshape(1, 1, *m.shape[-2:])
    w = F.interpolate(m, size=(side, side), mode="area").reshape(hw)
    d = (w @ vision_feat[:, 0, :].float()) / w.sum()
    return F.normalize(d, dim=0)


def is_scene_change_hsv(img1, img2, threshold=0.35):
    return hsv_distance(img1, img2) > threshold


def hsv_distance(img1, img2):
    """Bhattacharyya distance between H-S histograms (body of the original is_scene_change_hsv)."""
    img1 = cv2.resize(np.array(img1), (1024, 1024))
    img2 = cv2.resize(np.array(img2), (1024, 1024))

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [60, 80], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [60, 80], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return distance


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation_phi3(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs