#    Copyright 2024 Orion Project
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import Qwen2VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaForCausalLM, IGNORE_INDEX, IMAGE_TOKEN_INDEX

# Qwen2-VL 特殊 Token 定义
VISION_START_TOKEN_INDEX = 151652  # <|vision_start|>
IMAGE_PAD_TOKEN_INDEX = 151655     # <|image_pad|>
VISION_END_TOKEN_INDEX = 151653    # <|vision_end|>


class OrionQwen2VlForCausalLM(Qwen2VLForConditionalGeneration, LlavaMetaForCausalLM):
    """
    Orion's adapter for Qwen2-VL, combining Qwen2-VL's architecture with
    LLaVA's multimodal input processing capability.
    
    This adapter enables seamless integration of Orion's BEV features with
    Qwen2-VL by reusing LLaVA's proven multimodal fusion mechanism.
    """
    
    def __init__(self, config, use_gen_token=False, use_critical_qa=False):
        # Initialize Qwen2-VL base model using proper super() call
        super().__init__(config)

        # 🔧 关键修复: 强制禁用内部模型的梯度检查点
        # Qwen2VLModel 的 HF 实现不支持标准的 gradient_checkpointing,
        # 但其配置可能默认启用了它。我们在这里强制禁用以避免 AttributeError。
        if hasattr(self, 'model') and hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = False
        
        # Store Orion-specific configurations
        self.use_gen_token = use_gen_token
        self.hidden_size = config.hidden_size
        
        # Setup weighted mask for special tokens
        # Qwen2-VL tokenizer 的数字 token IDs
        # 对应字符: +-0.123456789
        # 注意: 这些 token ID 与 LLaMA tokenizer 完全不同！
        number_tokens = [
            10,   # +
            12,   # -
            13,   # .
            15,   # 0
            16,   # 1
            17,   # 2
            18,   # 3
            19,   # 4
            20,   # 5
            21,   # 6
            22,   # 7
            23,   # 8
            24    # 9
        ]
        
        if use_gen_token:
            weighted_mask = torch.ones(self.config.vocab_size + 1)
            weighted_mask[number_tokens] = 1.0
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
        
        if use_critical_qa:
            weighted_mask[number_tokens] = 3.0
            
        self.register_buffer("weighted_mask", weighted_mask)
    
    def get_model(self):
        """
        Required by LlavaMetaForCausalLM to access the base language model.
        For Qwen2-VL, the base model is stored in self.model.
        """
        return self.model

    def prepare_inputs_labels_for_qwen2vl(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        """
        Prepare inputs for Qwen2-VL by fusing BEV features with text tokens.
        
        Key differences from LLaVA:
        - Uses vision_start/vision_end token pairs (151652/151653) instead of single IMAGE_TOKEN (-200)
        - Replaces entire vision token ranges rather than single positions
        - Compatible with Qwen2-VL's tokenization format
        
        Args:
            input_ids: Token IDs containing vision_start/vision_end markers
            image_features: BEV features from Orion (B, N_queries, hidden_size)
        
        Returns:
            Fused embeddings ready for Qwen2-VL processing
        """
        # Early return conditions
        if image_features is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # Handle list-based image features (multi-view)
        if isinstance(image_features, list):
            temp_image_features = []
            for b_id in range(len(image_features[0])):
                for img_id in range(len(image_features)):
                    temp_image_features.append(image_features[img_id][b_id])
            image_features = temp_image_features
        else:
            # Reshape to (B, N_queries, hidden_size) and ensure correct dtype
            image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.dtype)

        # Initialize None values
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 🔧 Qwen2-VL 适配: 检测 vision_start 和 vision_end
            vision_start_mask = (cur_input_ids == VISION_START_TOKEN_INDEX)
            vision_end_mask = (cur_input_ids == VISION_END_TOKEN_INDEX)
            num_images = vision_start_mask.sum().item()
            
            # 处理无图像的情况
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_input_ids.append(cur_input_ids)
                cur_image_idx += 1
                continue

            # 找到所有 vision_start 和 vision_end 的位置
            vision_start_positions = torch.where(vision_start_mask)[0].tolist()
            vision_end_positions = torch.where(vision_end_mask)[0].tolist()
            
            # 验证 start/end 成对出现
            assert len(vision_start_positions) == len(vision_end_positions), \
                f"Vision token mismatch: {len(vision_start_positions)} starts vs {len(vision_end_positions)} ends"
            
            # 🔧 Qwen2-VL 适配: 按 vision_start/end 范围切分文本
            # 构建切分点：包含所有文本段和视觉范围的边界
            cur_labels = labels[batch_idx]
            
            # 收集文本段和视觉范围
            text_segments = []
            label_segments = []
            vision_ranges = list(zip(vision_start_positions, vision_end_positions))
            
            # 添加第一个文本段（vision_start 之前）
            if vision_ranges[0][0] > 0:
                text_segments.append(cur_input_ids[:vision_ranges[0][0]])
                label_segments.append(cur_labels[:vision_ranges[0][0]])
            else:
                text_segments.append(torch.tensor([], dtype=cur_input_ids.dtype, device=cur_input_ids.device))
                label_segments.append(torch.tensor([], dtype=cur_labels.dtype, device=cur_labels.device))
            
            # 添加中间文本段（两个视觉范围之间）
            for i in range(len(vision_ranges) - 1):
                end_of_current = vision_ranges[i][1] + 1  # vision_end 之后
                start_of_next = vision_ranges[i + 1][0]    # 下一个 vision_start 之前
                if end_of_current < start_of_next:
                    text_segments.append(cur_input_ids[end_of_current:start_of_next])
                    label_segments.append(cur_labels[end_of_current:start_of_next])
                else:
                    text_segments.append(torch.tensor([], dtype=cur_input_ids.dtype, device=cur_input_ids.device))
                    label_segments.append(torch.tensor([], dtype=cur_labels.dtype, device=cur_labels.device))
            
            # 添加最后一个文本段（最后一个 vision_end 之后）
            last_end = vision_ranges[-1][1] + 1
            if last_end < len(cur_input_ids):
                text_segments.append(cur_input_ids[last_end:])
                label_segments.append(cur_labels[last_end:])
            else:
                text_segments.append(torch.tensor([], dtype=cur_input_ids.dtype, device=cur_input_ids.device))
                label_segments.append(torch.tensor([], dtype=cur_labels.dtype, device=cur_labels.device))
            
            # 获取文本 embeddings
            text_embeds_list = []
            for text_seg in text_segments:
                if text_seg.shape[0] > 0:
                    text_embeds_list.append(
                        self.get_model().embed_tokens(text_seg.to(image_features.device))
                    )
                else:
                    text_embeds_list.append(
                        torch.empty(0, self.hidden_size, device=image_features.device, dtype=self.dtype)
                    )
            
            # 🔧 Qwen2-VL 适配: 交错组合文本和图像
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_input_ids = []
            
            for i in range(len(vision_ranges) + 1):
                # 添加文本段
                if i < len(text_embeds_list):
                    cur_new_input_embeds.append(text_embeds_list[i])
                    cur_new_labels.append(label_segments[i])
                    cur_new_input_ids.append(text_segments[i])
                
                # 添加图像特征（范围替换）
                if i < len(vision_ranges):
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, 
                                  device=cur_labels.device, dtype=cur_labels.dtype)
                    )
                    # 使用 IMAGE_PAD_TOKEN_INDEX 填充（Qwen2-VL 风格）
                    cur_new_input_ids.append(
                        torch.full((cur_image_features.shape[0],), IMAGE_PAD_TOKEN_INDEX,
                                  device=cur_input_ids.device, dtype=cur_input_ids.dtype)
                    )
            
            # 拼接所有段
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            cur_new_input_ids = torch.cat(cur_new_input_ids, dim=0)
            
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids.append(cur_new_input_ids)
        
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_inputs_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels,cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]

            #padding
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_inputs_ids_padded[i, :cur_len] = cur_new_input_ids
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,  # BEV features from Orion
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_ego_feature: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass that handles both text and vision inputs.
        
        The 'images' parameter receives Orion's vision_embeded (BEV features),
        which are then fused with text tokens using LLaVA's fusion mechanism.
        """
        
        # Step 1: Use LLaVA's fusion mechanism to merge BEV features with text
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids
            ) = self.prepare_inputs_labels_for_qwen2vl(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,  # This is vision_embeded from Orion (B, N_queries, C)
                image_sizes
            )
        else:
            new_input_ids = None
        
        # Step 2: Call Qwen2-VL's base model with the fused inputs_embeds
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call self.model directly to get hidden states
        outputs = self.model(
            input_ids=input_ids,  # Will be None after fusion
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Fused text + vision embeddings
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Step 3: Extract ego_feature if needed (Orion-specific for trajectory planning)
        if return_ego_feature and new_input_ids is not None:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list, dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
        
        # Step 4: Compute logits and loss
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Use weighted cross entropy loss
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if return_ego_feature:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states
        else:
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
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate method for text generation with multimodal inputs.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_qwen2vl(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    @torch.no_grad()
    def inference_ego(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_ego_feature=False,
        **kwargs,
    ):
        """
        Orion-specific inference method for extracting ego features.
        This is used for trajectory planning in autonomous driving.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_qwen2vl(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict
        
        # Get model outputs
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list, dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            return selected_hidden_states
        else:
            raise ValueError("return_ego_feature must be True for inference_ego")
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        Prepare inputs for generation, including multimodal inputs.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


    def add_special_token(special_token_list, tokenizer, model):
        # 给新的token添加索引并用大模型的embeding的平均值来初始化token的embeding
        num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens = True)
        model.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg