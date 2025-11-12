# mmcv/utils/qwen_causallm.py
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import Qwen2Config, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaForCausalLM


class QwenVLMConfig(Qwen2Config):
    """自定义 Qwen VLM 配置类，用于序列化和注册"""
    model_type = "qwen_vlm"
    # 可选：Qwen2-VL 特殊视觉 token id（若未设置，将在拼接阶段使用默认回退值）
    vision_start_token_id: int = None
    vision_end_token_id: int = None
    image_pad_token_id: int = None


class QwenForCausalLMAdapter(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    """Qwen2-7B-Instruct 适配器，用于与 Orion 多模态架构集成"""
    config_class = QwenVLMConfig
    
    def __init__(self, config, use_gen_token: bool=False, use_critical_qa: bool=False):
        super().__init__(config)
        # HF结构与LLaMA一致：self.model 为骨干，self.lm_head 为输出头
        self.hidden_size = config.hidden_size
        
        # loss加权：Qwen2 的 vocab_size 约为 151936
        # 注意：原 LLaMA 的 number_tokens 对 Qwen 无效，这里初始化为统一权重
        # 实际的数字 token 权重将在 orion.py 中根据 tokenizer 动态设置
        if use_gen_token:
            weighted_mask = torch.ones(self.config.vocab_size + 1)
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
        
        if use_critical_qa:
            # 如需针对 critical_qa 任务的特殊权重，保持默认即可
            pass
        
        self.register_buffer("weighted_mask", weighted_mask)
        self.use_gen_token = use_gen_token
        # 注意：Qwen2ForCausalLM 已有 dtype 属性（property），无需手动设置

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        images: Optional[torch.FloatTensor]=None,
        image_sizes: Optional[List[List[int]]]=None,
        return_dict: Optional[bool]=None,
        return_ego_feature: Optional[bool]=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_input_ids = \
                self.prepare_inputs_labels_for_qwen2vl(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )
        else:
            new_input_ids = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 直接走骨干 + lm_head，拿到hidden_states便于抽取ego_feature
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if return_ego_feature:
            assert new_input_ids is not None
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

        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 动态匹配 weighted_mask 到实际 logits 的词表大小
            actual_vocab_size = shift_logits.shape[-1]
            if self.weighted_mask.shape[0] != actual_vocab_size:
                # 如果 weighted_mask 大小不匹配，截断或填充到实际大小
                if self.weighted_mask.shape[0] > actual_vocab_size:
                    weight = self.weighted_mask[:actual_vocab_size].float()
                else:
                    weight = torch.ones(actual_vocab_size, device=self.weighted_mask.device, dtype=self.weighted_mask.dtype)
                    weight[:self.weighted_mask.shape[0]] = self.weighted_mask
                    weight = weight.float()
            else:
                weight = self.weighted_mask.float()
            
            loss_fct = CrossEntropyLoss(weight=weight)
            shift_logits = shift_logits.view(-1, actual_vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = torch.nan_to_num(loss_fct(shift_logits, shift_labels))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if return_ego_feature:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _, _ = \
                self.prepare_inputs_labels_for_qwen2vl(
                    inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def inference_ego(self, inputs=None, images=None, image_sizes=None, return_ego_feature=False, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        inputs, position_ids, attention_mask, _, inputs_embeds, _, new_input_ids = \
            self.prepare_inputs_labels_for_qwen2vl(
                inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes
            )
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.use_return_dict,
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
            assert False

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


# 注册自定义模型到 Transformers（类似 llava_llama.py 的做法）
AutoConfig.register("qwen_vlm", QwenVLMConfig)
AutoModelForCausalLM.register(QwenVLMConfig, QwenForCausalLMAdapter)
