#    Copyright 2023 Haotian Liu
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

"""
============================================================================
LLaVA-LLaMA: 视觉-语言多模态模型
============================================================================

核心功能：
1. 将视觉特征（来自感知模块）与语言模型（LLaMA）结合
2. 支持自动驾驶场景理解和问答
3. 提取特定token位置的隐藏状态（用于规划）
4. 通过特殊token与感知模块交互

架构设计：
- 基于LLaMA因果语言模型
- 继承LlavaMetaModel实现多模态融合
- 支持视觉token嵌入
- 可提取waypoint特征用于规划

应用场景：
- 自动驾驶场景描述
- 驾驶决策解释
- 轨迹规划（通过waypoint token）
- 交互式问答
"""


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    """
    LLaVA配置类
    ============================================================================
    继承自LlamaConfig，添加了多模态相关配置
    
    关键配置项：
    - waypoint_token_idx: waypoint特殊token的索引（用于轨迹规划）
    - 其他LLaMA标准配置
    """
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """
    LLaVA-LLaMA模型主体
    ============================================================================
    
    多重继承：
    - LlavaMetaModel: 提供视觉-语言融合的元类功能
    - LlamaModel: 提供LLaMA的transformer实现
    
    功能：
    - 处理文本token和视觉token的混合输入
    - 通过多层transformer进行特征提取
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    LLaVA-LLaMA因果语言模型
    ============================================================================
    
    核心功能：
    1. 文本生成（因果语言建模）
    2. 视觉-语言多模态理解
    3. 提取waypoint特征用于规划
    4. 支持数字token的加权损失
    
    多重继承：
    - LlamaForCausalLM: 提供因果语言建模能力
    - LlavaMetaForCausalLM: 提供多模态处理能力
    
    特殊设计：
    - weighted_mask: 对数字token施加更高权重（轨迹坐标预测）
    - waypoint_token: 特殊token用于提取规划特征
    """
    config_class = LlavaConfig

    def __init__(self, config, use_gen_token=False, use_critical_qa=False):
        """
        初始化LLaVA模型
        
        Args:
            config: 模型配置
            use_gen_token: 是否使用生成token（影响损失权重）
            use_critical_qa: 是否使用关键QA模式（影响数字token权重）
        """
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.hidden_size = config.hidden_size
        # 语言模型头：hidden_size -> vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

        # ========== 数字token列表 ==========
        # 对应: + - 0 . 1 2 3 4 5 6 7 8 9
        # 用于轨迹坐标预测（如"x=1.5, y=2.3"）
        number_tokens = [
                718,    # +
                448,    # -
                29900,  # 0
                29889,  # .
                29896,  # 1
                29906,  # 2
                29941,  # 3
                29946,  # 4
                29945,  # 5
                29953,  # 6
                29955,  # 7
                29947,  # 8
                29929,  # 9
            ]  # +-0.123456789
        
        # ========== 损失权重mask ==========
        # 策略：对数字token施加更高权重，提升坐标预测精度
        if use_gen_token:
            weighted_mask = torch.ones(self.config.vocab_size + 1)
            weighted_mask[number_tokens] = 1.0  # 生成模式：标准权重
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0  # 训练模式：3倍权重
        if use_critical_qa:
            weighted_mask[number_tokens] = 3.0  # 关键QA模式：3倍权重
        
        self.register_buffer("weighted_mask", weighted_mask)
        self.use_gen_token = use_gen_token
        # 初始化权重
        self.post_init()

    def get_model(self):
        return self.model

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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_ego_feature: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播 - 多模态语言模型
        ============================================================================
        
        处理流程：
        1. 准备多模态输入（文本+视觉）
        2. LLaMA模型前向传播
        3. 提取waypoint特征（可选）
        4. 语言模型头预测
        5. 计算损失（加权交叉熵）
        
        Args:
            input_ids: 输入token ID [B, seq_len]
            attention_mask: 注意力mask [B, seq_len]
            position_ids: 位置ID
            past_key_values: 缓存的KV（用于生成）
            inputs_embeds: 输入嵌入（可直接提供，跳过embedding层）
            labels: 训练标签 [B, seq_len]
            use_cache: 是否缓存KV
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            images: 视觉特征 [B, num_extra+num_memory, vision_dim]
                   来自OrionHead和OrionHeadM的vlm_memory
            image_sizes: 图像尺寸信息
            return_dict: 是否返回字典格式
            return_ego_feature: 是否返回waypoint特征（用于规划）
            
        Returns:
            CausalLMOutputWithPast: 包含loss, logits等
            selected_hidden_states: waypoint位置的隐藏状态（可选）
        """

        # ========== 步骤1: 准备多模态输入 ==========
        # 将文本token和视觉token融合
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,  # 融合后的嵌入 [B, seq_len, hidden_size]
                labels,
                new_input_ids   # 更新后的token ID（包含特殊token）
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,         # 来自感知模块的视觉特征
                image_sizes
            )
        else:
            new_input_ids = None
        
        # 设置输出选项
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ========== 步骤2: LLaMA模型前向传播 ==========
        # 输入: 融合的文本+视觉嵌入
        # 输出: 每层的隐藏状态
        outputs = self.model(
            input_ids=input_ids,              # None (使用inputs_embeds)
            attention_mask=attention_mask,    # 例: (1, 588)
            position_ids=position_ids,        # None
            past_key_values=past_key_values,  # None
            inputs_embeds=inputs_embeds,      # 例: (1, 588, 4096) 融合嵌入
            use_cache=use_cache,              # False
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 提取最后一层的隐藏状态
        hidden_states = outputs[0]  # [B, seq_len, hidden_size]

        # ========== 步骤3: 提取Waypoint特征（可选）==========
        # 用于规划模块：提取特定token位置的隐藏状态
        # 这些特征可以用于预测车辆的未来轨迹
        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                # 单个waypoint token的情况
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                # 多个waypoint token的情况
                # 例如：不同时间步的waypoint使用不同的特殊token
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    # 遍历所有waypoint token，找到它们的位置
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list, dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
        
        # ========== 步骤4: 语言模型头预测 ==========
        # hidden_states -> logits (词表概率)
        if self.pretraining_tp > 1:
            # 张量并行：将lm_head分片计算（多GPU）
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # 标准计算
            logits = self.lm_head(hidden_states)  # 例: (1, 588, 32001)
        logits = logits.float()

        # ========== 步骤5: 计算损失 ==========
        loss = None
        if labels is not None:
            # 因果语言建模：预测下一个token
            # token[i]用于预测token[i+1]
            shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置
            shift_labels = labels[..., 1:].contiguous()      # 去掉第一个位置
            
            # 使用加权交叉熵损失
            # weighted_mask: 对数字token施加更高权重（轨迹坐标预测）
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)  # 处理NaN值

        # ========== 步骤6: 返回结果 ==========
        if not return_dict:
            # tuple格式输出
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        # 字典格式输出
        if return_ego_feature:
            # 返回waypoint特征（用于规划模块）
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states  # waypoint位置的隐藏状态
        else:
            # 标准输出
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
        文本生成方法
        ============================================================================
        
        用途：
        - 场景描述生成
        - 驾驶决策解释
        - 问答系统
        - 轨迹坐标生成（通过数字token）
        
        流程：
        1. 准备多模态输入（文本+视觉）
        2. 调用LLaMA的生成方法
        3. 自回归生成token序列
        
        Args:
            inputs: 输入token ID
            images: 视觉特征（来自感知模块）
            image_sizes: 图像尺寸信息
            **kwargs: 生成参数（temperature, top_p等）
            
        Returns:
            generated_ids: 生成的token序列
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # 准备多模态输入
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,  # 融合后的嵌入
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            # 纯文本输入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类的生成方法（LLaMA的自回归生成）
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
        return_ego_feature = False,
        **kwargs,
    ):
        """
        自车特征推理方法
        ============================================================================
        
        核心功能：
        - 提取waypoint token位置的隐藏状态
        - 用于规划模块的特征提取
        - 不进行文本生成，只提取特征
        
        与generate方法的区别：
        - generate: 自回归生成完整文本序列
        - inference_ego: 只提取特定位置的特征（waypoint）
        
        应用场景：
        - 轨迹规划：提取规划相关的语义特征
        - 决策模块：提取决策相关的上下文特征
        - 端到端规划：将LLM特征传递给规划器
        
        Args:
            inputs: 输入token ID
            images: 视觉特征（来自感知模块）
            image_sizes: 图像尺寸信息
            return_ego_feature: 是否返回ego特征（必须为True）
            
        Returns:
            selected_hidden_states: waypoint位置的隐藏状态
                                   shape: [num_waypoints, hidden_size]
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # ========== 步骤1: 准备多模态输入 ==========
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids  # 包含waypoint token的ID序列
            ) = self.prepare_inputs_labels_for_multimodal(
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
        
        # 配置输出选项
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,  # 启用KV缓存（虽然只推理一次）
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]

        # ========== 步骤3: 提取Waypoint特征 ==========
        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                # 单个waypoint token
                loc_positions = (new_input_ids == self.config.waypoint_token_idx)
                selected_hidden_states = hidden_states[loc_positions.to(device=hidden_states.device)]
            else:
                # 多个waypoint token（不同时间步）
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
            assert False, "return_ego_feature must be True for inference_ego"
        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
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

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

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
