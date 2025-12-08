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

# ============================================================================
# 分布式训练相关常量
# ============================================================================
CONTROLLER_HEART_BEAT_EXPIRATION = 30  # 控制器心跳过期时间（秒）
WORKER_HEART_BEAT_INTERVAL = 15        # 工作节点心跳间隔（秒）

LOGDIR = "."  # 日志目录

# ============================================================================
# 模型常量定义
# ============================================================================
IGNORE_INDEX = -100              # 忽略索引，用于标记不计算损失的位置（如图像token位置）
IMAGE_TOKEN_INDEX = -200         # 图像token索引，用于标记文本中图像的位置
DEFAULT_IMAGE_TOKEN = "<image>"  # 默认图像占位符
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # 图像patch token
DEFAULT_IM_START_TOKEN = "<im_start>"     # 图像开始token
DEFAULT_IM_END_TOKEN = "<im_end>"         # 图像结束token


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ============================================================================
# LLaVA元模型类定义
# ============================================================================

class LlavaMetaModel:
    """
    LLaVA元模型基类
    用于多模态模型的基础架构，提供配置初始化功能
    """
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)


class LlavaMetaForCausalLM(ABC):
    """
    LLaVA因果语言模型元类（抽象基类）
    提供多模态输入处理的核心功能
    """
    
    @abstractmethod
    def get_model(self):
        """获取底层语言模型（抽象方法，需要子类实现）"""
        pass

    def get_vision_tower(self):
        """获取视觉编码器（Vision Tower）"""
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        """
        准备多模态输入、标签的核心方法
        将图像特征融合到文本序列中，处理图像token占位符
        
        参数:
            input_ids: 文本token IDs，形状 (batch_size, seq_len)
            position_ids: 位置编码IDs
            attention_mask: 注意力掩码
            past_key_values: 缓存的键值对（用于生成）
            labels: 训练标签
            image_features: 图像特征，形状 (batch_size, num_patches, hidden_size)
            image_sizes: 图像尺寸信息
            
        返回:
            (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_input_ids)
        """
        
        # ========================================================================
        # 第一步：处理无图像或生成场景
        # ========================================================================
        if image_features is None or input_ids.shape[1] == 1:
            # 情况1：没有图像特征
            # 情况2：生成阶段，每次只输入一个token (input_ids.shape[1] == 1)
            # 这些情况下直接返回，不需要融合图像特征
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # ========================================================================
        # 第二步：处理图像特征格式
        # ========================================================================
        if isinstance(image_features, list):
            # 如果图像特征是列表形式（多个视图/相机），需要重组
            # 例如：自动驾驶中有多个摄像头，每个摄像头一个特征列表
            temp_image_features = []
            for b_id in range(len(image_features[0])):  # 遍历batch
                for img_id in range(len(image_features)):  # 遍历每个图像/相机
                    temp_image_features.append(image_features[img_id][b_id])
            image_features = temp_image_features
        else:
            # 标准格式：重塑为 (batch_size, num_patches, hidden_size)
            # 例如：(B, 513, 4096) - 513是图像patches数量（如224x224图像分成16x16 patches）
            image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.dtype)

        # ========================================================================
        # 第三步：初始化默认值（避免处理None的麻烦）
        # ========================================================================
        # 保存原始值，用于后续判断是否需要返回None
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        # 如果attention_mask不存在，创建全1掩码（所有位置都有效）
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()  # 转换为布尔类型，例如 (B, 76)
        
        # 如果position_ids不存在，创建顺序位置编码 [0, 1, 2, ..., seq_len-1]
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        # 如果labels不存在，创建全IGNORE_INDEX的标签（训练时会忽略这些位置）
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # ========================================================================
        # 第四步：移除padding，保留有效token
        # ========================================================================
        # 使用attention_mask过滤掉padding的部分，只保留真实的token
        # 结果是列表，每个元素是一个样本的有效tokens
        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # ========================================================================
        # 第五步：遍历batch，融合图像特征和文本embeddings
        # ========================================================================
        new_input_embeds = []  # 存储融合后的输入embeddings
        new_labels = []        # 存储融合后的标签
        new_input_ids = []     # 存储融合后的input_ids
        cur_image_idx = 0      # 当前处理的图像索引
        
        for batch_idx, cur_input_ids in enumerate(input_ids):  # 遍历batch中的每个样本
            # 统计当前样本中有多少个图像token（IMAGE_TOKEN_INDEX=-200）
            # 例如：文本 "Describe <image> in detail" 中有1个图像占位符
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # 特殊情况：文本中没有图像占位符（但有图像特征）
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # 使用一个空切片[0:0]来保证维度一致性
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # ====================================================================
            # 5.1 找到图像token的位置，分割文本序列
            # ====================================================================
            # 例如：如果序列长度76，图像token在位置35
            # 则 image_token_indices = [-1, 35, 76]
            # 这样可以方便地分割出：[0:35]和[36:76]两个文本块
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []  # 存储分割后的文本块（不含图像token）
            cur_labels = labels[batch_idx]
            cur_labels_noim = []     # 存储分割后的标签块
            
            # 以图像token位置为分界，分割出文本块
            for i in range(len(image_token_indices) - 1):
                # 提取两个图像token之间（或开头/结尾）的文本
                # 例如：[(35,), (40,)] - 第一块35个token，第二块40个token
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            
            # ====================================================================
            # 5.2 将文本token转换为embeddings
            # ====================================================================
            split_sizes = [x.shape[0] for x in cur_labels_noim]  # 记录每块的大小，例如 [35, 40]
            # 将所有文本块拼接，一次性转换为embeddings
            # (75,) -> (75, 4096)，其中75 = 35 + 40
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim).to(image_features.device))
            # 再按原来的分块大小切分：[(35, 4096), (40, 4096)]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # ====================================================================
            # 5.3 在正确位置插入图像特征
            # ====================================================================
            cur_new_input_embeds = []  # 存储融合后的embeddings
            cur_new_labels = []         # 存储融合后的labels
            cur_new_input_ids = []      # 存储融合后的input_ids

            # 遍历文本块，在相邻块之间插入图像特征
            # 例如：文本块1 + 图像特征 + 文本块2
            for i in range(num_images + 1):
                # 添加第i个文本块
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])
                
                # 如果还有图像，插入图像特征
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]  # 获取当前图像特征 (513, 4096)
                    cur_image_idx += 1
                    
                    # 添加图像特征
                    cur_new_input_embeds.append(cur_image_features)
                    # 图像位置的label设为IGNORE_INDEX（不计算损失）
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, 
                                                     device=cur_labels.device, dtype=cur_labels.dtype))
                    # 图像位置的input_id设为IMAGE_TOKEN_INDEX
                    cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, 
                                                        device=cur_labels.device, dtype=cur_labels.dtype))
            
            # ====================================================================
            # 5.4 拼接当前样本的所有部分
            # ====================================================================
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)  # (588, 4096)
            cur_new_labels = torch.cat(cur_new_labels)              # (588,)
            cur_new_input_ids = torch.cat(cur_new_input_ids)        # (588,)
            # 其中 588 = 35 (文本块1) + 513 (图像特征) + 40 (文本块2)
            
            # 添加到batch列表
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_input_ids.append(cur_new_input_ids)
        # ========================================================================
        # 第六步：将batch内所有样本对齐到相同长度（padding）
        # ========================================================================
        # 找到batch内最长的序列长度
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        # 初始化padded张量
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, 
                                       dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_inputs_ids_padded = torch.zeros((batch_size, max_len), 
                                            dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), 
                                     dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), 
                                   dtype=position_ids.dtype, device=position_ids.device)

        # 对每个样本进行padding
        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]  # 当前样本的实际长度

            # Padding embeddings：在末尾补零到max_len
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), 
                           dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            
            # 只在有效长度范围内填充真实值
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_inputs_ids_padded[i, :cur_len] = cur_new_input_ids
                attention_mask[i, :cur_len] = True  # 有效位置标记为True
                # 位置编码：0, 1, 2, ..., cur_len-1
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # 将列表转换为张量，形状: (batch_size, max_len, hidden_size)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # ========================================================================
        # 第七步：根据原始输入决定是否返回None
        # ========================================================================
        # 如果原始输入中某些值是None，则保持返回None（保持接口一致性）
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

        # ========================================================================
        # 返回值说明：
        # - None: input_ids设为None（因为已经转换为embeddings）
        # - position_ids: 位置编码
        # - attention_mask: 注意力掩码（标记有效位置）
        # - past_key_values: 缓存的键值对（原样返回）
        # - new_input_embeds: 融合后的输入embeddings（文本+图像）
        # - new_labels: 融合后的标签（图像位置标记为IGNORE_INDEX）
        # - new_inputs_ids_padded: 融合后的input_ids（用于后续处理）
        # ========================================================================
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded
