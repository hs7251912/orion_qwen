# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
# Modified from OmniDrive(https://github.com/NVlabs/OmniDrive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.models.bricks import Linear
from mmcv.models.utils import bias_init_with_prob

from mmcv.core import build_assigner, build_sampler
from mmcv.core.utils.dist_utils import reduce_mean
from mmcv.core.utils.misc import multi_apply

from mmcv.models.utils import build_transformer,xavier_init
from mmcv.models import HEADS, build_loss
from mmcv.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmcv.models.utils.transformer import inverse_sigmoid

from math import factorial

from mmcv.models.utils import NormedLinear

from mmcv.models.utils.positional_encoding import pos2posemb1d, nerf_positional_encoding
from mmcv.utils.misc import MLN, topk_gather, transform_reference_points_lane, memory_refresh
import numpy as np
import os
@HEADS.register_module()
class OrionHeadM(AnchorFreeHead):
    """
    ============================================================================
    OrionHeadM: 地图元素检测头部（Map Detection Head）
    ============================================================================
    
    核心功能：
    1. 车道线检测 - 使用贝塞尔曲线表示车道线
    2. 控制点回归 - 预测n_control个控制点的3D坐标
    3. 时序建模 - 利用Memory Bank存储历史帧的车道线
    4. One2One & One2Many匹配 - 借鉴HybridTaskCascade的思想
    
    与OrionHead的区别：
    - OrionHead: 检测动态目标（车辆、行人等）+ 轨迹预测
    - OrionHeadM: 检测静态地图元素（车道线、路沿等）
    
    车道线表示方法：
    - 使用贝塞尔曲线的控制点表示（默认n_control=4）
    - 每个控制点是3D坐标 (x, y, z)
    - 可以转换为更多采样点（num_pts_vector=20）用于可视化
    
    Args:
        num_classes (int): 车道线类别数（例如：实线、虚线、双黄线等）
        in_channels (int): 输入特征通道数
        num_lane (int): 车道线查询数量（类似目标检测的num_query）
        n_control (int): 贝塞尔曲线控制点数量，默认4
        num_pts_vector (int): 用于可视化的采样点数量，默认20
        num_lanes_one2one (int): one2one匹配的车道线数量
        k_one2many (int): one2many匹配时，每个GT复制的倍数
        memory_len (int): 记忆库长度
        topk_proposals (int): 每帧保留的top-k车道线
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_lane=100,
                 memory_len=1000,
                 topk_proposals=500,
                 num_lanes_one2one=0,
                 k_one2many=0,
                 lambda_one2many=1.0,
                 num_extra=256,
                 with_ego_pos=True,
                 with_mask=False,
                 pc_range=None,
                 num_reg_fcs=2,
                 n_control=4,
                 num_pts_vector=20, #16
                 dir_interval=1,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 match_costs=None,
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 normedlinear=False,
                 score_threshold=0.,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights
            
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is OrionHeadM):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.with_ego_pos = with_ego_pos
        self.with_mask = with_mask
        self.output_dims = out_dims
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.dir_interval = dir_interval
        self.num_pts_vector = num_pts_vector
        self.n_control = n_control
        self.num_lane = num_lane
        self.num_lanes_one2one = num_lanes_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_extra = num_extra 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(OrionHeadM, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)

        # NMS的指标
        self.max_num = 50
        self.score_threshold = score_threshold
        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """
        初始化网络层
        ============================================================================
        主要组件：
        1. 分类分支 (cls_branch) - 预测车道线类别（实线/虚线等）
        2. 回归分支 (reg_branch) - 预测n_control个控制点的(x,y,z)坐标
        3. 查询嵌入 (embeddings) - 为每条车道线和每个控制点生成嵌入
        
        与OrionHead的主要区别：
        - 回归分支输出: n_control*3（控制点坐标）而非边界框参数
        - 使用点嵌入和实例嵌入的组合生成查询
        """

        # ============= 分类分支 =============
        # 预测车道线类别（例如：实线、虚线、双黄线等）
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        # ============= 控制点回归分支 =============
        # 输出: [n_control个控制点的(x,y,z)] = n_control*3维
        # 例如：n_control=4时，输出12维 -> 4个3D控制点
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.n_control*3))  # 控制点坐标
        reg_branch = nn.Sequential(*reg_branch)

        # 为每层decoder克隆预测头（实现iterative refinement）
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        # ============= 特征投影层 =============
        self.input_projection = nn.Linear(self.in_channels, self.embed_dims)
        if self.output_dims is not None:
            self.output_projection = nn.Linear(self.embed_dims, self.output_dims)

        # ============= 车道线查询相关 =============
        # 用于生成初始的参考点
        self.reference_points_lane = nn.Linear(self.embed_dims, 3)
        
        # 点嵌入：为每个控制点生成不同的嵌入（n_control个）
        self.points_embedding_lane = nn.Embedding(self.n_control, self.embed_dims)#(11,256)
        # 实例嵌入：为每条车道线生成不同的嵌入（num_lane个）
        self.instance_embedding_lane = nn.Embedding(self.num_lane, self.embed_dims)  #(1800,256)
        
        # 额外查询嵌入（用于捕获场景信息，供VLM使用）
        self.query_embedding = nn.Embedding(self.num_extra, self.embed_dims)
        self.query_pos = None

        self.time_embedding = None
        self.ego_pose_pe = None

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        xavier_init(self.reference_points_lane, distribution='uniform', bias=0.)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.sample_time = None
        self.memory_mask = None
        self.memory_scene_tokens = None

    def pre_update_memory(self, img_metas, data):
        """
        记忆库预更新 - 管理历史帧的车道线信息
        ============================================================================
        
        功能：
        1. 初始化记忆库（首次调用）
        2. 场景切换检测 - 新场景时清空记忆
        3. 坐标变换 - 将历史车道线坐标转到当前帧坐标系
        4. 时间戳更新
        
        与OrionHead的区别：
        - memory_reference_point的shape: [B, memory_len, n_control, 3]
          因为每条车道线有n_control个控制点
        - 使用transform_reference_points_lane而非transform_reference_points
          专门处理车道线控制点的变换
        """
        B = data['img_feats'].size(0)
        
        # ========== 情况1: 首次初始化记忆库 ==========
        if self.memory_embedding is None:
            self.memory_embedding = data['img_feats'].new_zeros(B, self.memory_len, self.embed_dims)
            # 车道线的参考点是n_control个3D控制点
            self.memory_reference_point = data['img_feats'].new_zeros(B, self.memory_len, self.n_control, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(B, self.memory_len, 4, 4)
            self.sample_time = data['timestamp'].new_zeros(B)
            self.memory_mask = data['img_feats'].new_zeros(B, self.memory_len, 1)  # 标记哪些记忆是有效的
            x = self.sample_time.to(data['img_feats'].dtype)
            self.memory_scene_tokens = ['' for meta in img_metas]
        # ========== 情况2: 更新已有记忆库 ==========
        else:
            # 更新时间戳
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            
            # 检测是否需要刷新记忆（时间间隔<2秒 且 场景相同）
            x = (torch.abs(self.sample_time) < 2.0)
            y = [meta['scene_token'] == memory_tokens for meta, memory_tokens in zip(img_metas, self.memory_scene_tokens)]
            y = torch.tensor(y,device=x.device)
            x = torch.logical_and(x,y).to(data['img_feats'].dtype)
            
            # 坐标变换：历史帧 -> 当前帧坐标系
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            # 专门用于车道线控制点的变换（处理多个控制点）
            self.memory_reference_point = transform_reference_points_lane(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            
            # 根据场景切换标志刷新记忆
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_mask = memory_refresh(self.memory_mask[:, :self.memory_len], x)
            self.sample_time = data['timestamp'].new_zeros(B)

    def post_update_memory(self, img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec):
        rec_reference_points = all_bbox_preds[-1].reshape(outs_dec.shape[1], -1, self.n_control, 3)
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_mask = torch.cat([torch.ones_like(rec_timestamp), self.memory_mask], dim=1)
        self.memory_reference_point = transform_reference_points_lane(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        self.memory_scene_tokens = [meta['scene_token'] for meta in img_metas]
        
        return out_memory
    
    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.flatten(-2))) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(tgt[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(tgt[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is OrionHeadM:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    def forward(self, img_metas, pos_embed, **data):
        """
        前向传播 - 车道线检测的核心流程
        ============================================================================
        
        处理流程：
        1. 记忆库预更新 - 处理历史车道线信息
        2. 特征处理 - 提取图像token序列
        3. 查询初始化 - 使用点嵌入+实例嵌入生成车道线查询
        4. Attention Mask构建 - 实现One2One和One2Many的隔离
        5. 时序对齐 - 融合历史车道线信息
        6. Transformer解码 - 多层refinement
        7. 控制点预测 - 输出每条车道线的控制点坐标
        
        Args:
            img_metas: 图像元信息
            pos_embed: 位置编码
            data: 包含img_feats、ego_pose、timestamp等
            
        Returns:
            outs (dict): 包含one2one和one2many的预测结果
            vlm_memory: VLM输入特征
        """
        # ========== 步骤1: 记忆库预更新 ==========
        self.pre_update_memory(img_metas, data)

        # ========== 步骤2: 图像特征处理 ==========
        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)  # [B, N*H*W, C]
        memory = self.input_projection(memory)

        # ========== 步骤3: 车道线查询初始化 ==========
        # 关键设计：点嵌入 + 实例嵌入
        # instance_embedding: [num_lane, embed_dims] - 区分不同车道线
        # points_embedding: [n_control, embed_dims] - 区分同一车道线的不同控制点
        # lane_embedding shape: [num_lane, n_control, embed_dims]
        lane_embedding = self.instance_embedding_lane.weight.unsqueeze(-2) + self.points_embedding_lane.weight.unsqueeze(0) 
        
        # 生成初始参考点（归一化坐标）
        # reference_points_lane shape: [B, num_lane, n_control*3]
        reference_points_lane = self.reference_points_lane(lane_embedding).sigmoid().flatten(-2).unsqueeze(0).repeat(B, 1, 1)
        
        # 位置编码和查询特征
        query_pos = self.query_pos(nerf_positional_encoding(reference_points_lane))
        tgt = self.instance_embedding_lane.weight.unsqueeze(0).repeat(B, 1, 1)
        query_embedding = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)  # 额外查询

        # ========== 步骤4: 构建Attention Mask（One2One & One2Many）==========
        # 设计思想：类似HybridTaskCascade
        # - One2One查询：用于最终输出，严格的1对1匹配
        # - One2Many查询：用于辅助训练，每个GT匹配多个查询
        # - 两者之间相互屏蔽，避免信息泄漏
        self_attn_mask = (
            torch.zeros([self.num_lane+self.num_extra, self.num_lane+self.num_extra]).bool().to(x.device)
        )
        # One2One和One2Many查询相互屏蔽
        self_attn_mask[self.num_lanes_one2one+self.num_extra:, 0: self.num_lanes_one2one+self.num_extra] = True
        self_attn_mask[0: self.num_lanes_one2one+self.num_extra, self.num_lanes_one2one+self.num_extra:] = True
        
        # 扩展mask以支持时序建模
        temporal_attn_mask = (
            torch.zeros([self.num_lane+self.num_extra, self.num_lane+self.num_extra+self.memory_len]).bool().to(x.device)
        )
        temporal_attn_mask[:self_attn_mask.size(0), :self_attn_mask.size(1)] = self_attn_mask
        if self.with_mask:
            # 额外查询的mask规则
            temporal_attn_mask[self.num_extra:, :self.num_extra] = True

        # ========== 步骤5: 时序对齐 ==========
        tgt, query_pos, reference_points_lane, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points_lane)

        # 将额外查询添加到序列开头
        tgt = torch.cat([query_embedding, tgt], dim=1)
        query_pos = torch.cat([torch.zeros_like(query_embedding), query_pos], dim=1)
        
        # ========== 步骤6: Transformer解码器 ==========
        outs_dec = self.transformer(tgt, memory, query_pos, pos_embed, temporal_attn_mask, temp_memory, temp_pos)

        # 分离额外查询和车道线查询
        vlm_memory = outs_dec[-1, :, :self.num_extra, :]  # VLM特征
        outs_dec = outs_dec[:, :, self.num_extra:, :]  # 车道线查询特征
        
        outs_dec = torch.nan_to_num(outs_dec)
        
        # ========== 步骤7: 预测控制点坐标和类别 ==========
        lane_queries = outs_dec
        outputs_lane_preds = []
        outputs_lane_clses = []
        
        for lvl in range(outs_dec.shape[0]):  # 遍历每层decoder
            # 参考点 + 残差 -> 最终坐标（DETR的标准做法）
            reference = inverse_sigmoid(reference_points_lane.clone())
            reference = reference.view(B, self.num_lane, self.n_control*3)
            
            # 预测残差
            tmp = self.reg_branches[lvl](lane_queries[lvl])
            outputs_lanecls = self.cls_branches[lvl](lane_queries[lvl])

            # 加上参考点，得到最终坐标
            tmp = tmp.reshape(B, self.num_lane, self.n_control*3)
            tmp += reference
            tmp = tmp.sigmoid()  # 归一化到[0,1]

            # reshape为控制点格式
            outputs_coord = tmp
            outputs_coord = outputs_coord.reshape(B, self.num_lane, self.n_control, 3)
            outputs_lane_preds.append(outputs_coord)
            outputs_lane_clses.append(outputs_lanecls)

        # 堆叠所有层的输出
        all_lane_preds = torch.stack(outputs_lane_preds)  # [num_dec, B, num_lane, n_control, 3]
        all_lane_clses = torch.stack(outputs_lane_clses)  # [num_dec, B, num_lane, num_classes]

        # ========== 步骤8: 坐标反归一化 ==========
        # 归一化坐标[0,1] -> 真实世界坐标（米）
        all_lane_preds[..., 0:3] = (all_lane_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        all_lane_preds = all_lane_preds.flatten(-2)  # [num_dec, B, num_lane, n_control*3]

        # ========== 步骤9: 分离One2One和One2Many结果 ==========
        # One2One: 用于最终预测，严格1对1匹配
        all_lane_cls_one2one = all_lane_clses[:, :, 0: self.num_lanes_one2one, :]
        all_lane_preds_one2one = all_lane_preds[:, :, 0: self.num_lanes_one2one, :]
        # One2Many: 用于辅助训练，提供更多正样本
        all_lane_cls_one2many = all_lane_clses[:, :, self.num_lanes_one2one:, :]
        all_lane_preds_one2many = all_lane_preds[:, :, self.num_lanes_one2one:, :]
        outs_dec_one2one = outs_dec[:, :, 0: self.num_lanes_one2one, :]
        outs_dec_one2many = outs_dec[:, :, self.num_lanes_one2one:, :]
        
        # ========== 步骤10: 记忆库后更新 ==========
        # 只使用One2One的结果更新记忆库
        out_memory = self.post_update_memory(img_metas, data, rec_ego_pose, all_lane_cls_one2one, all_lane_preds_one2one, outs_dec_one2one)
        
        # 组装输出字典
        outs = {
            'all_lane_cls_one2one': all_lane_cls_one2one,
            'all_lane_preds_one2one': all_lane_preds_one2one,
            'all_lane_cls_one2many': all_lane_cls_one2many,
            'all_lane_preds_one2many': all_lane_preds_one2many,
            'outs_dec_one2one': outs_dec_one2one,
            'outs_dec_one2many':outs_dec_one2many,
        }
        
        # VLM特征投影
        if self.output_dims is not None:
            vlm_memory = self.output_projection(vlm_memory)
        return outs, vlm_memory


    def loss(self,
             gt_lanes,
             gt_lanes_label,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """
        损失函数 - One2One & One2Many混合训练
        ============================================================================
        
        损失组成：
        1. One2One损失（主要）：
           - loss_cls_lane: 分类损失（Focal Loss）
           - loss_bbox_lane: 控制点坐标L1损失
           - loss_dir: 方向损失（控制点间的方向一致性）
           
        2. One2Many损失（辅助）：
           - loss_cls_H: One2Many分类损失 * lambda_one2many
           - loss_bbox_H: One2Many坐标损失 * lambda_one2many
           
        One2Many训练策略：
        - 每个GT车道线复制k_one2many次（例如k=3）
        - 提供更多正样本，加速收敛
        - 通过attention mask隔离，不影响One2One预测
        
        Args:
            gt_lanes: GT车道线控制点 [B个list，每个包含num_gt条车道线]
            gt_lanes_label: GT车道线类别标签
            preds_dicts: 预测结果字典
            img_metas: 图像元信息
            
        Returns:
            loss_dict: 包含各项损失的字典
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_lane_cls_one2one']
        all_bbox_preds = preds_dicts['all_lane_preds_one2one']
        all_cls_scores_one2many_list = preds_dicts['all_lane_cls_one2many']
        all_bbox_preds_one2many_list = preds_dicts['all_lane_preds_one2many']
        gt_lanes = [lane.reshape(-1, self.n_control*3) for lane in gt_lanes]
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_lanes for _ in range(num_dec_layers)]
        gt_labels = gt_lanes_label
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        # for one2many
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []
        
        for gt_bboxes in gt_lanes:
            one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))
        for gt_labels in gt_labels:
            one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))
        all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)
        
        img_metas_list_one2many = img_metas_list
        all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list
        losses_cls_one2many, losses_bbox_one2many, losses_dir_one2many = multi_apply(
            self.loss_single, all_cls_scores_one2many_list, all_bbox_preds_one2many_list,
            all_gt_bboxes_list_one2many, all_gt_labels_list_one2many, img_metas_list_one2many,
            all_gt_bboxes_ignore_list_one2many)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls_lane'] = losses_cls[-1]
        loss_dict['loss_cls_H'] = losses_cls_one2many[-1] * self.lambda_one2many
        loss_dict['loss_bbox_lane'] = losses_bbox[-1]
        loss_dict['loss_bbox_H'] = losses_bbox_one2many[-1] * self.lambda_one2many
        # loss_dict['loss_dir'] = losses_dir[-1]
        # loss_dict['loss_dir_H'] = losses_dir_one2many[-1] * self.lambda_one2many
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dir_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_lane'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_cls_H'] = losses_cls_one2many[num_dec_layer] * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_bbox_lane'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_H'] = losses_bbox_one2many[num_dec_layer] * self.lambda_one2many
            # loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            # loss_dict[f'd{num_dec_layer}.loss_dir_H'] = losses_dir_one2many[num_dec_layer] * self.lambda_one2many
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()


        dir_weights = bbox_weights.reshape(-1, self.n_control, 3)[:, :-1,0]
        pts_preds_dir = bbox_preds.reshape(-1, self.n_control, 3)[:,1:,:] - bbox_preds.reshape(-1, self.n_control, 3)[:,:-1,:] # 逐向量相减，得到dir
        pts_targets_dir = bbox_targets.reshape(-1, self.n_control, 3)[:, 1:,:] - bbox_targets.reshape(-1, self.n_control, 3)[:,:-1,:]
        loss_dir = self.loss_dir(
            pts_preds_dir, pts_targets_dir,
            dir_weights,
            avg_factor=num_total_pos)
        bbox_preds = bbox_preds.reshape(-1, self.n_control * 3)
        
        # bbox_preds = self.control_points_to_lane_points(bbox_preds)
        # bbox_targets = self.control_points_to_lane_points(bbox_targets)
        bbox_weights = bbox_weights.mean(-1).unsqueeze(-1).repeat(1, bbox_preds.shape[-1])
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        

        
        return loss_cls, loss_bbox, loss_dir


    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
    
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):

        num_preds = bbox_pred.size(0)
        # import pdb;pdb.set_trace()
        # bbox_pred = self.control_points_to_lane_points(bbox_pred)
        # gt_bboxes = self.control_points_to_lane_points(gt_bboxes)
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_preds,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_preds)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        
        if sampling_result.num_gts > 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            bbox_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights,
                bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """
        后处理 - 从预测结果生成最终的车道线
        ============================================================================
        
        流程：
        1. 提取One2One的预测结果（用于最终输出）
        2. 对每个样本单独处理
        3. Top-K选择最高置信度的车道线
        4. 阈值过滤
        5. 坐标裁剪（确保在有效范围内）
        
        Args:
            preds_dicts: 预测结果字典
            img_metas: 图像元信息
            rescale: 是否需要rescale（本方法中未使用）
            
        Returns:
            predictions_list: 每个样本的检测结果列表
        """
        # 只使用One2One的结果（最后一层decoder的输出）
        cls_scores = preds_dicts['all_lane_cls_one2one'][-1]
        bbox_preds = preds_dicts['all_lane_preds_one2one'][-1]

        predictions_list = []
        for img_id in range(len(img_metas)):
            # 从num_lanes_one2one个候选中选择max_num=50条最优的
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            predictions_list.append(self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale))

        return predictions_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        # 修改为输出50条概率最大的，modify by fuhaoyu
        max_num = self.max_num
        assert len(cls_score) == len(bbox_pred)
        
        cls_score = cls_score.sigmoid()
        scores, indexs = cls_score.view(-1).topk(max_num)

        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes

        det_bboxes = bbox_pred
        for p in range(self.n_control):
            det_bboxes[..., 3 * p].clamp_(min=self.pc_range[0], max=self.pc_range[3])
            det_bboxes[..., 3 * p + 1].clamp_(min=self.pc_range[1], max=self.pc_range[4])
            
        # det_bboxes = self.control_points_to_lane_points(det_bboxes)
        det_bboxes = det_bboxes.reshape(det_bboxes.shape[0], -1, 3)
        
        det_bboxes = det_bboxes[bbox_index]

        if os.getenv('DEBUG_SHOW_PRED', None) is not None:
            score_threshold = self.score_threshold
        else:
            score_threshold = 0.
        if isinstance(score_threshold, list):
            assert len(score_threshold) == self.num_classes, \
                "score_threshold length must = class_names, len class_names: {}".format(self.num_classes)
        elif isinstance(score_threshold, dict):
            for dist_range, cls_scores_thr in score_threshold.items():
                assert len(cls_scores_thr) == self.num_classes, \
                "dist_range ---> score_threshold length must = class_names, class_names: {}".format(self.num_classes)
        else:
            score_threshold = [score_threshold] * self.num_classes
        cur_bboxes = []
        cur_labels = []
        cur_scores = []
        for cid in range(self.num_classes):
            cid_mask = labels == cid
            score_thrs = scores.new_ones(scores.shape) * score_threshold[cid]
            score_mask = scores > score_thrs
            mask = cid_mask & score_mask
            cur_bboxes.append(det_bboxes[mask])
            cur_labels.append(labels[mask])
            cur_scores.append(scores[mask])
        det_bboxes = torch.cat(cur_bboxes)
        scores = torch.cat(cur_scores)
        labels = torch.cat(cur_labels)

        predictions_dict = {
            'map_scores_3d': scores.cpu(),
            'map_labels_3d': labels.cpu(),
            'map_pts_3d': det_bboxes.cpu(),
        }

        return predictions_dict
        # return det_bboxes.cpu().numpy(), cls_score.cpu().numpy()

    def onnx_export(self, **kwargs):
        raise NotImplementedError(f'TODO: replace 4 with self.n_control : {self.n_control}')

    def control_points_to_lane_points(self, lanes):
        """
        贝塞尔曲线转换 - 将控制点转换为采样点
        ============================================================================
        
        功能：
        - 将n_control个控制点表示的贝塞尔曲线转换为n_points个采样点
        - 用于可视化或评估（采样点更密集，曲线更平滑）
        
        贝塞尔曲线公式：
        B(t) = Σ C(n-1,j) * (1-t)^(n-1-j) * t^j * P_j
        其中：
        - P_j: 第j个控制点
        - C(n-1,j): 二项式系数
        - t ∈ [0, 1]: 参数
        
        示例：
        - 输入: [num_lanes, n_control*3] (例如：[600, 12] - 4个控制点)
        - 输出: [num_lanes, n_points*3] (例如：[600, 33] - 11个采样点)
        
        Args:
            lanes: 控制点坐标 [num_lanes, n_control*3]
            
        Returns:
            lanes: 采样点坐标 [num_lanes, n_points*3]
        """
        if lanes.shape[-1] == 0:
            return lanes.reshape(-1, 33)
        lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)  # [num_lanes, n_control, 3]

        # 二项式系数计算
        def comb(n, k):
            return factorial(n) // (factorial(k) * factorial(n - k))

        # 生成n_points个采样点
        n_points = 11  # 目标采样点数量
        n_control = lanes.shape[1]  # 控制点数量
        
        # 构建贝塞尔矩阵A: [n_points, n_control]
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)  # t ∈ [0, 1]
        
        # 计算贝塞尔基函数
        for i in range(n_points):
            for j in range(n_control):
                # 贝塞尔基函数: B_{n-1,j}(t)
                A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        
        bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
        # 矩阵乘法: [n_points, n_control] @ [num_lanes, n_control, 3] -> [num_lanes, n_points, 3]
        lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
        lanes = lanes.reshape(lanes.shape[0], -1)  # [num_lanes, n_points*3]

        return lanes