import torch
import torch.nn as nn
import numpy as np


class HumanDrivingPreferenceRewards(nn.Module):
    """
    Human driving behavior preference modeling for GRPO training.
    
    This module implements reward functions that align with human driving preferences,
    including safety, comfort, efficiency, and social norms.
    """
    
    def __init__(self, 
                 safety_weight=4.0,      # 安全性：最高优先级
                 comfort_weight=2.0,     # 舒适性：重要
                 efficiency_weight=1.5,  # 效率性：适中
                 social_weight=1.0,      # 社会规范：基础
                 legality_weight=3.0):   # 合法性：很高
        """
        Args:
            safety_weight: 安全性权重 (碰撞避免、安全距离)
            comfort_weight: 舒适性权重 (平滑驾驶、乘客舒适)
            efficiency_weight: 效率性权重 (路径优化、时间效率)
            social_weight: 社会规范权重 (礼让行为、交通礼仪)
            legality_weight: 合法性权重 (交通规则、法规遵守)
        """
        super().__init__()
        self.safety_weight = safety_weight
        self.comfort_weight = comfort_weight
        self.efficiency_weight = efficiency_weight
        self.social_weight = social_weight
        self.legality_weight = legality_weight
        
    def forward(self, 
                trajectories,          # (B, K, fut_ts, 2) 采样轨迹
                gt_trajectories,       # (B, fut_ts, 2) 专家轨迹
                lane_preds,           # 车道预测
                lane_scores,          # 车道置信度
                agent_preds=None,     # 其他车辆预测
                traffic_states=None,  # 交通信号状态
                ego_fut_masks=None):  # 有效性掩码
        """
        计算基于人类驾驶偏好的奖励。
        
        Returns:
            total_rewards: 总奖励 (B, K)
            reward_components: 各组件奖励详情
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        # 初始化总奖励
        total_rewards = torch.zeros(B, K, device=device)
        reward_components = {}
        
        # 1. 安全性奖励 (最高优先级)
        safety_reward = self._compute_safety_rewards(
            trajectories, agent_preds, lane_preds, lane_scores
        )
        total_rewards += self.safety_weight * safety_reward
        reward_components['safety'] = safety_reward
        
        # 2. 舒适性奖励
        comfort_reward = self._compute_comfort_rewards(trajectories)
        total_rewards += self.comfort_weight * comfort_reward
        reward_components['comfort'] = comfort_reward
        
        # 3. 效率性奖励
        efficiency_reward = self._compute_efficiency_rewards(
            trajectories, gt_trajectories, lane_preds
        )
        total_rewards += self.efficiency_weight * efficiency_reward
        reward_components['efficiency'] = efficiency_reward
        
        # 4. 社会规范奖励
        social_reward = self._compute_social_rewards(
            trajectories, agent_preds, lane_preds
        )
        total_rewards += self.social_weight * social_reward
        reward_components['social'] = social_reward
        
        # 5. 合法性奖励
        legality_reward = self._compute_legality_rewards(
            trajectories, traffic_states, lane_preds
        )
        total_rewards += self.legality_weight * legality_reward
        reward_components['legality'] = legality_reward
        
        # 应用有效性掩码
        if ego_fut_masks is not None:
            valid_ratio = ego_fut_masks.float().mean(dim=-1, keepdim=True)
            total_rewards = total_rewards * valid_ratio
        
        reward_components['total'] = total_rewards
        return total_rewards, reward_components
    
    def _compute_safety_rewards(self, trajectories, agent_preds, lane_preds, lane_scores):
        """
        计算安全性奖励：
        - 与其他车辆保持安全距离
        - 避免危险机动
        - 保持在车道内行驶
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        safety_rewards = torch.zeros(B, K, device=device)
        
        # 1. 车辆间安全距离奖励
        if agent_preds is not None:
            # 计算与所有其他车辆的最小距离
            for t in range(fut_ts):
                ego_pos = trajectories[:, :, t, :]  # (B, K, 2)
                agent_pos = agent_preds  # (B, num_agents, 2)
                
                # 扩展维度进行距离计算
                ego_expanded = ego_pos.unsqueeze(2)  # (B, K, 1, 2)
                agent_expanded = agent_pos.unsqueeze(1)  # (B, 1, num_agents, 2)
                
                distances = torch.norm(ego_expanded - agent_expanded, dim=-1)  # (B, K, num_agents)
                min_distances = distances.min(dim=-1)[0]  # (B, K)
                
                # 安全距离奖励 (指数衰减，距离越近奖励越低)
                safe_distance_threshold = 3.0  # 3米安全距离
                safety_rewards += torch.exp(min_distances / safe_distance_threshold - 1)
        
        # 2. 车道保持奖励
        if lane_preds is not None:
            lane_keeping_reward = self._compute_lane_keeping_reward(
                trajectories, lane_preds, lane_scores
            )
            safety_rewards += lane_keeping_reward
        
        # 3. 急刹车/急转弯惩罚
        emergency_penalty = self._compute_emergency_maneuver_penalty(trajectories)
        safety_rewards -= emergency_penalty
        
        return safety_rewards
    
    def _compute_comfort_rewards(self, trajectories):
        """
        计算舒适性奖励：
        - 平滑的加速度变化
        - 避免急刹车和急转弯
        - 稳定的驾驶节奏
        """
        B, K, fut_ts, _ = trajectories.shape
        
        # 计算速度、加速度、急动度(jerk)
        velocities = trajectories[:, :, 1:] - trajectories[:, :, :-1]  # (B, K, fut_ts-1, 2)
        accelerations = velocities[:, :, 1:] - velocities[:, :, :-1]  # (B, K, fut_ts-2, 2)
        jerks = accelerations[:, :, 1:] - accelerations[:, :, :-1]  # (B, K, fut_ts-3, 2)
        
        # 1. 平滑加速度奖励
        accel_smoothness = -torch.norm(accelerations, dim=-1).std(dim=-1)  # (B, K)
        
        # 2. 低急动度奖励 (减少颠簸感)
        jerk_penalty = -torch.norm(jerks, dim=-1).mean(dim=-1)  # (B, K)
        
        # 3. 稳定速度奖励
        speed_stability = -torch.norm(velocities, dim=-1).std(dim=-1)  # (B, K)
        
        comfort_reward = accel_smoothness + jerk_penalty + speed_stability
        return comfort_reward
    
    def _compute_efficiency_rewards(self, trajectories, gt_trajectories, lane_preds):
        """
        计算效率性奖励：
        - 朝目标方向的进度
        - 避免不必要的绕行
        - 合理的速度选择
        """
        B, K, fut_ts, _ = trajectories.shape
        
        # 1. 目标导向进度奖励
        start_pos = trajectories[:, :, 0, :]  # (B, K, 2)
        end_pos = trajectories[:, :, -1, :]   # (B, K, 2)
        
        # 假设专家轨迹代表最优路径
        if gt_trajectories is not None:
            gt_start = gt_trajectories[:, 0, :]  # (B, 2)
            gt_end = gt_trajectories[:, -1, :]   # (B, 2)
            gt_direction = gt_end - gt_start  # (B, 2)
            gt_direction = gt_direction / (torch.norm(gt_direction, dim=-1, keepdim=True) + 1e-8)
            
            # 计算轨迹方向与专家方向的一致性
            traj_direction = end_pos - start_pos  # (B, K, 2)
            traj_direction = traj_direction / (torch.norm(traj_direction, dim=-1, keepdim=True) + 1e-8)
            
            direction_alignment = torch.sum(
                traj_direction * gt_direction.unsqueeze(1), dim=-1
            )  # (B, K)
        else:
            # 默认向前进度
            direction_alignment = (end_pos - start_pos)[:, :, 1]  # Y方向进度
        
        # 2. 路径效率奖励 (避免过度绕行)
        total_distance = torch.norm(
            trajectories[:, :, 1:] - trajectories[:, :, :-1], dim=-1
        ).sum(dim=-1)  # (B, K)
        
        straight_distance = torch.norm(end_pos - start_pos, dim=-1)  # (B, K)
        path_efficiency = straight_distance / (total_distance + 1e-8)  # (B, K)
        
        # 3. 速度适应性奖励
        velocities = torch.norm(
            trajectories[:, :, 1:] - trajectories[:, :, :-1], dim=-1
        )  # (B, K, fut_ts-1)
        
        # 适中速度奖励 (不要太快也不要太慢)
        optimal_speed = 5.0  # 5 m/s (约18 km/h，城市驾驶)
        speed_penalty = -torch.abs(velocities.mean(dim=-1) - optimal_speed)  # (B, K)
        
        efficiency_reward = direction_alignment + path_efficiency + speed_penalty * 0.3
        return efficiency_reward
    
    def _compute_social_rewards(self, trajectories, agent_preds, lane_preds):
        """
        计算社会规范奖励：
        - 礼让行为
        - 变道礼仪
        - 与其他车辆的协调
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        social_rewards = torch.zeros(B, K, device=device)
        
        if agent_preds is not None:
            # 1. 礼让行为奖励
            # 检测是否在其他车辆附近减速让行
            for t in range(1, fut_ts):
                current_pos = trajectories[:, :, t, :]  # (B, K, 2)
                prev_pos = trajectories[:, :, t-1, :]   # (B, K, 2)
                
                ego_speed = torch.norm(current_pos - prev_pos, dim=-1)  # (B, K)
                
                # 计算与前方车辆的距离
                agent_pos = agent_preds  # (B, num_agents, 2)
                distances = torch.norm(
                    current_pos.unsqueeze(2) - agent_pos.unsqueeze(1), dim=-1
                )  # (B, K, num_agents)
                
                min_front_distance = distances.min(dim=-1)[0]  # (B, K)
                
                # 礼让奖励：在车辆较近时适当减速
                yielding_bonus = torch.where(
                    (min_front_distance < 8.0) & (ego_speed < 3.0),  # 8米内减速到3m/s以下
                    torch.tensor(1.0, device=device),
                    torch.tensor(0.0, device=device)
                )
                social_rewards += yielding_bonus * 0.1
        
        # 2. 平滑变道奖励
        if lane_preds is not None:
            lane_change_smoothness = self._compute_lane_change_smoothness(
                trajectories, lane_preds
            )
            social_rewards += lane_change_smoothness
        
        return social_rewards
    
    def _compute_legality_rewards(self, trajectories, traffic_states, lane_preds):
        """
        计算合法性奖励：
        - 遵守交通信号
        - 遵守车道规则
        - 遵守速度限制
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        legality_rewards = torch.zeros(B, K, device=device)
        
        # 1. 交通信号遵守奖励
        if traffic_states is not None:
            # 红灯停车、绿灯通行奖励
            signal_compliance = self._compute_traffic_signal_compliance(
                trajectories, traffic_states
            )
            legality_rewards += signal_compliance
        
        # 2. 车道合规性奖励
        if lane_preds is not None:
            lane_compliance = self._compute_lane_compliance(
                trajectories, lane_preds
            )
            legality_rewards += lane_compliance
        
        # 3. 速度限制遵守
        speed_limit_compliance = self._compute_speed_limit_compliance(trajectories)
        legality_rewards += speed_limit_compliance
        
        return legality_rewards
    
    def _compute_lane_keeping_reward(self, trajectories, lane_preds, lane_scores):
        """计算车道保持奖励"""
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        lane_keeping_rewards = torch.zeros(B, K, device=device)
        
        # 为每个轨迹计算到最近车道中心的距离
        for b in range(B):
            for k in range(K):
                traj = trajectories[b, k]  # (fut_ts, 2)
                lanes = lane_preds[b]  # (num_lanes, num_pts, 2)
                scores = lane_scores[b] if lane_scores is not None else None
                
                total_distance_penalty = 0.0
                for t in range(fut_ts):
                    point = traj[t]
                    min_dist = float('inf')
                    
                    for lane_idx, lane in enumerate(lanes):
                        if torch.allclose(lane, torch.zeros_like(lane)):
                            continue
                        
                        # 计算点到车道中心线的距离
                        distances = torch.norm(lane - point.unsqueeze(0), dim=-1)
                        lane_dist = distances.min().item()
                        
                        # 考虑车道置信度
                        if scores is not None:
                            lane_dist = lane_dist / (scores[lane_idx].item() + 1e-8)
                        
                        min_dist = min(min_dist, lane_dist)
                    
                    # 距离车道中心越近奖励越高
                    if min_dist < 2.0:  # 2米内认为在车道内
                        total_distance_penalty += min_dist
                
                lane_keeping_rewards[b, k] = -total_distance_penalty / fut_ts
        
        return lane_keeping_rewards
    
    def _compute_emergency_maneuver_penalty(self, trajectories):
        """计算急刹车/急转弯惩罚"""
        B, K, fut_ts, _ = trajectories.shape
        
        # 计算加速度幅值
        velocities = trajectories[:, :, 1:] - trajectories[:, :, :-1]
        accelerations = velocities[:, :, 1:] - velocities[:, :, :-1]
        accel_magnitudes = torch.norm(accelerations, dim=-1)  # (B, K, fut_ts-2)
        
        # 急刹车阈值 (>3 m/s²)
        emergency_threshold = 3.0
        emergency_mask = accel_magnitudes > emergency_threshold
        
        # 急刹车惩罚
        emergency_penalty = emergency_mask.float().sum(dim=-1)  # (B, K)
        
        return emergency_penalty
    
    def _compute_lane_change_smoothness(self, trajectories, lane_preds):
        """计算变道平滑性奖励"""
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        # 计算横向移动平滑性
        lateral_velocities = trajectories[:, :, 1:, 0] - trajectories[:, :, :-1, 0]  # X方向速度
        lateral_smoothness = -torch.std(lateral_velocities, dim=-1)  # (B, K)
        
        return lateral_smoothness
    
    def _compute_traffic_signal_compliance(self, trajectories, traffic_states):
        """计算交通信号遵守奖励"""
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        # 简化实现：基于速度和信号状态
        velocities = torch.norm(
            trajectories[:, :, 1:] - trajectories[:, :, :-1], dim=-1
        )  # (B, K, fut_ts-1)
        
        avg_speed = velocities.mean(dim=-1)  # (B, K)
        
        # 红灯时低速奖励，绿灯时正常速度奖励
        # 这里需要根据实际的traffic_states格式进行调整
        signal_compliance = torch.ones_like(avg_speed)  # 占位符
        
        return signal_compliance
    
    def _compute_lane_compliance(self, trajectories, lane_preds):
        """计算车道合规性奖励"""
        # 检查是否违反车道边界
        return self._compute_lane_keeping_reward(trajectories, lane_preds, None)
    
    def _compute_speed_limit_compliance(self, trajectories):
        """计算速度限制遵守奖励"""
        B, K, fut_ts, _ = trajectories.shape
        
        velocities = torch.norm(
            trajectories[:, :, 1:] - trajectories[:, :, :-1], dim=-1
        )  # (B, K, fut_ts-1)
        
        # 城市道路速度限制 (约15 m/s = 54 km/h)
        speed_limit = 15.0
        speed_violations = torch.relu(velocities - speed_limit)  # 超速部分
        speed_compliance = -speed_violations.mean(dim=-1)  # (B, K)
        
        return speed_compliance


class AdaptiveRewardScheduler(nn.Module):
    """
    自适应奖励调度器，根据训练进度调整不同奖励权重。
    
    在训练初期更重视安全性和合法性，
    随着训练进展逐渐增加效率性和社会性的权重。
    """
    
    def __init__(self, 
                 initial_weights,
                 final_weights,
                 transition_steps=10000):
        super().__init__()
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.transition_steps = transition_steps
        self.step_count = 0
    
    def get_current_weights(self):
        """获取当前训练步数对应的权重"""
        if self.step_count >= self.transition_steps:
            return self.final_weights
        
        # 线性插值
        alpha = self.step_count / self.transition_steps
        current_weights = {}
        
        for key in self.initial_weights:
            initial_val = self.initial_weights[key]
            final_val = self.final_weights[key]
            current_weights[key] = initial_val + alpha * (final_val - initial_val)
        
        return current_weights
    
    def step(self):
        """更新训练步数"""
        self.step_count += 1


def create_human_preference_calculator(cfg):
    """
    创建人类驾驶偏好奖励计算器的工厂函数。
    
    Args:
        cfg: 配置字典，包含权重和参数设置
        
    Returns:
        HumanDrivingPreferenceRewards: 配置好的奖励计算器
    """
    return HumanDrivingPreferenceRewards(
        safety_weight=cfg.get('safety_weight', 4.0),
        comfort_weight=cfg.get('comfort_weight', 2.0),
        efficiency_weight=cfg.get('efficiency_weight', 1.5),
        social_weight=cfg.get('social_weight', 1.0),
        legality_weight=cfg.get('legality_weight', 3.0)
    )




