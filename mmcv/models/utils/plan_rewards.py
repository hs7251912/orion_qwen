import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PlanningRewardCalculator(nn.Module):
    """
    Calculate rewards for trajectory planning using various criteria.
    This is a vectorized implementation that can handle batched computations.
    """
    
    def __init__(self, 
                 reward_weights=None,
                 collision_threshold=0.5,
                 boundary_threshold=0.5,
                 smoothness_weights=(1.0, 1.0, 1.0),  # weights for velocity, acceleration, jerk
                 progress_weight=1.0,
                 l1_gt_weight=0.2):
        """
        Args:
            reward_weights (dict): Weights for different reward components
            collision_threshold (float): Distance threshold for collision detection
            boundary_threshold (float): Distance threshold for boundary violations
            smoothness_weights (tuple): Weights for velocity, acceleration, jerk penalties
            progress_weight (float): Weight for progress reward
            l1_gt_weight (float): Weight for L1 distance to ground truth
        """
        super().__init__()
        
        # Default reward weights
        default_weights = {
            'collision': 3.0,
            'boundary': 1.0, 
            'smoothness': 0.5,
            'progress': 1.0,
            'l1_gt': 0.2
        }
        
        if reward_weights is not None:
            default_weights.update(reward_weights)
        
        self.reward_weights = default_weights
        self.collision_threshold = collision_threshold
        self.boundary_threshold = boundary_threshold
        self.smoothness_weights = smoothness_weights
        self.progress_weight = progress_weight
        self.l1_gt_weight = l1_gt_weight
    
    def forward(self, 
                trajectories, 
                gt_trajectories=None,
                lane_preds=None,
                lane_scores=None,
                agent_preds=None,
                agent_fut_preds=None,
                agent_scores=None,
                ego_fut_masks=None):
        """
        Compute rewards for a batch of trajectory samples.
        
        Args:
            trajectories (torch.Tensor): Sampled trajectories [B, K, fut_ts, 2]
            gt_trajectories (torch.Tensor, optional): Ground truth trajectories [B, fut_ts, 2]
            lane_preds (torch.Tensor, optional): Lane predictions [B, num_lanes, num_pts, 2]
            lane_scores (torch.Tensor, optional): Lane confidence scores [B, num_lanes]
            agent_preds (torch.Tensor, optional): Agent predictions [B, num_agents, ...]
            agent_fut_preds (torch.Tensor, optional): Agent future predictions [B, num_agents, fut_ts, 2]
            agent_scores (torch.Tensor, optional): Agent confidence scores [B, num_agents]
            ego_fut_masks (torch.Tensor, optional): Valid timestep masks [B, fut_ts]
            
        Returns:
            torch.Tensor: Reward values [B, K]
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        # Initialize total rewards
        total_rewards = torch.zeros(B, K, device=device)
        
        # 1. Collision reward (negative penalty)
        if agent_preds is not None and self.reward_weights['collision'] > 0:
            collision_penalty = self._compute_collision_penalty(
                trajectories, agent_preds, agent_fut_preds, agent_scores
            )
            total_rewards -= self.reward_weights['collision'] * collision_penalty
        
        # 2. Boundary reward (negative penalty) 
        if lane_preds is not None and self.reward_weights['boundary'] > 0:
            boundary_penalty = self._compute_boundary_penalty(
                trajectories, lane_preds, lane_scores
            )
            total_rewards -= self.reward_weights['boundary'] * boundary_penalty
        
        # 3. Smoothness reward (negative penalty for non-smooth trajectories)
        if self.reward_weights['smoothness'] > 0:
            smoothness_penalty = self._compute_smoothness_penalty(trajectories)
            total_rewards -= self.reward_weights['smoothness'] * smoothness_penalty
        
        # 4. Progress reward (positive for forward motion)
        if self.reward_weights['progress'] > 0:
            progress_reward = self._compute_progress_reward(trajectories)
            total_rewards += self.reward_weights['progress'] * progress_reward
        
        # 5. L1 distance to ground truth (negative penalty)
        if gt_trajectories is not None and self.reward_weights['l1_gt'] > 0:
            l1_penalty = self._compute_l1_gt_penalty(trajectories, gt_trajectories)
            total_rewards -= self.reward_weights['l1_gt'] * l1_penalty
        
        # Apply validity masks if provided
        if ego_fut_masks is not None:
            # Apply mask to reduce reward for invalid timesteps
            valid_ratio = ego_fut_masks.float().mean(dim=-1, keepdim=True)  # [B, 1]
            total_rewards = total_rewards * valid_ratio
        
        return total_rewards
    
    def _compute_collision_penalty(self, trajectories, agent_preds, agent_fut_preds=None, agent_scores=None):
        """
        Compute collision penalty based on distance to other agents.
        
        Args:
            trajectories (torch.Tensor): Ego trajectories [B, K, fut_ts, 2]
            agent_preds (torch.Tensor): Agent current positions [B, num_agents, 2] 
            agent_fut_preds (torch.Tensor, optional): Agent future trajectories [B, num_agents, fut_ts, 2]
            agent_scores (torch.Tensor, optional): Agent confidence scores [B, num_agents]
            
        Returns:
            torch.Tensor: Collision penalties [B, K]
        """
        B, K, fut_ts, _ = trajectories.shape
        device = trajectories.device
        
        if agent_fut_preds is not None:
            # Use predicted agent trajectories
            agent_positions = agent_fut_preds  # [B, num_agents, fut_ts, 2]
            num_agents = agent_positions.shape[1]
            
            # Expand trajectories for comparison: [B, K, fut_ts, 1, 2]
            ego_expanded = trajectories.unsqueeze(3)  
            # Expand agent positions: [B, 1, fut_ts, num_agents, 2]
            agent_expanded = agent_positions.unsqueeze(1)
            
            # Compute distances: [B, K, fut_ts, num_agents]
            distances = torch.norm(ego_expanded - agent_expanded, dim=-1)
            
        else:
            # Use current agent positions, assume they stay static
            num_agents = agent_preds.shape[1]
            # Expand for all timesteps: [B, 1, fut_ts, num_agents, 2]
            agent_static = agent_preds.unsqueeze(1).unsqueeze(2).expand(B, 1, fut_ts, num_agents, 2)
            ego_expanded = trajectories.unsqueeze(3)  # [B, K, fut_ts, 1, 2]
            
            # Compute distances: [B, K, fut_ts, num_agents]
            distances = torch.norm(ego_expanded - agent_static, dim=-1)
        
        # Apply confidence weights if available
        if agent_scores is not None:
            confidence_weights = agent_scores.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, num_agents]
            distances = distances * confidence_weights
        
        # Collision penalty (inverse exponential of minimum distance)
        min_distances = distances.min(dim=-1)[0]  # [B, K, fut_ts]
        collision_mask = min_distances < self.collision_threshold
        
        # Penalty increases exponentially as distance decreases
        penalties = torch.exp(-min_distances / self.collision_threshold) * collision_mask.float()
        
        # Average over timesteps
        collision_penalty = penalties.mean(dim=-1)  # [B, K]
        
        return collision_penalty
    
    def _compute_boundary_penalty(self, trajectories, lane_preds, lane_scores=None):
        """
        Compute penalty for going out of lane boundaries.
        
        Args:
            trajectories (torch.Tensor): Ego trajectories [B, K, fut_ts, 2]
            lane_preds (torch.Tensor): Lane boundary predictions [B, num_lanes, num_pts, 2]
            lane_scores (torch.Tensor, optional): Lane confidence scores [B, num_lanes]
            
        Returns:
            torch.Tensor: Boundary penalties [B, K]
        """
        B, K, fut_ts, _ = trajectories.shape
        num_lanes, num_pts = lane_preds.shape[1:3]
        device = trajectories.device
        
        boundary_penalties = torch.zeros(B, K, device=device)
        
        for b in range(B):
            for k in range(K):
                traj = trajectories[b, k]  # [fut_ts, 2]
                lanes = lane_preds[b]  # [num_lanes, num_pts, 2]
                
                # For each trajectory point, find distance to closest lane boundary
                traj_penalties = []
                for t in range(fut_ts):
                    point = traj[t]  # [2]
                    min_dist = float('inf')
                    
                    for lane_idx in range(num_lanes):
                        lane = lanes[lane_idx]  # [num_pts, 2]
                        
                        # Skip invalid lanes (all zeros)
                        if torch.allclose(lane, torch.zeros_like(lane)):
                            continue
                        
                        # Compute distance to lane boundary
                        distances = torch.norm(lane - point.unsqueeze(0), dim=-1)
                        min_lane_dist = distances.min()
                        
                        if lane_scores is not None:
                            # Weight by lane confidence
                            min_lane_dist = min_lane_dist / (lane_scores[b, lane_idx] + 1e-8)
                        
                        min_dist = min(min_dist, min_lane_dist.item())
                    
                    # Penalty for being too close to boundaries
                    if min_dist < self.boundary_threshold:
                        penalty = torch.exp(-(min_dist / self.boundary_threshold))
                        traj_penalties.append(penalty)
                    else:
                        traj_penalties.append(0.0)
                
                boundary_penalties[b, k] = torch.tensor(traj_penalties).mean()
        
        return boundary_penalties
    
    def _compute_smoothness_penalty(self, trajectories):
        """
        Compute penalty for non-smooth trajectories (high curvature, acceleration, jerk).
        
        Args:
            trajectories (torch.Tensor): Trajectories [B, K, fut_ts, 2]
            
        Returns:
            torch.Tensor: Smoothness penalties [B, K]
        """
        B, K, fut_ts, _ = trajectories.shape
        
        # Compute velocity (first derivative)
        velocity = trajectories[:, :, 1:] - trajectories[:, :, :-1]  # [B, K, fut_ts-1, 2]
        velocity_magnitude = torch.norm(velocity, dim=-1)  # [B, K, fut_ts-1]
        
        # Compute acceleration (second derivative)
        acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]  # [B, K, fut_ts-2, 2]
        acceleration_magnitude = torch.norm(acceleration, dim=-1)  # [B, K, fut_ts-2]
        
        # Compute jerk (third derivative) 
        jerk = acceleration[:, :, 1:] - acceleration[:, :, :-1]  # [B, K, fut_ts-3, 2]
        jerk_magnitude = torch.norm(jerk, dim=-1)  # [B, K, fut_ts-3]
        
        # Smoothness penalties
        vel_penalty = (velocity_magnitude ** 2).mean(dim=-1) * self.smoothness_weights[0]
        acc_penalty = (acceleration_magnitude ** 2).mean(dim=-1) * self.smoothness_weights[1]
        jerk_penalty = (jerk_magnitude ** 2).mean(dim=-1) * self.smoothness_weights[2]
        
        total_penalty = vel_penalty + acc_penalty + jerk_penalty
        
        return total_penalty
    
    def _compute_progress_reward(self, trajectories):
        """
        Compute reward for forward progress along desired direction.
        
        Args:
            trajectories (torch.Tensor): Trajectories [B, K, fut_ts, 2]
            
        Returns:
            torch.Tensor: Progress rewards [B, K]
        """
        # Simple progress: distance from start to end point
        start_points = trajectories[:, :, 0, :]  # [B, K, 2]
        end_points = trajectories[:, :, -1, :]   # [B, K, 2]
        
        # Forward progress (positive x direction assumed)
        forward_progress = end_points[:, :, 0] - start_points[:, :, 0]  # [B, K]
        
        # Total distance traveled
        total_distance = torch.norm(end_points - start_points, dim=-1)  # [B, K]
        
        # Reward combination of forward progress and total distance
        progress_reward = forward_progress + 0.5 * total_distance
        
        return progress_reward
    
    def _compute_l1_gt_penalty(self, trajectories, gt_trajectories):
        """
        Compute L1 distance penalty to ground truth trajectories.
        
        Args:
            trajectories (torch.Tensor): Predicted trajectories [B, K, fut_ts, 2]
            gt_trajectories (torch.Tensor): Ground truth trajectories [B, fut_ts, 2]
            
        Returns:
            torch.Tensor: L1 penalties [B, K]
        """
        B, K, fut_ts, _ = trajectories.shape
        
        # Expand GT trajectories for comparison
        gt_expanded = gt_trajectories.unsqueeze(1).expand(B, K, fut_ts, 2)  # [B, K, fut_ts, 2]
        
        # Compute L1 distance
        l1_distances = torch.abs(trajectories - gt_expanded).sum(dim=-1)  # [B, K, fut_ts]
        
        # Average over timesteps
        l1_penalty = l1_distances.mean(dim=-1)  # [B, K]
        
        return l1_penalty


def create_planning_reward_calculator(cfg):
    """
    Factory function to create reward calculator from config.
    
    Args:
        cfg (dict): Configuration dictionary
        
    Returns:
        PlanningRewardCalculator: Configured reward calculator
    """
    return PlanningRewardCalculator(
        reward_weights=cfg.get('reward_weights', None),
        collision_threshold=cfg.get('collision_threshold', 0.5),
        boundary_threshold=cfg.get('boundary_threshold', 0.5),
        smoothness_weights=cfg.get('smoothness_weights', (1.0, 1.0, 1.0)),
        progress_weight=cfg.get('progress_weight', 1.0),
        l1_gt_weight=cfg.get('l1_gt_weight', 0.2)
    )
