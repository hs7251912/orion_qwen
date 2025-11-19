import os
import cv2
import numpy as np
import torch
from mmcv.utils import mkdir_or_exist, ProgressBar
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def project_3d_to_2d(points_3d, lidar2img, min_depth=0.1):
    """Project 3D points in lidar coordinates to 2D image coordinates.
    
    This function performs the complete projection pipeline:
    1. Convert 3D points to homogeneous coordinates
    2. Apply lidar2img transformation (combines lidar2cam and camera intrinsics)
    3. Normalize by depth to get pixel coordinates
    
    Args:
        points_3d (torch.Tensor): 3D points in lidar coordinates [N, 3]
            - X: forward, Y: left, Z: up in lidar frame
        lidar2img (torch.Tensor): Projection matrix from lidar to image [3, 4]
            - Combines camera intrinsics K and lidar2cam extrinsics
        min_depth (float): Minimum valid depth in camera frame (default: 0.1m)
    
    Returns:
        torch.Tensor: 2D points in image pixel coordinates [N, 2]
            Points behind camera (depth < min_depth) will have invalid coordinates
    """
    # Convert to homogeneous coordinates [N, 4]
    N = points_3d.shape[0]
    ones = torch.ones((N, 1), device=points_3d.device, dtype=points_3d.dtype)
    points_3d_homo = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    # Apply transformation: points_img = lidar2img @ points_lidar
    # lidar2img is [3, 4], points_3d_homo.T is [4, N]
    points_img = torch.matmul(lidar2img, points_3d_homo.T).T  # [N, 3]
    
    # Normalize by depth (z-coordinate in camera frame) to get pixel coordinates
    # points_img[:, 2] is the depth
    depth = points_img[:, 2:3]  # [N, 1]
    
    # Filter points behind camera: set invalid depth to a large value to push them off-screen
    valid_mask = depth.squeeze() >= min_depth
    depth_safe = depth.clone()
    depth_safe[~valid_mask.unsqueeze(1)] = 1.0  # Avoid division by very small numbers
    
    # Get 2D pixel coordinates [u, v]
    points_2d = points_img[:, :2] / depth_safe  # [N, 2]
    
    # Mark invalid points by setting them far outside image bounds
    points_2d[~valid_mask] = -999999.0
    
    return points_2d


def compute_future_track(data_info, dataset, actual_idx, future_frames=6, sample_rate=1):
    """计算未来轨迹的绝对位置（在当前帧LiDAR坐标系中）
    
    Args:
        data_info (dict): 当前帧的数据信息
        dataset: 数据集对象
        actual_idx (int): 当前帧在数据集中的实际索引
        future_frames (int): 未来帧数量
        sample_rate (int): 采样间隔（默认1表示连续帧）
    
    Returns:
        np.ndarray or None: 未来轨迹点 [future_frames, 2]，如果数据无效返回 None
    """
    cur_frame = dataset.data_infos[actual_idx]
    future_track = []
    world2lidar_cur = np.array(cur_frame['sensors']['LIDAR_TOP']['world2lidar'])
    
    # 遍历未来帧（从下一帧开始，连续采样）
    for i in range(1, future_frames + 1):
        fut_idx = actual_idx + i * sample_rate
        
        # 边界检查
        if fut_idx >= len(dataset.data_infos):
            break
        
        fut_frame = dataset.data_infos[fut_idx]
        
        # 场景连续性检查
        if fut_frame['folder'] != cur_frame['folder']:
            break
        
        # 计算未来帧在当前帧坐标系中的位置
        world2lidar_fut = np.array(fut_frame['sensors']['LIDAR_TOP']['world2lidar'])
        fut2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_fut)
        xy = fut2cur_lidar[0:2, 3]
        future_track.append(xy)
    
    # 如果没有有效的未来帧，返回 None
    if len(future_track) == 0:
        return None
    
    return np.array(future_track)


def visualize_trajectory(image, traj_2d, save_path, sample_idx=0, cmd_idx=None, gt_traj_2d=None):
    """Draw trajectory on image and save.
    
    Args:
        image (np.ndarray): Image array [H, W, 3]
        traj_2d (torch.Tensor): 2D predicted trajectory points in pixel coordinates [num_points, 2]
        save_path (str): Path to save the visualization
        sample_idx (int): Sample index for filename
        cmd_idx (int, optional): Command index for the trajectory
        gt_traj_2d (np.ndarray, optional): 2D ground truth trajectory points in pixel coordinates [num_points, 2]
    """
    # Convert image to BGR for OpenCV
    img_vis = image.copy()
    if img_vis.shape[2] == 3:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    # Draw ground truth trajectory first (if available) so prediction appears on top
    if gt_traj_2d is not None:
        gt_points = gt_traj_2d.astype(int)
        num_gt = len(gt_points)
        
        # Draw GT trajectory lines using blue color
        for i in range(num_gt - 1):
            p1, p2 = gt_points[i], gt_points[i + 1]
            if (0 <= p1[0] < img_vis.shape[1] and 0 <= p1[1] < img_vis.shape[0] and
                0 <= p2[0] < img_vis.shape[1] and 0 <= p2[1] < img_vis.shape[0]):
                cv2.line(img_vis, tuple(p1), tuple(p2), (255, 0, 0), 2)  # 蓝色线
        
        # Draw GT trajectory points (with clamping for out-of-bounds points)
        for i in range(num_gt):
            point = gt_points[i]
            
            # Skip invalid points (marked as -999999 by projection)
            if point[0] < -999000 or point[1] < -999000:
                continue
            
            # Clamp point to image boundaries
            point_clamped = np.array([
                np.clip(point[0], 0, img_vis.shape[1] - 1),
                np.clip(point[1], 0, img_vis.shape[0] - 1)
            ])
            
            # Check if within bounds
            is_in_bounds = (0 <= point[0] < img_vis.shape[1] and 0 <= point[1] < img_vis.shape[0])
            
            # Draw GT point
            cv2.circle(img_vis, tuple(point_clamped), 6, (255, 0, 0), -1)  # 蓝色圆点
            cv2.circle(img_vis, tuple(point_clamped), 8, (255, 255, 255), 2)  # 白色边框
            
            # Mark if out of bounds
            if not is_in_bounds:
                cv2.drawMarker(img_vis, tuple(point_clamped), (255, 0, 0), 
                              cv2.MARKER_TRIANGLE_DOWN, 15, 2)
    
    # Draw predicted trajectory (if provided)
    if traj_2d is not None:
        # Get predicted trajectory points as integers
        traj_points = traj_2d.cpu().numpy().astype(int)
        num_points = len(traj_points)
        
        # Create color gradient from green (start) to red (end)
        colors = cm.get_cmap('RdYlGn_r')(np.linspace(0, 1, num_points))[:, :3] * 255
        
        # Draw trajectory lines first (so they appear under the points)
        for i in range(num_points - 1):
            p1 = traj_points[i]
            p2 = traj_points[i + 1]
            
            # Check if both points are within bounds
            if (0 <= p1[0] < img_vis.shape[1] and 0 <= p1[1] < img_vis.shape[0] and
                0 <= p2[0] < img_vis.shape[1] and 0 <= p2[1] < img_vis.shape[0]):
                color = tuple(map(int, colors[i][::-1]))  # Convert RGB to BGR
                cv2.line(img_vis, (p1[0], p1[1]), (p2[0], p2[1]), color, 4)
        
        # Draw trajectory points
        for i in range(num_points):
            point = traj_points[i]
            color = tuple(map(int, colors[i][::-1]))  # Convert RGB to BGR
            
            # Skip invalid points (marked as -999999 by projection)
            if point[0] < -999000 or point[1] < -999000:
                continue
            
            # Clamp point to image boundaries for visualization
            point_clamped = np.array([
                np.clip(point[0], 0, img_vis.shape[1] - 1),
                np.clip(point[1], 0, img_vis.shape[0] - 1)
            ])
            
            # Check if original point was within bounds
            is_in_bounds = (0 <= point[0] < img_vis.shape[1] and 0 <= point[1] < img_vis.shape[0])
            
            # Draw circle at point (clamped to image boundary if needed)
            cv2.circle(img_vis, tuple(point_clamped), 8, color, -1)
            cv2.circle(img_vis, tuple(point_clamped), 10, (255, 255, 255), 2)  # White border
            
            # If point is out of bounds, draw a small arrow indicator
            if not is_in_bounds:
                # Draw arrow pointing to the clamped position
                cv2.drawMarker(img_vis, tuple(point_clamped), (0, 0, 255), 
                              cv2.MARKER_TRIANGLE_DOWN, 15, 2)
            
            # Add timestamp label
            label = f't{i}'
            label_pos = (int(point_clamped[0]) + 15, int(point_clamped[1]) - 5)
            # Keep label within image
            label_pos = (
                np.clip(label_pos[0], 0, img_vis.shape[1] - 50),
                np.clip(label_pos[1], 20, img_vis.shape[0] - 10)
            )
            cv2.putText(img_vis, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add legend and information
    title = 'Ego Trajectory (3D -> 2D Projection)'
    cv2.putText(img_vis, title, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_vis, f'Sample: {sample_idx}', (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if cmd_idx is not None:
        cmd_names = ['Turn Right', 'Go Straight', 'Turn Left']
        cmd_name = cmd_names[cmd_idx] if 0 <= cmd_idx < len(cmd_names) else f'Command {cmd_idx}'
        cv2.putText(img_vis, f'Command: {cmd_name}', (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend based on what's being displayed
    legend_y = 120
    if traj_2d is not None and gt_traj_2d is not None:
        # Both prediction and GT
        cv2.circle(img_vis, (20, legend_y), 6, (0, 0, 255), -1)  # 红色：预测
        cv2.putText(img_vis, 'Prediction', (35, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(img_vis, (20, legend_y + 25), 6, (255, 0, 0), -1)  # 蓝色：GT
        cv2.putText(img_vis, 'Ground Truth', (35, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif traj_2d is not None:
        # Only prediction
        cv2.putText(img_vis, 'Prediction Only', (20, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif gt_traj_2d is not None:
        # Only GT
        cv2.putText(img_vis, 'Ground Truth Only', (20, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save image
    cv2.imwrite(save_path, img_vis)


def visualize_trajectories(outputs, data_loader, save_dir, show_pred=True, show_gt=True):
    """Main function to visualize trajectories from model outputs.
    
    Args:
        outputs (dict): Model outputs containing bbox_results
        data_loader: DataLoader used for testing
        save_dir (str): Directory to save visualization results
        show_pred (bool): Whether to show predicted trajectory (default: True)
        show_gt (bool): Whether to show ground truth trajectory (default: True)
    """
    mkdir_or_exist(save_dir)
    
    bbox_results = outputs['bbox_results']
    dataset = data_loader.dataset
    
    # Handle Subset wrapper (common in validation/test splits)
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        actual_dataset = dataset.dataset
        subset_indices = dataset.indices
    else:
        actual_dataset = dataset
        subset_indices = None
    
    print(f"\nVisualizing trajectories to {save_dir}")
    prog_bar = ProgressBar(len(bbox_results))
    
    # Iterate through results
    for idx, bbox_result in enumerate(bbox_results):
        try:
            # Check if trajectory predictions exist
            # ego_fut_preds is stored directly in bbox_result, not in pts_bbox
            if 'ego_fut_preds' not in bbox_result['pts_bbox']:
                prog_bar.update()
                continue
                
            ego_fut_preds = bbox_result['pts_bbox']['ego_fut_preds']
            
            # Skip if no valid predictions
            if ego_fut_preds is None or (hasattr(ego_fut_preds, '__len__') and len(ego_fut_preds) == 0):
                prog_bar.update()
                continue
            
            # Get ego_fut_cmd if available to select the correct trajectory
            if 'ego_fut_cmd' in bbox_result:
                ego_fut_cmd = bbox_result['ego_fut_cmd']
                # ego_fut_cmd shape: [1, 3] (batch, num_commands)
                if ego_fut_cmd.ndim >= 2:
                    ego_fut_cmd = ego_fut_cmd[0, 0] if ego_fut_cmd.ndim == 3 else ego_fut_cmd[0]
                ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0].item() if torch.any(ego_fut_cmd) else 0
            else:
                ego_fut_cmd_idx = 0  # Default to first trajectory
            
            # ego_fut_preds shape: [num_modes, time_steps, 2]
            # Select the trajectory corresponding to the command
            if ego_fut_preds.ndim == 3:
                ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]  # [time_steps, 2]
            else:
                ego_fut_pred = ego_fut_preds  # Already [time_steps, 2]
            
            # Apply cumsum to get absolute positions (predictions are deltas)
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)  # [time_steps, 2]
            
            # Map index to actual dataset index if using Subset
            actual_idx = subset_indices[idx] if subset_indices is not None else idx
            
            # Get corresponding data info
            if actual_idx >= len(actual_dataset.data_infos):
                print(f"Warning: No data info for index {actual_idx}")
                prog_bar.update()
                continue
                
            data_info = actual_dataset.data_infos[actual_idx]
            
            # Check if front camera exists
            if 'CAM_FRONT' not in data_info['sensors']:
                print(f"Warning: No front camera info for sample {idx}")
                prog_bar.update()
                continue
            
            # Get image path
            cam_info = data_info['sensors']['CAM_FRONT']
            img_filename = cam_info.get('data_path', '')
            
            # Build full image path
            if hasattr(actual_dataset, 'data_root') and actual_dataset.data_root:
                img_path = os.path.join(actual_dataset.data_root, img_filename)
            else:
                img_path = img_filename
            
            # Load image
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                prog_bar.update()
                continue
                
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Failed to load image: {img_path}")
                prog_bar.update()
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get camera transformation matrix
            # Check if we have the required camera parameters
            if 'intrinsic' not in cam_info or 'cam2ego' not in cam_info:
                print(f"Warning: Missing camera transformation for sample {idx}")
                prog_bar.update()
                continue
            
            # Check if LIDAR_TOP sensor exists
            if 'LIDAR_TOP' not in data_info['sensors']:
                print(f"Warning: Missing LIDAR_TOP sensor for sample {idx}")
                prog_bar.update()
                continue
                
            # Get transformation matrices
            intrinsic = np.array(cam_info['intrinsic'])
            cam2ego = np.array(cam_info['cam2ego'])
            lidar2ego = np.array(data_info['sensors']['LIDAR_TOP']['lidar2ego'])
            
            # Compute lidar2cam = inv(cam2ego) @ lidar2ego
            lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
            
            # Create projection matrix: lidar2img = intrinsic @ lidar2cam
            # Pad intrinsic to 4x4 if needed
            intrinsic_pad = np.eye(4)
            intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img = intrinsic_pad @ lidar2cam
            
            # Extract only the 3x4 projection matrix for projection
            lidar2img_tensor = torch.from_numpy(lidar2img[:3, :]).float()
            
            # Convert trajectory to tensor if needed
            if isinstance(ego_fut_pred, torch.Tensor):
                traj_2d = ego_fut_pred.cpu()
            elif isinstance(ego_fut_pred, np.ndarray):
                traj_2d = torch.from_numpy(ego_fut_pred).float()
            else:
                print(f"Warning: Unknown trajectory format for sample {idx}")
                prog_bar.update()
                continue
            
            # Ensure correct shape [time_steps, 2]
            if len(traj_2d.shape) == 1:
                traj_2d = traj_2d.reshape(-1, 2)
            
            # Create 3D trajectory points in EGO coordinate system
            # Model outputs trajectory in ego coords with format [Y, X]
            # Need to swap to [X, Y] format: X=forward, Y=left in ego frame
            traj_3d_ego = torch.zeros((traj_2d.shape[0], 3), dtype=torch.float32)
            traj_3d_ego[:, 0] = traj_2d[:, 1]  # ego_x (forward) <- from traj_2d[:, 1]
            traj_3d_ego[:, 1] = traj_2d[:, 0]  # ego_y (left) <- from traj_2d[:, 0]
            traj_3d_ego[:, 2] = 0.0  # ego_z (ground plane)
            
            # Create ego2img projection matrix directly
            ego2cam = np.linalg.inv(cam2ego)
            ego2img = intrinsic_pad @ ego2cam
            ego2img_tensor = torch.from_numpy(ego2img[:3, :]).float()
            
            # Project 3D ego trajectory points to 2D image coordinates (if enabled)
            traj_2d_proj = None
            if show_pred:
                traj_2d_proj = project_3d_to_2d(traj_3d_ego, ego2img_tensor)
            
            # Compute and project ground truth future trajectory (if enabled)
            gt_traj_2d_proj = None
            if show_gt:
                try:
                    # Use sample_rate=1 to match model prediction frequency (every frame)
                    future_track = compute_future_track(data_info, actual_dataset, actual_idx, 
                                                        future_frames=6, sample_rate=1)
                    if future_track is not None:
                        # future_track is in lidar coords with format [X, Y] (already correct order from world2lidar transform)
                        # Create 3D points in lidar coords
                        gt_traj_3d_lidar = np.zeros((len(future_track), 3), dtype=np.float32)
                        gt_traj_3d_lidar[:, :2] = future_track  # X, Y in lidar
                        gt_traj_3d_lidar[:, 2] = 0.0  # Z = ground plane
                        
                        # Transform from lidar to ego coordinates
                        gt_traj_3d_lidar_homo = np.concatenate([gt_traj_3d_lidar, np.ones((len(future_track), 1))], axis=1)
                        gt_traj_3d_ego_homo = (lidar2ego @ gt_traj_3d_lidar_homo.T).T
                        gt_traj_3d_ego = gt_traj_3d_ego_homo[:, :3]
                        
                        # Project to 2D image coordinates using ego2img
                        gt_traj_3d_ego_tensor = torch.from_numpy(gt_traj_3d_ego).float()
                        gt_traj_2d_proj = project_3d_to_2d(gt_traj_3d_ego_tensor, ego2img_tensor)
                        gt_traj_2d_proj = gt_traj_2d_proj.cpu().numpy()
                except Exception as e:
                    print(f"Warning: Failed to compute GT trajectory for sample {idx}: {e}")
            
            # Skip if both prediction and GT are disabled or unavailable
            if traj_2d_proj is None and gt_traj_2d_proj is None:
                prog_bar.update()
                continue
            
            # Save visualization (with prediction and/or GT based on settings)
            save_path = os.path.join(save_dir, f'trajectory_{idx:06d}.jpg')
            visualize_trajectory(image, traj_2d_proj, save_path, idx, ego_fut_cmd_idx, gt_traj_2d_proj)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            
        prog_bar.update()
    
    print(f"Trajectory visualization completed! Results saved to {save_dir}")
