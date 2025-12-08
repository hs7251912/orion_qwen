"""
增强版 TensorBoard Hook
支持更丰富的可视化，包括：
- 标量指标（Loss、学习率等）
- 直方图（参数分布、梯度分布）
- 图像（特征图可视化）
- 文本（模型预测结果）
- 自定义图表（特征对齐监控）
"""
import os
import os.path as osp
import torch
import torch.nn.functional as F
from collections import defaultdict

from mmcv.utils import TORCH_VERSION, digit_version, master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class EnhancedTensorboardLoggerHook(LoggerHook):
    """增强版 TensorBoard Logger Hook
    
    Args:
        log_dir (str): TensorBoard 日志目录
        interval (int): 日志记录间隔
        log_grad_histogram (bool): 是否记录梯度分布直方图
        log_param_histogram (bool): 是否记录参数分布直方图
        log_alignment_metrics (bool): 是否记录特征对齐指标
        histogram_interval (int): 直方图记录间隔（通常比普通日志更大）
        ignore_last (bool): 是否忽略最后一次迭代
        reset_flag (bool): 是否清空输出缓冲
        by_epoch (bool): 是否按 epoch 记录
    """

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 log_grad_histogram=True,
                 log_param_histogram=True,
                 log_alignment_metrics=True,
                 histogram_interval=100,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(EnhancedTensorboardLoggerHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        self.log_grad_histogram = log_grad_histogram
        self.log_param_histogram = log_param_histogram
        self.log_alignment_metrics = log_alignment_metrics
        self.histogram_interval = histogram_interval
        
        # 用于累积特定指标
        self.alignment_buffer = defaultdict(list)

    @master_only
    def before_run(self, runner):
        super(EnhancedTensorboardLoggerHook, self).before_run(runner)
        
        # 导入 SummaryWriter
        if digit_version(TORCH_VERSION) < digit_version('1.1'):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"📊 TensorBoard 日志保存到: {self.log_dir}")
        print(f"📌 启动命令: tensorboard --logdir={self.log_dir} --port=6006")

    @master_only
    def log(self, runner):
        """记录标量指标"""
        tags = self.get_loggable_tags(runner, allow_text=True)
        global_step = self.get_iter(runner)
        
        # 1. 记录标量指标
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, global_step)
            else:
                self.writer.add_scalar(tag, val, global_step)
        
        # 2. 记录学习率（更详细）
        self._log_learning_rates(runner, global_step)
        
        # 3. 记录显存使用
        if torch.cuda.is_available():
            self._log_memory_usage(runner, global_step)
        
        # 4. 记录梯度和参数（每 histogram_interval 步）
        if global_step % self.histogram_interval == 0:
            if self.log_grad_histogram:
                self._log_gradients(runner, global_step)
            if self.log_param_histogram:
                self._log_parameters(runner, global_step)
        
        # 5. 记录特征对齐指标
        if self.log_alignment_metrics:
            self._log_alignment_metrics(runner, global_step)
    
    def _log_learning_rates(self, runner, global_step):
        """详细记录各模块的学习率"""
        lr_dict = runner.current_lr()
        
        if isinstance(lr_dict, list):
            for i, lr in enumerate(lr_dict):
                self.writer.add_scalar(f'train/lr_group_{i}', lr, global_step)
        elif isinstance(lr_dict, dict):
            for name, lrs in lr_dict.items():
                if isinstance(lrs, list):
                    for i, lr in enumerate(lrs):
                        self.writer.add_scalar(f'train/lr_{name}_{i}', lr, global_step)
                else:
                    self.writer.add_scalar(f'train/lr_{name}', lrs, global_step)
    
    def _log_memory_usage(self, runner, global_step):
        """记录 GPU 显存使用情况"""
        for device_id in range(torch.cuda.device_count()):
            # 已分配的显存
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
            # 缓存的显存
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
            # 最大已分配的显存
            max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
            
            self.writer.add_scalar(f'memory/device_{device_id}_allocated_GB', 
                                  allocated, global_step)
            self.writer.add_scalar(f'memory/device_{device_id}_reserved_GB', 
                                  reserved, global_step)
            self.writer.add_scalar(f'memory/device_{device_id}_max_allocated_GB', 
                                  max_allocated, global_step)
    
    def _log_gradients(self, runner, global_step):
        """记录梯度分布直方图"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 梯度直方图
                self.writer.add_histogram(f'gradients/{name}', 
                                         param.grad.data.cpu().numpy(), 
                                         global_step)
                
                # 梯度范数
                grad_norm = param.grad.data.norm(2).item()
                self.writer.add_scalar(f'grad_norm/{name}', grad_norm, global_step)
                
                # 梯度统计
                grad_mean = param.grad.data.mean().item()
                grad_std = param.grad.data.std().item()
                self.writer.add_scalar(f'grad_stats/{name}_mean', grad_mean, global_step)
                self.writer.add_scalar(f'grad_stats/{name}_std', grad_std, global_step)
    
    def _log_parameters(self, runner, global_step):
        """记录参数分布直方图"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 参数直方图
                self.writer.add_histogram(f'parameters/{name}', 
                                         param.data.cpu().numpy(), 
                                         global_step)
                
                # 参数统计
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_norm = param.data.norm(2).item()
                
                self.writer.add_scalar(f'param_stats/{name}_mean', param_mean, global_step)
                self.writer.add_scalar(f'param_stats/{name}_std', param_std, global_step)
                self.writer.add_scalar(f'param_stats/{name}_norm', param_norm, global_step)
    
    def _log_alignment_metrics(self, runner, global_step):
        """记录特征对齐相关指标"""
        # 尝试从 runner.log_buffer 中获取对齐指标
        alignment_keys = [
            'bev_feature_norm', 'projected_feature_norm', 'norm_ratio',
            'internal_similarity', 'feature_diversity', 'llm_alignment_score',
            'projection_condition_number'
        ]
        
        for key in alignment_keys:
            if key in runner.log_buffer.output:
                val = runner.log_buffer.output[key]
                if self.is_scalar(val):
                    self.writer.add_scalar(f'alignment/{key}', val, global_step)
        
        # 如果有自定义的对齐监控器
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model, 'alignment_monitor'):
            monitor = model.alignment_monitor
            if hasattr(monitor, 'metrics_history'):
                for metric_name, values in monitor.metrics_history.items():
                    if len(values) > 0:
                        self.writer.add_scalar(f'alignment/{metric_name}', 
                                              values[-1], global_step)
    
    @master_only
    def after_train_epoch(self, runner):
        """训练 epoch 结束后的操作"""
        super().after_train_epoch(runner)
        
        # 记录 epoch 级别的统计
        if runner.log_buffer.ready:
            epoch = runner.epoch
            self.writer.add_scalar('epoch', epoch, self.get_iter(runner))
    
    @master_only
    def after_val_epoch(self, runner):
        """验证 epoch 结束后的操作"""
        # 记录验证指标
        if hasattr(runner, 'eval_results') and runner.eval_results:
            global_step = self.get_iter(runner)
            for metric_name, metric_val in runner.eval_results.items():
                if self.is_scalar(metric_val):
                    self.writer.add_scalar(f'val/{metric_name}', 
                                          metric_val, global_step)
    
    @master_only
    def after_run(self, runner):
        """训练结束后关闭 writer"""
        self.writer.close()
        print(f"✅ TensorBoard 日志已保存完成")
        print(f"📊 查看日志: tensorboard --logdir={self.log_dir} --port=6006")


@HOOKS.register_module()
class AlignmentMonitorHook(LoggerHook):
    """特征对齐专用监控 Hook
    
    专门用于监控 BEV 到 Qwen2-VL 的特征对齐质量
    """
    
    def __init__(self,
                 interval=10,
                 save_dir='./alignment_logs',
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(AlignmentMonitorHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch)
        self.save_dir = save_dir
        self.monitor = None
    
    @master_only
    def before_run(self, runner):
        """初始化对齐监控器"""
        from mmcv.utils.alignment_monitor import get_alignment_monitor
        self.monitor = get_alignment_monitor(self.save_dir, self.interval)
        print(f"🔍 特征对齐监控已启动，日志保存到: {self.save_dir}")
    
    @master_only
    def after_train_iter(self, runner):
        """训练迭代后记录对齐指标"""
        if not self.every_n_iters(runner, self.interval):
            return
        
        # 从模型中提取特征和投影层
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        metrics = {}
        
        # 尝试获取 BEV 特征和投影特征
        if hasattr(model, 'pts_bbox_head'):
            head = model.pts_bbox_head
            
            # 检查是否有缓存的特征（需要在模型中添加）
            if hasattr(head, '_last_bev_features'):
                bev_features = head._last_bev_features
                metrics['bev_feature_norm'] = bev_features.norm(dim=-1).mean().item()
            
            if hasattr(head, '_last_projected_features'):
                proj_features = head._last_projected_features
                metrics['projected_feature_norm'] = proj_features.norm(dim=-1).mean().item()
                
                if hasattr(head, '_last_bev_features'):
                    metrics['norm_ratio'] = (
                        metrics['projected_feature_norm'] / 
                        (metrics['bev_feature_norm'] + 1e-8)
                    )
            
            # 投影层统计
            if hasattr(head, 'output_projection'):
                proj_layer = head.output_projection
                weight_norm = proj_layer.weight.norm().item()
                metrics['projection_weight_norm'] = weight_norm
                
                if proj_layer.weight.grad is not None:
                    grad_norm = proj_layer.weight.grad.norm().item()
                    metrics['projection_grad_norm'] = grad_norm
        
        # 记录到监控器
        if metrics:
            self.monitor.log_step(metrics, self.get_iter(runner))
            
            # 也记录到 runner.log_buffer 供 TensorBoard 使用
            for key, val in metrics.items():
                runner.log_buffer.output[key] = val
    
    @master_only
    def after_run(self, runner):
        """训练结束后保存监控数据"""
        if self.monitor:
            self.monitor.save_metrics()
            print(f"✅ 特征对齐监控数据已保存")



