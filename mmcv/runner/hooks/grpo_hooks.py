import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .hook import HOOKS, Hook


@HOOKS.register_module()
class GRPOLoggingHook(Hook):
    """
    Custom hook for logging GRPO-specific metrics and visualizations.
    """
    
    def __init__(self, 
                 log_dir=None,
                 log_metrics=None,
                 log_interval=50,
                 plot_interval=500,
                 save_trajectory_vis=True):
        """
        Args:
            log_dir (str): Directory to save logs and visualizations
            log_metrics (list): List of metric names to log
            log_interval (int): Interval for logging metrics
            plot_interval (int): Interval for saving plots
            save_trajectory_vis (bool): Whether to save trajectory visualizations
        """
        self.log_dir = log_dir
        self.log_metrics = log_metrics or [
            'reward_mean', 'reward_std', 'advantage_mean', 
            'advantage_std', 'kl_divergence'
        ]
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.save_trajectory_vis = save_trajectory_vis
        
        # Storage for metrics over time
        self.metrics_history = defaultdict(list)
        self.iteration_history = []
        
        # Create log directory
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, 'grpo_metrics.json')
            self.plot_dir = os.path.join(self.log_dir, 'plots')
            os.makedirs(self.plot_dir, exist_ok=True)
    
    def before_train_iter(self, runner):
        """Called before each training iteration."""
        pass
    
    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if runner.iter % self.log_interval == 0:
            self._log_grpo_metrics(runner)
        
        if runner.iter % self.plot_interval == 0:
            self._save_plots(runner)
    
    def _log_grpo_metrics(self, runner):
        """Log GRPO-specific metrics."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Check if GRPO is enabled and we're in training mode
        if not (hasattr(model, 'use_grpo') and model.use_grpo and model.training):
            return
        
        # Try to compute metrics if model has the capability
        if hasattr(model, 'grpo_loss') and hasattr(model.grpo_loss, 'compute_metrics'):
            # Get recent batch data for metrics computation
            # This is a simplified approach - in practice you'd want to store batch data
            runner.logger.info(f"Iter {runner.iter}: GRPO training active")
        
        # Log additional GRPO-specific information
        if hasattr(model, 'reference_model'):
            ref_status = "Loaded" if model.reference_model is not None else "Not loaded"
            runner.logger.info(f"Reference model status: {ref_status}")
        
        # Log reward calculator status
        if hasattr(model, 'reward_calculator'):
            runner.logger.info(f"Reward calculator weights: {model.reward_calculator.reward_weights}")
    
    def _save_plots(self, runner):
        """Save training progress plots."""
        if not self.log_dir:
            return
        
        # Plot metrics history
        if self.metrics_history:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(self.log_metrics):
                if metric in self.metrics_history and i < len(axes):
                    axes[i].plot(self.iteration_history, self.metrics_history[metric])
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_xlabel('Iteration')
                    axes[i].grid(True)
            
            # Remove unused subplots
            for i in range(len(self.log_metrics), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plot_path = os.path.join(self.plot_dir, f'grpo_metrics_iter_{runner.iter}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            runner.logger.info(f"Saved GRPO metrics plot to {plot_path}")
    
    def _visualize_trajectories(self, runner, trajectories, rewards, iteration):
        """Visualize sampled trajectories with reward information."""
        if not self.save_trajectory_vis or not self.log_dir:
            return
        
        B, K, T, _ = trajectories.shape
        
        for b in range(min(B, 2)):  # Visualize first 2 batches
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot all K trajectories for this batch
            for k in range(K):
                traj = trajectories[b, k].cpu().numpy()
                reward = rewards[b, k].item()
                
                # Color based on reward (red for low, green for high)
                color = plt.cm.RdYlGn(reward / (rewards.max().item() + 1e-8))
                
                ax.plot(traj[:, 0], traj[:, 1], 
                       color=color, alpha=0.7, linewidth=2,
                       label=f'Traj {k}: R={reward:.3f}')
                
                # Mark start and end points
                ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=50, marker='o')
                ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=50, marker='x')
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Sampled Trajectories (Batch {b}, Iter {iteration})')
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            traj_path = os.path.join(self.plot_dir, f'trajectories_batch_{b}_iter_{iteration}.png')
            plt.savefig(traj_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def after_train_epoch(self, runner):
        """Called after each training epoch."""
        if self.log_dir:
            # Save metrics history to JSON
            metrics_data = {
                'iterations': self.iteration_history,
                'metrics': dict(self.metrics_history)
            }
            with open(self.log_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)


@HOOKS.register_module()
class KLSchedulerHook(Hook):
    """
    Hook for dynamically adjusting KL divergence coefficient in GRPO training.
    """
    
    def __init__(self,
                 initial_kl_coeff=0.05,
                 target_kl=0.1,
                 kl_tolerance=2.0,
                 adaptation_rate=1.5,
                 min_kl_coeff=0.001,
                 max_kl_coeff=1.0,
                 update_interval=100):
        """
        Args:
            initial_kl_coeff (float): Initial KL coefficient
            target_kl (float): Target KL divergence value
            kl_tolerance (float): Tolerance factor for KL divergence
            adaptation_rate (float): Rate of adaptation for KL coefficient
            min_kl_coeff (float): Minimum KL coefficient
            max_kl_coeff (float): Maximum KL coefficient
            update_interval (int): Interval for updating KL coefficient
        """
        self.initial_kl_coeff = initial_kl_coeff
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.adaptation_rate = adaptation_rate
        self.min_kl_coeff = min_kl_coeff
        self.max_kl_coeff = max_kl_coeff
        self.update_interval = update_interval
        
        self.current_kl_coeff = initial_kl_coeff
        self.kl_history = []
    
    def before_train_iter(self, runner):
        """Called before each training iteration."""
        # Update model's KL coefficient
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        if hasattr(model, 'grpo_loss') and hasattr(model.grpo_loss, 'kl_coeff'):
            model.grpo_loss.kl_coeff = self.current_kl_coeff
    
    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if runner.iter % self.update_interval == 0:
            self._update_kl_coefficient(runner)
    
    def _update_kl_coefficient(self, runner):
        """Update KL coefficient based on recent KL divergence values."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # This is a simplified version - in practice, you'd track KL divergence
        # from recent batches and adjust accordingly
        
        if len(self.kl_history) > 0:
            recent_kl = np.mean(self.kl_history[-10:])  # Average of last 10 values
            
            if recent_kl > self.target_kl * self.kl_tolerance:
                # KL too high, decrease coefficient
                self.current_kl_coeff = max(
                    self.current_kl_coeff / self.adaptation_rate,
                    self.min_kl_coeff
                )
                runner.logger.info(f"Decreased KL coeff to {self.current_kl_coeff:.6f} (KL={recent_kl:.6f})")
            
            elif recent_kl < self.target_kl / self.kl_tolerance:
                # KL too low, increase coefficient
                self.current_kl_coeff = min(
                    self.current_kl_coeff * self.adaptation_rate,
                    self.max_kl_coeff
                )
                runner.logger.info(f"Increased KL coeff to {self.current_kl_coeff:.6f} (KL={recent_kl:.6f})")


@HOOKS.register_module()
class TrajectoryVisualizationHook(Hook):
    """
    Hook for visualizing trajectory samples during GRPO training.
    """
    
    def __init__(self,
                 save_dir=None,
                 vis_interval=1000,
                 num_samples_to_vis=3,
                 include_rewards=True):
        """
        Args:
            save_dir (str): Directory to save visualizations
            vis_interval (int): Interval for saving visualizations
            num_samples_to_vis (int): Number of trajectory samples to visualize
            include_rewards (bool): Whether to include reward information in plots
        """
        self.save_dir = save_dir
        self.vis_interval = vis_interval
        self.num_samples_to_vis = num_samples_to_vis
        self.include_rewards = include_rewards
        
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if self.save_dir and runner.iter % self.vis_interval == 0:
            self._save_trajectory_visualization(runner)
    
    def _save_trajectory_visualization(self, runner):
        """Save trajectory visualization."""
        # This would be called from the model during training
        # when trajectory samples and rewards are available
        runner.logger.info(f"Trajectory visualization hook triggered at iter {runner.iter}")
        
        # In practice, you'd store trajectory data in the model and access it here
        # For now, just log that the hook is working
        
        # Example of what this would do:
        # 1. Get recent trajectory samples from model
        # 2. Get corresponding rewards
        # 3. Create visualization plots
        # 4. Save to disk
        
        placeholder_path = os.path.join(self.save_dir, f'trajectories_iter_{runner.iter}.png')
        
        # Create a simple placeholder plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Trajectory Visualization\nIteration: {runner.iter}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('GRPO Trajectory Samples')
        
        plt.savefig(placeholder_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        runner.logger.info(f"Saved trajectory visualization to {placeholder_path}")


# Helper function to register all GRPO hooks
def register_grpo_hooks():
    """Register all GRPO-related hooks."""
    # Hooks are automatically registered via the @HOOKS.register_module() decorator
    pass
