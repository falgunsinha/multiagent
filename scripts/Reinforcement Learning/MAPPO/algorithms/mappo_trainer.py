"""
MAPPO Trainer

Adapted from on-policy/onpolicy/algorithms/r_mappo/r_mappo.py
Simplified for two-agent cube reshuffling task.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


def huber_loss(e, d):
    """Huber loss"""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    """MSE loss"""
    return e**2 / 2


class MAPPO:
    """
    MAPPO Trainer for reshuffling agent.
    
    Implements PPO algorithm with:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy regularization
    - Gradient clipping
    """
    
    def __init__(
        self,
        policy,
        device=torch.device("cpu"),
        clip_param=0.2,
        ppo_epoch=10,
        num_mini_batch=4,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        huber_delta=10.0,
    ):
        """
        Initialize MAPPO trainer.
        
        Args:
            policy: MAPPO policy (actor-critic)
            device: Device to run on
            clip_param: PPO clipping parameter
            ppo_epoch: Number of PPO epochs per update
            num_mini_batch: Number of mini-batches
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Max gradient norm for clipping
            use_clipped_value_loss: Whether to use clipped value loss
            use_huber_loss: Whether to use Huber loss
            huber_delta: Huber loss delta
        """
        self.device = device
        self.policy = policy
        
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
    
    def cal_value_loss(
        self,
        values: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate value function loss.
        
        Args:
            values: Current value predictions
            value_preds_batch: Old value predictions
            return_batch: Returns (rewards-to-go)
            
        Returns:
            Value loss
        """
        # Clipped value loss
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values
        
        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        
        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        
        return value_loss.mean()
    
    def ppo_update(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
        old_action_log_probs_batch: torch.Tensor,
        adv_targ: torch.Tensor,
        action_mask_batch: torch.Tensor = None,
    ) -> Tuple[float, float, float, float]:
        """
        Perform one PPO update.
        
        Args:
            obs_batch: Observation batch
            actions_batch: Action batch
            value_preds_batch: Old value predictions
            return_batch: Returns
            old_action_log_probs_batch: Old action log probabilities
            adv_targ: Advantage targets
            action_mask_batch: Action mask (optional)
            
        Returns:
            value_loss, policy_loss, dist_entropy, ratio
        """
        # Evaluate actions
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs_batch, actions_batch, action_mask_batch
        )
        
        # Actor update (PPO clipped objective)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
        
        # Backprop
        self.policy.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        
        return value_loss.item(), policy_loss.item(), dist_entropy.item(), imp_weights.mean().item()

