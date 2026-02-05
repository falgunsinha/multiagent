import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Union
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class ActionMaskingWrapper(gym.Wrapper):
    """Gym wrapper that applies action masking to prevent invalid actions"""

    def __init__(self, env):
        super().__init__(env)
        self.action_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_mask = info.get("action_mask", None)
        return obs, info

    def step(self, action):
        if self.action_mask is not None:
            if not self.action_mask[action]:
                valid_actions = np.where(self.action_mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.action_mask = info.get("action_mask", None)
        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        """Get current action mask"""
        return self.action_mask


class VecActionMaskingWrapper(VecEnvWrapper):
    """Vectorized environment wrapper for action masking"""

    def __init__(self, venv):
        super().__init__(venv)
        self.action_masks = None

    def reset(self):
        obs = self.venv.reset()
        self.action_masks = self._get_action_masks()
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.action_masks = np.array([info.get("action_mask", None) for info in infos])
        return obs, rewards, dones, infos

    def _get_action_masks(self):
        """Get action masks from all environments"""
        try:
            masks = self.venv.env_method("get_action_mask")
            return np.array(masks)
        except:
            return None

    def get_action_masks(self):
        """Get current action masks for all environments"""
        return self.action_masks


def apply_action_mask(logits: torch.Tensor, action_mask: np.ndarray) -> torch.Tensor:
    """Apply action mask to logits by setting invalid actions to -inf"""
    if action_mask is None:
        return logits
    if isinstance(action_mask, np.ndarray):
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
    masked_logits = logits.clone()
    masked_logits[~action_mask] = -1e8
    return masked_logits


class MaskablePPOCallback:
    """Callback for PPO training that applies action masking during rollout collection"""

    def __init__(self, vec_env_wrapper: VecActionMaskingWrapper):
        self.vec_env_wrapper = vec_env_wrapper

    def on_rollout_start(self, model):
        """Called at the start of rollout collection"""
        pass

    def on_step(self, model):
        """Called after each environment step during rollout"""
        action_masks = self.vec_env_wrapper.get_action_masks()
        if action_masks is not None:
            model.policy.action_masks = action_masks
        return True

