"""
MASAC Agent adapted for Cube Reshuffling Task
Based on multiagent-sac but adapted for our two-agent environment
"""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from buffers.buffer import ReplayBuffer
from models.mactor_critic import Actor, CriticQ, CriticV


class MASACReshufflingAgent:
    """
    Multi-Agent SAC for cube reshuffling.
    
    Agent 1: DDQN (cube selection) - pretrained, frozen
    Agent 2: MASAC (reshuffling decisions) - this agent
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 100,
        policy_update_frequency: int = 2,
        pretrained_model_path: str = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_frequency = policy_update_frequency
        
        # Replay buffer
        self.memory = ReplayBuffer(state_dim, action_dim, memory_size, batch_size)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Temperature parameter (alpha) for entropy regularization
        self.target_alpha = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Actor network
        self.actor = Actor(state_dim, action_dim).to(self.device)
        
        # Value function networks
        self.vf = CriticV(state_dim).to(self.device)
        self.vf_target = CriticV(state_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        
        # Q function networks (twin critics)
        self.qf1 = CriticQ(state_dim + action_dim).to(self.device)
        self.qf2 = CriticQ(state_dim + action_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-4)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-4)
        
        # Training state
        self.total_step = 0
        self.is_test = False
        
        # Load pretrained model if provided
        if pretrained_model_path:
            self.load_pretrained(pretrained_model_path)
    
    def load_pretrained(self, model_path: str):
        """Load pretrained MASAC models"""
        try:
            actor_path = Path(model_path) / "mactor.pt"
            qf1_path = Path(model_path) / "mqf1.pt"
            qf2_path = Path(model_path) / "mqf2.pt"
            vf_path = Path(model_path) / "mvf.pt"
            
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.qf1.load_state_dict(torch.load(qf1_path, map_location=self.device))
            self.qf2.load_state_dict(torch.load(qf2_path, map_location=self.device))
            self.vf.load_state_dict(torch.load(vf_path, map_location=self.device))
            self.vf_target.load_state_dict(self.vf.state_dict())
            
            print(f"✅ Loaded pretrained MASAC models from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained models: {e}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action for reshuffling.
        
        Args:
            state: Current state observation
            deterministic: If True, use mean action (for testing)
        
        Returns:
            Selected action
        """
        if self.total_step < self.initial_random_steps and not self.is_test and not deterministic:
            # Random exploration
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Use policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.actor(state_tensor)
                action = action.cpu().numpy()[0]
                action = np.clip(action, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if not self.is_test:
            self.memory.store(state, action, reward, next_state, done)
    
    def update_model(self) -> Tuple[float, float, float, float]:
        """
        Update MASAC networks.
        
        Returns:
            Tuple of (actor_loss, qf_loss, v_loss, alpha_loss)
        """
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        # Sample batch
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device)

        # Get new action and log probability
        new_action, log_prob = self.actor(state)

        # Update alpha (temperature parameter)
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_alpha).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        # Update Q functions
        mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        q_target = reward + self.gamma * vf_target * mask
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Update V function
        v_pred = self.vf(state)
        q_pred = torch.min(self.qf1(state, new_action), self.qf2(state, new_action))
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        # Update actor (policy)
        if self.total_step % self.policy_update_frequency == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target network
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        qf_loss = qf1_loss + qf2_loss

        return (
            actor_loss.item(),
            qf_loss.item(),
            v_loss.item(),
            alpha_loss.item()
        )

    def _target_soft_update(self):
        """Soft update of target network"""
        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

    def save(self, save_path: str):
        """Save model weights"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), save_path / "mactor.pt")
        torch.save(self.qf1.state_dict(), save_path / "mqf1.pt")
        torch.save(self.qf2.state_dict(), save_path / "mqf2.pt")
        torch.save(self.vf.state_dict(), save_path / "mvf.pt")

        print(f"✅ Saved MASAC models to {save_path}")

    def set_test_mode(self, is_test: bool):
        """Set test mode"""
        self.is_test = is_test

