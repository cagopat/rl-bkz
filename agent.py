"""
DQN agent: Q-network + target network + experience replay.
"""

import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        # small init for stable early training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs,  dtype=np.float32),
            np.array(act,  dtype=np.int64),
            np.array(rew,  dtype=np.float32),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)



class DQNAgent:
    """
    Standard DQN with:
      - epsilon-greedy exploration (linear annealing)
      - periodic hard target-network sync
      - Huber loss + gradient clipping
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 2000,   # env steps to reach eps_end
        buffer_size: int = 20_000,
        batch_size: int = 128,
        target_update_freq: int = 200,  # learn steps between target syncs
        hidden: int = 128,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._env_steps = 0    # total environment steps seen
        self._learn_steps = 0  # total gradient updates

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q(t).argmax(dim=1).item())

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition and update epsilon."""
        self.buffer.push(obs, action, reward, next_obs, float(done))
        self._env_steps += 1
        # linear epsilon annealing
        frac = min(self._env_steps / self._eps_decay_steps, 1.0)
        self.eps = self._eps_start + frac * (self._eps_end - self._eps_start)

    def learn(self) -> Optional[float]:
        """One gradient step; returns loss or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)
        obs_t  = torch.from_numpy(obs).to(self.device)
        act_t  = torch.from_numpy(act).to(self.device)
        rew_t  = torch.from_numpy(rew).to(self.device)
        nobs_t = torch.from_numpy(nobs).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        # Current Q-values
        q_vals = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient)
        with torch.no_grad():
            next_q = self.q_target(nobs_t).max(1)[0]
            target = rew_t + self.gamma * next_q * (1.0 - done_t)

        loss = F.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "q_state_dict": self.q.state_dict(),
                "env_steps": self._env_steps,
                "learn_steps": self._learn_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_target.load_state_dict(self.q.state_dict())
        self._env_steps = ckpt.get("env_steps", 0)
        self._learn_steps = ckpt.get("learn_steps", 0)
        # eps is not saved: use select_action(greedy=True) for eval,
        # or set agent.eps = agent._eps_end before continued training.