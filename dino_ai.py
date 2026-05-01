from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(NamedTuple):
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool


@dataclass(frozen=True)
class AgentConfig:
    state_size: int = 9
    action_size: int = 3
    hidden_size: int = 256
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    memory_size: int = 50_000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.9997
    gradient_clip: float = 10.0


class DQN(nn.Module):
    """Small MLP that turns the game state into one score per action."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DinoAI:
    """The learning agent.

    This file is mostly the standard DQN pieces: replay memory, epsilon-greedy
    actions, a target network, and checkpoint saving/loading.
    """

    def __init__(self, config: AgentConfig | None = None, seed: int | None = None):
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.model = DQN(self.config.state_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self.target_model = DQN(self.config.state_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self.update_target()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.memory: Deque[Transition] = deque(maxlen=self.config.memory_size)
        self.epsilon = self.config.epsilon_start
        self.training_steps = 0

    def act(self, state: List[float], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.config.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def replay(self) -> float | None:
        if len(self.memory) < self.config.batch_size:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        predicted_q = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Double DQN: use the main net to pick the action, but the target
            # net to score it. This helped reduce jumpy Q-value estimates.
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.training_steps += 1
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())

    def imitate(
        self,
        samples: list[tuple[List[float], int]],
        epochs: int,
        batch_size: int | None = None,
    ) -> float | None:
        """Train on expert examples before doing RL updates.

        I added this because learning when to duck under birds was unstable
        from reward alone. This is basically behavior cloning for a short warm
        start, then DQN can keep fine-tuning from there.
        """
        if not samples or epochs <= 0:
            return None

        batch_size = batch_size or self.config.batch_size
        states = torch.tensor([state for state, _ in samples], dtype=torch.float32, device=self.device)
        actions = torch.tensor([action for _, action in samples], dtype=torch.long, device=self.device)

        counts = torch.bincount(actions, minlength=self.config.action_size).float()
        class_weights = counts.sum() / counts.clamp_min(1.0)
        class_weights = class_weights / class_weights.mean()

        last_loss = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(samples), device=self.device)
            for start in range(0, len(samples), batch_size):
                indices = order[start : start + batch_size]
                logits = self.model(states[indices])
                loss = F.cross_entropy(logits, actions[indices], weight=class_weights)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                last_loss = float(loss.item())

        self.update_target()
        self.epsilon = min(self.epsilon, 0.15)
        return last_loss

    def update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(model_state)
        self.target_model.load_state_dict(checkpoint.get("target_model", model_state))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_min))
        self.training_steps = int(checkpoint.get("training_steps", 0))
