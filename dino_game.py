from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


# I keep Pygame quiet here because the training logs get noisy otherwise.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


Action = int
INK = (83, 83, 83)
PAPER = (247, 247, 247)


@dataclass(frozen=True)
class GameConfig:
    width: int = 760
    height: int = 300
    fps: int = 60
    ground_y: int = 238
    gravity: float = 1.15
    jump_velocity: float = -18.5
    obstacle_speed: float = 7.0
    obstacle_spawn_chance: float = 0.022
    max_obstacles: int = 2
    speedup_per_point: float = 0.055


@dataclass
class Obstacle:
    rect: pygame.Rect
    kind: str
    variant: int = 0


@dataclass
class Cloud:
    x: float
    y: int
    speed: float


class DinoGame:
    """Small Chrome-Dino style environment.

    I made this look a little like a Gym environment, so training code can call
    reset() and step(action). The agent gets a small hand-built state vector
    instead of raw pixels.
    """

    ACTION_STAY: Action = 0
    ACTION_JUMP: Action = 1
    ACTION_DUCK: Action = 2

    def __init__(self, config: GameConfig | None = None, render_mode: bool = True, seed: int | None = None):
        self.config = config or GameConfig()
        self.render_mode = render_mode
        self.random = random.Random(seed)

        pygame.init()
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("Dino Trainer")
        else:
            self.screen = pygame.Surface((self.config.width, self.config.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("couriernew", 18, bold=True)

        self.running = True
        self.curriculum_stage: Literal["cacti", "high_birds", "full"] = "full"
        self.dino: pygame.Rect
        self.obstacles: List[Obstacle]
        self.clouds: List[Cloud]
        self.velocity_y: float
        self.ducking: bool
        self.score: int
        self.frames_alive: int
        self.ground_scroll: float
        self.reset()

    @property
    def dino_floor_y(self) -> int:
        return self.config.ground_y - 46

    @property
    def obstacle_speed(self) -> float:
        return self.config.obstacle_speed + self.score * self.config.speedup_per_point

    def reset(self) -> List[float]:
        self.dino = pygame.Rect(52, self.dino_floor_y, 44, 46)
        self.obstacles = []
        self.clouds = [
            Cloud(180, 58, 0.35),
            Cloud(430, 82, 0.25),
            Cloud(660, 48, 0.4),
        ]
        self.velocity_y = 0.0
        self.ducking = False
        self.score = 0
        self.frames_alive = 0
        self.ground_scroll = 0.0
        self.running = True
        return self.get_state()

    def set_curriculum_stage(self, episode: int, total_episodes: int) -> str:
        progress = episode / max(1, total_episodes)
        if progress < 0.55:
            self.curriculum_stage = "cacti"
        elif progress < 0.88:
            self.curriculum_stage = "high_birds"
        else:
            self.curriculum_stage = "full"
        return self.curriculum_stage

    def set_full_curriculum(self) -> None:
        self.curriculum_stage = "full"

    def step(self, action: Action) -> Tuple[List[float], float, bool, Dict[str, int]]:
        obstacle_before_action = self._next_obstacle()
        on_ground = self._on_ground()
        self.ducking = action == self.ACTION_DUCK and on_ground
        if on_ground:
            self._resize_dino_for_pose()

        if action == self.ACTION_JUMP and on_ground:
            self.velocity_y = self.config.jump_velocity
            self.ducking = False
            self._resize_dino_for_pose()

        self._apply_physics()
        self._move_scenery()
        reward = self._action_reward(action, obstacle_before_action)
        reward += self._move_obstacles()
        self._maybe_spawn_obstacle()
        self.frames_alive += 1

        collided = any(obstacle.rect.colliderect(self.dino) for obstacle in self.obstacles)
        if collided:
            return self.get_state(), -10.0, True, self._info()

        reward += 0.05
        return self.get_state(), reward, False, self._info()

    def get_state(self) -> List[float]:
        obstacle = self._next_obstacle()
        if obstacle is None:
            distance = self.config.width
            obstacle_y = self.config.ground_y
            obstacle_width = 0
            obstacle_height = 0
            obstacle_is_bird = 0.0
        else:
            distance = max(0, obstacle.rect.x - self.dino.x)
            obstacle_y = obstacle.rect.y
            obstacle_width = obstacle.rect.width
            obstacle_height = obstacle.rect.height
            obstacle_is_bird = 1.0 if obstacle.kind == "bird" else 0.0

        return [
            self.dino.y / self.config.height,
            (self.velocity_y + 25.0) / 50.0,
            1.0 if self.ducking else 0.0,
            distance / self.config.width,
            obstacle_y / self.config.height,
            obstacle_width / 80.0,
            obstacle_height / 80.0,
            self.obstacle_speed / 18.0,
            obstacle_is_bird,
        ]

    def render(self) -> None:
        self._handle_events()
        self.screen.fill(PAPER)
        self._draw_clouds()
        self._draw_ground()
        self._draw_dino()

        for obstacle in self.obstacles:
            if obstacle.kind == "bird":
                self._draw_bird(obstacle)
            else:
                self._draw_cactus(obstacle)

        score_text = self.font.render(f"{self.score:05d}", True, INK)
        self.screen.blit(score_text, (self.config.width - score_text.get_width() - 22, 18))

        if self.render_mode:
            pygame.display.flip()
        self.clock.tick(self.config.fps)

    def close(self) -> None:
        pygame.quit()

    def _apply_physics(self) -> None:
        self.velocity_y += self.config.gravity
        self.dino.y += int(self.velocity_y)
        if self.dino.y >= self.dino_floor_y:
            self.dino.y = self.dino_floor_y
            self.velocity_y = 0.0
            self._resize_dino_for_pose()

    def _move_scenery(self) -> None:
        self.ground_scroll = (self.ground_scroll + self.obstacle_speed) % 28
        for cloud in self.clouds:
            cloud.x -= cloud.speed
            if cloud.x < -80:
                cloud.x = self.config.width + self.random.randint(20, 180)
                cloud.y = self.random.choice([44, 58, 72, 88])
                cloud.speed = self.random.choice([0.25, 0.35, 0.45])

    def _move_obstacles(self) -> float:
        reward = 0.0
        remaining = []
        for obstacle in self.obstacles:
            obstacle.rect.x -= int(self.obstacle_speed)
            if obstacle.rect.right < 0:
                self.score += 1
                reward += 2.0
            else:
                remaining.append(obstacle)
        self.obstacles = remaining
        return reward

    def _maybe_spawn_obstacle(self) -> None:
        if len(self.obstacles) >= self.config.max_obstacles:
            return
        if self.obstacles and self.obstacles[-1].rect.x > self.config.width * 0.52:
            return
        if self.random.random() > self.config.obstacle_spawn_chance:
            return

        bird_chance = 0.10 if self.curriculum_stage == "high_birds" else 0.12
        if self.curriculum_stage != "cacti" and self.score >= 18 and self.random.random() < bird_chance:
            self._spawn_bird()
        else:
            self._spawn_cactus()

    def _spawn_cactus(self) -> None:
        variant = self.random.randint(0, 3)
        width = [20, 28, 42, 56][variant]
        height = [42, 52, 44, 58][variant]
        x = self.config.width + self.random.randint(20, 90)
        y = self.config.ground_y - height
        self.obstacles.append(Obstacle(pygame.Rect(x, y, width, height), "cactus", variant))

    def _spawn_bird(self) -> None:
        width = 46
        height = 30
        x = self.config.width + self.random.randint(20, 110)
        if self.curriculum_stage == "high_birds":
            y = self.config.ground_y - 88
        elif self.score < 35:
            y = self.config.ground_y - 88
        else:
            y = self.config.ground_y - 62 if self.random.random() < 0.22 else self.config.ground_y - 88
        self.obstacles.append(Obstacle(pygame.Rect(x, y, width, height), "bird", self.random.randint(0, 1)))

    def _next_obstacle(self) -> Obstacle | None:
        candidates = [obstacle for obstacle in self.obstacles if obstacle.rect.right >= self.dino.x]
        return min(candidates, key=lambda obstacle: obstacle.rect.x, default=None)

    def _on_ground(self) -> bool:
        return self.dino.y >= self.dino_floor_y and self.velocity_y == 0.0

    def _resize_dino_for_pose(self) -> None:
        bottom = self.config.ground_y if self._on_ground() else self.dino.bottom
        if self.ducking:
            self.dino.update(self.dino.x, bottom - 30, 58, 30)
        else:
            self.dino.update(self.dino.x, bottom - 46, 44, 46)

    def _action_reward(self, action: Action, obstacle: Obstacle | None) -> float:
        if obstacle is None:
            return -0.02 if action in (self.ACTION_JUMP, self.ACTION_DUCK) else 0.0

        distance = obstacle.rect.x - self.dino.x
        if distance < 0 or distance > 150:
            return -0.015 if action in (self.ACTION_JUMP, self.ACTION_DUCK) else 0.0

        if obstacle.kind == "cactus":
            if 45 <= distance <= 125:
                if action == self.ACTION_JUMP:
                    return 0.20
                if action == self.ACTION_DUCK:
                    return -0.12
            return 0.0

        low_bird = obstacle.rect.y > self.config.ground_y - 75
        if low_bird and obstacle.rect.right >= self.dino.x - 4 and distance <= 135:
            if action == self.ACTION_DUCK:
                return 0.30
            if action == self.ACTION_JUMP:
                return -0.08
            return -0.04

        if not low_bird and 35 <= distance <= 125:
            if action == self.ACTION_DUCK:
                return -0.06
            if action == self.ACTION_STAY:
                return 0.04
        return 0.0

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _info(self) -> Dict[str, int]:
        return {"score": self.score, "frames_alive": self.frames_alive}

    def _draw_clouds(self) -> None:
        for cloud in self.clouds:
            x = int(cloud.x)
            y = cloud.y
            pygame.draw.line(self.screen, INK, (x + 8, y + 14), (x + 58, y + 14), 3)
            pygame.draw.line(self.screen, INK, (x + 18, y + 6), (x + 30, y + 6), 3)
            pygame.draw.line(self.screen, INK, (x + 30, y + 2), (x + 44, y + 2), 3)
            pygame.draw.line(self.screen, INK, (x + 44, y + 8), (x + 58, y + 8), 3)
            pygame.draw.line(self.screen, INK, (x + 8, y + 14), (x + 18, y + 6), 3)
            pygame.draw.line(self.screen, INK, (x + 58, y + 14), (x + 66, y + 10), 3)

    def _draw_ground(self) -> None:
        pygame.draw.line(self.screen, INK, (0, self.config.ground_y), (self.config.width, self.config.ground_y), 2)
        start = -int(self.ground_scroll)
        for x in range(start, self.config.width, 28):
            y = self.config.ground_y + self.randomish_ground_offset(x)
            pygame.draw.line(self.screen, INK, (x, y), (x + 8, y), 2)
        for x in range(start + 13, self.config.width, 56):
            pygame.draw.line(self.screen, INK, (x, self.config.ground_y + 9), (x + 3, self.config.ground_y + 9), 2)

    def randomish_ground_offset(self, x: int) -> int:
        return 7 + ((x * 17 + self.frames_alive // 8) % 5)

    def _draw_dino(self) -> None:
        x = self.dino.x
        y = self.dino.y
        leg_lift = (self.frames_alive // 6) % 2 if self._on_ground() else 0

        if self.ducking:
            self._draw_ducking_dino(x, y, leg_lift)
            return

        # Pixel-ish body, neck, and head.
        pygame.draw.rect(self.screen, INK, (x + 8, y + 18, 24, 23))
        pygame.draw.rect(self.screen, INK, (x + 24, y + 8, 12, 22))
        pygame.draw.rect(self.screen, INK, (x + 32, y + 4, 25, 18))
        pygame.draw.rect(self.screen, INK, (x + 52, y + 10, 8, 7))
        pygame.draw.rect(self.screen, PAPER, (x + 39, y + 8, 4, 4))

        # Back, tail, and tiny arms.
        pygame.draw.rect(self.screen, INK, (x + 2, y + 24, 12, 8))
        pygame.draw.rect(self.screen, INK, (x - 5, y + 27, 8, 5))
        pygame.draw.rect(self.screen, INK, (x + 31, y + 28, 10, 4))

        # Two-frame leg animation.
        if leg_lift:
            pygame.draw.rect(self.screen, INK, (x + 12, y + 39, 7, 12))
            pygame.draw.rect(self.screen, INK, (x + 25, y + 39, 7, 7))
            pygame.draw.rect(self.screen, INK, (x + 23, y + 46, 13, 5))
        else:
            pygame.draw.rect(self.screen, INK, (x + 12, y + 39, 7, 7))
            pygame.draw.rect(self.screen, INK, (x + 8, y + 46, 13, 5))
            pygame.draw.rect(self.screen, INK, (x + 26, y + 39, 7, 12))

    def _draw_ducking_dino(self, x: int, y: int, leg_lift: int) -> None:
        pygame.draw.rect(self.screen, INK, (x + 5, y + 9, 36, 17))
        pygame.draw.rect(self.screen, INK, (x + 34, y + 3, 29, 16))
        pygame.draw.rect(self.screen, INK, (x + 58, y + 9, 8, 6))
        pygame.draw.rect(self.screen, PAPER, (x + 43, y + 7, 4, 4))
        pygame.draw.rect(self.screen, INK, (x - 4, y + 14, 13, 5))
        pygame.draw.rect(self.screen, INK, (x + 31, y + 20, 10, 4))

        if leg_lift:
            pygame.draw.rect(self.screen, INK, (x + 12, y + 24, 14, 5))
            pygame.draw.rect(self.screen, INK, (x + 34, y + 24, 7, 7))
        else:
            pygame.draw.rect(self.screen, INK, (x + 12, y + 24, 7, 7))
            pygame.draw.rect(self.screen, INK, (x + 30, y + 24, 14, 5))

    def _draw_cactus(self, obstacle: Obstacle) -> None:
        rect = obstacle.rect
        stem_w = 8
        centers = self._cactus_centers(rect)
        for cx, height in centers:
            top = self.config.ground_y - height
            pygame.draw.rect(self.screen, INK, (cx - stem_w // 2, top, stem_w, height))
            pygame.draw.rect(self.screen, INK, (cx - stem_w // 2 - 5, top + 14, 6, 8))
            pygame.draw.rect(self.screen, INK, (cx - stem_w // 2 - 9, top + 8, 5, 14))
            pygame.draw.rect(self.screen, INK, (cx + stem_w // 2 - 1, top + 22, 6, 8))
            pygame.draw.rect(self.screen, INK, (cx + stem_w // 2 + 4, top + 15, 5, 15))

    def _cactus_centers(self, rect: pygame.Rect) -> List[Tuple[int, int]]:
        if rect.width <= 22:
            return [(rect.centerx, rect.height)]
        if rect.width <= 34:
            return [(rect.x + 9, rect.height), (rect.x + 22, rect.height - 10)]
        return [
            (rect.x + 9, rect.height - 8),
            (rect.x + rect.width // 2, rect.height),
            (rect.right - 9, rect.height - 12),
        ]

    def _draw_bird(self, obstacle: Obstacle) -> None:
        rect = obstacle.rect
        flap = (self.frames_alive // 8 + obstacle.variant) % 2
        body_y = rect.y + 14
        pygame.draw.rect(self.screen, INK, (rect.x + 12, body_y, 22, 7))
        pygame.draw.rect(self.screen, INK, (rect.x + 32, body_y - 4, 10, 8))
        pygame.draw.rect(self.screen, PAPER, (rect.x + 36, body_y - 2, 3, 3))

        if flap:
            pygame.draw.line(self.screen, INK, (rect.x + 16, body_y), (rect.x + 2, rect.y + 2), 4)
            pygame.draw.line(self.screen, INK, (rect.x + 25, body_y), (rect.x + 39, rect.y + 3), 4)
        else:
            pygame.draw.line(self.screen, INK, (rect.x + 16, body_y + 6), (rect.x + 2, rect.y + 27), 4)
            pygame.draw.line(self.screen, INK, (rect.x + 25, body_y + 6), (rect.x + 39, rect.y + 27), 4)
