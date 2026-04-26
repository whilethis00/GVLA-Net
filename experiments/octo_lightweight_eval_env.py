import math
from dataclasses import dataclass
from typing import Any

import gym
import numpy as np


@dataclass(frozen=True)
class ReachTask:
    language_instruction: str
    target_xy: np.ndarray


class OctoLightweightReachEnv(gym.Env):
    """A MuJoCo-free Gym env for lightweight Octo rollout sanity checks.

    This environment is intentionally simple: a point-effector moves in a 2D plane
    using the first two action dimensions. The observation format always includes
    `image_primary` and can optionally include `proprio`, which makes it useful as a
    lightweight integration harness before switching to a heavier simulator.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        action_dim: int,
        proprio_dim: int,
        image_size: int = 256,
        step_scale: float = 0.08,
        success_tolerance: float = 0.05,
        max_steps: int = 100,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError(f"action_dim must be >= 2, got {action_dim}")
        if proprio_dim < 0:
            raise ValueError(f"proprio_dim must be >= 0, got {proprio_dim}")

        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.image_size = int(image_size)
        self.step_scale = float(step_scale)
        self.success_tolerance = float(success_tolerance)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state_xy = np.zeros(2, dtype=np.float32)
        self._task = ReachTask(
            language_instruction="move the point effector to the target",
            target_xy=np.zeros(2, dtype=np.float32),
        )

        obs_spaces = {
            "image_primary": gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8,
            ),
        }
        if self.proprio_dim > 0:
            obs_spaces["proprio"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.proprio_dim,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def get_task(self) -> dict[str, Any]:
        return {
            "language_instruction": self._task.language_instruction,
            "target_xy": self._task.target_xy.copy(),
        }

    def _sample_xy(self) -> np.ndarray:
        return self._rng.uniform(low=-0.8, high=0.8, size=(2,)).astype(np.float32)

    def _render_image(self) -> np.ndarray:
        image = np.full((self.image_size, self.image_size, 3), 245, dtype=np.uint8)
        self._draw_disc(image, self._task.target_xy, radius=8, color=(220, 40, 40))
        self._draw_disc(image, self._state_xy, radius=8, color=(30, 60, 220))
        return image

    def _draw_disc(
        self,
        image: np.ndarray,
        xy: np.ndarray,
        *,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cx = int((xy[0] * 0.5 + 0.5) * (self.image_size - 1))
        cy = int((xy[1] * 0.5 + 0.5) * (self.image_size - 1))
        yy, xx = np.ogrid[: self.image_size, : self.image_size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = color

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {"image_primary": self._render_image()}
        if self.proprio_dim > 0:
            proprio = np.zeros(self.proprio_dim, dtype=np.float32)
            proprio[0:2] = self._state_xy
            if self.proprio_dim >= 4:
                proprio[2:4] = self._task.target_xy - self._state_xy
            if self.proprio_dim >= 5:
                proprio[4] = np.linalg.norm(self._task.target_xy - self._state_xy)
            obs["proprio"] = proprio
        return obs

    def compute_success(self) -> bool:
        return np.linalg.norm(self._task.target_xy - self._state_xy) <= self.success_tolerance

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state_xy = self._sample_xy()
        target_xy = self._sample_xy()
        self._task = ReachTask(
            language_instruction="move the point effector to the red target",
            target_xy=target_xy,
        )
        return self._get_obs(), {"task": self.get_task()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        delta = np.clip(action[:2], -1.0, 1.0) * self.step_scale
        self._state_xy = np.clip(self._state_xy + delta, -1.0, 1.0)

        distance = float(np.linalg.norm(self._task.target_xy - self._state_xy))
        success = distance <= self.success_tolerance
        reward = 1.0 if success else -distance
        done = bool(success)
        truncated = self._step_count >= self.max_steps
        info = {
            "distance": distance,
            "success": success,
            "step_count": self._step_count,
            "task": self.get_task(),
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        return self._render_image()
