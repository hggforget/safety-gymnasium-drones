# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Env builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import gymnasium
import numpy as np

from safety_gymnasium_drones.bases.base_task import BaseTask
from safety_gymnasium_drones.utils.task_utils import get_task_class_name


@dataclass
class RenderConf:
    r"""Render options.

    Attributes:
        mode (str): render mode, can be 'human', 'rgb_array', 'depth_array'.
        width (int): width of the rendered image.
        height (int): height of the rendered image.
        camera_id (int): camera id to render.
        camera_name (str): camera name to render.

        Note:
            ``camera_id`` and ``camera_name`` can only be set one of them.
    """

    mode: str = None
    width: int = 256
    height: int = 256
    camera_id: int = None
    camera_name: str = None


# pylint: disable-next=too-many-instance-attributes
class BaseBuilder(gymnasium.Env, gymnasium.utils.EzPickle):
    r"""An entry point to organize different environments, while showing unified API for users.

    The Builder class constructs the basic control framework of environments, while
    the details were hidden. There is another important parts, which is **task module**
    including all task specific operation.

    Methods:

    - :meth:`_setup_simulation`: Set up mujoco the simulation instance.
    - :meth:`_get_task`: Instantiate a task object.
    - :meth:`set_seed`: Set the seed for the environment.
    - :meth:`reset`: Reset the environment.
    - :meth:`step`: Step the environment.
    - :meth:`_reward`: Calculate the reward.
    - :meth:`_cost`: Calculate the cost.
    - :meth:`render`: Render the environment.

    Attributes:

    - :attr:`task_id` (str): Task id.
    - :attr:`config` (dict): Pre-defined configuration of the environment, which is passed via
      :meth:`safety_gymnasium_drones.register()`.
    - :attr:`render_parameters` (RenderConf): Render parameters.
    - :attr:`action_space` (gymnasium.spaces.Box): Action space.
    - :attr:`observation_space` (gymnasium.spaces.Dict): Observation space.
    - :attr:`obs_space_dict` (dict): Observation space dictionary.
    - :attr:`done` (bool): Whether the episode is done.
    """

    metadata: ClassVar[dict[str, Any]] = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 30,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        task_id: str,
        config: dict | None = None,
        render_mode: str | None = None,
        width: int = 256,
        height: int = 256,
        camera_id: int | None = None,
        camera_name: str | None = None,
    ) -> None:
        """Initialize the builder.

        Note:
            The ``camera_name`` parameter can be chosen from:
              - **human**: The camera used for freely moving around and can get input
                from keyboard real time.
              - **vision**: The camera used for vision observation, which is fixed in front of the
                agent's head.
              - **track**: The camera used for tracking the agent.
              - **fixednear**: The camera used for top-down observation.
              - **fixedfar**: The camera used for top-down observation, but is further than **fixednear**.

        Args:
            task_id (str): Task id.
            config (dict): Pre-defined configuration of the environment, which is passed via
              :meth:`safety_gymnasium_drones.register`.
            render_mode (str): Render mode, can be 'human', 'rgb_array', 'depth_array'.
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
            camera_id (int): Camera id to render.
            camera_name (str): Camera name to render.
        """
        gymnasium.utils.EzPickle.__init__(self, config=config)

        self.task_id: str = task_id
        self.config: dict = config
        self._seed: int = None
        self._setup_simulation()

        self.first_reset: bool = None
        self.steps: int = None
        self.cost: float = None
        self.terminated: bool = True
        self.truncated: bool = False

        self.render_parameters = RenderConf(render_mode, width, height, camera_id, camera_name)

    def _setup_simulation(self) -> None:
        """Set up mujoco the simulation instance."""
        self.task = self._get_task()
        self.set_seed()

    def _get_task(self) -> BaseTask:
        """Instantiate a task object."""
        class_name = get_task_class_name(self.task_id)
        assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
        task_class = getattr(tasks, class_name)
        task = task_class(config=self.config)

        task.build_observation_space()
        return task

    def set_seed(self, seed: int | None = None) -> None:
        """Set internal random state seeds."""
        self._seed = np.random.randint(2**32, dtype='int64') if seed is None else seed
        self.task.random_generator.set_random_seed(self._seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:  # pylint: disable=arguments-differ
        """Reset the environment and return observations."""

        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        """Take a step and return observation, reward, cost, terminated, truncated, info."""

        raise NotImplementedError

    def _reward(self, **kwargs) -> float:
        """Calculate the current rewards.

        Call exactly once per step.
        """
        raise NotImplementedError

    def _cost(self) -> dict:
        """Calculate the current costs and return a dict.

        Call exactly once per step.
        """
        cost = self.task.calculate_cost()

        # Optionally remove shaping from reward functions.
        if self.task.cost_conf.constrain_indicator:
            for k in list(cost.keys()):
                cost[k] = float(cost[k] > 0.0)  # Indicator function

        self.cost = cost

        return cost

    def render(self) -> np.ndarray | None:
        """Call underlying :meth:`safety_gymnasium_drones.bases.underlying.Underlying.render` directly.

        Width and height in parameters are constant defaults for rendering
        frames for humans. (not used for vision)

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if render_mode is:

        - None (default): no render is computed.
        - human: render return None.
          The environment is continuously rendered in the current display or terminal. Usually for human consumption.
        - rgb_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
        - rgb_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y, 3), as with `rgb_array`.
        - depth_array: return a single frame representing the current state of the environment.
          A frame is a numpy.ndarray with shape (x, y) representing depth values for an x-by-y pixel image.
        - depth_array_list: return a list of frames representing the states of the environment since the last reset.
          Each frame is a numpy.ndarray with shape (x, y), as with `depth_array`.
        """
        assert self.render_parameters.mode, 'Please specify the render mode when you make env.'
        assert (
            not self.task.observe_vision
        ), 'When you use vision envs, you should not call this function explicitly.'
        return self.task.render(cost=self.cost, **asdict(self.render_parameters))

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        """Helper to get action space."""
        return self.task.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box | gymnasium.spaces.Dict:
        """Helper to get observation space."""
        return self.task.observation_space

    @property
    def obs_space_dict(self) -> dict[str, gymnasium.spaces.Box]:
        """Helper to get observation space dictionary."""
        return self.task.obs_info.obs_space_dict

    @property
    def done(self) -> bool:
        """Whether this episode is ended."""
        return self.terminated or self.truncated

    @property
    def render_mode(self) -> str:
        """The render mode."""
        return self.render_parameters.mode

    def __deepcopy__(self, memo) -> Builder:
        """Make class instance copyable."""
        other = Builder(
            self.task_id,
            self.config,
            self.render_parameters.mode,
            self.render_parameters.width,
            self.render_parameters.height,
            self.render_parameters.camera_id,
            self.render_parameters.camera_name,
        )
        other._seed = self._seed
        other.first_reset = self.first_reset
        other.steps = self.steps
        other.cost = self.cost
        other.terminated = self.terminated
        other.truncated = self.truncated
        other.task = deepcopy(self.task)  # pylint: disable=attribute-defined-outside-init
        return other
