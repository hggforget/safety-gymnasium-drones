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
"""Quad-rotor."""

from __future__ import annotations

import glfw
import numpy as np
from safety_gymnasium_drones.space.bases.base_agent import BaseSpaceAgent
from safety_gymnasium_drones.utils.random_generator import RandomGenerator


class Quadrotor(BaseSpaceAgent):
    """
    Quad rotor
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list | None = None,
        locations: list | None = None,
        keepout: float = 0.4,
        rot: float | None = None,
    ) -> None:
        super().__init__(
            self.__class__.__name__,
            random_generator,
            placements,
            locations,
            keepout,
            rot,
        )

    @property
    def mass(self) -> float:
        """Mass of the environment-body or robot.

        Returns:
            float: mass of the robot

        """
        return self.model.body_mass.sum()

    def apply_action(self, action: np.ndarray, noise: np.ndarray | None = None) -> None:
        """Apply an action to the agent.

        Just fill up the control array in the engine data.

        Args:
            action (np.ndarray): The action to apply.
            noise (np.ndarray): The noise to add to the action.
        """
        action = np.array(action, copy=False)  # Cast to ndarray

        # Set action
        action_range = self.engine.model.actuator_ctrlrange

        action = np.clip(action, action_range[:, 0], action_range[:, 1])
        # self.engine.data.ctrl[:] = self.get_motor_input(action=action)
        self.engine.data.ctrl[:] = action
        if noise:
            self.engine.data.ctrl[:] += noise


    def is_alive(self):
        """Point runs until timeout."""
        if self.pos[2] < 0.1 and abs((np.degrees(self.euler)[0] + 360) % 360 - 180) <= 1.0:
            return False
        return True

    def reset(self):
        """No need to reset anything."""

    def debug(self):
        """Apply action which inputted from keyboard."""
        action = np.array([0., 0., 0., 0.])
        for key in self.debug_info.keys:
            if key == glfw.KEY_I:
                action += np.array([1., 0., 0., 1.])
            elif key == glfw.KEY_K:
                action += np.array([0., 1., 1., 0.])
            elif key == glfw.KEY_J:
                action += np.array([1., 1., 0, 0.])
            elif key == glfw.KEY_L:
                action += np.array([1., 1., 1., 1.])
        self.apply_action(action)
