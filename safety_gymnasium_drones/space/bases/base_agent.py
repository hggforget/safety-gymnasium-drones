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
"""Base class for agents."""

from __future__ import annotations

import os
from abc import ABC

import numpy as np

import safety_gymnasium_drones
from safety_gymnasium_drones.bases.base_agent import BaseAgent



BASE_DIR = os.path.dirname(safety_gymnasium_drones.__file__)


class BaseSpaceAgent(BaseAgent, ABC):  # pylint: disable=too-many-instance-attributes

    def dist(self, pos: np.ndarray) -> float:
        """Return the distance from the agent to an XYZ position.

        Args:
            pos (np.ndarray): The position to measure the distance to.

        Returns:
            float: The distance from the agent to the position.
        """
        return self.dist_xyz(pos)

    def dist_xy(self, pos: np.ndarray) -> float:
        """Return the distance from the agent to an XYZ position.

        Args:
            pos (np.ndarray): The position to measure the distance to.

        Returns:
            float: The distance from the agent to the position.
        """
        pos = np.asarray(pos)
        agent_pos = self.pos
        return np.sqrt(np.sum(np.square(pos - agent_pos)))

    def dist_xyz(self, pos: np.ndarray) -> float:
        """Return the distance from the agent to an XYZ position.

        Args:
            pos (np.ndarray): The position to measure the distance to.

        Returns:
            float: The distance from the agent to the position.
        """
        pos = np.asarray(pos)
        agent_pos = self.pos
        return np.sqrt(np.sum(np.square(pos - agent_pos)))

    def world(self, pos: np.ndarray) -> np.ndarray:
        """Return the world XYZ vector to a position from the agent.

        Args:
            pos (np.ndarray): The position to measure the vector to.

        Returns:
            np.ndarray: The world XY vector to the position.
        """
        assert pos.shape == (3,)
        return pos - self.agent.agent_pos()  # pylint: disable=no-member