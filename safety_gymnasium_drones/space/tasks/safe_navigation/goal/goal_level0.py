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
"""Goal level 0."""
import math
import mujoco

from safety_gymnasium_drones.space.assets.geoms import Goal
from safety_gymnasium_drones.bases.base_task import BaseTask
import numpy as np
from safety_gymnasium_drones.utils.task_utils import angle_between_vectors


class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]
        self._add_geoms(Goal(keepout=0.305))

        self._hover_reward = 0.01
        self._euler_penalty = 2e-3
        self._angular_vel_penalty = 5e-5
        self._perception_reward = 2e-3
        self._large_action_penalty = 2e-5
        self._action_smoothness_penalty = 2e-5
        self._floor_contact_cost = 0.1

        self.last_dist_goal = None
        self.last_action = None

    def calc_floor_cost(self):

        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if 'floor' in geom_names:
                geom_names.remove('floor')
                if any(n in geom_names for n in self.agent.body_info.geom_names):
                    return self._floor_contact_cost
        return 0.0

    def calculate_cost(self) -> dict:
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Calculate constraint violations
        for obstacle in self._obstacles:
            cost.update(obstacle.cal_cost())

        cost['cost_floor_contact'] = self.calc_floor_cost()

        if self._is_load_static_geoms and self.static_geoms_contact_cost:
            cost['cost_static_geoms_contact'] = 0.0
            for contact in self.data.contact[: self.data.ncon]:
                geom_ids = [contact.geom1, contact.geom2]
                geom_names = sorted([self.model.geom(g).name for g in geom_ids])
                if any(n in self.static_geoms_names for n in geom_names) and any(
                        n in self.agent.body_info.geom_names for n in geom_names
                ):
                    # pylint: disable-next=no-member
                    cost['cost_static_geoms_contact'] += self.static_geoms_contact_cost

        # Sum all costs into single total cost
        cost['cost_sum'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        return cost

    def calculate_reward(self, **kwargs):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        action = kwargs['action']
        reward = 0.0
        reward += self._perception_reward * \
                  math.exp(- math.pow(self.optical_axis_angle[0], 4))
        reward -= self._angular_vel_penalty * np.sqrt(np.sum(np.square(self.agent.angular_vel)))
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal
        if self.last_action is None:
            self.last_action = action
        # reward -= self._large_action_penalty * np.sqrt(np.sum(np.square(action))) + \
        #           self._action_smoothness_penalty * np.sqrt(np.sum(np.square(action - self.last_action)))
        self.last_action = action

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self, **kwargs):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_action = kwargs['action']

    @property
    def optical_axis_angle(self):
        camera_axis = self.agent.camera('vision').optical_axis
        goal_axis = self.goal.pos - self.agent.pos
        angle_rad, angle_deg = angle_between_vectors(camera_axis, goal_axis)
        return angle_rad, angle_deg

    @property
    def euler(self):
        return self.agent.camera('vision').euler

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size
