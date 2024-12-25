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

        self.last_dist_goal = None
        self.last_action = None

    def is_out_of_bounds(self):
        agent_pos = self.agent.pos
        bounds = self.placements_conf.extents
        assert len(bounds) % 2 == 0
        dims = int(len(bounds) / 2)
        min_bounds = bounds[:dims]
        max_bounds = bounds[dims:]
        if dims == 2:
            min_bounds.append(0)
            max_bounds.append(2.0)
        if all([min_bound - 5.0 <= pos <= max_bound + 5.0 for min_bound, pos, max_bound
                in zip(min_bounds, agent_pos, max_bounds)]):
            return False
        return True

    def calc_hover_reward(self):

        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if 'floor' in geom_names:
                geom_names.remove('floor')
                if any(n in geom_names for n in self.agent.body_info.geom_names):
                    return 0.0
        return self._hover_reward

    def calculate_reward(self, **kwargs):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        action = kwargs['action']
        reward = 0.0
        reward += self.calc_hover_reward()
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
