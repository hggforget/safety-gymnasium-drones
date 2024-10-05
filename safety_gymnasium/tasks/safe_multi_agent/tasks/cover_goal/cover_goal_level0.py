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
"""CoverGoal level 0."""

from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import Goals
from safety_gymnasium.tasks.safe_multi_agent.bases.base_task import BaseTask
import numpy as np


class CoverGoalLevel0(BaseTask):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.last_dist_goal = None
        self.placements_conf.extents = [-3, -3, 3, 3]
        self._add_geoms(
            Goals(keepout=0.305, num=self.agents.num),
        )
        self.goal_achieved_index = np.zeros(self.agents.num, dtype=bool)

        self._reward_for_staying_alive = 0.01
        self._euler_penalty = 1e-3
        self._angular_vel_penalty = 5e-4
        self._perception_reward = 0.01
        self._perception_reward_lambda = 2e-2

    def is_out_of_bounds(self):
        agent_pos = self.agents.pos
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

    def is_collision(self, agent1, agent2) -> bool:
        delta_pos = agent1.pos - agent2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def dist_index_goals(self, index) -> list:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goals'), 'Please make sure you have added goal into env.'
        return [self.agents.dist_xyz(pos, index) for pos in self.goals.pos]

    def dist_goal(self):
        """Return the smallest distance from the agent to the goal XY position."""
        assert hasattr(self, 'goals'), 'Please make sure you have added goal into env.'
        return [min(self.dist_index_goals(i)) for i in range(self.agents.num)]

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {f'agent_{i}': 0.0 for i in range(self.agents.num)}
        for index in range(self.agents.num):
            reward[f'agent_{index}'] += self._reward_for_staying_alive if self.agents.is_alive(index) else 0

        dist_goal = self.dist_goal()
        reward_dist = [(self.last_dist_goal[i] - dist_goal[i]) * self.goals.reward_distance
                       for i in range(self.agents.num)]
        self.last_dist_goal = dist_goal

        for index in range(self.agents.num):
            reward[f'agent_{index}'] += reward_dist[index]

        if self.goal_achieved.all():
            for index in range(self.agents.num):
                reward[f'agent_{index}'] += self.goals.reward_goal
        self.last_dist_goal = dist_goal
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goals_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member
        self.goal_achieved_index = np.zeros(self.agents.num, dtype=bool)

        for index in range(self.agents.num):
            dist_goal = np.array(self.dist_index_goals(index))
            local_achieved = dist_goal <= self.goals.size
            self.goal_achieved_index |= local_achieved


        return self.goal_achieved_index.all()
