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
"""HalfCheetah environment with a safety constraint on velocity."""

from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

from safety_gymnasium_drones.utils.task_utils import add_velocity_marker, clear_viewer


class SafetyHalfCheetahVelocityEnv(HalfCheetahEnv):
    """HalfCheetah environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._velocity_threshold = 2.8795
        self.model.light(0).castshadow = False

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
        }

        cost = float(x_velocity > self._velocity_threshold)

        if self.mujoco_renderer.viewer:
            clear_viewer(self.mujoco_renderer.viewer)
            add_velocity_marker(
                viewer=self.mujoco_renderer.viewer,
                pos=self.get_body_com('torso')[:3].copy(),
                vel=x_velocity,
                cost=cost,
                velocity_threshold=self._velocity_threshold,
            )
        if self.render_mode == 'human':
            self.render()
        return observation, reward, cost, terminated, False, info
