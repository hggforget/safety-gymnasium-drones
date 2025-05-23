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
"""Hazard."""

from dataclasses import dataclass

import numpy as np

from safety_gymnasium_drones.assets.color import COLOR
from safety_gymnasium_drones.assets.group import GROUP
from safety_gymnasium_drones.bases.base_object import Geom


@dataclass
class Sigwalls(Geom):  # pylint: disable=too-many-instance-attributes
    """None collision walls.

    This class is used for showing the boundary which is forbidden for entering.
    """

    name: str = 'sigwalls'
    num: int = 2
    locate_factor: float = 1.125
    size: float = 3.5
    placements: list = None
    keepout: float = 0.0

    color: np.array = COLOR['sigwall']
    alpha: float = 0.1
    group: np.array = GROUP['sigwall']
    is_lidar_observed: bool = False
    is_constrained: bool = False
    is_meshed: bool = False
    mesh_name: str = name[:-1]

    def __post_init__(self) -> None:
        assert self.num in (2, 4), 'Sigwalls are specific for Circle and Run tasks.'
        assert (
            self.locate_factor >= 0
        ), 'For cost calculation, the locate_factor must be greater than or equal to zero.'
        self.locations: list = [
            (self.locate_factor, 0),
            (-self.locate_factor, 0),
            (0, self.locate_factor),
            (0, -self.locate_factor),
        ]

        self.index: int = 0

    def index_tick(self):
        """Count index."""
        self.index += 1
        self.index %= self.num

    def get_config(self, xy_pos, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        body = {
            'name': self.name,
            'pos': np.r_[xy_pos, 0.25],
            'rot': 0,
            'geoms': [
                {
                    'name': self.name,
                    'size': np.array([0.05, self.size, 0.3]),
                    'type': 'box',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': self.group,
                    'rgba': self.color * np.array([1, 1, 1, self.alpha]),
                },
            ],
        }
        if self.index >= 2:
            body.update({'rot': np.pi / 2})
        self.index_tick()
        if self.is_meshed:
            body['geoms'][0].update(
                {
                    'type': 'mesh',
                    'mesh': self.mesh_name,
                    'material': self.mesh_name,
                    'euler': [0, 0, 0],
                },
            )
        return body

    def cal_cost(self):
        """Contacts Processing."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_out_of_boundary'] = np.abs(self.agent.pos[0]) > self.locate_factor
        if self.num == 4:
            cost['cost_out_of_boundary'] = (
                cost['cost_out_of_boundary'] or np.abs(self.agent.pos[1]) > self.locate_factor
            )

        return cost

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""
