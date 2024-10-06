"""Safety-Gymnasium Environments."""

import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register

from safety_gymnasium_drones.utils.registration import make, register


__all__ = [
    'register',
    'make',
    'gymnasium_make',
    'gymnasium_register',
]

VERSION = 'v0'
ROBOT_NAMES = ('Quadrotor',)
MAKE_VISION_ENVIRONMENTS = True
MAKE_DEBUG_ENVIRONMENTS = True

# ========================================
# Helper Methods for Easy Registration
# ========================================

PREFIX = 'Safety'


def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium_drones.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **copy.deepcopy(spec_kwargs)},
        **kwargs,
    )


def __combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = copy.deepcopy(task_config)
            combined_config.update({'agent_name': robot_name})

            __register_helper(
                env_id=env_id,
                entry_point='safety_gymnasium_drones.space.builder:SpaceBuilder',
                spec_kwargs={'config': combined_config, 'task_id': env_id},
                max_episode_steps=max_episode_steps,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision inputs
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                }
                vision_config.update(combined_config)
                __register_helper(
                    env_id=vision_env_name,
                    entry_point='safety_gymnasium_drones.space.builder:SpaceBuilder',
                    spec_kwargs={'config': vision_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )

            if MAKE_DEBUG_ENVIRONMENTS and robot_name in ['Quadrotor']:
                # Keyboard inputs for debugging
                debug_env_name = f'{PREFIX}{robot_name}{task_name}Debug-{VERSION}'
                debug_config = {'debug': True}
                debug_config.update(combined_config)
                __register_helper(
                    env_id=debug_env_name,
                    entry_point='safety_gymnasium_drones.space.builder:SpaceBuilder',
                    spec_kwargs={'config': debug_config, 'task_id': env_id},
                    max_episode_steps=max_episode_steps,
                )


# ----------------------------------------
# Safety Navigation
# ----------------------------------------

# Goal Environments
# ----------------------------------------
goal_tasks = {'GoalBase': {'task_name': 'GoalBase'}, 'Goal0': {}, 'Goal1': {}, 'Goal2': {}}
__combine(goal_tasks, ROBOT_NAMES, max_episode_steps=1000)
