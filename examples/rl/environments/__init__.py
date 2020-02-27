from gym.envs.registration import register

register(
    id = 'RobotLocomotion-v0',
    entry_point = 'environments.robot_locomotion:RobotLocomotionEnv',
    max_episode_steps=128,
)
