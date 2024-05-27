from gymnasium.envs.registration import register

register(
    id="WormWorld-v0",
    entry_point="worm_world.worms.env.worm_world:WormWorldEnv",
    max_episode_steps=5_000,
    kwargs={}
)
