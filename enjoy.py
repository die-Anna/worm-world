from rl_zoo3.enjoy import enjoy
import gymnasium
import worm_world.worms  # noqa: F401 pylint: disable=unused-import

if __name__ == "__main__":
    env = gymnasium.make("WormWorld-v0")
    enjoy()
