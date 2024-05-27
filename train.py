import gymnasium
from rl_zoo3.train import train
import rl_zoo3.utils as utils
import worm_world.worms  # noqa: F401 pylint: disable=unused-import


if __name__ == "__main__":
    env = gymnasium.make("WormWorld-v0")
    train()

