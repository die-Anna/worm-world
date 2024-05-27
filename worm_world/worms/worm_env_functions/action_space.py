import numpy as np
from typing import List
from gymnasium import spaces


class CustomActionSpace:
    """
    Custom Action Space for controlling an agent with a normalized continuous action space.

    This class transforms a given continuous action space, represented as speed and rotation,
    into a normalized and bounded action space.

    Parameters:
    - max_speed (float): Maximum speed allowed for the agent.
    - min_speed (float): Minimum speed allowed for the agent.
    - max_rotation (float): Maximum rotational speed (rotation) allowed for the agent.
    - p (float, optional): Exponent value for the norm calculation. Defaults to 2.

    Attributes:
    - original_action_space (gym.spaces.Box): Original continuous action space before transformation.

    Methods:
    - norm(v: List[float]) -> float:
        Calculates the p-norm of a given vector v.

    - transform_action(action: Tuple[float, float]) -> Tuple[float, float]:
        Transforms the original action into a normalized action within the specified speed and rotation constraints.

    """
    def __init__(self, max_speed, min_speed, max_rotation, p=2):
        self.p = p
        self.max_speed = max_speed - min_speed
        self.min_speed = min_speed
        self.max_rotation = max_rotation
        self.original_action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

    def norm(self, v: List[float]):
        # if sum([x ** 2 for x in v]) == 0:
        #    return 0
        return sum([x ** self.p for x in v]) ** (1 / self.p)

    # def transform(self, v: List[float], p=2):
    def transform_action(self, action):
        speed, rotation = action
        max_norm = max([abs(x) for x in (speed, rotation)])
        norm = self.norm([speed, rotation])
        factor = max_norm
        if norm > 0:
            factor /= norm
        new_speed = speed * factor * self.max_speed
        if new_speed >= 0:
            return new_speed + self.min_speed, rotation * factor * self.max_rotation
        else:
            return new_speed - self.min_speed, rotation * factor * self.max_rotation
