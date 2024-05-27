import gymnasium
import numpy as np
import worm_world.worms  # noqa: F401 pylint: disable=unused-import


class ChemotaxisAlgorithmWrapper:

    def __init__(self, initial_step_size=0.15, momentum=0.8, beta=2.0, sigma=3.0, min_number_food_objects=2,
                 max_number_food_objects=4, height=35, width=35,
                 reward_type='math', test_eat_fixed_amount=True,
                 hunger_penalty=True, gradient_reward=True, use_multiplicative_noise=True,
                 use_additive_noise=True, frame_stacking=False, centroids=None, start_position=None, start_angle=None):
        self.initial_step_size = initial_step_size
        self.height = height
        self.width = width
        self.momentum = momentum
        self.beta = beta
        self.sigma = sigma
        self.min_number_food_objects = min_number_food_objects
        self.max_number_food_objects = max_number_food_objects
        self.reward_type = reward_type
        self.test_eat_fixed_amount = test_eat_fixed_amount
        self.hunger_penalty = hunger_penalty
        self.gradient_reward = gradient_reward

        self.use_multiplicative_noise = use_multiplicative_noise
        self.use_additive_noise = use_additive_noise
        self.frame_stacking = frame_stacking
        self.action_solution = np.random.uniform(-1, 1, size=(2,))
        self.velocity = np.zeros_like(self.action_solution)

        self.centroids = centroids
        self.start_position = start_position
        self.start_angle = start_angle

    def get_action2(self, gradient):
        self.action_solution[0] = self.action_solution[0] + self.initial_step_size * abs(gradient)
        if gradient > 0:
            self.action_solution[1] = self.action_solution[1] * self.momentum
        else:
            self.action_solution[1] = np.random.uniform(-1, 1)  # Randomize rotation
        # Ensure actions are within bounds
        self.action_solution = np.clip(self.action_solution, -1, 1)
        return self.action_solution

    def get_action3(self, gradient):
        self.initial_step_size = 0.15
        self.momentum = 0.55
        if gradient > 0:
            # Positive gradient: Encourage maintaining direction
            # Maintain/increase speed
            self.action_solution[0] = np.clip(self.action_solution[0] + self.initial_step_size * gradient, -1, 1)
            self.action_solution[1] *= 0.9  # Reduce rotation
        else:
            # Negative gradient: Encourage direction change
            self.action_solution[0] = np.clip(self.action_solution[0] - self.initial_step_size * gradient, -1,
                                              1)  # Reduce speed
            self.action_solution[1] = np.random.uniform(-1, 1)  # Randomize rotation
        # if self.use_momentum:
        self.velocity = self.momentum * self.velocity + self.initial_step_size * gradient
        #     print(self.velocity)
        self.action_solution += self.velocity
        #     print(self.action_solution)
        #     # input('press Enter')
        # else:
        # self.action_solution += self.initial_step_size * gradient

        # Ensure actions are within bounds
        self.action_solution = np.clip(self.action_solution, -1, 1)
        return self.action_solution

    def get_action(self, gradient):
        if gradient > 0:
            # Positive gradient: Encourage maintaining direction
            # Maintain/increase speed
            self.action_solution[0] = self.action_solution[0] + self.initial_step_size
            self.action_solution[1] *= 0.9  # Reduce rotation
        else:
            # Negative gradient: Encourage direction change, reduce speed
            self.action_solution[0] = self.action_solution[0] - self.initial_step_size
            self.action_solution[1] = np.random.uniform(-1, 1)  # Randomize rotation

        self.velocity = self.momentum * self.velocity + self.initial_step_size * gradient
        self.action_solution += self.velocity

        # Ensure actions are within bounds
        self.action_solution = np.clip(self.action_solution, -1, 1)
        return self.action_solution

    def get_action4(self, gradient):
        if gradient > 0:
            # Positive gradient: Encourage maintaining direction
            # Maintain/increase speed
            self.action_solution[0] = np.clip(self.action_solution[0] + self.initial_step_size, -1, 1)
            self.action_solution[1] *= 0.9  # Reduce rotation
        else:
            # Negative gradient: Encourage direction change
            self.action_solution[0] = np.clip(self.action_solution[0] - self.initial_step_size, -1, 1)  # Reduce speed
            self.action_solution[1] = np.random.uniform(-1, 1)  # Randomize rotation

        self.action_solution += self.initial_step_size * gradient

            # Ensure actions are within bounds
        self.action_solution = np.clip(self.action_solution, -1, 1)
        return self.action_solution

    def create_env(self):
        return gymnasium.make("WormWorld-v0", render_mode="rgb_array", beta=self.beta, sigma=self.sigma,
                              width=self.width, height=self.height,
                              min_number_food_objects=self.min_number_food_objects, hunger_penalty=self.hunger_penalty,
                              gradient_reward=self.gradient_reward, use_multiplicative_noise=
                              self.use_multiplicative_noise, max_number_food_objects=self.max_number_food_objects,
                              use_additive_noise=self.use_additive_noise, reward_type=self.reward_type,
                              frame_stacking=self.frame_stacking,
                              test_eat_fixed_amount=self.test_eat_fixed_amount, centroids=self.centroids,
                              start_angle=self.start_angle, start_position=self.start_position
                              )
