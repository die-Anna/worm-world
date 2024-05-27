import math
from collections import deque

import gymnasium

from worm_world.worms.worm_env_functions import *
from worm_world.worms.worm_env_functions.environment_class import EnvironmentClass
from worm_world.worms.worm_env_functions.render_env_class import RenderWormClass


class WormWorldEnv(gymnasium.Env):
    """
Gym environment for a nematode agent searching for food objects.
    The nematode, represented by the head, explores the environment to find and consume food objects.

    Parameters:
    - height & width (floats): size of the environment
    - max_speed & min_speed (floats): speed of the agent
    - max_rotation=math.pi / 4 (float): maximum rotation of the agent (0, pi/2)
    - detector_distance (float): distance of the detector from the head of the agent
    - eat_distance (float): radius of the food elements
    - min_number_food_objects & max_number_food_objects (ints): number of food clusters
    - min_food_elements & max_food_elements (ints): number of food elements per cluster
    - food_element_range (float): radius / standard deviation of the food cluster (uniform / gaussian)
    - sampling_method (string): 'gaussian' or 'uniform' distribution of food elements in the cluster
    - min_eat_height & max_eat_height (floats): 'height' of the food elements
    - eat_amount (float): amount the agent can consume from a food object at one step
    - noise_std_dev (float): noise of the detector
    - sigma (float): decay parameter for the food objects
    - action_space_p (int): power of the action space transfer function
    - centroids (array): None or array of initial centroids (array of four floats)
    - start_position tuple(float, float): start position of the agent
    - start_angle (float): start angle of the agent [0, 2pi]

    Observation Space:
        - Unbounded continuous space representing the environment (wrap around). Food objects are randomly distributed.

    Action Space:
        - Continuous actions representing the movement of the nematode's head: speed and rotation.

    Rewards:
        - Eat amount for each time the nematode's head overlaps with a food object (max height of food object).
        - Negative reward for movement (speed, rotation) according to the given values.

    Episode Termination:
        - When the nematode consumes all the food objects or reaches a maximum number of time steps.

    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, height=35, width=35, pixel_per_mm=15,
                 max_speed=0.25, min_speed=0.00, max_rotation=math.pi * 0.25, detector_distance=0.1,
                 test_eat_fixed_amount=True, hunger_penalty=True, gradient_reward=True,
                 reward_type='math', eat_distance=0.2,
                 min_number_food_objects=2, max_number_food_objects=4,
                 min_food_elements=1, max_food_elements=1, food_element_range=0.4,
                 eat_amount=10, min_eat_height=1, max_eat_height=10,
                 use_multiplicative_noise=True, use_additive_noise=True,
                 multiplicative_noise=0.003, additive_noise=0.003456, sigma=3, action_space_p=2,
                 sampling_method='uniform', centroids=None, start_position=None, start_angle=None,
                 beta=2.0, food_memory=False,
                 frame_stacking=True, frame_stack_size=8, decay=False, decay_factor=100, decay_rate=0.03):
        self.beta = beta
        reward_types = ['physics', 'math']
        if reward_type not in reward_types:
            raise ValueError("Invalid reward type. Expected one of: %s" % reward_types)
        self.reward_type = reward_type
        if test_eat_fixed_amount not in [True, False]:
            raise ValueError("Invalid value for test_eat_fixed_amount. Expected boolean")
        self.test_eat_fixed_amount = test_eat_fixed_amount
        if hunger_penalty not in [True, False]:
            raise ValueError("Invalid value for hunger_penalty. Expected boolean")
        self.hunger_penalty = hunger_penalty
        if gradient_reward not in [True, False]:
            raise ValueError("Invalid value for gradient_reward. Expected boolean")
        self.gradient_reward = gradient_reward
        if frame_stacking not in [True, False]:
            raise ValueError("Invalid value for frame_stacking. Expected boolean")
        self.frame_stacking = frame_stacking
        self.frame_stack_size = frame_stack_size
        self.pixel_per_mm = pixel_per_mm
        # initialize all adjusted values
        self.height = height
        self.width = width
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.detector_distance = detector_distance
        self.eat_distance = eat_distance
        self.food_element_range = food_element_range

        self.sigma = sigma
        self.max_rotation = max_rotation
        self.min_eat_height = min_eat_height
        self.max_eat_height = max_eat_height
        self.min_number_centroids = min_number_food_objects
        self.max_number_centroids = max_number_food_objects
        self.min_food_elements = min_food_elements
        self.max_food_elements = max_food_elements
        self.initial_agent_angle = start_angle
        self.initial_agent_location = start_position
        self.eat_amount = eat_amount
        self.decay = decay
        self.decay_factor = decay_factor

        # noise params
        self.multiplicative_noise_std_dev = multiplicative_noise
        self.additive_noise = additive_noise
        self.use_additive_noise = use_additive_noise
        self.use_multiplicative_noise = use_multiplicative_noise

        # initialize data structures
        self.last_observation_absolute_value = None
        self.last_observation = 0
        self.success = False
        self.test_step_food_found = 0
        self.steps = 0
        self._path = []
        self._agent_direction = None
        self._agent_angle = None
        self._agent_location = None
        self._detector_pos = None
        self._speed_path = []
        self._speed_path.append(0)
        self.centroids = centroids
        self.reward = 0
        # save current observation, speed, rotation
        self.info = [0, 0]
        # Define the observation space
        self.food_memory = food_memory
        if frame_stacking:
            if self.food_memory:
                self.observation_space = spaces.Box(low=-1, high=1, shape=(self.frame_stack_size + 1,), dtype=np.float32)
                self.observation_deque = deque([0.] * (self.frame_stack_size + 1), maxlen=(self.frame_stack_size + 1))
            else:
                self.observation_space = spaces.Box(low=-1, high=1, shape=(self.frame_stack_size,), dtype=np.float32)
                self.observation_deque = deque([0.] * self.frame_stack_size, maxlen=self.frame_stack_size)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define the action space
        self.custom_action_space = CustomActionSpace(max_speed, min_speed, max_rotation, action_space_p)
        self.action_space = self.custom_action_space.original_action_space
        # initialize environment
        self.environment_class = EnvironmentClass(self.np_random,
                                                  centroids=centroids, sigma=sigma, width=self.width,
                                                  height=self.height, pixel_per_mm=pixel_per_mm,
                                                  min_number_food_objects=min_number_food_objects,
                                                  max_number_food_objects=max_number_food_objects,
                                                  min_food_elements=min_food_elements,
                                                  max_food_elements=max_food_elements,
                                                  food_element_range=self.food_element_range,
                                                  min_eat_height=min_eat_height, max_eat_height=max_eat_height,
                                                  sampling_method=sampling_method, decay=decay, decay_rate=decay_rate)
        # initialize render class if necessary
        self.call_render_function = False
        self.renderWormClass = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode in ["human", "rgb_array"]:
            self.call_render_function = True
            self.renderWormClass = RenderWormClass(self.environment_class, render_mode, pixel_per_mm,
                                                   max_speed, self.metadata["render_fps"], sigma, eat_distance)

    def _get_obs(self):
        """
        Returns the odor value at the current position of the detector.

        Returns:
        numpy float32 array with one value.
        """
        height, width = self._detector_pos
        background_value = self.environment_class.get_value_from_coordinates(height, width)
        # print(f'background value: {background_value}')
        noise = self.np_random.normal(0, self.multiplicative_noise_std_dev)
        if self.use_multiplicative_noise:
            old_value = background_value
            background_value *= max(0., (1 + noise))
        gradient = 0
        if self.last_observation_absolute_value is not None:
            gradient = background_value - self.last_observation_absolute_value
        self.last_observation_absolute_value = background_value
        # gradient *= max(0., (1 + noise))
        # if gradient == 0:
        #     gradient = noise
        # print(f'gradient: {gradient}')
        if self.use_additive_noise:
            old_value = gradient
            additive_noise = self.np_random.normal(0, self.additive_noise)
            gradient += additive_noise
            # print(f'additive noise: {additive_noise}, new gradient value: {gradient}, difference: {old_value - gradient}')
        # input()
        self.last_observation = gradient
        if self.frame_stacking:
            self.observation_deque.append(gradient)
            if self.food_memory:
                self.observation_deque[0] = np.exp(-(self.steps - self.test_step_food_found +
                                                     0 if self.test_step_food_found > 0 else 1000) / self.decay_factor)
            return np.array(self.observation_deque, dtype=np.float32)
        else:
            return np.array([gradient], dtype=np.float32)

    def _get_info(self):
        """
        Calculates how many food elements still remain in the environment
        Returns:
        number of food elements left in the environment
        """
        return {'speed': self.info[0], 'rotation': self.info[1],
                'agent_location': self._agent_location, 'success': self.success, 'total_reward': self.reward}

    def reset(self, seed=None, options=None):
        """
        Resets all data structures
        Args:
            seed: (optional int) – The seed that is used to initialize the environment’s PRNG (np_random)
            options: (optional dict) - not used in this env
        Returns:
            odor value at the initial detector position, number of food elements in the environment
        """
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.last_observation_absolute_value = None
        self.last_observation = 0
        self.success = False
        self.test_step_food_found = 0
        self.environment_class.np_random = self.np_random
        # reset speed and position arrays
        self._path = []
        self._speed_path = []
        self._speed_path.append(0)
        self.reward = 0
        if self.food_memory:
            self.observation_deque = deque([0.] * (self.frame_stack_size + 1), maxlen=(self.frame_stack_size + 1))
        else:
            self.observation_deque = deque([0.] * self.frame_stack_size, maxlen=self.frame_stack_size)
        # initialize centroids and set detector eat amount (for observation calculation)
        self.environment_class.create_centroids()
        # division_factor = self.environment_class.create_centroids()
        # self.eat_amount_detector = self.eat_amount / division_factor
        # initialize grids for graphic representation if necessary
        if self.call_render_function:
            self.environment_class.fill_periodic_grid_with_bumps()
        # Randomly initialize the agent's location within the environment bounds if not set by param
        if self.initial_agent_location is not None:
            self._agent_location = self.initial_agent_location
        else:
            width = self.np_random.uniform(0, self.width, size=1).astype(np.float32)
            height = self.np_random.uniform(0, self.height, size=1).astype(np.float32)
            self._agent_location = [height[0], width[0]]
        # Randomly initialize the agent direction if not set by param
        if self.initial_agent_angle is not None:
            angle = self.initial_agent_angle
        else:
            angle = self.np_random.uniform(0, 2 * np.pi)
        self._agent_angle = angle

        # calculate agent direction and detector position
        self._agent_direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        self._detector_pos = self._agent_location + np.multiply(self._agent_direction, self.detector_distance)

        # fetch observation and info
        observation = self._get_obs()
        info = self._get_info()
        # initialize frame
        # if self.render_mode == "human" or self.render_mode == "rgb_array":
        if self.render_mode == "human":
            self.renderWormClass.background_surface = None
            if self.frame_stacking:
                self.renderWormClass.render_frame(self._path, self._agent_location,
                                                  self._agent_angle, self._speed_path, self.reward, observation[-1])
            else:
                self.renderWormClass.render_frame(self._path, self._agent_location,
                                                  self._agent_angle, self._speed_path, self.reward, observation)

        self.info = [0, 0]
        return observation, info

    def step(self, action):
        """
        Run one timestep of the environment’s dynamics using the agent actions (speed and rotation).
        When the end of an episode is reached (terminated or truncated), it is necessary to call reset() to
        reset this environment’s state for the next episode.
        Args:
            action: (ActType) – an action provided by the agent to update the environment state.
                speed [-1, 1], rotation change [-1, 1]
        Returns:  observation (np.array[1] float32), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # perform action space transformation on speed and rotation
        self.steps += 1
        speed, rotation = self.custom_action_space.transform_action(action)
        self._speed_path.append(speed)
        # calculate new angle and new agent + detector position
        angle = self._agent_angle + rotation
        self._agent_angle = angle % (2 * math.pi)
        self.info[0] = speed
        self.info[1] = rotation
        self._agent_direction = np.array([np.cos(self._agent_angle), np.sin(self._agent_angle)], dtype=np.float32)
        # Update the agent's position based on velocity (wrap around if it reaches boundaries)
        location = (self._agent_location + np.multiply(self._agent_direction, speed))
        lower_bound = np.array([0, 0])
        upper_bound = np.array([self.height, self.width])
        self._agent_location = (location - lower_bound) % (upper_bound - lower_bound) + lower_bound
        self._detector_pos = self._agent_location + np.multiply(self._agent_direction, self.detector_distance)
        # Append the current agent's position to the path
        self._path.append(tuple(self._agent_location))
        # check reward
        terminated = truncated = False
        eat_reward = 0
        # initialize with value greater than min value
        distance = self.eat_distance + 1
        found_index = -1
        # loop through all centroids to get the nearest one in eating range
        # centroid: 0: height, 1: width, 2: detector eat amount, 3: eat amount
        for i, centroid in enumerate(self.environment_class.centroids):
            # skip already consumed centroids
            if self.environment_class.consumed_centroids[i] == 1:
                continue
            d_height = min(abs(self._agent_location[0] - centroid[0]),
                           self.height - abs(self._agent_location[0] - centroid[0]))
            d_width = min(abs(self._agent_location[1] - centroid[1]),
                          self.width - abs(self._agent_location[1] - centroid[1]))
            c_distance = math.sqrt(d_width ** 2 + d_height ** 2)
            if c_distance <= self.eat_distance:
                if c_distance < distance:
                    distance = c_distance
                    found_index = i
        # if a food element is in eating range - consume
        if found_index >= 0:
            eat_reward = self.eat_amount
            # self.environment_class.centroids[found_index][3] -= self.eat_amount
            self.environment_class.centroids[found_index][2] -= self.eat_amount
            # self.environment_class.centroids[found_index][2] -= self.eat_amount_detector
            if self.environment_class.centroids[found_index][2] <= 0:
                # subtract value below zero from reward, set values to zero and centroid to 'consumed'
                eat_reward += self.environment_class.centroids[found_index][3]
                # self.environment_class.centroids[found_index][3] = 0
                self.environment_class.centroids[found_index][2] = 0
                self.environment_class.consume_centroid(found_index)
                # self.environment_class.consumed_centroids[found_index] = 1  # mark as consumed
                # terminate when no centroids are left
                if (sum(self.environment_class.consumed_centroids) ==
                        len(self.environment_class.consumed_centroids)):
                    terminated = True
                    self.success = True
                if self.call_render_function:
                    self.environment_class.recalculate_periodic_grids()
                if self.test_eat_fixed_amount:
                    test = (self.max_eat_height * self.max_food_elements * self.max_number_centroids) / len(
                        self.environment_class.consumed_centroids)
                    # eat_reward = self.max_eat_height
                    eat_reward = test
                eat_reward *= 1000
                path_length = np.sum([abs(x) for x in self._speed_path[self.test_step_food_found:]])
                if self.reward_type == 'physics':
                    eat_reward = eat_reward * math.exp(-path_length / self.beta)
                else:
                    eat_reward = eat_reward / math.pow(path_length, 1 / self.beta)
                self.test_step_food_found = self.steps
        # calculate rewards
        # speed_change = (self._speed_path[-1] - self._speed_path[-2])
        # reward = (eat_reward - abs(speed) * self._speed_costs -
        #           abs(rotation) * self._rotation_costs -
        #           pow(abs(speed_change), 2) * self.speed_change_costs)
        observation = self._get_obs()
        reward = eat_reward
        if self.hunger_penalty:
            step_estimate = 10000
            reward -= math.exp((self.steps - self.test_step_food_found) / step_estimate)
        if self.gradient_reward:
            gradient_bonus = self.last_observation
            if eat_reward == 0:  # when nothing has been eaten at the last step
                if gradient_bonus > 0:
                    reward += gradient_bonus
                else:
                    reward += gradient_bonus * 2
        self.reward += reward
        info = self._get_info()
        if self.render_mode == "human":
            if self.frame_stacking:
                self.renderWormClass.render_frame(self._path, self._agent_location,
                                                  self._agent_angle, self._speed_path, self.reward, observation[-1])
            else:
                self.renderWormClass.render_frame(self._path, self._agent_location,
                                                  self._agent_angle, self._speed_path, self.reward, observation)
        # observation, reward, terminated, truncated, info
        if self.decay:
            self.environment_class.trigger_decay()
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Compute the render frames
        """
        if self.render_mode == "rgb_array":
            return self.renderWormClass.render_frame(self._path, self._agent_location,
                                                     self._agent_angle, self._speed_path, self.reward,
                                                     self.last_observation)
        elif self.render_mode == 'human':
            self.renderWormClass.render_frame(self._path, self._agent_location,
                                              self._agent_angle, self._speed_path, self.reward, self.last_observation)
        else:
            super().render()

    def close(self):
        """
        closes pygame for rendering
        """
        if self.call_render_function:
            self.renderWormClass.close()
