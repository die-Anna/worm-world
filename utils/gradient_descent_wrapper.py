import math
from autograd import grad
import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from worm_world.worms.env.worm_world import WormWorldEnv
from functions import *
import matplotlib.pyplot as plt


class GradientDescentWrapper(WormWorldEnv):

    def __init__(self):
        super(GradientDescentWrapper, self).__init__(render_mode='rgb_array')
        self.lower_bound = np.array([0, 0])
        self.upper_bound = np.array([self.height, self.width])
        self.position = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.position = None
        self._path = []
        self._speed_path = []

    def set_agent_position(self, position):
        if isinstance(position, ArrayBox):
            self._agent_location = np.array(position._value)
        else:
            self._agent_location = np.array(position)

    def step(self, action):
        x2, y2 = action
        dx = min(abs(x2 - self._agent_location[0]), self.width - abs(x2 - self._agent_location[0]))
        dy = min(abs(y2 - self._agent_location[1]), self.height - abs(y2 - self._agent_location[1]))
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if isinstance(distance, ArrayBox):
            distance = distance._value
        self._speed_path.append(distance)
        angle = np.arctan2(dy, dx)
        if isinstance(angle, ArrayBox):
            angle = angle._value
        angle += math.pi
        print(angle)
        self._agent_angle = angle
        location = (action - self.lower_bound) % (self.upper_bound - self.lower_bound) + self.lower_bound
        self.set_agent_position(location)
        self._detector_pos = self._agent_location + np.multiply(self._agent_direction, self.detector_distance)
        self._path.append(self._agent_location)

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
            c_distance = np.sqrt(d_width ** 2 + d_height ** 2)
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
                eat_reward = (self.max_eat_height * self.max_food_elements * self.max_number_centroids) / len(
                    self.environment_class.consumed_centroids)
                eat_reward *= 1000
                path_length = np.sum([abs(x) for x in self._speed_path[self.test_step_food_found:]])
                eat_reward = eat_reward / np.power(path_length, 1 / self.beta)
                self.test_step_food_found = self.steps
        reward = eat_reward
        step_estimate = 10000
        reward -= np.exp((self.steps - self.test_step_food_found) / step_estimate)
        gradient_bonus = self.last_observation
        if eat_reward == 0:  # when nothing has been eaten at the last step
            if gradient_bonus > 0:
                reward += gradient_bonus
            else:
                reward += gradient_bonus * 2
        self.reward += reward
        return reward, terminated

    def get_agent_location_p(self):
        return self._agent_location

    def get_obs(self):
        return self.last_observation


wrapper = GradientDescentWrapper()
wrapper.reset()
centroids = wrapper.environment_class.centroids


def function(xy):
    x, y = xy
    summed_effects = 0
    for index, c in enumerate(centroids):
        d_x = min(abs(x - c[0]), wrapper.width - abs(x - c[0]))
        d_y = min(abs(y - c[1]), wrapper.height - abs(y - c[1]))
        distance = np.sqrt(d_x * d_x + d_y * d_y)
        summed_effects += c[2] * np.exp(-distance / (2 * wrapper.sigma ** 2))
    return summed_effects


def norm(x):
    return np.sqrt(sum(i ** 2 for i in x))


def resizeVector(x, length):
    return (length / norm(x)) * x


# Get gradient function
grad_of_function = grad(function)

beta = momentum_param = 0.9
my = 1.0
start = np.array(wrapper.get_agent_location_p())
learning_rate = 0.9
total_reward = 0
for _ in range(10000):
    grad_value = grad_of_function(start)
    my = my * beta + (1 - beta) * grad_value
    step = learning_rate * my
    if norm(step) > 0.23:
        step = resizeVector(step, 0.23)
    start += step
    reward, terminated = wrapper.step(start)
    total_reward += reward
    wrapper.render()
    if terminated:
        img = wrapper.render()
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{get_save_plot_directory("gradient_descent")}/model_gd_plots.png', bbox_inches='tight')
        break

# Should lead to value near zero
print(f"Total reward: {total_reward}")
