import math
import numpy as np
import torch
from numpy.random import Generator


def round_down(num, digits):
    factor = 10.0 ** digits
    return math.floor(num * factor) / factor


class EnvironmentClass:
    """
     Custom environment class for simulating an agent's interaction with an environment containing food sources.

    Parameters:
    - sigma (float): Standard deviation for generating Gaussian distributions in the environment.
    - width (float): Width of the environment in arbitrary units.
    - height (float): Height of the environment in arbitrary units.
    - pixel_per_mm (int, optional): Number of pixels per millimeter for graphic representation. Defaults to 10.
    - centroids (List[Tuple[float, float, float, float]], optional): Initial positions and properties of food sources.
    - min_number_food_objects (int): Minimum number of food objects in the environment.
    - max_number_food_objects (int): Maximum number of food objects in the environment.
    - min_food_elements (int): Minimum number of food elements per food object.
    - max_food_elements (int): Maximum number of food elements per food object.
    - food_element_range (float): Range for distributing food elements around a food object.
    - min_eat_height (float): Minimum height of food elements.
    - max_eat_height (float): Maximum height of food elements.
    - sampling_method (str): Method for sampling points around food objects ('gaussian' or 'uniform').

    Attributes:
    - max_value (float): Maximum value used for normalization.
    - initial_centroids (List[Tuple[float, float, float, float]], optional): Initial positions and properties
        of food sources.
    - centroids (List[Tuple[float, float, float, float]]): Current positions and properties of food sources.
    - grids (List[np.ndarray]): Grids representing the distribution of food odor around each food object.
    - centroid_grid (np.ndarray): Grid representing the combined distribution of food odor from all unconsumed
        food objects.
    - consumed_centroids_grid (np.ndarray): Grid representing the combined distribution of food odor from
        all consumed food objects.
    - consumed_centroids (np.ndarray): Array indicating whether each food object has been consumed (1) or not (0).

    Methods:
    - fill_grid_for_centroid(centroid: Tuple[float, float, float, float]) -> np.ndarray:
        Generates a grid representing the distribution of food odor around a given food object.

    - fill_periodic_grid_with_bumps():
        Fills the environment grids with food odor distributions around all food objects.

    - recalculate_periodic_grids():
        Recalculates the combined grids representing the distribution of food odor from all unconsumed and
        consumed food objects.

    - get_value_from_coordinates(height: float, width: float) -> np.float32:
        Calculates the sum of food odor effects at a given point in the environment.

    - create_centroids() -> float:
        Initializes or sets the positions and properties of food sources in the environment.

    - uniform_sampling_around_point(height: float, width: float, radius: float, num_points: int = 10)
        -> List[Tuple[float, float]]:
        Performs uniform sampling of points around a given location within a specified radius.

    - gaussian_sampling_around_point(height: float, width: float, radius: float, num_points: int = 10)
        -> List[Tuple[float, float]]:
        Performs Gaussian sampling of points around a given location within a specified radius.

    """
    def __init__(self, np_random: Generator, sigma, width, height, pixel_per_mm, centroids,
                 min_number_food_objects, max_number_food_objects,
                 min_food_elements, max_food_elements, food_element_range,
                 min_eat_height, max_eat_height, sampling_method='gaussian', decay=False,
                 decay_rate=0.03
                 ):
        self.np_random = np_random
        self.min_number_food_objects = min_number_food_objects
        self.max_number_food_objects = max_number_food_objects
        self.min_food_elements = min_food_elements
        self.max_food_elements = max_food_elements
        self.food_element_range = food_element_range
        self.min_eat_height = min_eat_height
        self.max_eat_height = max_eat_height
        self.width = width
        self.height = height
        self.pixel_per_mm = pixel_per_mm
        self.sigma = sigma
        self.sampling_method = sampling_method

        self.max_value = 0
        self.initial_centroids = centroids
        self.centroids = centroids
        if self.centroids is None:
            self.centroids = []
        self.grids = None
        self.centroid_grid = None
        self.consumed_centroids_grid = None
        self.consumed_centroids = np.zeros(len(self.centroids), dtype=int)
        self.consumed_centroids_time = np.zeros(len(self.centroids), dtype=int)
        self.background_changed = False  # used for graphical representation only
        self.decay_rate = decay_rate
        if decay:
            self.active_get_value_function = self.get_value_from_coordinates_with_decay
        else:
            self.active_get_value_function = self.get_value_from_coordinates_without_decay

    def trigger_decay(self):
        self.consumed_centroids_time[self.consumed_centroids_time > 0] += 1

    def consume_centroid(self, index):
        self.consumed_centroids[index] = 1
        self.consumed_centroids_time[index] = 1

    # used for graphic representation only
    def fill_grid_for_centroid(self, centroid):
        centroid_grid = torch.zeros((self.width * self.pixel_per_mm, self.height * self.pixel_per_mm),
                                    dtype=torch.float64)
        centroid_height, centroid_width, height, _ = centroid
        centroid_height *= self.pixel_per_mm
        centroid_width *= self.pixel_per_mm
        # Generate coordinate grids
        w_coords, h_coords = torch.meshgrid(torch.arange(self.width * self.pixel_per_mm),
                                            torch.arange(self.height * self.pixel_per_mm), indexing='ij')
        d_width = torch.min(torch.abs(w_coords - centroid_width),
                            self.width * self.pixel_per_mm - torch.abs(w_coords - centroid_width))
        d_height = torch.min(torch.abs(h_coords - centroid_height),
                             self.height * self.pixel_per_mm - torch.abs(h_coords - centroid_height))
        distance = torch.sqrt(d_width ** 2 + d_height ** 2) / self.pixel_per_mm
        mask = distance > 0
        centroid_grid[mask] = height * torch.exp(-distance[mask] ** 2 / (2 * self.sigma ** 2)).double()
        if len(centroid_grid[~mask]) > 0:
            centroid_grid[~mask] = height
        # Transpose the centroid_grid
        centroid_grid = centroid_grid.T
        return centroid_grid.numpy()

    def fill_periodic_grid_with_bumps(self):
        self.grids = []
        for i in range(len(self.centroids)):
            self.grids.append(self.fill_grid_for_centroid(self.centroids[i]))

        self.recalculate_periodic_grids()

    def fill_periodic_grid_with_bumps_old(self):
        self.grids = []
        for index, centroid in enumerate(self.centroids):
            centroid_grid = np.zeros((self.height * self.pixel_per_mm,
                                      self.width * self.pixel_per_mm), dtype=float)
            for h in range(self.height * self.pixel_per_mm):
                for w in range(self.width * self.pixel_per_mm):
                    centroid_height, centroid_width, height, _ = centroid
                    centroid_height *= self.pixel_per_mm
                    centroid_width *= self.pixel_per_mm
                    # get minimum distance to centroid
                    d_width = min(abs(w - centroid_width), self.width * self.pixel_per_mm - abs(w - centroid_width))
                    d_height = min(abs(h - centroid_height), self.height * self.pixel_per_mm - abs(h - centroid_height))
                    distance = math.sqrt(d_width ** 2 + d_height ** 2) / self.pixel_per_mm
                    if distance > 0:
                        centroid_grid[h][w] = height * np.exp(-distance ** 2 / (2 * self.sigma ** 2))
                    else:
                        centroid_grid[h][w] = height
            self.grids.append(centroid_grid)
        self.recalculate_periodic_grids()

    def recalculate_periodic_grids(self):
        """
        used for graphic representation only - recalculates grid after changes
        """
        self.centroid_grid = np.zeros((self.height * self.pixel_per_mm,
                                       self.width * self.pixel_per_mm), dtype=float)
        self.consumed_centroids_grid = np.zeros((self.height * self.pixel_per_mm,
                                                 self.width * self.pixel_per_mm), dtype=float)
        for i in range(len(self.centroids)):
            if self.consumed_centroids[i] == 0:
                self.centroid_grid += self.grids[i]
            else:
                self.consumed_centroids_grid += self.grids[i]
        if self.max_value == 0:
            self.max_value = 0
            for row in self.centroid_grid:
                for e in row:
                    self.max_value = max(e, self.max_value)
        self.centroid_grid /= self.max_value
        self.consumed_centroids_grid /= self.max_value
        self.background_changed = True

    def get_value_from_coordinates(self, height, width):
        """
        calculates detector value form coordinates
        Args:
            height: y-coordinate
            width: x-coordinate

        Returns: detector value

        """
        return self.active_get_value_function(height, width)

    def get_value_from_coordinates_without_decay(self, height, width):
        """
        function to calculate food odor at a given point
        Args:
            height: y position
            width: x position

        Returns: sum of the odor of all remaining food elements at the given point
        """
        centroid_effects = 0
        for centroid in self.centroids:
            centroid_height, centroid_width, bump_height, _ = centroid
            d_width = min(abs(width - centroid_width), self.width - abs(width - centroid_width))
            d_height = min(abs(height - centroid_height), self.height - abs(height - centroid_height))
            distance = math.sqrt(d_width ** 2 + d_height ** 2)
            bump_value = bump_height * np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            centroid_effects += bump_value
        return np.float32(centroid_effects)

    def get_value_from_coordinates_with_decay(self, height, width):
        """
        function to calculate food odor at a given point
        Args:
            height: y position
            width: x position

        Returns: sum of the odor of all remaining food elements at the given point
        """
        centroid_effects = 0
        for centroid in self.centroids:
            centroid_height, centroid_width, bump_height, _ = centroid
            d_width = min(abs(width - centroid_width), self.width - abs(width - centroid_width))
            d_height = min(abs(height - centroid_height), self.height - abs(height - centroid_height))
            distance = math.sqrt(d_width ** 2 + d_height ** 2)
            bump_value = bump_height * np.exp(-distance ** 2 / (2 * self.sigma ** 2))
            centroid_effects += bump_value
        for i in range(len(self.consumed_centroids)):
            if self.consumed_centroids[i] == 1:
                centroid_height, centroid_width, _, bump_height = self.centroids[i]
                d_width = min(abs(width - centroid_width), self.width - abs(width - centroid_width))
                d_height = min(abs(height - centroid_height), self.height - abs(height - centroid_height))
                temporal_component = np.exp(-self.decay_rate * self.consumed_centroids_time[i])
                distance = math.sqrt(d_width ** 2 + d_height ** 2)
                centroid_effects += bump_height * np.exp(-distance ** 2 / (2 * self.sigma ** 2)) * temporal_component
        return np.float32(centroid_effects)

    def create_centroids(self):
        """
        Creates the food objects for the environment, initializes necessary data structures

        """
        # set centroids to predefined array if set - else initialize randomly
        if self.initial_centroids is not None:
            new_centroids = np.array(self.initial_centroids).copy()
        else:
            number_of_centroids = self.np_random.integers(self.min_number_food_objects, self.max_number_food_objects,
                                                          endpoint=True)
            new_centroids = []
            # eat_sum = 0
            for i in range(number_of_centroids):
                # set number of food elements for the current object and set position
                number_of_elements = self.np_random.integers(self.min_food_elements, self.max_food_elements,
                                                             endpoint=True)
                random_width = self.np_random.uniform(0, self.width, size=1).astype(np.float32)
                random_height = self.np_random.uniform(0, self.height, size=1).astype(np.float32)
                # distribute elements
                if self.sampling_method == 'gaussian':
                    points = self.gaussian_sampling_around_point(random_height[0], random_width[0],
                                                                 self.food_element_range, number_of_elements)
                else:
                    points = self.uniform_sampling_around_point(random_height[0], random_width[0],
                                                                self.food_element_range, number_of_elements)
                for p in points:
                    random_eat_height = (self.np_random.uniform
                                         (self.min_eat_height, self.max_eat_height, size=1).astype(np.float32))
                    new_centroids.append([p[0], p[1], random_eat_height[0], random_eat_height[0]])

        self.centroids = new_centroids
        self.consumed_centroids = np.zeros(len(self.centroids), dtype=int)
        self.consumed_centroids_time = np.zeros(len(self.centroids), dtype=int)

    def uniform_sampling_around_point(self, height, width, radius, num_points):
        """
        creates cluster of food object at given position (uniform distribution)
        Args:
            height: y-coordinate
            width: x-coordinate
            radius: radius in which food objects are distributed
            num_points: number of created food objects

        Returns: created food objects
        """
        points = []
        for _ in range(num_points):
            angle = self.np_random.uniform(0, 2 * math.pi)
            distance = math.sqrt(self.np_random.uniform(0, 1)) * radius
            new_width = width + distance * math.cos(angle)
            new_height = height + distance * math.sin(angle)
            new_width = new_width % self.width
            new_height = new_height % self.height
            point = new_height, new_width
            points.append(point)
        return points

    def gaussian_sampling_around_point(self, height, width, radius, num_points):
        """
        creates cluster of food object at given position (gaussian distribution)
        Args:
            height: y-coordinate
            width: x-coordinate
            radius: standard deviation in which food objects are distributed
            num_points: number of created food objects

        Returns: created food objects
        """
        points = []
        for _ in range(num_points):
            angle = self.np_random.uniform(0, 2 * math.pi)
            distance = self.np_random.normal(0, 1) * radius
            new_width = width + distance * math.cos(angle)
            new_height = height + distance * math.sin(angle)
            new_width = new_width % self.width
            new_height = new_height % self.height
            points.append((new_height, new_width))
        return points
