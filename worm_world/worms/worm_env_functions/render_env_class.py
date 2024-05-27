import copy
import math

import numpy as np
import pygame


class RenderWormClass:
    """
     Class for rendering the environment and agent interactions in a Pygame window.

    Parameters:
    - environment_class (EnvironmentClass): Instance of the EnvironmentClass providing the environment information.
    - render_mode (str): Rendering mode, either 'human' for displaying the window or 'rgb_array' for obtaining frames.
    - pixel_per_mm (int): Number of pixels per millimeter for graphic representation.
    - max_speed (float): Maximum speed allowed for the agent.
    - render_fps (int): Frames per second for rendering when in 'human' mode.
    - sigma (float): Standard deviation for generating Gaussian distributions in the environment.
    - eat_distance (float): Distance for rendering the detection area of the agent.

    Attributes:
    - window (pygame.Surface): Pygame window surface for rendering.
    - clock (pygame.time.Clock): Pygame clock for controlling rendering speed.

    Methods:
    - calculate_background_surface() -> pygame.Surface:
        Calculates and returns the background surface based on the current state of the environment.

    - render_frame(path: List[np.ndarray], agent_location: np.ndarray, agent_direction: np.ndarray,
                   agent_angle: float, speed_array: np.ndarray):
        Renders and displays a frame of the environment with the agent's path, location, and detection area.

    - close():
        Closes the Pygame window when rendering is done.
    """

    def __init__(self, environment_class, render_mode, pixel_per_mm, max_speed, render_fps, sigma, eat_distance):
        self.render_fps = render_fps
        self.environment_class = environment_class
        self.max_speed = max_speed
        self.pixel_per_mm = pixel_per_mm
        self.render_mode = render_mode
        self.eat_distance = eat_distance
        self.sigma = sigma
        self.window = None
        self.clock = None
        self.background_surface = None

    def calculate_background_surface_old(self):
        """
        Create Pygame surface with alpha channel -- OLD VERSION
        Returns: pygame Surface (background with visible odor)
        """
        # create pygame surface with alpha channel
        # Surface((width, height), flags=0, depth=0, masks=None) -> Surface
        background_surface = pygame.Surface(
            (self.environment_class.width * self.pixel_per_mm,
             self.environment_class.height * self.pixel_per_mm), pygame.SRCALPHA)
        # self.environment_class.recalculate_periodic_grids()
        for h in range(self.environment_class.height * self.pixel_per_mm):
            for w in range(self.environment_class.width * self.pixel_per_mm):
                value_grid = self.environment_class.centroid_grid[h][w]
                value_consumed_grid = self.environment_class.consumed_centroids_grid[h][w]
                color = int(255 * value_grid)  # Scale the value to a color between 0 and 255
                color_consumed = np.clip(int(255 * value_consumed_grid), 0, 255)
                mix_color = np.clip(color + color_consumed, 0, 255)
                # Rect(left, top, width, height) -> Rect
                pygame.draw.rect(background_surface, (255 - mix_color, 255 - mix_color, 255 - color_consumed),
                                 (w, h, 1, 1))
        return background_surface

    def calculate_background_surface(self):
        """
        Create Pygame surface with alpha channel
        Returns: pygame Surface (background with visible odor)
        """
        # Create Pygame surface with alpha channel
        background_surface = pygame.Surface(
            (self.environment_class.width * self.pixel_per_mm,
             self.environment_class.height * self.pixel_per_mm), pygame.SRCALPHA)

        for h in range(self.environment_class.height * self.pixel_per_mm):
            for w in range(self.environment_class.width * self.pixel_per_mm):
                value_grid = self.environment_class.centroid_grid[h][w]
                value_consumed_grid = self.environment_class.consumed_centroids_grid[h][w]

                color = int(255 * value_grid)  # Scale the value to a color between 0 and 255
                color_consumed = min(max(int(255 * value_consumed_grid), 0), 255)  # Clamp the value between 0 and 255
                mix_color = min(max(color + color_consumed, 0), 255)  # Clamp the value between 0 and 255

                # Set pixel color
                background_surface.set_at((w, h), (255 - mix_color, 255 - mix_color, 255 - color_consumed, 255))

        return background_surface

    def render_frame(self, path, agent_location, agent_angle, speed_array, reward, observation):
        """
        Render pygame frame
        Args:
            path: array with agent positions
            agent_location: current agent position
            agent_angle: current agent angle
            speed_array: array with speed actions
            reward: current reward
            observation: current observation

        Returns: rgb image when render_mode 'rgb_array' is set

        """
        # pygame.display.set_mode(WIDTH, HEIGHT)
        if self.window is None and self.render_mode == "human":
            self.window = pygame.display.set_mode((self.environment_class.width * self.pixel_per_mm,
                                                   self.environment_class.height * self.pixel_per_mm))
            pygame.display.set_caption("Worm Environment")
            pygame.font.init()  # you have to call this at the start,
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.window is None and self.render_mode == "rgb_array":
            self.window = pygame.display.set_mode((self.environment_class.width * self.pixel_per_mm,
                                                   self.environment_class.height * self.pixel_per_mm))
            pygame.display.set_caption("Worm Environment")
            pygame.font.init()  # you have to call this at the start,
            # Get the background surface from calculate_background_surface()
        if self.background_surface is None or self.environment_class.background_changed:
            self.background_surface = self.calculate_background_surface()
            self.environment_class.background_changed = False

        # Create a transparent overlay surface to draw the agent, target, and path
        overlay_surface = pygame.Surface((self.environment_class.width * self.pixel_per_mm,
                                          self.environment_class.height * self.pixel_per_mm), pygame.SRCALPHA)

        # Draw the path as a line with alpha channel
        if len(path) > 1:
            last_point = copy.copy(path[0])
            last_point = np.multiply(last_point, self.pixel_per_mm)
            for i, point in enumerate(path):
                p = copy.copy(point)
                p = np.multiply(p, self.pixel_per_mm)
                distance = np.linalg.norm(np.array(p) - np.array(last_point)) / (self.max_speed * self.pixel_per_mm)
                if distance <= 1:
                    speed_factor = np.clip(speed_array[i] / self.max_speed, -1, 1)
                    if speed_factor > 0:
                        pygame.draw.line(overlay_surface,
                                         (255, int(255 - 255 * speed_factor), 0, 255),
                                         (last_point[1], last_point[0]), (p[1], p[0]), width=2)
                    else:
                        speed_factor = abs(speed_factor)
                        pygame.draw.line(overlay_surface,
                                         (0, 255, int(255 - 255 * speed_factor), 255),
                                         (last_point[1], last_point[0]), (p[1], p[0]), width=2)
                last_point = p
        # to prevent freezing
        pygame.event.get()

        # Draw the agent as a blue circle with alpha channel
        pygame.draw.circle(
            overlay_surface,
            (0, 0, 255, 255),  # Blue color with alpha channel
            (int(agent_location[1] * self.pixel_per_mm), int(agent_location[0] * self.pixel_per_mm)),
            2  # Radius of the circle
        )
        # draw the center of the centroids
        for i, c in enumerate(self.environment_class.centroids):
            color = (255, 255, 0, 255)  # food color
            if self.environment_class.consumed_centroids[i] == 1:  # consumed food color
                color = (0, 0, 0, 255)
            else:
                pygame.draw.circle(overlay_surface, color,
                                   (int(c[1] * self.pixel_per_mm),
                                    int(c[0] * self.pixel_per_mm)), self.eat_distance * self.pixel_per_mm, 1)
            pygame.draw.circle(
                overlay_surface,
                color,
                (int(c[1] * self.pixel_per_mm),
                 int(c[0] * self.pixel_per_mm)),
                1  # Radius of the circle
            )

        # draw arc
        pygame.draw.arc(overlay_surface, (255, 255, 0, 255),
                        [int(agent_location[1] * self.pixel_per_mm - 10),
                         int(agent_location[0] * self.pixel_per_mm - 10),
                         20, 20],
                        agent_angle - 5 * math.pi / 8, agent_angle - 3 * math.pi / 8, 10)

        # draw labels
        mm = 5
        pygame.draw.line(overlay_surface, (0, 0, 0, 255),
                         (20, 15), (20, 19), 1)
        pygame.draw.line(overlay_surface, (0, 0, 0, 255),
                         (20, 17), (20 + self.pixel_per_mm * mm, 17), 1)
        pygame.draw.line(overlay_surface, (0, 0, 0, 255),
                         (20 + self.pixel_per_mm * mm, 15), (20 + self.pixel_per_mm * mm, 19), 1)

        my_font = pygame.font.SysFont('FreeMono, Monospace', 15)
        text_top_right = my_font.render(f'{mm} mm', True, (0, 0, 0, 255))
        text_bottom_right = my_font.render(f'{len(path)}s', True, (0, 0, 0, 255))
        text_top_left = my_font.render(f'reward: {reward:.2f}', True, (0, 0, 0, 255))
        text_bottom_left = my_font.render(f'obs: {observation: .3f}', True, (0, 0, 0, 255))

        # Blit the background surface onto the display
        self.window.blit(self.background_surface, (0, 0))

        # Blit the overlay surface (agent, target, and path) onto the display
        self.window.blit(overlay_surface, (0, 0))
        self.window.blit(text_top_right, (10 + (self.pixel_per_mm * mm) / 2 - 10, 0))
        rect_bottom_right = text_bottom_right.get_rect()
        rect_bottom_right.right = self.environment_class.width * self.pixel_per_mm - 20
        rect_bottom_right.bottom = self.environment_class.height * self.pixel_per_mm - 20
        rect_top_left = text_top_left.get_rect()
        rect_top_left.right = self.environment_class.width * self.pixel_per_mm - 20
        rect_top_left.top = 20
        rect_bottom_left = text_bottom_left.get_rect()
        rect_bottom_left.left = 20
        rect_bottom_left.bottom = self.environment_class.height * self.pixel_per_mm - 20
        self.window.blit(text_bottom_right, rect_bottom_right)
        self.window.blit(text_top_left, rect_top_left)
        self.window.blit(text_bottom_left, rect_bottom_left)
        pygame.display.update()  # Update the screen
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """
        close pygame
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
