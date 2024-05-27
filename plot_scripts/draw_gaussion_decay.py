import math

import matplotlib.pyplot as plt

from worm_world.worms.env.worm_world import WormWorldEnv

width = 5
height = 5
pixel_per_mm = 10

centroids = [[pixel_per_mm * height * 0.5, pixel_per_mm * width * 0.5, 0.9, 0.9],
             [pixel_per_mm * height * 0.5 * 0.5, pixel_per_mm * width * 0.5 * 0.25, 0.9, 0.9]]
start_position = [25, 50]
start_angle = 3 * math.pi / 4

test = WormWorldEnv(render_mode="human", sigma=1., height=height, width=width,
                    pixel_per_mm=pixel_per_mm, centroids=None,
                    min_number_food_objects=1, max_number_food_objects=1,
                    min_food_elements=1, max_food_elements=1,
                    food_element_range=.5,
                    min_eat_height=4, max_eat_height=4,
                    sampling_method='gaussian')

test.reset()

plt.figure(figsize=(6, 6))

# plt.subplot(1, 2, 1)
plt.title('Gaussian Decay Food Odor')
# plt.plot(test.environment_class.centroid_grid)

plt.imshow(test.environment_class.centroid_grid, origin='lower', aspect='auto')
for c in test.environment_class.centroids:
    x = (c[1] * pixel_per_mm) % (width * pixel_per_mm)
    y = (c[0] * pixel_per_mm) % (height * pixel_per_mm)

    plt.scatter([x], [y], marker='x', color='black', linewidths=1)
    # print(f'test1: {c[1]} - {c[0]}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.tight_layout()
plt.show()
