import math

import matplotlib.pyplot as plt

from worms.env import worm_world

width = 7
height = 5
pixel_per_mm = 10

centroids = [[pixel_per_mm * height * 0.5, pixel_per_mm * width * 0.5, 0.9, 0.9],
             [pixel_per_mm * height * 0.5 * 0.5, pixel_per_mm * width * 0.5 * 0.25, 0.9, 0.9]]
start_position = [25, 50]
start_angle = 3 * math.pi / 4

test1 = worm_world.WormWorldEnv(render_mode="human", sigma=1., height=height, width=width,
                                pixel_per_mm=pixel_per_mm, centroids=None,
                                min_number_food_objects=1, max_number_food_objects=1,
                                min_food_elements=40, max_food_elements=40,
                                food_element_range=.5,
                                min_eat_height=4, max_eat_height=4,
                                sampling_method='gaussian')
test2 = worm_world.WormWorldEnv(render_mode="human", sigma=2., height=height, width=width,
                                pixel_per_mm=pixel_per_mm, centroids=None,
                                min_number_food_objects=1, max_number_food_objects=1,
                                min_food_elements=40, max_food_elements=40,
                                food_element_range=.5,
                                min_eat_height=4, max_eat_height=4,
                                sampling_method='rejection_sampling')
test1.reset()
print("done")
test2.reset()

# Apply the transformation to each point in the grid

# Plot the original square action space
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Gaussian Sampling')
# plt.plot(test.environment_class.centroid_grid)

plt.imshow(test1.environment_class.centroid_grid, cmap='YlOrBr', origin='lower', aspect='auto')
for c in test1.environment_class.centroids:
    x = (c[1] * pixel_per_mm) % (width * pixel_per_mm)
    y = (c[0] * pixel_per_mm) % (height * pixel_per_mm)

    plt.scatter([x], [y], marker='x', color='black', linewidths=1)
    # print(f'test1: {c[1]} - {c[0]}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# Plot the transformed circle action space
plt.subplot(1, 2, 2)
plt.title('Uniform Sampling')
plt.imshow(test2.environment_class.centroid_grid, cmap='YlOrBr', origin='lower')
for c in test2.environment_class.centroids:
    x = (c[1] * pixel_per_mm) % (width * pixel_per_mm)
    y = (c[0] * pixel_per_mm) % (height * pixel_per_mm)

    plt.scatter([x], [y], marker='x', color='black', linewidths=1)
    # print(f'test2: {c[1]} - {c[0]}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.tight_layout()
plt.show()
