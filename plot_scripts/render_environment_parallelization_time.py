import math
import time

import matplotlib.pyplot as plt

from worms.env import worm_world

width = 10
height = 7
pixel_per_mm = 50

centroids = [[pixel_per_mm * height * 0.5, pixel_per_mm * width * 0.5, 0.9, 0.9],
             [pixel_per_mm * height * 0.5 * 0.5, pixel_per_mm * width * 0.5 * 0.25, 0.9, 0.9]]
start_position = [25, 50]
start_angle = 3 * math.pi / 4

test1 = worm_world.WormWorldEnv(render_mode="human", sigma=1., height=height, width=width,
                                pixel_per_mm=pixel_per_mm, centroids=None,
                                min_number_food_objects=10, max_number_food_objects=13,
                                min_food_elements=20, max_food_elements=30,
                                food_element_range=.1,
                                min_eat_height=3, max_eat_height=5,
                                sampling_method='gaussian')


def main():
    test1.environment_class.create_centroids()
    # Plot the original square action space
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    # plt.plot(test.environment_class.centroid_grid)
    start_time = time.time()
    test1.environment_class.fill_periodic_grid_with_bumps_old()
    elapsed_time = time.time() - start_time
    # print(f'grid1: {test1.environment_class.grids}')
    plt.title(f'Elapsed time: {elapsed_time:.2f}')
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
    start_time = time.time()
    test1.environment_class.fill_periodic_grid_with_bumps()
    # print('testing first...')
    elapsed_time = time.time() - start_time
    plt.title(f'Elapsed time parallel: {elapsed_time:.2f}')
    # print(f'grid[3][2]: {test1.environment_class.grids[3][2]}')
    plt.imshow(test1.environment_class.centroid_grid, cmap='YlOrBr', origin='lower')
    for c in test1.environment_class.centroids:
        x = (c[1] * pixel_per_mm) % (width * pixel_per_mm)
        y = (c[0] * pixel_per_mm) % (height * pixel_per_mm)

        plt.scatter([x], [y], marker='x', color='black', linewidths=1)
        # print(f'test2: {c[1]} - {c[0]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
