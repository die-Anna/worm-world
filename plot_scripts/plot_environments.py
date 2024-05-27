import matplotlib.pyplot as plt
import math

from worms.env import worm_world

width = 55
height = 55
pixel_per_mm = 15
sigma = 1

centroids = [[height * 0.5, width * 0.5, 1, 1]]
start_position = [width * 0.75, height * 0.75]
start_angle = 3 * math.pi / 4

test1 = worm_world.WormWorldEnv(render_mode="human", sigma=sigma, height=height, width=width,
                                                 pixel_per_mm=pixel_per_mm,
                                                 # centroids=centroids, start_angle=start_angle, start_position=start_position,
                                                 min_number_food_objects=15, max_number_food_objects=15,
                                                 min_food_elements=25, max_food_elements=25,
                                                 # food_element_range=.1,
                                                 )


def main():
    test1.environment_class.create_centroids()
    # Plot the original square action space
    plt.figure()

    test1.environment_class.fill_periodic_grid_with_bumps()
    plt.title(f'Environment ({width}mm x {height}mm), sigma={sigma}')
    # print(f'grid[3][2]: {test1.environment_class.grids[3][2]}')
    plt.imshow(test1.environment_class.centroid_grid, cmap='YlOrBr', origin='lower')
    for c in test1.environment_class.centroids:
        x = (c[1] * pixel_per_mm) % (width * pixel_per_mm)
        y = (c[0] * pixel_per_mm) % (height * pixel_per_mm)

        plt.scatter([x], [y], marker='.', color='black', linewidths=0.1)
        # print(f'test2: {c[1]} - {c[0]}')
    plt.xlabel('x')
    plt.ylabel('y')
    ticks_x = range(0, width * pixel_per_mm + 1, 5 * pixel_per_mm)
    ticks_y = range(0, height * pixel_per_mm + 1, 5 * pixel_per_mm)
    labels_x = [f'{tick / 15:.0f}' for tick in ticks_x]
    labels_y = [f'{tick / 15:.0f}' for tick in ticks_y]
    plt.xticks(ticks=ticks_x, labels=labels_x)
    plt.yticks(ticks=ticks_y, labels=labels_y)
    plt.axis('equal')

    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

