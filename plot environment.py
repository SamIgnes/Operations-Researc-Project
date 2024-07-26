import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import math
from RandomPolygon import create_non_intersecting_polygons

path =[]

def plot_environment_and_path(barriers, path):    
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    ax.plot(0,0,'bo')
    ax.plot(90,90,'bo')    

    if path:
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        ax.plot(x_coords, y_coords, 'b-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Environment representation')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


barriers = create_non_intersecting_polygons(10, seed=4)
plot_environment_and_path(barriers, path)