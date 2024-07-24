import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from RandomPolygon import create_non_intersecting_polygons



def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def calculate_penalty(point, barriers, safe_distance = 2):
    min_distance = float('inf')
    for barrier in barriers:
        for vertex in barrier:
            distance = np.linalg.norm(np.array(point) - np.array(vertex))
            if distance < min_distance:
                min_distance = distance
    if min_distance < safe_distance:
        penalty = 1 / (min_distance + 0.01)  # Inverse penalty within safe distance
    else:
        penalty = 0  # No penalty beyond safe distance
    return penalty




def EdgeCost(node, neighbor, barriers):
    edgecost = heuristic(node, neighbor) + calculate_penalty(neighbor, barriers) 
    return edgecost

def create_graph(barriers, width, height):
    G = nx.Graph()
    step_size = 1  # Step size for moving to neighbors

    # Add nodes
    for x in range(0, width, step_size):
        for y in range(0, height, step_size):
            if not is_collision((x, y), barriers):
                G.add_node((x, y))

    # Add edges
    for node in G.nodes():
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor in G.nodes():
                    G.add_edge(node, neighbor, weight= EdgeCost(node, neighbor, barriers))

    return G

def is_collision(point, barriers):
    for barrier in barriers:
        if point_in_polygon(point, barrier):
            return True
    return False

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def plot_environment_and_path(barriers, paths):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    colors = ['b', 'g', 'm', 'c', 'y']  # Esempio di colori diversi per i percorsi

    for idx, path in enumerate(paths):
        if path:
            x_coords = [point[0] for point in path]
            y_coords = [point[1] for point in path]
            ax.plot(x_coords, y_coords, linestyle='-', color=colors[idx % len(colors)])

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')


width = 100
height = 100

start = (0,0)
end = (90,90)

barriers = create_non_intersecting_polygons(5, seed = 1)

# Create graph
graph = create_graph(barriers, width, height)

# Find the path using A* algorithm
best_path_astar = nx.astar_path(graph, start, end, heuristic=heuristic)

plot_environment_and_path(barriers, [best_path_astar])

plt.show()