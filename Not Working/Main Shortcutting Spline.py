import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def calculate_penalty(point, barriers):
    min_distance = float('inf')
    for barrier in barriers:
        for vertex in barrier:
            distance = heuristic(point, vertex)
            if distance < min_distance:
                min_distance = distance
    return 10 / (min_distance + 1)  # Penalizzazione inversa

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

def is_line_intersecting_polygon(p1, p2, polygon):
    for i in range(len(polygon)):
        p3 = polygon[i]
        p4 = polygon[(i + 1) % len(polygon)]
        if do_lines_intersect(p1, p2, p3, p4):
            return True
    return False

def do_lines_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def is_collision_with_line(p1, p2, barriers):
    for barrier in barriers:
        if is_line_intersecting_polygon(p1, p2, barrier):
            return True
    return False

def shortcut_path(path, barriers):
    if not path:
        return path

    new_path = [path[0]]
    i = 0

    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if not is_collision_with_line(path[i], path[j], barriers):
                new_path.append(path[j])
                i = j
                break
        else:
            i += 1
            new_path.append(path[i])

    return new_path

def smooth_path(path):
    if not path or len(path) < 3:
        return path

    x = [point[0] for point in path]
    y = [point[1] for point in path]

    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, len(path) * 10)
    x_new, y_new = splev(unew, tck)

    smooth_path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return smooth_path

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

def generate_random_polygon(max_sides, min_radius, max_radius):
    num_sides = random.randint(3, max_sides)
    angle_step = 360 / num_sides
    points = []
    for i in range(num_sides):
        angle = math.radians(i * angle_step)
        radius = random.uniform(min_radius, max_radius)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points

def is_intersecting(polygon, barriers):
    poly_patch = patches.Polygon(polygon, closed=True)
    for barrier in barriers:
        barrier_patch = patches.Polygon(barrier, closed=True)
        if poly_patch.get_path().intersects_path(barrier_patch.get_path()):
            return True
    return False

def create_non_intersecting_polygons(n, max_sides=6, min_radius=10, max_radius=20, width=100, height=100, start=(0, 0), end=(100, 100)):
    barriers = []
    attempts = 0
    while len(barriers) < n and attempts < 1000:
        attempts += 1
        polygon = generate_random_polygon(max_sides, min_radius, max_radius)
        centroid_x = random.uniform(0, width)
        centroid_y = random.uniform(0, height)
        translated_polygon = [(x + centroid_x, y + centroid_y) for x, y in polygon]
        
        # Check if start and end points are inside the translated polygon
        if not point_in_polygon(start, translated_polygon) and not point_in_polygon(end, translated_polygon):
            if not is_intersecting(translated_polygon, barriers):
                barriers.append(translated_polygon)
    
    return barriers

def plot_polygons(barriers):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Generated Non-Intersecting Polygons')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

width = 100
height = 100
# Generate 10 non-intersecting polygons excluding start and end points
start = (random.randint(0, width), random.randint(0, height))
end = (random.randint(0, width), random.randint(0, height))
barriers = create_non_intersecting_polygons(10, start=start, end=end)

# Create graph
graph = create_graph(barriers, width, height)

# Find the path using A* algorithm
best_path_astar = nx.astar_path(graph, start, end, heuristic=heuristic)

# Shortcut the path to make it smoother
shortcutted_path = shortcut_path(best_path_astar, barriers)

# Further smooth the path using splines
smooth_best_path = smooth_path(shortcutted_path)

plot_environment_and_path(barriers, [best_path_astar, shortcutted_path, smooth_best_path])

plt.show()
