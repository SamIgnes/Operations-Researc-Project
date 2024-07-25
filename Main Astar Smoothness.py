import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import math
from RandomPolygon import create_non_intersecting_polygons


class Node:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent
        self.g = 0  # Distance from start node
        self.h = 0  # Heuristic distance to end node
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def astar(start, end, barriers):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    end_node = Node(end)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.point)

        if current_node.point == end_node.point:
            path = []
            while current_node:
                path.append(current_node.point)
                current_node = current_node.parent
            return path[::-1]

        neighbors = get_neighbors(current_node.point, barriers)
        for next_point in neighbors:
            if next_point in closed_list:
                continue

            neighbor_node = Node(next_point, current_node)
            neighbor_node.g = current_node.g + heuristic(current_node.point, neighbor_node.point) + smoothness_penalty(current_node, neighbor_node) + calculate_penalty(next_point, barriers)
            neighbor_node.h = heuristic(neighbor_node.point, end_node.point)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if add_to_open(open_list, neighbor_node):
                heapq.heappush(open_list, neighbor_node)

    return None

def calculate_penalty(point, barriers, safe_distance=2):
    min_distance = float('inf')
    for barrier in barriers:
        for vertex in barrier:
            distance = heuristic(point, vertex)
            if distance < min_distance:
                min_distance = distance
    if min_distance < safe_distance:
        penalty = 1 / (min_distance + 0.01)  # Inverse penalty within safe distance
    else:
        penalty = 0  # No penalty beyond safe distance
    return penalty

def smoothness_penalty(current_node, neighbor_node):
    if not current_node.parent:
        return 0

    prev_point = current_node.parent.point
    curr_point = current_node.point
    next_point = neighbor_node.point

    prev_direction = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
    next_direction = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

    angle1 = math.atan2(prev_direction[1], prev_direction[0])
    angle2 = math.atan2(next_direction[1], next_direction[0])

    angle_difference = abs(angle2 - angle1)
    if angle_difference > math.pi:
        angle_difference = 2 * math.pi - angle_difference

    return angle_difference  # Penalize the change in direction

def get_neighbors(point, barriers):
    neighbors = []
    step_size = 1  # Step size for moving to neighbors

    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (point[0] + dx, point[1] + dy)
            if not is_collision(neighbor, barriers):
                neighbors.append(neighbor)

    return neighbors

def is_collision(point, barriers):
    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True)
        if polygon.contains_point(point):
            return True
    return False

def add_to_open(open_list, neighbor_node):
    for node in open_list:
        if neighbor_node.point == node.point and neighbor_node.g >= node.g:
            return False
    return True

def plot_environment_and_path(barriers, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    if path:
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        ax.plot(x_coords, y_coords, 'b-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding with A* Algorithm')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Define the barriers as lists of (x, y) coordinates (irregular polygons)
barriers = create_non_intersecting_polygons(5, seed=2)

# Define start and end points
start = (0, 0)
end = (90, 90)

# Find the path using A* algorithm
path = astar(start, end, barriers)

def calculate_path_metrics(path, barriers):
    total_distance = 0
    total_penalty = 0
    smooth = 0

    for i in range(len(path) - 1):
        total_distance += heuristic(path[i + 1], path[i])
        total_penalty += calculate_penalty(path[i], barriers)
        if i > 0:
            # Create nodes to calculate smoothness penalty
            current_node = Node(path[i], Node(path[i - 1]))
            neighbor_node = Node(path[i + 1], current_node)
            smooth += smoothness_penalty(current_node, neighbor_node)
    return total_distance, smooth, total_penalty

# Calculate metrics for the optimal path
total_distance, smooth, total_penalty = calculate_path_metrics(path, barriers)

# Print the metrics
print("Optimal Path Metrics:")
print(f"Total Distance: {total_distance}")
print(f"Smoothness: {smooth}")
print(f"Penalty: {total_penalty}")

# Plot the environment and the path found
plot_environment_and_path(barriers, path)
