import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from RandomPolygon import create_non_intersecting_polygons

class PathFindingProblem(Problem):

    def __init__(self, start, end, barriers, n_points = 10): 
        self.start = start
        self.end = end
        self.barriers = barriers
        self.n_points = n_points
        self.safe_distance = 2.0  # Safe distance threshold
        super().__init__(n_var=2 * n_points,
                         n_obj=3,  # Change to 4 objectives
                         n_constr=1,
                         xl=0.0,
                         xu=100.0)
    
    def calculate_penalty(self, point):
        min_distance = float('inf')
        for barrier in self.barriers:
            for vertex in barrier:
                distance = np.linalg.norm(np.array(point) - np.array(vertex))
                if distance < min_distance:
                    min_distance = distance
        if min_distance < self.safe_distance:
            penalty = 1 / (min_distance + 0.01)  # Inverse penalty within safe distance
        else:
            penalty = 0  # No penalty beyond safe distance
        return penalty
    
    def point_in_polygon(self, point, polygon):
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
    
    def do_lines_intersect(self, A, B, C, D):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def is_collision(self, point1, point2=None):
        for barrier in self.barriers:
            if self.point_in_polygon(point1, barrier):
                return True
            if point2 is not None:
                for i in range(len(barrier)):
                    if self.do_lines_intersect(point1, point2, barrier[i], barrier[(i + 1) % len(barrier)]):
                        return True
        return False   
    
    def smoothness(self, position):
        smooth = 0
        for i in range(1, len(position) - 1):
            alpha = math.atan2(position[i + 1][1] - position[i][1], position[i + 1][0] - position[i][0])
            beta = math.atan2(position[i][1] - position[i - 1][1], position[i][0] - position[i - 1][0])
            angle_diff = abs(alpha - beta)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            smooth += angle_diff
        return smooth / len(position)  # Normalize smoothness score

    def uniformity(self, path):
        distances = [np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)]
        variance = np.var(distances)
        return variance

    def _evaluate(self, X, out, *args, **kwargs):
        paths = X.reshape(-1, self.n_points, 2)
        total_distances = []
        penalties = []
        smoothness_scores = []
        uniformity_scores = []
        collision_constraints = []

        for path in paths:
            path = np.vstack([self.start, path, self.end])
            total_distance = 0
            total_penalty = 0
            collision = 0

            for i in range(len(path) - 1):
                total_distance += np.linalg.norm(path[i + 1] - path[i])
                total_penalty += self.calculate_penalty(path[i])
                if self.is_collision(path[i], path[i + 1]):
                    collision += 1  # Increment collision count

            smooth = self.smoothness(path)
            uniform = self.uniformity(path)
            total_distances.append(total_distance)
            penalties.append(total_penalty)
            smoothness_scores.append(smooth)
            #uniformity_scores.append(uniform)
            collision_constraints.append(collision)  # Add collision count to constraints

        out["F"] = np.column_stack([total_distances, penalties, smoothness_scores])
        out["G"] = np.column_stack([collision_constraints])  # Set constraints

def plot_path(barriers, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    for barrier in barriers:
        polygon = patches.Polygon(barrier, closed=True, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)

    if path is not None:
        x_coords = path[:, 0]
        y_coords = path[:, 1]
        ax.plot(x_coords, y_coords, 'b-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding with Genetic Algorithm')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Define the barriers as lists of (x, y) coordinates (irregular polygons)

# Define start and end points
start = np.array([0, 0])
end = np.array([90, 90])

barriers = create_non_intersecting_polygons(5,seed = 13)

# Define the problem
problem = PathFindingProblem(start, end, barriers)

# Define the algorithm
algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)

# Perform the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

# Get the optimal path
optimal_path = res.X.reshape(-1, problem.n_points, 2)[0]
optimal_path = np.vstack([start, optimal_path, end])

# compute smoothness, penalty, tot distance
def calculate_path_metrics(path, problem):
    total_distance = 0
    total_penalty = 0
    collision = 0

    for i in range(len(path) - 1):
        total_distance += np.linalg.norm(path[i + 1] - path[i])
        total_penalty += problem.calculate_penalty(path[i])


    smooth = problem.smoothness(path)
    return total_distance, smooth, total_penalty

# Calculate metrics for the optimal path
total_distance, smooth, total_penalty = calculate_path_metrics(optimal_path, problem)

# Print the metrics
print("Optimal Path Metrics:")
print(f"Total Distance: {total_distance}")
print(f"Smoothness: {smooth}")
print(f"Penalty: {total_penalty}")



# Plot the environment and the path found
plot_path(barriers, optimal_path)
