import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

class PathFindingProblem(Problem):

    def __init__(self, start, end, barriers, n_points=10):
        self.start = start
        self.end = end
        self.barriers = barriers
        self.n_points = n_points  # Number of points in the path excluding start and end
        super().__init__(n_var=2 * n_points,
                         n_obj=2,
                         xl=0.0,
                         xu=100.0)
    
    def calculate_penalty(self, point):
        min_distance = float('inf')
        for barrier in self.barriers:
            for vertex in barrier:
                distance = np.linalg.norm(np.array(point) - np.array(vertex))
                if distance < min_distance:
                    min_distance = distance
        return 1 / (min_distance + 1)  # Inverse penalty

    def _evaluate(self, X, out, *args, **kwargs):
        paths = X.reshape(-1, self.n_points, 2)
        total_distances = []
        penalties = []
        
        for path in paths:
            path = np.vstack([self.start, path, self.end])
            total_distance = 0
            total_penalty = 0
            
            for i in range(len(path) - 1):
                total_distance += np.linalg.norm(path[i+1] - path[i])
                total_penalty += self.calculate_penalty(path[i])
            
            total_distances.append(total_distance)
            penalties.append(total_penalty)
        
        out["F"] = np.column_stack([total_distances, penalties])

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
        ax.plot(x_coords, y_coords, 'bo-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Path Finding with Genetic Algorithm')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Define the barriers as lists of (x, y) coordinates (irregular polygons)
barriers = [
    [(10, 10), (20, 10), (15, 20)],  # Triangle barrier
    [(30, 30), (50, 30), (50, 50), (30, 50)],  # Rectangle barrier
    [(70, 70), (80, 65), (85, 75), (75, 80)],  # Irregular quadrilateral barrier
    [(10,10), (80,80),(10, 85), (10,85)] #Irregular triangle
]

# Define start and end points
start = np.array([0, 0])
end = np.array([90, 90])

# Define the problem
problem = PathFindingProblem(start, end, barriers)

# Define the algorithm
algorithm = GA(pop_size=100, eliminate_duplicates=True)

# Perform the optimization
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

# Get the optimal path
optimal_path = res.X.reshape(-1, problem.n_points, 2)[0]
optimal_path = np.vstack([start, optimal_path, end])

# Plot the environment and the path found
plot_path(barriers, optimal_path)
