class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        self.best_cost = float('inf')

def smoothness(position):
    smooth = 0
    for i in range(1,len(position)-2):

        alpha = math.atan2(position[i+1][0]-position[i][0], position[i+1][1]-position[i][0])
        beta = math.atan2(position[i][0]-position[i-1][0], position[i][1]-position[i-1][1])

        if abs(alpha-beta) > math.pi/8: smooth += 100

        smooth += abs(alpha-beta)
    return smooth

def evaluate_cost(position, end, barriers):
    total_cost = 0
    for i in range(len(position) - 1):
        total_cost += heuristic(position[i], position[i+1])
    
    total_cost += heuristic(position[-1], end)
    total_cost += calculate_penalty(position[-1], barriers)
    total_cost += smoothness(position)

    return total_cost

def pso_optimization(start, end, barriers, best_path_astar):
    num_particles = 100
    max_iterations = 2000
    inertia_weight = 0.7
    cognitive_param = 1.5
    social_param = 1.5

    particles = []
    for _ in range(num_particles):
        # Initialize particles around the best path found by A*
        initial_position = [best_path_astar[0]] + \
                   [(point[0] + random.uniform(-0.1, +0.1), point[1] + random.uniform(-0.1 , +0.1)) 
                    for point in best_path_astar[1:-2]] + \
                   [best_path_astar[-1]]
        initial_velocity = [(0, 0) for _ in initial_position]
        particles.append(Particle(initial_position, initial_velocity))

    global_best_position = best_path_astar

    global_best_cost = evaluate_cost(global_best_position, end, barriers)

    for _ in range(max_iterations):
        for particle in particles:
            # Update particle velocity
            for i in range(len(particle.position)):
                r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
                cognitive_velocity = (cognitive_param * r1 * 
                                      (particle.best_position[i][0] - particle.position[i][0]), 
                                      cognitive_param * r1 * 
                                      (particle.best_position[i][1] - particle.position[i][1]))
                social_velocity = (social_param * r2 * 
                                   (global_best_position[i][0] - particle.position[i][0]), 
                                   social_param * r2 * 
                                   (global_best_position[i][1] - particle.position[i][1]))
                particle.velocity[i] = (inertia_weight * particle.velocity[i][0] + cognitive_velocity[0] + social_velocity[0],
                                        inertia_weight * particle.velocity[i][1] + cognitive_velocity[1] + social_velocity[1])
            
            # Update particle position
            new_position = [(particle.position[i][0] + particle.velocity[i][0], 
                             particle.position[i][1] + particle.velocity[i][1]) 
                            for i in range(len(particle.position))]
            
            # Evaluate new position
            new_cost = evaluate_cost(new_position, end, barriers)

            # Update particle's best position if improved
            if new_cost < particle.best_cost:
                particle.best_position = new_position
                particle.best_cost = new_cost
            
            # Update global best if improved
            if new_cost < global_best_cost:
                global_best_position = new_position
                global_best_cost = new_cost

    return global_best_position