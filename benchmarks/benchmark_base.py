import tsplib95
import numpy as np
from joblib import Parallel, delayed
import time

def load_dataset(PROBLEM_SIZE):
    if PROBLEM_SIZE == "SMALL":
        return tsplib95.load('../datasets/ulysses22.tsp')
    elif PROBLEM_SIZE == "MEDIUM":
        return tsplib95.load('../datasets/berlin52.tsp')
    return tsplib95.load('../datasets/lin105.tsp')

def create_tsp_instance(problem):
    problem_name = problem.name
    num_nodes = problem.dimension
    # variables for clarity

    positions = np.zeros((num_nodes, 2))
    distance_matrix = np.zeros((num_nodes, num_nodes))
    # create numpy arrays

    for tsp_id, coord in problem.node_coords.items():
        positions[tsp_id - 1, 0] = coord[0]
        positions[tsp_id - 1, 1] = coord[1]
        # extract out positions to 2d numpy array

    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i, j] = problem.get_weight(i + 1, j + 1) # get_weight using library defined function. changes depending on geo coordinates or euclidean distance
            # add one since library uses one indexed

    return problem_name, num_nodes, positions, distance_matrix

def max_normalize(matrix, epsilon=1e-10):
    max_val = np.max(matrix)
    normalized_mat = matrix / (max_val + epsilon) # normalize the matrix by dividing by max value in the matrix
    return normalized_mat, max_val # also return max_val to unscale to real distances later

def get_edge_list(route):
    shifted_route = np.roll(route, -1) # shift the route to get the next node in the route
    route_edges = np.column_stack((route, shifted_route)) # stack the route and shifted route to get the edge list
    return route_edges

def get_route_distance(distance_matrix, route):
    edge_list = get_edge_list(route)
    edge_distances = distance_matrix[edge_list[:,0], edge_list[:,1]] # get the distances for each edge in the route
    total_distance = np.sum(edge_distances) # sum to get total distance
    return total_distance

def create_random_route(num_nodes):
    return np.random.permutation(num_nodes) # random permutation of the nodes

def order_by_pheromone_values(pheromone_up_list, edgelist):
    indices = np.argsort(pheromone_up_list) # get indices as if they were sorted

    pheromone_up_list = pheromone_up_list[indices] # sort the pheromone values
    edgelist = edgelist[indices] # sort the edges as well

    return edgelist, pheromone_up_list

def get_desirability_matrix(distance_matrix, epsilon=1e-10):
    desirability_matrix = 1.0 / (distance_matrix + epsilon) # invert distance. higher distance means less desirability and vice versa
    np.fill_diagonal(desirability_matrix, 0) # set diagonal to 0 since we can't go to the same node as we started from
    return desirability_matrix

def construct_single_ant_solution(total_influence_matrix, identity_route_permutation, start_node):
    num_nodes = total_influence_matrix.shape[0]
    
    unvisited_nodes = np.ones(num_nodes, dtype=bool)
    route = np.full(num_nodes, -1, dtype=int)
    route[0] = start_node
    unvisited_nodes[route[0]] = False # mark the start node as visited

    for i in range(num_nodes-1):
        current_node = route[i]
        
        unvisited_influences = total_influence_matrix[current_node] * unvisited_nodes # get influences for unvisited nodes
        unvisited_probabilities = unvisited_influences / np.sum(unvisited_influences) # normalize to get probabilities

        picked_node = np.random.choice(identity_route_permutation, p=unvisited_probabilities) # sample random node based on probability distribution
        
        route[i+1] = picked_node # add the node to the route
        unvisited_nodes[picked_node] = False # mark the node as picked

    return route

def update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances, PHEROMONE_DEPOSIT, EVAPORATION_RATE, epsilon=1e-10):
    pheromone_matrix *= (1 - EVAPORATION_RATE) # pheromone evaporation to soft forget previous routes

    for route, dist in zip(ant_routes, route_distances):
        route_edges = get_edge_list(route)

        pheromone_delta = PHEROMONE_DEPOSIT / (dist + epsilon) # how much to update pheromone

        r1_nodes = route_edges[:, 0]
        r2_nodes = route_edges[:, 1]
        # caching for vectorization

        pheromone_matrix[r1_nodes, r2_nodes] += pheromone_delta
        pheromone_matrix[r2_nodes, r1_nodes] += pheromone_delta
        # update all edges in pheromone matrix simultaneously

def get_best_route_and_distance(ant_routes, route_distances):
    idx = np.argmin(route_distances)
    return ant_routes[idx], route_distances[idx]

def construct_ant_system_solution(MAX_ITERATIONS, NUMBER_OF_ANTS, INITIAL_PHEROMONE_VALUE, PHEROMONE_INFLUENCE, DISTANCE_INFLUENCE, PHEROMONE_DEPOSIT, EVAPORATION_RATE, PARALLELIZE, distance_matrix, start_node):
    num_nodes = distance_matrix.shape[0]
    identity_route_permutation = np.arange(num_nodes) # used for caching purposes
    desirability_matrix = get_desirability_matrix(distance_matrix) # create desirability matrix based on distance matrix
    distance_influenced_matrix = desirability_matrix ** DISTANCE_INFLUENCE # influence for distance matrix

    pheromone_matrix = np.full(distance_matrix.shape, INITIAL_PHEROMONE_VALUE) # create pheromone matrix based on initial pheromone value
    ant_routes = np.zeros((NUMBER_OF_ANTS, num_nodes), dtype=int) # initialize ant routes
    route_distances = np.zeros(NUMBER_OF_ANTS) # route distances array

    best_route = create_random_route(num_nodes) # create a random route
    best_route_distance = get_route_distance(distance_matrix, best_route)

    for i in range(MAX_ITERATIONS):
        pheromone_influenced_matrix = pheromone_matrix ** PHEROMONE_INFLUENCE # influence for pheromone matrix
        total_influence_matrix = distance_influenced_matrix * pheromone_influenced_matrix # get total influence by multiplying

        if PARALLELIZE:
            ant_routes = Parallel(n_jobs=-1)(delayed(construct_single_ant_solution)(total_influence_matrix, identity_route_permutation, start_node) for _ in range(NUMBER_OF_ANTS)) # parallelize ant route construction with with all cpu cores
        else:
            for j in range(NUMBER_OF_ANTS):
                ant_routes[j] = construct_single_ant_solution(total_influence_matrix, identity_route_permutation, start_node) # manually construct ant routes without parallelization. better for smaller number of ants

        for j in range(NUMBER_OF_ANTS):
            route_distances[j] = get_route_distance(distance_matrix, ant_routes[j]) # don't parallelize this since it's fast enough

        update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances, PHEROMONE_DEPOSIT, EVAPORATION_RATE)
        current_iteration_best_route, current_iteration_best_route_distance = get_best_route_and_distance(ant_routes, route_distances) # get the best route and distance for the current iteration

        if current_iteration_best_route_distance < best_route_distance:
            best_route = current_iteration_best_route.copy() # copy it to avoid reference issues
            best_route_distance = current_iteration_best_route_distance
        # update best route if current best iteration route is better

    return best_route

def construct_nearest_neighbor_solution(distance_matrix, start_node):
    num_nodes = distance_matrix.shape[0]
    unvisited = set(range(num_nodes))
    route = [start_node]
    unvisited.remove(start_node) # remove starting node from unvisited set as it's now visited
    current_node = start_node

    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current_node, node]) # choose the nearest unvisited node
        route.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
        # update the route to include the next node

    return np.array(route)

def sample_main(PROBLEM_SIZE, MAX_ITERATIONS, NUMBER_OF_ANTS, INITIAL_PHEROMONE_VALUE, DISTANCE_INFLUENCE, PHEROMONE_INFLUENCE, PHEROMONE_DEPOSIT, EVAPORATION_RATE, PARALLELIZE=False):
    problem = load_dataset(PROBLEM_SIZE) # load dataset with specified dataset based on size
    problem_name, num_nodes, positions, distance_matrix_orig = create_tsp_instance(problem)
    distance_matrix_scaled, scaled_distance = max_normalize(distance_matrix_orig) # max normalize the distance matrix for stability
    start_node = np.random.randint(0, num_nodes) # choose random starting node for both algorithms

    # start timer for Ant System Algorithm
    start_time_as = time.perf_counter()
    best_route_as = construct_ant_system_solution(MAX_ITERATIONS, NUMBER_OF_ANTS, INITIAL_PHEROMONE_VALUE, PHEROMONE_INFLUENCE, DISTANCE_INFLUENCE, PHEROMONE_DEPOSIT, EVAPORATION_RATE, PARALLELIZE, distance_matrix_scaled, start_node)
    end_time_as = time.perf_counter() # end timer for Ant System Algorithm
    time_taken_as_ms = (end_time_as - start_time_as) * 1000 # calculate time taken in milliseconds

    as_distance_scaled = get_route_distance(distance_matrix_scaled, best_route_as) # get scaled distance for Ant System Algorithm
    as_distance = as_distance_scaled * scaled_distance # unscale to get real distance

    start_time_nn = time.perf_counter() # start timer for Nearest Neighbor Algorithm
    nearest_neighbor_route = construct_nearest_neighbor_solution(distance_matrix_scaled, start_node)
    end_time_nn = time.perf_counter() # end timer for Nearest Neighbor Algorithm

    nearest_neighbor_distance_scaled = get_route_distance(distance_matrix_scaled, nearest_neighbor_route) # get scaled distance for Nearest Neighbor Algorithm
    nn_distance = nearest_neighbor_distance_scaled * scaled_distance # unscale to get real distance
    time_taken_nn_ms = (end_time_nn - start_time_nn) * 1000 # calculate time taken in milliseconds

    dist_diff = nn_distance - as_distance # get distance difference between the two algorithms
    time_difference_s = (time_taken_as_ms - time_taken_nn_ms) / 1000.0 # get time difference in seconds

    improvement_percent = ((nn_distance - as_distance) / nn_distance) * 100 # percent increase formula between the two algorithms

    return as_distance, time_taken_as_ms, nn_distance, time_taken_nn_ms, dist_diff, time_difference_s, improvement_percent

PROBLEM_SIZE = "LARGE" # "SMALL", "MEDIUM", "LARGE"
MAX_ITERATIONS = 400 # number of iterations for the algorithm
NUMBER_OF_ANTS = 25 # number of ants in the system
INITIAL_PHEROMONE_VALUE = 1.0 # initial pheromone value for each edge
DISTANCE_INFLUENCE = 2.0 # influence of distance on route
PHEROMONE_INFLUENCE = 1.0 # influence of pheromone on route
PHEROMONE_DEPOSIT = 1.0 # pheromone deposit factor
EVAPORATION_RATE = 0.2 # pheromone evaporation rate
PARALLELIZE = False # use parallelization for ant route construction. beneficial if number of ants is large otherwise set to False