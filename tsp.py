import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import time

PROBLEM_SIZE = "LARGE" # "SMALL", "MEDIUM", "LARGE"
MAX_ITERATIONS = 400 # number of iterations for the algorithm
NUMBER_OF_ANTS = 25 # number of ants in the System
INITIAL_PHEROMONE_VALUE = 1.0 # initial pheromone value for each edge
DISTANCE_INFLUENCE = 2.0 # influence of distance on route
PHEROMONE_INFLUENCE = 1.0 # influence of pheromone on route
PHEROMONE_DEPOSIT = 1.0 # pheromone deposit factor
EVAPORATION_RATE = 0.2 # pheromone evaporation rate
# ALGORITHM PARAMETERS

PARALLELIZE = False # use parallelization for ant route construction. beneficial if number of ants is large otherwise set to False
# OPTIONAL PARAMETERS

DISABLE_PLOTS = True # disable all plots for benchmarking
ANIMATE_ROUTE = False
PLOT_EVERY_K_ITERATIONS = 10
ENABLE_NODE_LABELS = True
NODE_LABELS_THRESHOLD = 100
ALL_EDGES_THRESHOLD = 100
# DRAWING PARAMETERS

NODE_LABELS_NODE_SIZE = 200 # size of node labels on plot when labels are enabled
NO_NODE_LABELS_NODE_SIZE = 100 # size of node labels on plot when labels are disabled
# PLOT PARAMETERS

EPSILON = 1e-10 # small value to avoid division by zero. useful for many operations
# STABILITY PARAMETERS

def load_dataset():
    if PROBLEM_SIZE == "SMALL":
        return tsplib95.load('datasets/ulysses22.tsp')
    elif PROBLEM_SIZE == "MEDIUM":
        return tsplib95.load('datasets/berlin52.tsp')
    return tsplib95.load('datasets/lin105.tsp')

def print_problem(problem):
    print("\nTSP Problem Details")
    print(f"NAME: {problem.name}")
    print(f"COMMENT: {problem.comment}")
    print(f"NODES: {problem.dimension}")
    print(f"EDGE_WEIGHT_TYPE: {problem.edge_weight_type}")
    print("")

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

def max_normalize(matrix):
    max_val = np.max(matrix)
    normalized_mat = matrix / (max_val + EPSILON) # normalize the matrix by dividing by max value in the matrix
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

def get_upper_triangular_list(mat):
    n = mat.shape[0]
    up_indices = np.triu_indices(n) # get the upper triangular indices of the matrix
    return mat[up_indices] # return the values at those indices as a numpy array

def create_networkX_graph(num_nodes, positions, distance_matrix):
    G = nx.Graph() # create the graph object

    for i in range(num_nodes):
        G.add_node(i, pos=tuple(positions[i])) # add each position of the node to the graph

    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if i == j:
                continue # skip self loops
            G.add_edge(i, j, weight=distance_matrix[i, j]) # add each edge between all pairs of nodes

    return G

def order_by_pheromone_values(pheromone_up_list, edgelist):
    indices = np.argsort(pheromone_up_list) # get indices as if they were sorted

    pheromone_up_list = pheromone_up_list[indices] # sort the pheromone values
    edgelist = edgelist[indices] # sort the edges as well

    return edgelist, pheromone_up_list

def plot_ant_tsp_route(ax, G, pheromone_matrix, route, total_distance, problem_name, num_nodes, current_iteration, best_found=False, force_draw_edges=False):
    graph_edge_list = np.array(G.edges())
    route_edges = get_edge_list(route)
    pos = nx.get_node_attributes(G, 'pos') # position dictionary for networkX

    is_node_labels_enabled = ENABLE_NODE_LABELS and num_nodes <= NODE_LABELS_THRESHOLD
    is_all_edges_enabled = num_nodes <= ALL_EDGES_THRESHOLD or force_draw_edges
    node_size = NODE_LABELS_NODE_SIZE if is_node_labels_enabled else NO_NODE_LABELS_NODE_SIZE

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightcoral') # nodes

    if is_node_labels_enabled:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold') # node labels

    pheromone_up_list = pheromone_matrix[graph_edge_list[:,0], graph_edge_list[:,1]] # upper triangular pheromone values without diagonal
    sorted_edgelist, sorted_pheromone_up_list = order_by_pheromone_values(pheromone_up_list, graph_edge_list) # sort the pheromone values and edges based on increasing pheromone values for drawing
    # ordering is crucial for not drawing lower pheromone values on top of higher pheromone values!

    if is_all_edges_enabled:
        nx.draw_networkx_edges(G, pos, edgelist=sorted_edgelist, width=1.0, edge_color=sorted_pheromone_up_list, edge_cmap=plt.cm.Greens, edge_vmin=sorted_pheromone_up_list.min(), edge_vmax=sorted_pheromone_up_list.max()) # draw all edges in order of pheromone values

    pheromone_up_list = pheromone_matrix[route_edges[:,0], route_edges[:,1]]
    sorted_edgelist, sorted_pheromone_up_list = order_by_pheromone_values(pheromone_up_list, route_edges)
    # do same pheromone sorting for the route edges

    nx.draw_networkx_edges(G, pos, edgelist=sorted_edgelist, width=2.0, edge_color=sorted_pheromone_up_list, edge_cmap=plt.cm.copper, edge_vmin=sorted_pheromone_up_list.min(), edge_vmax=sorted_pheromone_up_list.max(), arrows=True, arrowstyle='-|>') # draw route edges in order of pheromone values
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=400, node_color='limegreen') # highlight starting node in the route

    ax.tick_params(axis='both', which='major', left=True, bottom=True, labelleft=True, labelbottom=True) # enable tick marks for both axes

    plt.title(f"{best_found*"Best"} Ant System TSP Route - {problem_name} ({num_nodes} nodes) \n{NUMBER_OF_ANTS} ants, {MAX_ITERATIONS} iterations, Current Iteration - {current_iteration}\nDistance Power - {DISTANCE_INFLUENCE}, Pheromone Power - {PHEROMONE_INFLUENCE}, Evaporation Rate - {EVAPORATION_RATE}\nDistance - {total_distance:.2f}", fontsize=14)

    plt.xlabel("Relative X Coord", fontsize=12)
    plt.ylabel("Relative Y Coord", fontsize=12) # plot labels
    plt.grid(True, linestyle='-', alpha=0.8) # add grid

def plot_nearest_neighbour_route(ax, G, route, total_distance, problem_name, num_nodes, force_draw_edges=False):
    route_edges = get_edge_list(route)
    pos = nx.get_node_attributes(G, 'pos') # position dictionary for networkX

    is_node_labels_enabled = ENABLE_NODE_LABELS and num_nodes <= NODE_LABELS_THRESHOLD
    is_all_edges_enabled = num_nodes <= ALL_EDGES_THRESHOLD or force_draw_edges
    node_size = NODE_LABELS_NODE_SIZE if is_node_labels_enabled else NO_NODE_LABELS_NODE_SIZE

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightcoral') # nodes

    if is_node_labels_enabled:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold') # node labels

    if is_all_edges_enabled:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.1) # all edges

    nx.draw_networkx_edges(G,pos,edgelist=route_edges,edge_color='black',width=2.0,arrows=True, arrowstyle='-|>') # route edges
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=400, node_color='limegreen') # highlight starting node in the route

    ax.tick_params(axis='both', which='major', left=True, bottom=True, labelleft=True, labelbottom=True) # enable tick marks for both axes

    plt.title(f"Nearest Neighbour TSP Route - {problem_name} ({num_nodes} nodes) \nDistance - {total_distance:.2f}", fontsize=14)

    plt.xlabel("Relative X Coord", fontsize=12)
    plt.ylabel("Relative Y Coord", fontsize=12) # plot labels
    plt.grid(True, linestyle='-', alpha=0.8) # add grid

def get_desirability_matrix(distance_matrix):
    desirability_matrix = 1.0 / (distance_matrix + EPSILON) # invert distance. higher distance means less desirability and vice versa
    np.fill_diagonal(desirability_matrix, 0) # set diagonal to 0 since we can't go to the same node as we started from
    return desirability_matrix

def construct_ant_solution(total_influence_matrix, identity_route_permutation, start_node):
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

def update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances):
    pheromone_matrix *= (1 - EVAPORATION_RATE) # pheromone evaporation to soft forget previous routes

    for route, dist in zip(ant_routes, route_distances):
        route_edges = get_edge_list(route)

        pheromone_delta = PHEROMONE_DEPOSIT / (dist + EPSILON) # how much to update pheromone

        r1_nodes = route_edges[:, 0]
        r2_nodes = route_edges[:, 1]
        # caching for vectorization

        pheromone_matrix[r1_nodes, r2_nodes] += pheromone_delta
        pheromone_matrix[r2_nodes, r1_nodes] += pheromone_delta
        # update all edges in pheromone matrix simultaneously

def get_best_route_and_distance(ant_routes, route_distances):
    idx = np.argmin(route_distances)
    return ant_routes[idx], route_distances[idx]

def construct_nearest_neighbor_solution(distance_matrix, start_node):
    num_nodes = distance_matrix.shape[0]
    unvisited = set(range(num_nodes))
    route = [start_node]
    unvisited.remove(start_node) # remove starting node from unvisited set as it's now visited
    current_node = start_node

    with tqdm(total=num_nodes, desc="Running Nearest Neighbour") as pbar:
        pbar.update(1)
        while unvisited:
            next_node = min(unvisited, key=lambda node: distance_matrix[current_node, node]) # choose the nearest unvisited node
            route.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node 
            # update the route to include the next node
            pbar.update(1)

    return np.array(route)

def print_ant_system_details():
    print("Ant System Parameters")
    print(f"Number of Ants: {NUMBER_OF_ANTS}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Initial Pheromone Value: {INITIAL_PHEROMONE_VALUE}")
    print(f"Distance Influence: {DISTANCE_INFLUENCE}")
    print(f"Pheromone Influence: {PHEROMONE_INFLUENCE}")
    print(f"Pheromone Deposit: {PHEROMONE_DEPOSIT}")
    print(f"Evaporation Rate: {EVAPORATION_RATE}")
    print("")

def main():
    np.set_printoptions(threshold=20) # shorten size since positions array is big

    problem = load_dataset()
    print_problem(problem)
    problem_name, num_nodes, positions, distance_matrix = create_tsp_instance(problem)
    original_distance_matrix = distance_matrix.copy() # keep original distance matrix for display
    distance_matrix, scaled_distance = max_normalize(distance_matrix) # normalize the distance matrix for stability
    identity_route_permutation = np.arange(num_nodes) # used for caching purposes
    pheromone_matrix = np.full(distance_matrix.shape, INITIAL_PHEROMONE_VALUE) # create pheromone matrix based on initial pheromone value
    desirability_matrix = get_desirability_matrix(distance_matrix) # create desirability matrix based on distance matrix
    distance_influenced_matrix = desirability_matrix ** DISTANCE_INFLUENCE # influence for distance matrix
    ant_routes = np.zeros((NUMBER_OF_ANTS, num_nodes), dtype=int) # ant route matrix
    route_distances = np.zeros(NUMBER_OF_ANTS) # route distances array

    start_node = np.random.randint(0, num_nodes) # random starting node to compare both algorithms

    print("2D Coordinates\n",positions,"\n")
    print("Distance Matrix\n",original_distance_matrix,"\n")
    # Print initial Problem

    fig = None

    if ANIMATE_ROUTE and not DISABLE_PLOTS:
        print("Animating route...")
        plt.ion() # enable interactive mode
        fig, ax = plt.subplots(figsize=(16, 9)) # create plot for animation
        fig.canvas.manager.set_window_title(f"Ant System TSP") # set window title

    G = create_networkX_graph(num_nodes, positions, distance_matrix) # create the precomputed graph object
    best_route = create_random_route(num_nodes) # create a random route
    best_route_distance = get_route_distance(distance_matrix, best_route)

    if DISABLE_PLOTS:
        print("Plots Disabled...")

    if not ANIMATE_ROUTE:
        print("Animation Disabled...")
    print("")

    print_ant_system_details()

    start_time_aco = time.perf_counter()
    continue_animation = True
    for i in tqdm(range(MAX_ITERATIONS), desc=f"Running Ant System", unit="iter"):
        if fig is not None:
            if not plt.fignum_exists(fig.number):
                continue_animation = False # disable animation and continue completing the algorithm

        pheromone_influenced_matrix = pheromone_matrix ** PHEROMONE_INFLUENCE # influence for pheromone matrix
        total_influence_matrix = distance_influenced_matrix * pheromone_influenced_matrix # get total influence by multiplying

        if PARALLELIZE:
            ant_routes = Parallel(n_jobs=-1)(delayed(construct_ant_solution)(total_influence_matrix, identity_route_permutation, start_node) for j in range(NUMBER_OF_ANTS)) # parallelize ant route construction with with all cpu cores
        else:
            for j in range(NUMBER_OF_ANTS):
                ant_routes[j] = construct_ant_solution(total_influence_matrix, identity_route_permutation, start_node) # manually construct ant routes without parallelization. better for smaller number of ants

        for j in range(NUMBER_OF_ANTS):
            route_distances[j] = get_route_distance(distance_matrix, ant_routes[j]) # don't parallelize this since it's fast enough

        update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances)
        current_iteration_best_route, current_iteration_best_route_distance = get_best_route_and_distance(ant_routes, route_distances) # get the best route and distance for the current iteration

        if current_iteration_best_route_distance < best_route_distance:
            best_route = current_iteration_best_route.copy() # copy it to avoid reference issues
            best_route_distance = current_iteration_best_route_distance
        # update best route if current best iteration route is better

        if ANIMATE_ROUTE and continue_animation and i % PLOT_EVERY_K_ITERATIONS == 0:
            ax.cla()
            plot_ant_tsp_route(ax, G, pheromone_matrix, best_route, best_route_distance * scaled_distance, problem_name, num_nodes, i, best_found=False) # plot the current route

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1) # short pause for viewing the window
    end_time_aco = time.perf_counter()

    plt.close(fig) # close animated figure
    plt.ioff() # disable interactive mode

    np.set_printoptions(threshold=np.inf) # for showing the tour

    as_distance = best_route_distance * scaled_distance
    print(f"Ant System Best Route Distance: {as_distance:.2f}") # unscale the distance to get the real distance
    print("Ant System Best Route:", best_route)

    time_taken_aco_ms = (end_time_aco - start_time_aco) * 1000 # report time taken in milliseconds
    print(f"Ant System Total Time Taken: {time_taken_aco_ms:.2f} ms")

    if not DISABLE_PLOTS:
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.canvas.manager.set_window_title(f"Ant System TSP") # set window title
        plot_ant_tsp_route(ax, G, pheromone_matrix, best_route, best_route_distance * scaled_distance, problem_name, num_nodes, MAX_ITERATIONS, best_found=True, force_draw_edges=True) # plot final route solution
        plt.show() # show optimal route

    print("")
    start_time_nn = time.perf_counter()
    nearest_neighbor_route = construct_nearest_neighbor_solution(distance_matrix, start_node)
    end_time_nn = time.perf_counter()
    nearest_neighbor_distance = get_route_distance(distance_matrix, nearest_neighbor_route)

    nn_distance = nearest_neighbor_distance * scaled_distance
    print(f"Nearest Neighbor Route Distance: {nn_distance:.2f}") # unscale the distance to get the real distance
    print("Nearest Neighbor Route:", nearest_neighbor_route)

    time_taken_nn_ms = (end_time_nn - start_time_nn) * 1000 # report time taken in milliseconds
    print(f"Nearest Neighbor Total Time Taken: {time_taken_nn_ms:.2f} ms")

    print("")

    if not DISABLE_PLOTS:
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        fig2.canvas.manager.set_window_title(f"Nearest Neighbor TSP")
        plot_nearest_neighbour_route(ax2, G, nearest_neighbor_route, nearest_neighbor_distance * scaled_distance, problem_name, num_nodes, force_draw_edges=False) # unscale the distance to get the real distance
        plt.show()

    print("Summary Comparison")

    dist_diff = nearest_neighbor_distance * scaled_distance - best_route_distance * scaled_distance
    print(f"Distance Difference: {dist_diff:.2f}")
    # unscale the stabilized distances to get the real distances

    time_difference_s = (time_taken_aco_ms - time_taken_nn_ms) / 1000.0
    print(f"Time Difference: {time_difference_s:.2f} sec")

    improvement_percent = ((nearest_neighbor_distance * scaled_distance - best_route_distance * scaled_distance) / (nearest_neighbor_distance * scaled_distance)) * 100 # percent increase formula between the two algorithms
    print(f"Improvement Percent: {improvement_percent:.2f}%")

    print("\nExiting program...")

main() # launch the program