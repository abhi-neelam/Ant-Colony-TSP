import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import mercantile

PROBLEM_SIZE = "SMALL" # "SMALL", "MEDIUM", "LARGE"
MAX_ITERATIONS = 1000 # number of iterations for the algorithm
NUMBER_OF_ANTS = 5 # number of ants in the colony
INITIAL_PHEROMONE_VALUE = 1.0 # initial pheromone value for each edge
DISTANCE_INFLUENCE = 4.0 # influence of distance on route
PHEROMONE_INFLUENCE = 1.0 # influence of pheromone on route
PHEROMONE_DEPOSIT = 1.0 # pheromone deposit factor
EVAPORATION_RATE = 0.3 # pheromone evaporation rate
# ALGORITHM PARAMETERS

ANIMATE_ROUTE = True
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
        return tsplib95.load('datasets/berlin52.tsp')
    elif PROBLEM_SIZE == "MEDIUM":
        return tsplib95.load('datasets/gil262.tsp')
    return tsplib95.load('datasets/ali535.tsp')

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

    if problem.edge_weight_type == 'GEO':
        print("Geo Coordinates detected. Projecting coordinates to mercator projection...")

    for tsp_id, coord in problem.node_coords.items():
        if problem.edge_weight_type == 'GEO': # project coordinates if GEO coords
            latitude = coord[0]
            longitude = coord[1]

            coord[0], coord[1] = mercantile.xy(longitude, latitude) # project lat long coords to 2d coordinates using web mercator projection

        positions[tsp_id - 1, 0] = coord[0]
        positions[tsp_id - 1, 1] = coord[1]
        # extract out positions to 2d numpy array

    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i, j] = problem.get_weight(i + 1, j + 1) # get_weight using library defined function. changes depending on geo coordinates or euclidean distance
            # add one since library uses one indexed

    return problem_name, num_nodes, positions, distance_matrix

def get_edge_list(route):
    route_edges = []
    num_nodes = len(route)

    for i in range(num_nodes):
        u = route[(i + 0) % num_nodes]
        v = route[(i + 1) % num_nodes]
        # handle wrap around by using mod

        route_edges.append((u, v)) # append each edge to the list

    return route_edges

def get_route_distance(distance_matrix, route):
    total = 0
    num_nodes = len(route)

    for i in range(num_nodes):
        u = route[(i + 0) % num_nodes]
        v = route[(i + 1) % num_nodes]
        # handle wrap around by using mod

        total += distance_matrix[u, v] # add each edge weight to the total route cost
    return total

def create_random_route(num_nodes):
    return np.random.permutation(num_nodes) # random permutation of the nodes

def create_networkX_graph(num_nodes, positions, distance_matrix):
    G = nx.Graph() # create the graph object

    for i in range(num_nodes):
        G.add_node(i, pos=tuple(positions[i])) # add each position of the node to the graph

    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if i == j:
                continue # skip same node
            G.add_edge(i, j, weight=distance_matrix[i, j]) # add each edge between all pairs of nodes

    return G

def plot_route(ax, G, route, total_distance, problem_name, num_nodes, distance_matrix, current_iteration, best_found=False):
    route_edges = get_edge_list(route)
    pos = nx.get_node_attributes(G, 'pos') # position dictionary for networkX

    is_node_labels_enabled =  ENABLE_NODE_LABELS and num_nodes <= NODE_LABELS_THRESHOLD
    node_size = NODE_LABELS_NODE_SIZE if is_node_labels_enabled else NO_NODE_LABELS_NODE_SIZE
    nx.draw_networkx_nodes(G,pos,node_size=node_size,node_color='lightcoral') # nodes

    if is_node_labels_enabled:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold') # node labels

    if num_nodes <= ALL_EDGES_THRESHOLD:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2) # all edges

    nx.draw_networkx_edges(G,pos,edgelist=route_edges,edge_color='black',width=1.5) # route edges
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=400, node_color='limegreen') # highlight starting node in the route

    ax.tick_params(axis='both', which='major', left=True, bottom=True, labelleft=True, labelbottom=True) # enable tick marks for both axes

    plt.title(f"{best_found*"Best"} Ant Colony TSP Route - {problem_name} ({num_nodes} nodes) \n{NUMBER_OF_ANTS} ants, {MAX_ITERATIONS} iterations, Current Iteration - {current_iteration}\nDistance Power - {DISTANCE_INFLUENCE}, Pheromone Power - {PHEROMONE_INFLUENCE}, Evaporation Rate - {EVAPORATION_RATE}\nDistance - {total_distance:.2f}", fontsize=14)

    plt.xlabel("Relative X Coord", fontsize=12)
    plt.ylabel("Relative Y Coord", fontsize=12) # plot labels
    plt.grid(True, linestyle='-', alpha=0.8) # add grid

def get_desirability_matrix(distance_matrix):
    desirability_matrix = 1.0 / (distance_matrix + EPSILON) # invert distance. higher distance means less desirability and vice versa
    np.fill_diagonal(desirability_matrix, 0) # set diagonal to 0 since we can't go to the same node as we started from
    return desirability_matrix

def construct_ant_solution(total_influence_matrix):
    num_nodes = total_influence_matrix.shape[0]
    
    unvisited_nodes = np.ones(num_nodes, dtype=bool)
    route = np.full(num_nodes, -1, dtype=int)
    route[0] = np.random.randint(0, num_nodes) # select a random start node for the ant
    unvisited_nodes[route[0]] = False # mark the start node as visited

    for i in range(num_nodes-1):
        current_node = route[i]
        
        unvisited_influences = total_influence_matrix[current_node] * unvisited_nodes # get influences for unvisited nodes
        unvisited_probabilities = unvisited_influences / np.sum(unvisited_influences) # normalize to get probabilities

        picked_node = np.random.choice(np.arange(num_nodes), p=unvisited_probabilities) # sample random node based on probability distribution
        
        route[i+1] = picked_node # add the node to the route
        unvisited_nodes[picked_node] = False # mark the node as picked

    return route

def update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances):
    pheromone_matrix *= (1 - EVAPORATION_RATE) # pheromone evaporation to soft forget previous routes

    for route, dist in zip(ant_routes, route_distances):
        route_edges = get_edge_list(route)

        for edge in route_edges:
            pheromone_matrix[edge[0], edge[1]] += PHEROMONE_DEPOSIT / (dist + EPSILON)
            pheromone_matrix[edge[1], edge[0]] += PHEROMONE_DEPOSIT / (dist + EPSILON)
            # update pheromone by inverse distance and default deposit

def get_best_route_and_distance(ant_routes, route_distances):
    idx = np.argmin(route_distances)
    return ant_routes[idx], route_distances[idx]

def construct_nearest_neighbor_solution(num_nodes, distance_matrix, start_node=0):
    pass

def main():
    np.set_printoptions(threshold=20) # shorten size since positions array is big

    problem = load_dataset()
    print_problem(problem)
    problem_name, num_nodes, positions, distance_matrix = create_tsp_instance(problem)
    pheromone_matrix = np.full(distance_matrix.shape, INITIAL_PHEROMONE_VALUE) # create pheromone matrix based on initial pheromone value
    desirability_matrix = get_desirability_matrix(distance_matrix) # create desirability matrix based on distance matrix
    distance_influenced_matrix = desirability_matrix ** DISTANCE_INFLUENCE # influence for distance matrix
    ant_routes = np.zeros((NUMBER_OF_ANTS, num_nodes), dtype=int) # ant route matrix
    route_distances = np.zeros(NUMBER_OF_ANTS) # route distances array

    print("2D Coordinates\n",positions,"\n")
    print("Distance Matrix\n",distance_matrix,"\n")
    # Print initial Problem

    fig = None

    if ANIMATE_ROUTE:
        print("Animating route...")
        plt.ion() # enable interactive mode
        fig, ax = plt.subplots(figsize=(16, 9)) # create plot for animation
        fig.canvas.manager.set_window_title(f"Ant Colony TSP") # set window title

    G = create_networkX_graph(num_nodes, positions, distance_matrix) # create the precomputed graph object
    best_route = create_random_route(num_nodes) # create a random route
    best_route_distance = get_route_distance(distance_matrix, best_route)

    if not ANIMATE_ROUTE:
        print("Animation Disabled...")

    continue_animation = True
    for i in tqdm(range(MAX_ITERATIONS), desc=f"Running Ant Colony", unit="iter"):
        if fig is not None:
            if not plt.fignum_exists(fig.number):
                continue_animation = False # disable animation and continue completing the algorithm

        pheromone_influenced_matrix = pheromone_matrix ** PHEROMONE_INFLUENCE # influence for pheromone matrix
        total_influence_matrix = distance_influenced_matrix * pheromone_influenced_matrix # get total influence by multiplying

        for j in range(NUMBER_OF_ANTS):
            ant_routes[j] = construct_ant_solution(total_influence_matrix)
            route_distances[j] = get_route_distance(distance_matrix, ant_routes[j])

        update_pheromone_matrix(pheromone_matrix, ant_routes, route_distances)
        current_iteration_best_route, current_iteration_best_route_distance = get_best_route_and_distance(ant_routes, route_distances) # get the best route and distance for the current iteration

        if current_iteration_best_route_distance < best_route_distance:
            best_route = current_iteration_best_route.copy() # copy it to avoid reference issues
            best_route_distance = current_iteration_best_route_distance
        # update best route if current best iteration route is better

        if ANIMATE_ROUTE and continue_animation and i % PLOT_EVERY_K_ITERATIONS == 0:
            ax.cla()
            plot_route(ax, G, best_route, best_route_distance, problem_name, num_nodes, distance_matrix, i, best_found=False) # plot the current route

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1) # short pause for viewing the window

    if ANIMATE_ROUTE:
        print("Animation finished...\n")

    plt.close(fig) # close animated figure
    plt.ioff() # disable interactive mode

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f"Ant Colony TSP") # set window title

    np.set_printoptions(threshold=np.inf) # for showing the tour

    print("\nBest Route Distance:", best_route_distance)
    print("Best Route:", best_route)

    print("\nPlotting Best Route...")
    plot_route(ax, G, best_route, best_route_distance, problem_name, num_nodes, distance_matrix, MAX_ITERATIONS, best_found=True) # plot final route solution
    plt.show() # show optimal route

    print("Exiting program...")

main() # launch the program