import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

PROBLEM_SIZE = "MEDIUM" # "SMALL", "MEDIUM", "LARGE"
ANIMATE_ROUTE = True
MAX_ITERATIONS = 20

ENABLE_NODE_LABELS = True
ENABLE_ALL_EDGES = False
# DRAWING PARAMETERS

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

    for tsp_id, coord in problem.node_coords.items():
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

def create_networkx_graph(num_nodes, positions, distance_matrix):
    G = nx.Graph() # create the graph object

    for i in range(num_nodes):
        G.add_node(i, pos=tuple(positions[i])) # add each position of the node to the graph

    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if i == j:
                continue # skip same node
            G.add_edge(i, j, weight=distance_matrix[i, j]) # add each edge between all pairs of nodes

    return G

def plot_route(ax, G, route, problem_name, num_nodes, distance_matrix):
    ax.cla() # Clear the axes for the new drawing

    route_edges = get_edge_list(route)
    total_distance = get_route_distance(distance_matrix, route)
    
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_nodes(G,pos,node_size=200,node_color='lightcoral') # nodes

    if ENABLE_NODE_LABELS:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold') # node labels

    if ENABLE_ALL_EDGES:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2) # all edges

    nx.draw_networkx_edges(G,pos,edgelist=route_edges,edge_color='black',width=1.5) # route edges
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=400, node_color='limegreen') # highlight starting node in the route

    ax.tick_params(axis='both', which='major', left=True, bottom=True, labelleft=True, labelbottom=True) # enable tick marks for both axes

    plt.title(f"Ant Colony TSP Route - {problem_name} ({num_nodes} nodes) \nDistance - {total_distance:.2f}", fontsize=14)
    plt.xlabel("Relative X Coord", fontsize=12)
    plt.ylabel("Relative Y Coord", fontsize=12) # plot labels
    plt.grid(True, linestyle='-', alpha=0.8) # add grid

def main():
    problem = load_dataset()
    print_problem(problem)
    problem_name, num_nodes, positions, distance_matrix = create_tsp_instance(problem)
    print("2D Coordinates\n",positions,"\n")
    print("Distance Matrix\n",distance_matrix,"\n")
    # Print initial Problem

    fig = None

    if ANIMATE_ROUTE:
        print("Animating route...")
        plt.ion() # enable interactive mode
        fig, ax = plt.subplots(figsize=(16, 9)) # create plot for animation
        fig.canvas.manager.set_window_title(f"Ant Colony TSP") # set window title

    G = create_networkx_graph(num_nodes, positions, distance_matrix) # create the precomputed graph object
    best_route = create_random_route(num_nodes) # create a random route at the beginning

    if not ANIMATE_ROUTE:
        print("Animation Disabled...")

    for i in tqdm(range(MAX_ITERATIONS), desc=f"Running Ant Colony", unit="iter"):
        if fig is not None:
            if not plt.fignum_exists(fig.number): # exit when window is closed
                break

        route = create_random_route(num_nodes) # random route for now. will change to ant colony later

        if ANIMATE_ROUTE:
            plot_route(ax, G, route, problem_name, num_nodes, distance_matrix) # plot the current route

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1) # short pause for viewing the window

    if ANIMATE_ROUTE:
        print("Animation finished...\n")

    plt.close(fig) # close animated figure
    plt.ioff() # disable interactive mode

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f"Ant Colony TSP") # set window title
    print("Plotting Best Found Route...")
    plot_route(ax, G, best_route, problem_name, num_nodes, distance_matrix) # plot final route solution
    plt.show() # show optimal route
    print("Exiting program...")

main() # launch the program