import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

PROBLEM_SIZE = "SMALL" # "SMALL", "MEDIUM", "LARGE"

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

def plot_route(route, problem_name, num_nodes, positions, distance_matrix, enable_node_labels=True, enable_all_edges=True):
    route_edges = get_edge_list(route)
    total_distance = get_route_distance(distance_matrix, route)
    G = create_networkx_graph(num_nodes, positions, distance_matrix)
    pos = nx.get_node_attributes(G, 'pos')

    fig, ax = plt.subplots(figsize=(16, 9)) # create plot

    nx.draw_networkx_nodes(G,pos,node_size=200,node_color='lightcoral') # nodes

    if enable_node_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold') # node labels

    if enable_all_edges:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2) # all edges
    nx.draw_networkx_edges(G,pos,edgelist=route_edges,edge_color='black',width=1.5) # route edges

    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=400, node_color='limegreen') # highlight starting node in the route
    # draw edges and nodes and labels using networkx functions

    ax.tick_params(axis='both', which='major', left=True, bottom=True, labelleft=True, labelbottom=True) # enable tick marks for both axes

    plt.title(f"Ant Colony TSP Route - {problem_name} ({num_nodes} nodes) \nDistance - {total_distance:.2f}", fontsize=14)
    plt.xlabel("Relative X Coord", fontsize=12)
    plt.ylabel("Relative Y Coord", fontsize=12) # plot labels
    plt.grid(True, linestyle='-', alpha=0.8) # add grid
    plt.show()

def main():
    problem = load_dataset()
    print_problem(problem)
    problem_name, num_nodes, positions, distance_matrix = create_tsp_instance(problem)
    print("2D Coordinates\n",positions,"\n")
    print("Distance Matrix\n",distance_matrix,"\n")

    route = create_random_route(num_nodes)
    plot_route(route, problem_name, num_nodes, positions, distance_matrix)

main()