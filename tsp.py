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

def main():
    problem = load_dataset()
    print_problem(problem)
    instance = create_tsp_instance(problem)
    print(instance)
    
main()