
template_program = '''
import numpy as np
from typing import List, Tuple
import random 

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    coords: np.ndarray,
    demand: np.ndarray,
    distance_matrix: np.ndarray,
    capacity: float
) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution from it.
    Args:
        archive: A list of tuples, where each tuple contains:
            - solution: A list of numpy arrays, each representing a vehicle route. 
                        Each route starts and ends at the depot (node index 0), e.g., [0, 3, 5, 0].
            - objective: A tuple of two float values (total_distance, makespan), 
                        representing the two objective values of the solution.
        
        coords: A numpy array of shape (n_nodes, 2), representing (x, y) coordinates of each node (depot + customers).
        demand: A numpy array of shape (n_nodes,), where demand[i] is the demand of node i. The depot has demand 0.
        distance_matrix: A numpy array of shape (n_nodes, n_nodes), where [i][j] is the Euclidean distance between node i and j.
        capacity: A float representing the maximum capacity of each vehicle.

    Returns:
        A new neighbor solution.
    """
    base_solution = archive[0][0].copy()
    new_solution = base_solution.copy()

    return new_solution

'''



task_description = "You are solving a Bi-objective Capacitated Vehicle Routing Problem (Bi-CVRP), where a single depot and multiple customers are located in 2D space. Each customer has a positive demand, and all vehicles in the fleet have identical capacity limits. The objective is to construct a set of routes, each starting and ending at the depot, such that all customers are served, vehicle capacities \
are not exceeded on any route, and two conflicting objectives are minimized: (1) the total travel distance across all routes, and (2) the makespan, defined as the length of the longest individual route. Each solution in the archive is represented as a list of NumPy arrays, where each array denotes a single route (starting and ending with depot index 0), and is paired with a tuple of two objective values (total_distance, makespan). Your task is to implement a function named 'select_neighbor' that selects one promising solution from the archive and apply a novel or hybrid local search operator to generate a feasible neighbor solution from it. \
Please perform an intelligent random selection from among the solutions that show promising potential for further local improvement. Using a creative local search strategy that you design yourself. Avoid standard methods like pure 2-opt; instead, invent or combine transformations, go beyond standard approaches to design a method that yields higher-quality solutions across multiple objectives. Ensure that the returned neighbor solution remains feasible under the vehicle capacity constraint. The function should return the new neighbor solution."





