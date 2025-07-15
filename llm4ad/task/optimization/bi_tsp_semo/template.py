
template_program = '''
import numpy as np
from typing import List, Tuple
import random 

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray
) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution from it.

    Args:
    archive: List of (solution, objective) pairs. Each solution is a numpy array of node IDs.
             Each objective is a tuple of two float values (cost in each space).
    instance: Numpy array of shape (N, 4). Each row corresponds to a node and contains its coordinates in two 2D spaces: (x1, y1, x2, y2).
    distance_matrix_1: Distance matrix in the first objective space.
    distance_matrix_2: Distance matrix in the second objective space.

    Returns:
    A new neighbor solution (numpy array).
    """
    base_solution = archive[0][0].copy()
    new_solution = base_solution.copy()
    new_solution[0], new_solution[1] = new_solution[1], new_solution[0]

    return new_solution
'''

task_description = "You are solving a Bi-objective Travelling Salesman Problem (bi-TSP), where each node has two different 2D coordinates: \
(x1, y1) and (x2, y2), representing its position in two objective spaces. The goal is to find a tour visiting each node exactly once and returning \
to the starting node, while minimizing two objectives simultaneously: the total tour length in each coordinate space. \
Given an archive of solutions, where each solution is a numpy array representing a TSP tour, and its corresponding objective \
is a tuple of two values (cost in each space), design a heuristic function named 'select_neighbor' that selects one solution from the archive \
and apply a novel or hybrid local search operator to generate a neighbor solution from it.  \
Must always ensure that the generated neighbor solution remains feasible, \
i.e., the solution must represent a valid TSP tour: it visits each node exactly once, ensuring no node is skipped or revisited.\
Please perform an intelligent random selection from among the solutions that show promising potential for further local improvement. Using a creative local search strategy that you design yourself, avoid 2-opt, \
go beyond standard approaches to design a method that yields higher-quality solutions across multiple objectives. The function should return the new neighbor solution."
