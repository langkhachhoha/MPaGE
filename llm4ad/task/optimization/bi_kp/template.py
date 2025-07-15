
template_program = '''
import numpy as np
from typing import List, Tuple
import random 

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    weight_lst: np.ndarray,
    value1_lst: np.ndarray,
    value2_lst: np.ndarray,
    capacity: float 
) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution from it.

    Args:
    archive: List of (solution, objective) pairs. Each solution is a binary numpy array (0/1) of item selections.
             Each objective is a tuple of two float values (total value1, total value2).
    weight_lst: Numpy array of shape (N, ), item weights.
    value1_lst: Numpy array of shape (N, ), item values for objective 1.
    value2_lst: Numpy array of shape (N, ), item values for objective 2.
    capacity: Maximum allowed total weight.

    Returns:
    A new neighbor solution (numpy array).
    """
    base_solution = archive[0][0].copy()
    new_solution = base_solution.copy()
    new_solution[0], new_solution[1] = new_solution[1], new_solution[0]

    return new_solution
'''

task_description = "You are solving a Bi-objective Knapsack Problem (BI-KP), where each item has a weight and two profit values: \
value1 and value2. The goal is to select a subset of items such that the total weight does not exceed a given capacity, while \
simultaneously maximizing the total value in both objective spaces. \
Given an archive of non-dominated solutions, where each solution is a binary numpy array indicating item inclusion (1) or exclusion (0), \
and its corresponding objective is a tuple of two values (total value1, total value2), design a heuristic function named 'select_neighbor' that selects one solution from the archive \
and apply a novel or hybrid local search operator to generate a neighbor solution from it. \
Must always ensure that the generated neighbor solution remains feasible, \
i.e., the total weight must not exceed the knapsack capacity \
Please perform an intelligent random selection from among the solutions that show promising potential for further local improvement. Using a creative local search strategy that you design yourself, avoid 2-opt, \
go beyond standard approaches to design a method that yields higher-quality solutions across multiple objectives. The function should return the new neighbor solution."


