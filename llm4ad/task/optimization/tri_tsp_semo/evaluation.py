# name: str: TSPEvaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tri_tsp_semo.get_instance import GetData
from llm4ad.task.optimization.tri_tsp_semo.template import template_program, task_description
from pymoo.indicators.hv import HV 
import random
import time 

__all__ = ['TRITSPEvolution']


def tour_cost(instance, solution, problem_size):

        cost_1 = 0
        cost_2 = 0
        cost_3 = 0
        
        for j in range(problem_size - 1):
            node1, node2 = int(solution[j]), int(solution[j + 1])
            
            coord_1_node1, coord_2_node1, coord_3_node1 = instance[node1][:2], instance[node1][2:4], instance[node1][4:]
            coord_1_node2, coord_2_node2, coord_3_node2 = instance[node2][:2], instance[node2][2:4], instance[node2][4:]

            cost_1 += np.linalg.norm(coord_1_node1 - coord_1_node2)
            cost_2 += np.linalg.norm(coord_2_node1 - coord_2_node2)
            cost_3 += np.linalg.norm(coord_3_node1 - coord_3_node2)
        
        node_first, node_last = int(solution[0]), int(solution[-1])
        
        coord_1_first, coord_2_first, coord_3_first = instance[node_first][:2], instance[node_first][2:4], instance[node_first][4:]
        coord_1_last, coord_2_last, coord_3_last = instance[node_last][:2], instance[node_last][2:4], instance[node_last][4:]

        cost_1 += np.linalg.norm(coord_1_last - coord_1_first)
        cost_2 += np.linalg.norm(coord_2_last - coord_2_first)
        cost_3 += np.linalg.norm(coord_3_last - coord_3_first)

        return cost_1, cost_2, cost_3
    

def dominates(a, b):
        """True if a dominates b (minimization)."""
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(problem_size):
        sol = list(range(problem_size))
        random.shuffle(sol)
        return np.array(sol)



def check_constraint(solution, problem_size):
    sol = list(solution)
    if len(sol) != problem_size:
        return False
    if len(set(sol)) != problem_size:
        return False
    if not all(0 <= x < problem_size for x in solution):
        return False
    return True



def evaluate(instance_data, n_instance, problem_size, ref_point, eva: callable):
        obj_1 = np.ones(n_instance)
        obj_2 = np.ones(n_instance)
        obj_3 = np.ones(n_instance)
        n_ins = 0
        for instance, distance_matrix_1, distance_matrix_2, distance_matrix_3 in instance_data:
            start = time.time()
            s = [random_solution(problem_size) for _ in range(100)]
            Archive = [(s_, tour_cost(instance, s_, problem_size)) for s_ in s]
            for _ in range(20000):
                s_prime = eva(Archive, instance, distance_matrix_1, distance_matrix_2, distance_matrix_3)
                f_s_prime = tour_cost(instance, s_prime, problem_size)
                if not check_constraint(s_prime, problem_size):
                    continue

                # Nếu không bị thống trị
                if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                    # Loại bỏ các phần tử bị thống trị bởi f_s_prime
                    Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                    # Thêm nghiệm mới
                    Archive.append((s_prime, f_s_prime))
            end = time.time()
            objs = np.array([obj for _, obj in Archive])
            # Tính HV
            hv_indicator = HV(ref_point=ref_point)
            hv_value = hv_indicator(objs)
            obj_1[n_ins] = -hv_value
            obj_2[n_ins] = end - start
            n_ins += 1
        return np.mean(obj_1), np.mean(obj_2)
            





class TRITSPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Traveling Salesman Problem (TSP) using a custom algorithm."""

    def __init__(self, **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90
        )

        self.n_instance = 20
        self.problem_size = 20 
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets = getData.generate_instances()
        self.ref_point = np.array([20.0,20.0, 20.0])

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets,self.n_instance,self.problem_size, self.ref_point, callable_func)
    
import numpy as np
from typing import List, Tuple
import random
import json
import multiprocessing
import os
import warnings
warnings.filterwarnings("ignore")



def run_exec_and_eval(code_str, result_queue):
    try:
        local_vars = {}
        exec(code_str, globals(), local_vars)
        select_neighbor_func = local_vars["select_neighbor"]
        tsp = TRITSPEvaluation()
        cst, tme = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")


import random
import numpy as np
import random
from typing import List, Tuple




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    with open(f"Illustration/Tri TSP/EoH/population_0/pop_1.json", "r") as f:
        data = json.load(f)
    for k in range(len(data)):
        if k == 9:
            for j in range(1):
                select_neighbor_code = data[k]["function"]
                result_queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=run_exec_and_eval, args=(select_neighbor_code, result_queue))
                p.start()
                p.join(timeout=3600)
                if p.is_alive():
                    print(f"Timeout on code {k+1}, skipping.")
                    p.terminate()
                    p.join()
                    data[k]["score"] = data[k-1]["score"]
                    continue
                result = result_queue.get()
                if isinstance(result, str) and result.startswith("Error"):
                    print(f"Error on code {k+1}: {result}")
                    continue
                result[0] = result[0] / (20*20*20)
                data[k]["score"] = result 
                print(f"Evaluating with code {k+1}...", result)







