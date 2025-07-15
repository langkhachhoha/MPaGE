from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_kp.get_instance import GetData
from llm4ad.task.optimization.bi_kp.template import template_program, task_description
from pymoo.indicators.hv import HV
import random
import time

__all__ = ['BIKPEvaluation']


def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    if np.sum(solution * weight_lst) > capacity:
        return -1e10, -1e10  # Penalize infeasible solutions
    # check if the solution is feasible
    if not np.all(np.isin(solution, [0, 1])):
        return -1e10, -1e10
    if len(solution) != len(weight_lst):
        return -1e10, -1e10
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return total_val1, total_val2


def dominates(a, b):
    """True if a dominates b (maximization)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def random_solution(weight_lst, capacity, problem_size):
    # Generate a permutation of the problem size and then select a subset of items in order with a probability of 0.5 till reaching the capacity
    # This is a simple random solution generator for the knapsack problem
    sol = list(range(problem_size))
    random.shuffle(sol)
    selected_items = []
    total_weight = 0
    for item in sol:
        if total_weight + weight_lst[item] <= capacity:
            selected_items.append(item)
            total_weight += weight_lst[item]
    return np.array([1 if i in selected_items else 0 for i in range(problem_size)])



    


def evaluate(instance_data, n_instance, problem_size, ref_point, capacity, eva: callable):
    obj_1 = np.ones(n_instance)
    obj_2 = np.ones(n_instance)
    n_ins = 0
    for weight_lst, value1_lst, value2_lst in instance_data:
        start = time.time()
        s = [random_solution(weight_lst, capacity, problem_size) for _ in range(20)]
        Archive = [(s_, knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)) for s_ in s if knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)[0] > -1e5]
        for _ in range(8000):
            s_prime = np.array(eva(Archive, weight_lst, value1_lst, value2_lst, capacity))
            f_s_prime = knapsack_value(s_prime, weight_lst, value1_lst, value2_lst, capacity)

            if f_s_prime[0] < -1e5:
                print("Here")
                continue  # Skip infeasible

            if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                Archive.append((s_prime, f_s_prime))
        end = time.time()
        objs = np.array([obj for _, obj in Archive]) * (-1)
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(objs)
        obj_1[n_ins] = -hv_value
        obj_2[n_ins] = end - start
        n_ins += 1
    return np.mean(obj_1), np.mean(obj_2)


class BIKPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Knapsack Problem (BI-KP) using a custom algorithm."""

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90
        )
        self.n_instance = 8
        self.problem_size = 200
        getData = GetData(self.n_instance, self.problem_size)
        self._datasets, self.cap = getData.generate_instances() 
        self.ref_point = np.array([-30, -30]) 

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(self._datasets, self.n_instance, self.problem_size, self.ref_point, self.cap, callable_func)
    
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
        tsp = BIKPEvaluation()
        cst, tme = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    with open(f"Illustration/Bi KP 50/EoH/population_0/pop_1.json", "r") as f:
        data = json.load(f)
    for k in range(len(data)):
        if k == 9:
            for _ in range(1):
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
                result[0] = result[0] / (45*45)
                data[k]["score"] = result
                print(f"Evaluating with code {k+1}...", result)




