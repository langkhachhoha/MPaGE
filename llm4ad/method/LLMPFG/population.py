from __future__ import annotations
from threading import Lock
from typing import List
import numpy as np
from ...base import *
import numpy as np
import random
from copy import deepcopy


# Population Management
def is_dominated(obj1, obj2):
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

def fast_non_dominated_sort(population):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for i, p in enumerate(population):
        S[i] = []
        n[i] = 0
        for j, q in enumerate(population):
            if i == j:
                continue
            if is_dominated(p.score, q.score):
                S[i].append(j)
            elif is_dominated(q.score, p.score):
                n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)

    i = 0
    while i < len(fronts):
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def calculate_crowding_distance(population, indices):
    distances = {i: 0.0 for i in indices}
    num_objectives = len(population[0].score)

    for m in range(num_objectives):
        indices.sort(key=lambda i: population[i].score[m])
        distances[indices[0]] = distances[indices[-1]] = float('inf')
        min_obj = population[indices[0]].score[m]
        max_obj = population[indices[-1]].score[m]
        if max_obj == min_obj:
            continue
        for k in range(1, len(indices) - 1):
            prev_obj = population[indices[k - 1]].score[m]
            next_obj = population[indices[k + 1]].score[m]
            distances[indices[k]] += (next_obj - prev_obj) / (max_obj - min_obj)
    return distances


def population_management(population, N):
    fronts = fast_non_dominated_sort(population)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            remaining = N - len(selected)
            distances = calculate_crowding_distance(population, front)
            sorted_front = sorted(front, key=lambda i: -distances[i])
            selected.extend(sorted_front[:remaining])
            break
    return [population[i] for i in selected]



# Selection
import numpy as np
import random
from copy import deepcopy



def cal_knee_point(pop):
    knee_point = np.zeros(len(pop[0].score))
    m = len(pop[0].score)
    for i in range(m):
        knee_point[i] = 1e9
    for indi in pop:
        for i in range(m):
            knee_point[i] = min(knee_point[i], indi.score[i])
    return knee_point


def cal_nadir_point(pop):
    m = len(pop[0].score)
    nadir_point = np.zeros(m)
    for i in range(m):
        nadir_point[i] = -1e9
    for indi in pop:
        for i in range(m):
            nadir_point[i] = max(nadir_point[i], indi.score[i])
    return nadir_point



def Generation_PFG(pop, GK, knee_point, nadir_point, sigma):
    m = len(knee_point)
    d = [(nadir_point[j] - knee_point[j] + 2 * sigma) / GK for j in range(m)]

    Grid = []
    for indi in pop:
        grid_indi = [(indi.score[j] - knee_point[j] + sigma) // d[j] for j in range(m)]
        Grid.append(grid_indi)

    PFG = [[[] for _ in range(GK)] for _ in range(m)]

    for i in range(m):
        for j in range(GK): 
            Sij = [idx for idx, g in enumerate(Grid) if g[i] == j]

            if not Sij:
                continue
            g_min = min(Grid[idx][i] for idx in Sij)

            for idx in Sij:
                if Grid[idx][i] == g_min:
                    PFG[i][j].append(pop[idx])

    return PFG


def parent_selection(pop, m, GK = 4, sigma = 0.01, epsilon = 0.8):
    pop_ = deepcopy(pop)
    knee_point = cal_knee_point(pop_)
    nadir_point = cal_nadir_point(pop_)

    for indi in pop_:
        if isinstance(indi.score, tuple):
            indi.score = list(indi.score)
    for indi in pop_:
        for i in range(len(indi.score)):
            indi.score[i] = (indi.score[i] - knee_point[i]) / (nadir_point[i] - knee_point[i])
    knee_point = np.array([0,0])
    nadir_point = np.array([1,1])
    
    PFG = Generation_PFG(pop_, GK, knee_point, nadir_point, sigma)

    if (random.random() > epsilon):
        funcs = [f for f in pop if f.score is not None]
        func = sorted(funcs, key=lambda f: f.score[0])
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)
        parents = random.choices(pop, k=m, weights=p)
    else:
        i = random.randint(0, len(knee_point)-1) 
        j = random.randint(0, len(PFG[i]) - 2)
        while len(PFG[i][j]) == 0:
            i = random.randint(0, len(knee_point)-1) 
            j = random.randint(0, len(PFG[i]) - 2)
        parents = PFG[i][j] + PFG[i][j + 1]
    
    if len(parents) > 5:
        parents = random.sample(parents, 5)
    return parents


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []

        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        if self._generation == 0 and func.score is None:
            return
        if func.score is None:
            return
        try:
            self._lock.acquire()
            self._next_gen_pop.append(func)

            if self._generation == 0 and len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                self._population = population_management(pop, self._pop_size)
                self._next_gen_pop = []
                self._generation += 1
            elif self._generation > 0 and len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                self._population = population_management(pop, self._pop_size)
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self, selection_num) -> List[Function]:
        try:
            return parent_selection(self._population, selection_num)
        except Exception as e:
            print(e)
            return []
        
    def selection_cluster(self, group, indivs) -> List[Function]:
        try:
            N = len(indivs)
            all_indices = [idx for subgroup in group for idx in subgroup]
            valid_group = (
            len(group) > 1 and
            sorted(all_indices) == list(range(N)) and
            len(set(all_indices)) == N
            )

            if not valid_group:
                return random.sample(indivs, 2)
                # Chọn một nhóm ngẫu nhiên để lấy parent1
            group1 = random.choice(group)
            idx1 = random.choice(group1)
            parent1 = indivs[idx1]

            # Loại bỏ nhóm chứa idx1 để chọn parent2 từ nhóm khác
            other_groups = [g for g in group if idx1 not in g and len(g) > 0]

            if other_groups:
                group2 = random.choice(other_groups)
                idx2 = random.choice(group2)
            else:
                # fallback nếu không còn nhóm khác
                idx2 = random.choice([i for i in range(N) if i != idx1])

            parent2 = indivs[idx2]

            return [parent1, parent2]

            
        except Exception as e:
            print(e)
            return []
        

