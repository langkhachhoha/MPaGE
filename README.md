# 🌟 MPaGE: Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization

🚀 **Official Codebase** for the paper:  
**[_MPaGE: Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization_](https://www.arxiv.org/abs/2507.20923)**

🧠 This repository provides the full implementation of **MPaGE**, the first framework that systematically integrates **Large Language Models (LLMs)** with the **Simple Evolutionary Multi-objective Optimization (SEMO)** paradigm and **Pareto Front Grid** guidance.

MPaGE is designed to solve **Multi-objective Combinatorial Optimization (MOCO)** problems by effectively balancing:
- ⏱ Runtime efficiency
- 🎯 Solution quality
- 🌐 Semantic diversity

---
## Overview 💡

Our approach curates heuristic algorithms for the **Simple Evolutionary Multi-objective Optimization (SEMO)** paradigm, leveraging the **Pareto Front Grid (PFG)** to guide the design of LLM-based variation heuristics.

![MPaGE Framework Overview](figure/Overview.png)

By partitioning the objective space into grids and retaining leading individuals from promising regions, MPaGE enhances both **solution quality** and **search efficiency**.
From these regions, **MPaGE** constructs a pool of elitist candidates and employs **Large Language Models (LLMs)** to assess their **semantic structures**, clustering them into groups with similar logic. Variation is then performed **with respect to these clusters**, promoting **semantic diversity** and reducing redundancy within the heuristic population.

![MPaGE Framework Overview](figure/Selection.png)

To the best of our knowledge, this is the **first comprehensive evaluation** of LLM-generated heuristics on standard **Multi-objective Combinatorial Optimization Problems (MOCOP)**, addressing solution quality, computational efficiency, and semantic diversity.

---

## 🔍 Main Contributions

- 🧩 We propose **MPaGE**, a novel framework that systematically integrates **LLMs** with the **SEMO paradigm** and **PFG** to solve **MOCOP** problems — balancing runtime, solution quality, and semantic diversity.

- 🧠 We leverage **LLMs** to verify the **logical structure** of heuristics and perform **cross-cluster recombination**, enhancing diversity and reducing redundancy through **logically dissimilar variations**.

- 📊 We conduct **extensive experiments** on standard **MOCOP benchmarks**, demonstrating consistent improvements in runtime efficiency, solution quality, and semantic diversity over both **LLM-based baselines** and traditional **multi-objective evolutionary algorithms (MOEAs)**.

---
## 🧪 Experiment Results

We evaluate the proposed **MPaGE** framework on four widely recognized **Multi-objective Combinatorial Optimization Problems (MOCOPs)** that have been extensively studied in the literature:

- 🛣 **Bi-objective Traveling Salesman Problem (Bi-TSP)**
- 🛣 **Tri-objective Traveling Salesman Problem (Tri-TSP)**
- 🚚 **Bi-objective Capacitated Vehicle Routing Problem (Bi-CVRP)**
- 🎒 **Bi-objective Knapsack Problem (Bi-KP)**

![MPaGE Framework Overview](figure/image3.png)
![MPaGE Framework Overview](figure/image1.png)
![MPaGE Framework Overview](figure/image2.png)
---
## How to use? 🔧
### 1. Install the dependencies

```bash
pip install -r requirements.txt
```
### 2. Set your LLM API key
### 3. Run MPaGE
```bash
python main.py
```
You can flexibly change the following hyperparameters:

| Parameter         | Description |
|------------------|-------------|
| `llm`            | The LLM model name used to generate heuristics. |
| `llm_cluster`    | The LLM model name is used specifically for clustering heuristics based on their semantic structure. |
| `evaluation`     | Defines the scoring mechanism for evaluating generated functions. |
| `max_generations`| Terminates the evolution after a fixed number of generations or earlier if `max_sample_nums` is reached. |
| `max_sample_nums`| Terminates after evaluating this many functions — valid or not — or earlier if `max_generations` is reached. |
| `pop_size`       | Population size used during the evolutionary process. |
| `selection_num`  | Number of individuals selected for crossover. |
| `use_e2_operator`| Boolean flag to enable the E2 operator for logic-driven exploration. |
| `use_m1_operator`| Boolean flag to activate the M1 mutation operator. |
| `use_m2_operator`| Boolean flag to activate the M2 mutation operator. |
| `llm_review`     | If `True`, LLM will review and filter the population before crossover to reduce low-quality logic. |


To set up **MPaGE** for your custom optimization problem, please refer to the task configuration files located in: llm4ad/task/optimization


## 📘 Glossary

- **LLM**: Large Language Model  
- **SEMO**: Simple Evolutionary Multi-objective Optimization  
- **PFG**: Pareto Front Grid  
- **MOCOP**: Multi-objective Combinatorial Optimization Problem  
- **MOEA**: Multi-objective Evolutionary Algorithm


If you find our work helpful for your research or applied projects:

```bibtex
@misc{ha2025paretogridguidedlargelanguagemodels,
      title={Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization}, 
      author={Minh Hieu Ha and Hung Phan and Tung Duy Doan and Tung Dao and Dao Tran and Huynh Thi Thanh Binh},
      year={2025},
      eprint={2507.20923},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2507.20923}, 
}












