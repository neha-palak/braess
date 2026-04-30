# Braess's Paradox in Transportation Networks
**CS-4231: Graph Theory and Network Sciences — Spring 2026**

Neha Palak · Ved Parekh

---

## Overview

This project simulates and detects Braess's paradox on the Sioux Falls benchmark transportation network. It computes a user equilibrium using the Frank-Wolfe algorithm, then tests whether adding new edges increases total system travel time under selfish routing.

---

## Running the Project

```bash
python scripts\run_all.py
```

This single command runs the full pipeline — data preprocessing, equilibrium assignment, and Braess analysis. Logs and results are saved automatically to the `outputs\` folder.

---

## Project Structure

├── data/                          # Input datasets
│   ├── SiouxFalls_net.tntp        # Original network file (TNTP format)
│   ├── SiouxFalls_trips.tntp      # Original OD demand file
│   ├── sioux_falls_net.csv        # Processed network (links, capacities, free-flow times)
│   └── sioux_falls_trips.csv      # Processed OD matrix
│
├── scripts/                       # Core implementation
│   ├── run_all.py                 # Main pipeline (executes full workflow end-to-end)
│   ├── initial_equilibrium.py     # Preprocessing + Frank–Wolfe user equilibrium assignment
│   └── edge_simulation.py         # Candidate edge generation and Braess effect detection
│
├── outputs/                       # Generated automatically per run
│   ├── traffic_log.txt            # Frank–Wolfe convergence details
│   ├── braess_log.txt             # β-sweep experiment logs
│   ├── summary.txt                # Key results (TTT, worst-case edge, etc.)
│   ├── initial_equilibrium_output.csv # Link-level equilibrium flows
│   └── beta_sweep_results.csv         # ΔTTT for each edge–β combination
│
└── figures.ipynb                  # Visualization and result plots


---

## Pipeline

### 1. `initial_equilibrium.py`
- Loads and cleans the raw network and OD data (removes self-loops, zero-capacity links, duplicates)
- Builds a directed graph (24 nodes, 76 links, 528 OD pairs)
- Runs Frank-Wolfe user equilibrium assignment with BPR latency functions (α=0.15, β=5.0)
- Exports equilibrium flows to `outputs\initial_equilibrium_output.csv`
- Passes the graph, flows, and baseline TTT to `edge_simulation.py` via a handoff dictionary

### 2. `edge_simulation.py`
- Generates candidate edges from the top 8 high-demand nodes (cap=1,000 veh/hr, fft=3.0 min)
- Runs a beta sweep (β ∈ {4.0, 5.0, 6.0}) — re-running Frank-Wolfe for each edge × beta combination
- Flags Braess cases where TTT increases by more than 0.5%
- Runs a deep-dive on the strongest detected Braess case (flow changes per link, travel time deltas)
- Exports results to `outputs\beta_sweep_results.csv`

---


## Dependencies

```bash
pip install networkx pandas numpy
```
