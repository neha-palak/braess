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

```
├── data\                          # Input data
│   ├── SiouxFalls_net.tntp        # Raw network file
│   ├── SiouxFalls_trips.tntp      # Raw OD demand file
│   ├── sioux_falls_net.csv        # Cleaned network (links, capacities, free-flow times)
│   └── sioux_falls_trips.csv      # Cleaned OD matrix
│
├── scripts\                       # Source code
│   ├── run_all.py                 # Entry point — runs the full pipeline
│   ├── initial_equilibrium.py     # Data preprocessing + Frank-Wolfe UE assignment
│   └── edge_simulation.py        # Candidate edge generation + Braess detection
│
└── outputs\                       # Auto-generated on each run
    ├── traffic_log_<timestamp>.txt         # Frank-Wolfe convergence log
    ├── braess_log_<timestamp>.txt          # Beta sweep experiment log
    ├── summary_<timestamp>.txt            # Pipeline summary (TTT, best Braess case)
    ├── initial_equilibrium_output.csv     # Equilibrium flows for all links
    └── beta_sweep_results.csv             # TTT change for each candidate edge × beta
```

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
