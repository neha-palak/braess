"""
=============================================================================
Sioux Falls Traffic Assignment
=============================================================================
Covers:
  1. Load & clean network + OD demand from CSV
  2. Build directed NetworkX graph
  3. BPR latency function
  4. Frank-Wolfe iterative traffic assignment (User Equilibrium)
  5. Run until convergence (gap < epsilon)
  6. Report equilibrium flows and total travel time
  7. Export results for Braess analysis
=============================================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NET_CSV   = "data/sioux_falls_net.csv"
TRIPS_CSV = "data/sioux_falls_trips.csv"

BPR_ALPHA = 0.15
BPR_BETA  = 5.0
MAX_ITER  = 1500
EPSILON   = 5e-4
VERBOSE   = True


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_network(path: str) -> pd.DataFrame:
    """Load and validate the network CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"init_node", "term_node", "capacity", "free_flow_time"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Network CSV missing columns: {missing}")

    df = df.dropna(subset=list(required))
    df["init_node"]      = df["init_node"].astype(int)
    df["term_node"]      = df["term_node"].astype(int)
    df["capacity"]       = df["capacity"].astype(float)
    df["free_flow_time"] = df["free_flow_time"].astype(float)

    df = df[df["init_node"] != df["term_node"]]
    df = df[df["capacity"]       > 0]
    df = df[df["free_flow_time"] > 0]
    df = df.drop_duplicates(subset=["init_node", "term_node"])
    df = df.reset_index(drop=True)

    print(f"[NET ] Loaded {len(df)} directed links "
          f"({df['init_node'].nunique()} unique origins, "
          f"{df['term_node'].nunique()} unique destinations)")
    return df


def load_trips(path: str) -> pd.DataFrame:
    """Load and validate the OD demand CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"origin", "destination", "demand"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Trips CSV missing columns: {missing}")

    df = df.dropna(subset=list(required))
    df["origin"]      = df["origin"].astype(int)
    df["destination"] = df["destination"].astype(int)
    df["demand"]      = df["demand"].astype(float)

    df = df[df["origin"] != df["destination"]]
    df = df[df["demand"] > 0]
    df = df.drop_duplicates(subset=["origin", "destination"])
    df = df.reset_index(drop=True)

    total_demand = df["demand"].sum()
    print(f"[OD  ] Loaded {len(df)} OD pairs, "
          f"total demand = {total_demand:,.1f} vehicles")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(net_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from the network DataFrame."""
    G = nx.DiGraph()
    for _, row in net_df.iterrows():
        G.add_edge(
            int(row["init_node"]),
            int(row["term_node"]),
            capacity       = float(row["capacity"]),
            free_flow_time = float(row["free_flow_time"]),
            flow           = 0.0,
        )
    print(f"[GRAPH] Built DiGraph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BPR LATENCY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def bpr_travel_time(free_flow_time: float,
                    capacity: float,
                    flow: float,
                    alpha: float = BPR_ALPHA,
                    beta:  float = BPR_BETA) -> float:
    """BPR volume-delay: t(x) = t0 * [1 + alpha * (x/c)^beta]"""
    return free_flow_time * (1.0 + alpha * (flow / capacity) ** beta)


def bpr_derivative(free_flow_time: float,
                   capacity: float,
                   flow: float,
                   alpha: float = BPR_ALPHA,
                   beta:  float = BPR_BETA) -> float:
    """Derivative of BPR w.r.t. flow."""
    return free_flow_time * alpha * beta * (flow ** (beta - 1)) / (capacity ** beta)


def beckmann_integral(free_flow_time: float,
                      capacity: float,
                      flow: float,
                      alpha: float = BPR_ALPHA,
                      beta:  float = BPR_BETA) -> float:
    """Integral of BPR from 0 to x (Beckmann objective term)."""
    return free_flow_time * (
        flow + alpha * (flow ** (beta + 1)) / ((beta + 1) * (capacity ** beta))
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAFFIC ASSIGNMENT CORE (Frank-Wolfe)
#     NOTE: alpha and beta are explicit parameters throughout so Ved's module
#           can sweep beta values without any global variable tricks.
# ──────────────────────────────────────────────────────────────────────────────

def update_edge_weights(G: nx.DiGraph,
                        alpha: float = BPR_ALPHA,
                        beta:  float = BPR_BETA) -> None:
    """Set 'weight' on every edge = current BPR travel time."""
    for u, v, data in G.edges(data=True):
        data["weight"] = bpr_travel_time(
            data["free_flow_time"], data["capacity"], data["flow"],
            alpha=alpha, beta=beta
        )


def all_or_nothing(G: nx.DiGraph,
                   trips_df: pd.DataFrame,
                   alpha: float = BPR_ALPHA,
                   beta:  float = BPR_BETA) -> dict:
    """
    All-or-Nothing assignment: route all demand for each OD pair onto
    the single shortest path under current travel times.
    Returns auxiliary flows {(u,v): flow}.
    """
    aux_flows = {(u, v): 0.0 for u, v in G.edges()}
    update_edge_weights(G, alpha=alpha, beta=beta)

    origins = trips_df["origin"].unique()
    for orig in origins:
        try:
            lengths, paths = nx.single_source_dijkstra(G, orig, weight="weight")
        except Exception:
            continue

        od_subset = trips_df[trips_df["origin"] == orig]
        for _, row in od_subset.iterrows():
            dest   = int(row["destination"])
            demand = float(row["demand"])
            if dest not in paths:
                continue
            path = paths[dest]
            for i in range(len(path) - 1):
                aux_flows[(path[i], path[i + 1])] += demand

    return aux_flows


def beckmann_objective(G: nx.DiGraph,
                       alpha: float = BPR_ALPHA,
                       beta:  float = BPR_BETA) -> float:
    """Beckmann objective Z = Σ_a ∫₀^{x_a} t_a(u) du."""
    total = 0.0
    for u, v, data in G.edges(data=True):
        total += beckmann_integral(
            data["free_flow_time"], data["capacity"], data["flow"],
            alpha=alpha, beta=beta
        )
    return total


def relative_gap(G: nx.DiGraph, aux_flows: dict) -> float:
    """
    Relative gap = |Σ_a t_a*y_a - Σ_a t_a*x_a| / (Σ_a t_a*x_a)
    where x = current flows, y = AON flows.
    """
    numerator   = 0.0
    denominator = 0.0
    for u, v, data in G.edges(data=True):
        tt = data["weight"]
        numerator   += tt * aux_flows.get((u, v), 0.0)
        denominator += tt * data["flow"]
    if denominator == 0:
        return float("inf")
    return abs(numerator - denominator) / denominator


def line_search(G: nx.DiGraph,
                aux_flows: dict,
                alpha: float = BPR_ALPHA,
                beta:  float = BPR_BETA,
                n_steps: int = 32) -> float:
    """
    Bisection line search for optimal step size λ ∈ [0,1].
    Finds root of dZ/dλ = Σ_a t_a(x + λ(y−x)) · (y_a − x_a) = 0.
    """
    lo, hi = 0.0, 1.0
    edges  = list(G.edges(data=True))

    for _ in range(n_steps):
        mid   = (lo + hi) / 2.0
        deriv = 0.0
        for u, v, data in edges:
            x   = data["flow"]
            y   = aux_flows.get((u, v), 0.0)
            xly = x + mid * (y - x)
            deriv += bpr_travel_time(
                data["free_flow_time"], data["capacity"], xly,
                alpha=alpha, beta=beta
            ) * (y - x)
        if deriv < 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def frank_wolfe(G: nx.DiGraph,
                trips_df: pd.DataFrame,
                max_iter: int   = MAX_ITER,
                epsilon:  float = EPSILON,
                verbose:  bool  = VERBOSE,
                alpha:    float = BPR_ALPHA,
                beta:     float = BPR_BETA) -> dict:
    """
    Frank-Wolfe user equilibrium assignment.

    alpha and beta are explicit parameters so Ved can sweep beta values
    without touching module globals — the original source of the bug.
    """
    print("\n" + "═" * 60)
    print("  FRANK-WOLFE USER EQUILIBRIUM ASSIGNMENT")
    print("═" * 60)
    print(f"  Max iterations : {max_iter}")
    print(f"  Convergence ε  : {epsilon:.2e}")
    print(f"  BPR (α, β)     : ({alpha}, {beta})")
    print("─" * 60)

    # Initialise with free-flow AON
    for u, v, data in G.edges(data=True):
        data["flow"]   = 0.0
        data["weight"] = data["free_flow_time"]

    aon_flows = all_or_nothing(G, trips_df, alpha=alpha, beta=beta)
    for u, v in G.edges():
        G[u][v]["flow"] = aon_flows.get((u, v), 0.0)
    update_edge_weights(G, alpha=alpha, beta=beta)

    history = []
    t_start = time.time()

    def beckmann_at_lambda(lam, cur, aux):
        total = 0.0
        for u2, v2, d2 in G.edges(data=True):
            x_new = cur[(u2, v2)] + lam * (aux.get((u2, v2), 0.0) - cur[(u2, v2)])
            total += beckmann_integral(
                d2["free_flow_time"], d2["capacity"], x_new,
                alpha=alpha, beta=beta
            )
        return total

    for iteration in range(1, max_iter + 1):

        aux_flows = all_or_nothing(G, trips_df, alpha=alpha, beta=beta)
        gap       = relative_gap(G, aux_flows)
        obj       = beckmann_objective(G, alpha=alpha, beta=beta)

        history.append({"iteration": iteration, "gap": gap, "objective": obj})

        if verbose:
            print(f"  Iter {iteration:4d} | Gap = {gap:.8f} | Z = {obj:,.4f}")

        if gap < epsilon:
            print(f"\n  ✓ Converged at iteration {iteration}  (gap = {gap:.2e})")
            break

        cur_flows = {(u, v): G[u][v]["flow"] for u, v in G.edges()}
        lam_ls    = line_search(G, aux_flows, alpha=alpha, beta=beta)

        if iteration <= 3:
            lam = lam_ls
        else:
            lam_msa = 1.0 / iteration
            z_ls    = beckmann_at_lambda(lam_ls,  cur_flows, aux_flows)
            z_msa   = beckmann_at_lambda(lam_msa, cur_flows, aux_flows)
            lam     = lam_ls if z_ls <= z_msa else lam_msa

        for u, v in G.edges():
            x = G[u][v]["flow"]
            y = aux_flows.get((u, v), 0.0)
            G[u][v]["flow"] = x + lam * (y - x)

        update_edge_weights(G, alpha=alpha, beta=beta)

    else:
        gap = history[-1]["gap"] if history else float("inf")
        print(f"\n  ⚠ Did not converge in {max_iter} iterations (gap = {gap:.2e})")

    elapsed = time.time() - t_start
    print(f"\n  Elapsed time : {elapsed:.2f} s")
    print("═" * 60 + "\n")

    return {
        "history":   history,
        "final_gap": gap,
        "converged": gap < epsilon,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  RESULTS & REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_travel_time(G: nx.DiGraph) -> float:
    """Total system travel time = Σ_a t_a(x_a) * x_a."""
    ttt = 0.0
    for u, v, data in G.edges(data=True):
        ttt += data["weight"] * data["flow"]
    return ttt


def flow_summary(G: nx.DiGraph) -> pd.DataFrame:
    """Return a tidy DataFrame of equilibrium link results."""
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "init_node":       u,
            "term_node":       v,
            "free_flow_time":  data["free_flow_time"],
            "capacity":        data["capacity"],
            "flow":            data["flow"],
            "travel_time":     data["weight"],
            "volume_capacity": data["flow"] / data["capacity"],
        })
    df = pd.DataFrame(rows).sort_values(["init_node", "term_node"])
    return df.reset_index(drop=True)


def print_top_congested(summary_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print the most congested links (by V/C ratio)."""
    top = summary_df.nlargest(top_n, "volume_capacity")
    print(f"{'─'*60}")
    print(f"  Top {top_n} Most Congested Links (by V/C ratio)")
    print(f"{'─'*60}")
    print(f"  {'From':>5} {'To':>5} {'Flow':>10} {'Cap':>10} "
          f"{'V/C':>7} {'TT':>8}")
    print(f"  {'─'*5} {'─'*5} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")
    for _, row in top.iterrows():
        print(f"  {int(row['init_node']):>5} {int(row['term_node']):>5} "
              f"{row['flow']:>10,.1f} {row['capacity']:>10,.1f} "
              f"{row['volume_capacity']:>7.3f} {row['travel_time']:>8.4f}")
    print()


def export_for_braess(G: nx.DiGraph,
                   summary_df: pd.DataFrame,
                   net_df: pd.DataFrame,
                   trips_df: pd.DataFrame,
                   total_travel_time: float,
                   output_path: str = "outputs/initial_equilibrium_output.csv") -> dict:
    """Save equilibrium flows to CSV and return handoff dict for further analysis."""
    summary_df.to_csv(output_path, index=False)
    print(f"[EXPORT] Equilibrium flows saved → {output_path}")

    handoff = {
        "net_df":               net_df,
        "trips_df":             trips_df,
        "equilibrium_flows_df": summary_df,
        "total_travel_time":    total_travel_time,
        "graph":                G,
        "bpr_alpha":            BPR_ALPHA,
        "bpr_beta":             BPR_BETA,
        "epsilon":              EPSILON,
        "max_iter":             MAX_ITER,
    }
    return handoff


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_traffic_pipeline(net_csv: str   = NET_CSV,
                      trips_csv: str = TRIPS_CSV) -> dict:
    """End-to-end pipeline. Returns the handoff dict for further analysis."""

    print("\n" + "█" * 60)
    print("  SIOUX FALLS TRAFFIC SIMULATION")
    print("█" * 60 + "\n")

    net_df   = load_network(net_csv)
    trips_df = load_trips(trips_csv)
    G        = build_graph(net_df)
    result   = frank_wolfe(G, trips_df)
    ttt      = compute_total_travel_time(G)

    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║  TOTAL SYSTEM TRAVEL TIME            ║")
    print(f"  ║  TTT = {ttt:>28,.4f}  ║")
    print(f"  ╚══════════════════════════════════════╝\n")

    summary_df = flow_summary(G)
    print_top_congested(summary_df)

    print(f"  Network statistics:")
    print(f"    Nodes             : {G.number_of_nodes()}")
    print(f"    Links             : {G.number_of_edges()}")
    print(f"    OD pairs          : {len(trips_df)}")
    print(f"    Total demand      : {trips_df['demand'].sum():,.1f} veh")
    print(f"    Converged         : {result['converged']}")
    print(f"    Final gap         : {result['final_gap']:.2e}")
    print(f"    Total travel time : {ttt:,.4f} veh·hr")
    print()

    handoff = export_for_braess(G, summary_df, net_df, trips_df, ttt)

    print("\n" + "█" * 60)
    print("  TRAFFIC PIPELINE COMPLETE — READY FOR BRAESS ANALYSIS")
    print("█" * 60 + "\n")

    return handoff


# ──────────────────────────────────────────────────────────────────────────────
# 7.  UTILITY HELPERS (exposed for Ved)
# ──────────────────────────────────────────────────────────────────────────────

def rebuild_graph_from_csv(net_csv: str) -> tuple:
    """Convenience: reload net + trips CSVs and build a fresh graph."""
    net_df   = load_network(net_csv)
    trips_df = load_trips(TRIPS_CSV)
    G        = build_graph(net_df)
    return G, net_df, trips_df


