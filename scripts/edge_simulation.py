"""
Sioux Falls Braess Analysis

Picks up from IE's handoff dict and covers:
  1. Add new edge(s) to the network
  2. Re-run Frank-Wolfe on the modified network
  3. Compare before vs after (TTT, V/C, link flows)
  4. Detect & report Braess-like behaviour
  5. Beta sweep experiments

KEY DESIGN DECISIONS (grounded in actual network stats):
  - Network free-flow times: 2–10 min (mean 4.1)
  - Network capacities: 4,800–25,900 veh/hr (median ~5,100)
  - New edge: fft=3.0 min (below average → tempting shortcut)
              cap=1,000 veh/hr (≈ 10× below median → congests quickly)
  This is the Braess trap: the edge looks attractive at free-flow but
  becomes a bottleneck once traffic shifts to it, raising system TTT.

  Beta sweep range: 4.0, 5.0, 6.0 only.
  Beta ≥ 7 causes non-convergence on this network (BPR curve too steep
  for Frank-Wolfe's line search to stabilise). Non-converged runs are
  skipped — their TTT values are meaningless.

  Braess threshold: 0.5% TTT increase.
  This filters out numerical noise while catching real effects.
  On a baseline TTT of ~10M veh·hr, 0.5% = 50,000 veh·hr — clearly real.
=============================================================================
"""

import itertools
import copy
import pandas as pd
import networkx as nx

from initial_equilibrium import (
    frank_wolfe,
    compute_total_travel_time,
    flow_summary,
    BPR_ALPHA,
    BPR_BETA,
    MAX_ITER,
    EPSILON,
)

# HELPERS

def clone_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Deep-copy graph so experiments are fully independent."""
    return copy.deepcopy(G)


# EDGE ADDITION

def add_edge(G, u, v, capacity, free_flow_time, bidirectional=False):
    G.add_edge(u, v,
               capacity=float(capacity),
               free_flow_time=float(free_flow_time),
               flow=0.0,
               weight=float(free_flow_time))
    if bidirectional:
        G.add_edge(v, u,
                   capacity=float(capacity),
                   free_flow_time=float(free_flow_time),
                   flow=0.0,
                   weight=float(free_flow_time))
    return G


def add_edges_from_list(G, edge_list):
    for item in edge_list:
        add_edge(G, *item)
    return G


# RUN MODIFIED NETWORK

def run_modified(handoff, new_edges, verbose=False,
                 bpr_beta=None, bpr_alpha=None):
    """
    Clone original graph, add new edges, re-run Frank-Wolfe.
    bpr_beta and bpr_alpha are passed directly to frank_wolfe()
    so beta sweeps actually take effect (no global variable tricks).
    """
    alpha = bpr_alpha if bpr_alpha is not None else handoff["bpr_alpha"]
    beta  = bpr_beta  if bpr_beta  is not None else handoff["bpr_beta"]

    G_mod    = clone_graph(handoff["graph"])
    trips_df = handoff["trips_df"]

    # Reset all flows to free-flow state before adding new edges
    for u, v, data in G_mod.edges(data=True):
        data["flow"]   = 0.0
        data["weight"] = data["free_flow_time"]

    add_edges_from_list(G_mod, new_edges)

    fw_result = frank_wolfe(
        G_mod, trips_df,
        verbose=verbose,
        alpha=alpha,
        beta=beta,
        max_iter=handoff.get("max_iter", MAX_ITER),
        epsilon=handoff.get("epsilon", EPSILON),
    )

    ttt     = compute_total_travel_time(G_mod)
    summary = flow_summary(G_mod)

    return {
        "graph":             G_mod,
        "summary_df":        summary,
        "total_travel_time": ttt,
        "fw_result":         fw_result,
        "new_edges":         new_edges,
    }


# BRAESS DETECTION

def detect_braess(handoff, modified, threshold_pct=0.5):
    """
    Flag Braess if TTT increases by more than threshold_pct.
    Default 0.5%: on baseline TTT ~10M veh·hr this is ~50k veh·hr —
    well above numerical noise, clearly a real effect.
    """
    ttt_before = handoff["total_travel_time"]
    ttt_after  = modified["total_travel_time"]
    pct_change = 100.0 * (ttt_after - ttt_before) / ttt_before

    return {
        "braess_detected": pct_change > threshold_pct,
        "pct_change":      pct_change,
        "ttt_before":      ttt_before,
        "ttt_after":       ttt_after,
    }


# CANDIDATE EDGE DESIGN

def design_braess_candidates(handoff):
    """
    Build candidate edges between high-demand node pairs that are not
    currently directly connected.

    Edge parameters chosen from actual network statistics:
      fft = 3.0 min  → below network mean (4.1 min), making it attractive
      cap = 1,000    → ~10x below median capacity (5,109), congests fast

    This is the Braess trap: at free-flow the edge looks like a shortcut,
    but once drivers switch to it, it saturates and travel times rise for
    everyone using the connecting corridors.
    """
    G        = handoff["graph"]
    trips_df = handoff["trips_df"]

    high_nodes = (
        trips_df.groupby("origin")["demand"]
        .sum()
        .nlargest(8)
        .index.tolist()
    )

    candidates = []
    for u, v in itertools.permutations(high_nodes, 2):
        if G.has_edge(u, v):
            continue
        try:
            nx.shortest_path(G, u, v, weight="free_flow_time")
        except nx.NetworkXNoPath:
            continue
        # One candidate per pair: tempting (fft=3) but low-capacity (cap=1000)
        candidates.append((u, v, 1000.0, 3.0, True))

    print(f"[CANDIDATES] {len(candidates)} candidate edges from "
          f"{len(high_nodes)} high-demand nodes")
    print(f"             Edge params: cap=1,000 veh/hr, fft=3.0 min")
    return candidates


# BETA SWEEP

def beta_sweep(handoff, candidates, beta_values=None, verbose=False):
    """
    For each (beta, candidate_edge) pair:
      1. Run modified Frank-Wolfe with that beta
      2. Skip if not converged (result is unreliable)
      3. Detect Braess at 0.5% threshold

    Beta range capped at 6.0: beta ≥ 7 causes non-convergence on this
    network due to the steep BPR curve overwhelming the line search.
    The interesting transition (0 → all Braess cases) happens at β=5→6.
    """
    if beta_values is None:
        beta_values = [4.0, 5.0, 6.0]

    test_candidates = candidates[:10]
    total = len(beta_values) * len(test_candidates)
    done  = 0

    print(f"\nBeta sweep: {len(beta_values)} betas × "
          f"{len(test_candidates)} edges = {total} experiments\n")

    results = []

    for beta in beta_values:
        print("\n" + "═" * 60)
        print(f"  BETA = {beta}")
        print("═" * 60)

        for edge in test_candidates:
            done += 1
            u, v = edge[0], edge[1]

            mod = run_modified(handoff, [edge], verbose=verbose, bpr_beta=beta)

            # Skip non-converged runs — TTT is unreliable
            if not mod["fw_result"]["converged"]:
                print(f"  [{done:>3}/{total}]  {u:>2}→{v:<2}  "
                      f"SKIPPED (did not converge, gap="
                      f"{mod['fw_result']['final_gap']:.2e})")
                continue

            verdict = detect_braess(handoff, mod, threshold_pct=0.5)
            flag    = "⚠  BRAESS" if verdict["braess_detected"] else "not braess"

            print(f"  [{done:>3}/{total}]  {u:>2}→{v:<2}  "
                  f"Δ={verdict['pct_change']:+.3f}%  {flag}")

            results.append({
                "beta":       beta,
                "from":       u,
                "to":         v,
                "pct_change": verdict["pct_change"],
                "braess":     verdict["braess_detected"],
                "ttt_before": verdict["ttt_before"],
                "ttt_after":  verdict["ttt_after"],
            })

    if not results:
        print("\n[NOTE] All runs were skipped (non-convergence).")
        print("       Try reducing beta_values to [4.0, 5.0] only.\n")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    print("\n" + "█" * 65)
    print("  BETA SWEEP SUMMARY")
    print("█" * 65)

    for beta in beta_values:
        sub = df[df["beta"] == beta]
        if sub.empty:
            print(f"\n  Beta = {beta:4.1f} │ all runs skipped (non-convergence)")
            continue
        count = sub["braess"].sum()
        print(f"\n  Beta = {beta:4.1f} │ Braess cases: {count:>2} / {len(sub)}", end="")
        if count > 0:
            best = sub[sub["braess"]].nlargest(1, "pct_change").iloc[0]
            print(f"  │ worst: {int(best['from'])}→{int(best['to'])}  "
                  f"Δ={best['pct_change']:+.3f}%", end="")
        print()

    print(f"\n  Total Braess detections: {df['braess'].sum()} / {len(df)}")

    df.to_csv("outputs/beta_sweep_results.csv", index=False)
    print("\n[EXPORT] beta_sweep_results.csv\n")

    return df


# SINGLE EDGE DEEP DIVE

def single_edge_deep_dive(handoff, edge, beta=BPR_BETA):
    """
    Detailed before/after comparison for one edge.
    Useful for the report section showing the Braess mechanism clearly.
    """
    u, v = edge[0], edge[1]
    print("\n" + "─" * 60)
    print(f"  DEEP DIVE: edge {u}→{v}  "
          f"cap={edge[2]:,.0f}  fft={edge[3]:.1f}  beta={beta}")
    print("─" * 60)

    mod = run_modified(handoff, [edge], verbose=True, bpr_beta=beta)

    if not mod["fw_result"]["converged"]:
        print(f"\n  ⚠ Did not converge — deep dive result is unreliable.")
        return None, mod

    verdict = detect_braess(handoff, mod, threshold_pct=0.5)

    print(f"\n  TTT before : {verdict['ttt_before']:>15,.2f} veh·min")
    print(f"  TTT after  : {verdict['ttt_after']:>15,.2f} veh·min")
    print(f"  Δ TTT      : {verdict['pct_change']:>+14.3f} %")
    print(f"  Braess?    : {'YES ⚠' if verdict['braess_detected'] else 'No'}")

    # Top 5 links with biggest flow change
    before_df = handoff["equilibrium_flows_df"].copy()
    after_df  = mod["summary_df"].copy()

    merged = before_df.merge(
        after_df, on=["init_node", "term_node"],
        suffixes=("_before", "_after"), how="inner"
    )
    merged["flow_delta"] = merged["flow_after"]        - merged["flow_before"]
    merged["tt_delta"]   = merged["travel_time_after"] - merged["travel_time_before"]

    print("\n  Top 5 links with biggest flow change:")
    top5 = merged.nlargest(5, "flow_delta")[
        ["init_node", "term_node",
         "flow_before", "flow_after", "flow_delta", "tt_delta"]
    ]
    print(top5.to_string(index=False))

    return verdict, mod


# main pipeline function

def run_braess_pipeline(handoff: dict):

    print("  BRAESS PARADOX ANALYSIS")

    # Step 1: Design candidates using actual network statistics
    candidates = design_braess_candidates(handoff)

    # Step 2: Beta sweep — the main experiment
    results_df = beta_sweep(
        handoff,
        candidates,
        beta_values=[4.0, 5.0, 6.0],
        verbose=False,
    )

    # Handle empty results cleanly
    if results_df.empty:
        print("\n[INFO] No valid results (likely due to non-convergence).")
        return results_df

    # Step 3: Deep dive on best Braess case
    braess_cases = results_df[results_df["braess"]]

    if not braess_cases.empty:
        # Choose most credible case (lowest beta with Braess)
        min_braess_beta = braess_cases["beta"].min()

        best_row = (
            braess_cases[braess_cases["beta"] == min_braess_beta]
            .nlargest(1, "pct_change")
            .iloc[0]
        )

        u, v = int(best_row["from"]), int(best_row["to"])
        best_edge = (u, v, 1000.0, 3.0, True)

        print(f"\n>>> Deep dive: {u}→{v}  beta={min_braess_beta}  "
              f"Δ={best_row['pct_change']:+.3f}%")

        single_edge_deep_dive(handoff, best_edge, beta=min_braess_beta)

    else:
        print("\n[INFO] No Braess cases found with current parameters.")
        print("       This is a valid result — report it as a conditional finding.")
        print("       Braess on Sioux Falls requires high congestion sensitivity")
        print("       (β ≥ 5) AND a low-capacity shortcut in a congested corridor.")

    return results_df
    
