import os
import sys
from datetime import datetime

from initial_equilibrium import run_traffic_pipeline
from edge_simulation import run_braess_pipeline


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    traffic_log_path    = f"outputs/traffic_log.txt"
    braess_log_path     = f"outputs/braess_log.txt"
    summary_log_path = f"outputs/summary.txt"

    summary_data = {}

    # 1. RUN TRAFFIC MODULE
    print("\nRunning traffic simulation...\n")

    with open(traffic_log_path, "w") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect ONLY to file (not Tee)
        sys.stdout = f
        sys.stderr = f

        try:
            handoff = run_traffic_pipeline()
            summary_data["ttt_before"] = handoff["total_travel_time"]
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Traffic simulation complete → {traffic_log_path}")

    # 2. RUN BRAESS MODULE
    print("\nRunning Braess analysis...\n")

    with open(braess_log_path, "w") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = f
        sys.stderr = f

        try:
            results_df = run_braess_pipeline(handoff)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Braess analysis complete → {braess_log_path}")

    # 3. SUMMARY
    print("\n Writing summary...\n")

    with open(summary_log_path, "w") as f:
        f.write("FULL PIPELINE SUMMARY\n")

        if "ttt_before" in summary_data:
            f.write(f"Initial Total Travel Time: {summary_data['ttt_before']:.4f}\n")

        if not results_df.empty:
            best = results_df.nlargest(1, "pct_change").iloc[0]
            f.write("\nBest Braess case:\n")
            f.write(f"Edge: {int(best['from'])} → {int(best['to'])}\n")
            f.write(f"Beta: {best['beta']}\n")
            f.write(f"Increase: {best['pct_change']:.3f}%\n")

        f.write("\nLogs:\n")
        f.write(f"Initial Equilibrium: {traffic_log_path}\n")
        f.write(f"Braess Analysis:  {braess_log_path}\n")

    print(f"Summary saved → {summary_log_path}")

    print("\nFULL PIPELINE COMPLETED\n")