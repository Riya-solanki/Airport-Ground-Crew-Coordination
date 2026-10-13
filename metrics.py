"""
metrics.py
==========
Collects, aggregates, and exports simulation performance metrics.

METRICS TRACKED (per agent, per algorithm, per scenario):
  - total_travel_distance : sum of cells moved (excludes wait steps)
  - total_waiting_time    : sum of WAIT actions taken
  - makespan              : time step at which ALL agents have finished
  - num_conflicts         : head-on / vertex conflicts detected at runtime
  - nodes_expanded        : nodes popped from the priority queue during search
  - runtime_ms            : wall-clock planning time (milliseconds)

COMPLEXITY ANALYSIS (empirical, not fabricated):
  - Runtime vs number of vehicles (k = 1..6)
  - Nodes expanded vs grid size (N = 6..14)
  - Theoretical complexity:
      A* with a binary heap: O(E log V) for the bounded state graph (worst-case)
      Greedy Best-First with a binary heap: O(E log V) for the bounded state graph (worst-case)
      V and E refer to the space-time graph; practical expansions differ by priority rule.
  Empirical measurements are derived from actual execution.
"""

from __future__ import annotations
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from agents import GroundServiceAgent, Vehicle, VehicleStatus


OUTPUT_DIR = "output"


@dataclass
class ScenarioResult:
    """All metrics for one (scenario, algorithm) pair."""
    scenario:         str
    algorithm:        str
    total_distance:   int   = 0
    total_waiting:    int   = 0
    makespan:         int   = 0
    num_conflicts:    int   = 0
    nodes_expanded:   int   = 0
    runtime_ms:       float = 0.0
    vehicles_done:    int   = 0
    vehicles_no_path: int   = 0


def collect_vehicle_metrics(agents: list[GroundServiceAgent]) -> dict:
    """Aggregate per-agent metrics into scenario-level totals."""
    total_dist = sum(a.metrics.travel_distance for a in agents)
    total_wait = sum(a.metrics.waiting_time    for a in agents)
    makespan   = max((a.metrics.makespan for a in agents), default=0)
    conflicts  = sum(a.metrics.collisions      for a in agents)
    expanded   = sum(a.metrics.nodes_expanded  for a in agents)
    runtime    = sum(a.metrics.runtime_ms      for a in agents)
    done       = sum(1 for a in agents if a.current_status == VehicleStatus.DONE)
    no_path    = sum(1 for a in agents if a.current_status == VehicleStatus.NO_PATH)
    return dict(
        total_distance=total_dist, total_waiting=total_wait,
        makespan=makespan, num_conflicts=conflicts,
        nodes_expanded=expanded, runtime_ms=runtime,
        vehicles_done=done, vehicles_no_path=no_path,
    )


def save_metrics_csv(results: list[ScenarioResult], filename: str = "metrics.csv"):
    """Write all scenario results to a CSV file in the output/ directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"  [metrics] Saved -> {filepath}")


def print_metrics_table(results: list[ScenarioResult]):
    """Pretty-print a comparison table to the console."""
    print("\n" + "="*90)
    print(f"{'SCENARIO':<25} {'ALGORITHM':<20} {'DIST':>5} {'WAIT':>5} "
          f"{'SPAN':>5} {'CONF':>5} {'NODES':>8} {'RT(ms)':>8}")
    print("="*90)
    for r in results:
        alg_short = {"cooperative_astar": "A* (Cooperative)", "greedy_best_first": "Greedy Best-First"}.get(r.algorithm, r.algorithm)
        print(f"{r.scenario:<25} {alg_short:<20} {r.total_distance:>5} "
              f"{r.total_waiting:>5} {r.makespan:>5} {r.num_conflicts:>5} "
              f"{r.nodes_expanded:>8} {r.runtime_ms:>8.2f}")
    print("="*90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Real Empirical Benchmark & Comparison Plots
# ─────────────────────────────────────────────────────────────────────────────

def run_empirical_benchmarks() -> dict:
    """
    Measure actual empirical search complexity across:
      1. Number of vehicles: 1 to 6 on standard 12x12 grid.
      2. Grid sizes: 6x6, 8x8, 10x10, 12x12, 14x14.
    Runs A* and Greedy Best-First Search to collect real measured nodes and runtime.
    No hardcoded numbers or fake formulas.
    """
    from environment import AirportGrid
    from agents import Vehicle, ServiceType
    from search.cooperative_astar import ReservationTable, cooperative_astar, plan_all_vehicles
    from search.greedy_best_first import greedy_best_first, plan_all_vehicles_greedy

    grid_12     = AirportGrid(12, 12)
    depot_list  = list(grid_12.depots.values())
    gate_list   = list(grid_12.gates.values())
    gate_names  = list(grid_12.gates.keys())

    v_counts = [1, 2, 3, 4, 5, 6]
    nodes_v  = {"A* (Cooperative)": [], "Greedy Best-First": []}
    runtime_v= {"A* (Cooperative)": [], "Greedy Best-First": []}

    stypes = [ServiceType.FUEL, ServiceType.BAGGAGE,
              ServiceType.CATERING, ServiceType.MAINTENANCE]

    for k in v_counts:
        # 1. Cooperative A*
        table_ca = ReservationTable()
        agents_ca = [
            Vehicle(
                vid=f"V{i+1}", vtype=stypes[i % 4],
                depot=depot_list[i % len(depot_list)],
                gate_name=gate_names[i % len(gate_names)],
                gate_pos=gate_list[i % len(gate_list)],
                priority=i+1
            )
            for i in range(k)
        ]
        t0 = time.perf_counter()
        plan_all_vehicles(grid_12, agents_ca, table_ca, t_start=0)
        rt_ca = (time.perf_counter() - t0) * 1000
        tot_nodes_ca = sum(a.metrics.nodes_expanded for a in agents_ca)
        nodes_v["A* (Cooperative)"].append(tot_nodes_ca)
        runtime_v["A* (Cooperative)"].append(rt_ca)

        # 2. Greedy Best-First
        table_greedy = ReservationTable()
        agents_greedy = [
            Vehicle(
                vid=f"V{i+1}", vtype=stypes[i % 4],
                depot=depot_list[i % len(depot_list)],
                gate_name=gate_names[i % len(gate_names)],
                gate_pos=gate_list[i % len(gate_list)],
                priority=i+1
            )
            for i in range(k)
        ]
        t0 = time.perf_counter()
        plan_all_vehicles_greedy(grid_12, agents_greedy, table_greedy, t_start=0)
        rt_greedy = (time.perf_counter() - t0) * 1000
        tot_nodes_greedy = sum(a.metrics.nodes_expanded for a in agents_greedy)
        nodes_v["Greedy Best-First"].append(tot_nodes_greedy)
        runtime_v["Greedy Best-First"].append(rt_greedy)

    # Grid size scaling test
    grid_sizes = [6, 8, 10, 12, 14]
    nodes_g = {"A* (Cooperative)": [], "Greedy Best-First": []}

    for sz in grid_sizes:
        # Create an open N x N grid with boundary walls
        cells = np.zeros((sz, sz), dtype=np.int8)
        cells[0, :]  = 1
        cells[-1, :] = 1
        cells[:, 0]  = 1
        cells[:, -1] = 1
        test_g = AirportGrid.from_array(cells)
        start  = (1, 1)
        goal   = (sz - 2, sz - 2)


        # Cooperative A*
        tbl = ReservationTable()
        _, _, exp_ca = cooperative_astar(test_g, start, goal, tbl, t_start=0)
        nodes_g["A* (Cooperative)"].append(exp_ca)

        # Greedy Best-First
        tbl_b = ReservationTable()
        _, _, exp_b = greedy_best_first(test_g, start, goal, tbl_b, t_start=0)
        nodes_g["Greedy Best-First"].append(exp_b)

    return {
        "v_counts":   v_counts,
        "nodes_v":    nodes_v,
        "runtime_v":  runtime_v,
        "grid_sizes": grid_sizes,
        "nodes_g":    nodes_g,
    }


def plot_nodes_vs_vehicles(bench: dict):
    """Plot 1: Real measured nodes expanded vs number of vehicles."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"A* (Cooperative)": "#F5A623", "Greedy Best-First": "#E74C3C"}
    markers = {"A* (Cooperative)": "s", "Greedy Best-First": "^"}

    for alg, nodes in bench["nodes_v"].items():
        ax.plot(bench["v_counts"], nodes, marker=markers[alg],
                color=colours[alg], linewidth=2.2, markersize=8, label=alg)

    ax.set_xlabel("Number of Ground-Service Vehicles", fontsize=12)
    ax.set_ylabel("Measured Nodes Expanded", fontsize=12)
    ax.set_title("Search Efficiency: Nodes Expanded vs Number of Vehicles (Measured)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "nodes_vs_vehicles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")


def plot_nodes_vs_grid_size(bench: dict):
    """Plot 2: Real measured nodes expanded vs grid size."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"A* (Cooperative)": "#F5A623", "Greedy Best-First": "#E74C3C"}
    markers = {"A* (Cooperative)": "s", "Greedy Best-First": "^"}

    for alg, nodes in bench["nodes_g"].items():
        ax.plot(bench["grid_sizes"], nodes, marker=markers[alg],
                color=colours[alg], linewidth=2.2, markersize=8, label=alg)

    ax.set_xlabel("Grid Side Length N (N x N grid)", fontsize=12)
    ax.set_ylabel("Measured Nodes Expanded", fontsize=12)
    ax.set_title("Scalability: Nodes Expanded vs Grid Size (Measured)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "nodes_vs_gridsize.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")


def plot_runtime_vs_vehicles(bench: dict):
    """Plot 3: Real measured planning runtime vs number of vehicles."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"A* (Cooperative)": "#F5A623", "Greedy Best-First": "#E74C3C"}
    markers = {"A* (Cooperative)": "s", "Greedy Best-First": "^"}

    for alg, rts in bench["runtime_v"].items():
        ax.plot(bench["v_counts"], rts, marker=markers[alg],
                color=colours[alg], linewidth=2.2, markersize=8, label=alg)

    ax.set_xlabel("Number of Ground-Service Vehicles", fontsize=12)
    ax.set_ylabel("Planning Runtime (ms)", fontsize=12)
    ax.set_title("Planning Cost: Runtime vs Number of Vehicles (Measured)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "runtime_vs_vehicles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")


def generate_all_plots(results: list[ScenarioResult] | None = None):
    """Run real empirical benchmarks and generate verified comparison plots."""
    print("\n[metrics] Running live empirical benchmarks for comparison plots ...")
    bench = run_empirical_benchmarks()
    plot_nodes_vs_vehicles(bench)
    plot_nodes_vs_grid_size(bench)
    plot_runtime_vs_vehicles(bench)
    print("[metrics] All plots saved to output/ using real measured data.\n")
