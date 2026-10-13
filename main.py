"""
main.py
=======
Entry point for the Airport Ground-Crew Coordination Multi-Agent simulation.

USAGE:
  python main.py                     # Launch desktop GUI (no args = GUI)
  python main.py --gui               # Launch desktop GUI explicitly
  python main.py --scenario 1        # Run Scenario 1: Normal Operation
  python main.py --scenario 3        # Run Scenario 3: Shared Gate Contention
  python main.py --scenario all      # Run all 5 report scenarios
  python main.py --compare           # Compare A* and Greedy Best-First Search on scenario 2
  python main.py --layout            # Save airport layout image only
  python main.py --scenario all --no-anim   # Skip GIF (faster)

ALGORITHM FLAGS:
  --algorithm cooperative_astar  (A*, default)
  --algorithm greedy_best_first  (Greedy Best-First Search)
  --algorithm all                # Run both algorithms

SCENARIOS:
  1  Normal Operation
  2  Corridor Conflict
  3  Gate Unavailable
  4  Emergency Priority
  5  Unreachable Goal

OUTPUT:
  output/animation_<scenario>_<algorithm>.gif
  output/frame_<scenario>_<algorithm>.png
  output/metrics.csv
  output/nodes_vs_vehicles.png
  output/nodes_vs_gridsize.png
  output/runtime_vs_vehicles.png
  output/airport_layout.png
"""

import argparse
import os
import sys
import random
import numpy as np

# ── Force UTF-8 output on Windows to avoid cp1252 codec errors ──────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Determinism: fixed seed for reproducibility ──────────────────────────────
random.seed(42)
np.random.seed(42)

from environment import AirportGrid
from scenarios import SCENARIO_REGISTRY, SCENARIO_DESCRIPTIONS
from simulation import run_simulation
from metrics import ScenarioResult, save_metrics_csv, print_metrics_table, generate_all_plots
from visualize import draw_grid_static


def launch_gui():
    """Lazy import and launch of the Tkinter GUI."""
    try:
        from gui import launch
        launch()
    except ImportError as e:
        print(f"[ERROR] Cannot launch GUI: {e}. Ensure tkinter and a display server are available.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] GUI launch error: {e}")
        sys.exit(1)


ALGORITHMS = ["cooperative_astar", "greedy_best_first"]


def run_scenario(scenario_id: str, algorithm: str,
                 save_anim: bool = True, verbose: bool = True) -> ScenarioResult:
    """Instantiate and run one (scenario, algorithm) combination."""
    name, factory = SCENARIO_REGISTRY[scenario_id]
    desc_title, desc_text = SCENARIO_DESCRIPTIONS.get(scenario_id, (name, ""))
    grid   = AirportGrid()
    agents, events = factory(grid)

    print(f"\n{'='*62}")
    print(f"  SCENARIO {scenario_id}: {desc_title}")
    print(f"  {desc_text[:100]}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Agents:    {[a.agent_id for a in agents]}")
    print(f"{'='*62}")

    result = run_simulation(
        grid, agents, events,
        scenario_name = name,
        algorithm     = algorithm,
        save_animation= save_anim,
        verbose       = verbose,
    )
    return result


def run_search_tree_demo():
    """
    Print the A* search tree for a tiny 4x4 example and the real 12x12 grid.
    """
    print("\n" + "="*60)
    print("  SEARCH TREE TRACE -- Airport Ground-Service A* Example")
    print("  (Shows how A* expands nodes guided by true-distance heuristic)")
    print("="*60)

    # Tiny 4x4 grid: boundary walls around a 2x2 open area
    mini_grid = AirportGrid.from_array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ])
    start = (1, 1)
    goal  = (2, 2)
    print_search_tree(mini_grid, start, goal, max_levels=3)

    # Real 12x12 grid
    big_grid = AirportGrid()
    print("\n--- A* tree on real 12x12 airport grid ---")
    print_search_tree(big_grid, (2, 2), (2, 9), max_levels=2)


def print_airport_board(grid, agents=None, title="AIRPORT GROUND-SERVICE COORDINATION"):
    """Compact text board for CLI visualization."""
    agents = agents or []
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)
    symbols = {0: ".", 1: "#", 2: "R", 3: "G", 4: "|"}
    occupied = {a.current_position: a.agent_id
                for a in agents
                if a.current_status.name not in {"DONE", "NO_PATH"}}
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            if (r, c) in occupied:
                vid = occupied[(r, c)]
                row.append(vid[:2])
            else:
                row.append(symbols.get(int(grid.grid[r, c]), "?"))
        print("  " + " ".join(row))
    print("-" * 62)
    if agents:
        for a in agents:
            gate = a.gate_name.replace("gate_", "Gate ").upper()
            stype = a.service_type.value if hasattr(a.service_type, "value") else str(a.service_type)
            print(f"  {a.agent_id:6} -> {gate:8} | {a.current_status.name:10} | "
                  f"P={a.priority} | {stype}")
    print("  R=Depot  G=Gate  |=Single-lane Taxiway  #=Wall")
    print("=" * 62)


def main():
    parser = argparse.ArgumentParser(
        description="Airport Ground-Crew Coordination Multi-Agent Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario",    default="1",
                        help="Scenario to run: 1-10, or 'all'")
    parser.add_argument("--algorithm",   default="cooperative_astar",
                        choices=ALGORITHMS + ["all"],
                        help="Search algorithm to use")
    parser.add_argument("--no-anim",     action="store_true",
                        help="Skip GIF animation (faster)")
    parser.add_argument("--layout",      action="store_true",
                        help="Save airport layout image and exit")
    parser.add_argument("--compare",     action="store_true",
                        help="Run scenario 2 with A* and Greedy Best-First and compare")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress verbose per-step output")
    parser.add_argument("--gui",         action="store_true",
                        help="Launch the desktop airport GUI")
    parser.add_argument("--play",        action="store_true",
                        help="Launch the desktop airport GUI (alias for --gui)")

    args = parser.parse_args()

    # No arguments -> open GUI
    if len(sys.argv) == 1:
        launch_gui()
        return

    os.makedirs("output", exist_ok=True)

    if args.gui or args.play:
        launch_gui()
        return

    all_results: list[ScenarioResult] = []

    # ── Layout-only mode ─────────────────────────────────────────────────────
    if args.layout:
        print("[main] Drawing airport layout ...")
        draw_grid_static(AirportGrid(), title="Airport Ground Layout (12x12)")
        return


    # ── Algorithm comparison mode ────────────────────────────────────────────
    if args.compare:
        print("\n[main] COMPARISON MODE: Running scenario 2 (Corridor Conflict) "
              "with A* and Greedy Best-First Search ...")
        for alg in ALGORITHMS:
            r = run_scenario("2", alg, save_anim=not args.no_anim,
                             verbose=not args.quiet)
            all_results.append(r)
        print_metrics_table(all_results)
        save_metrics_csv(all_results, "comparison_metrics.csv")
        generate_all_plots(all_results)
        return

    # ── Determine which scenarios to run ────────────────────────────────────
    if args.scenario.lower() == "all":
        scenario_ids = list(SCENARIO_REGISTRY.keys())
    else:
        scenario_ids = [args.scenario]

    # ── Determine which algorithms ──────────────────────────────────────────
    if args.algorithm == "all":
        algorithms = ALGORITHMS
    else:
        algorithms = [args.algorithm]

    # ── Run ─────────────────────────────────────────────────────────────────
    for sid in scenario_ids:
        if sid not in SCENARIO_REGISTRY:
            print(f"[ERROR] Unknown scenario '{sid}'. Choose 1-5 or 'all'.")
            sys.exit(1)
        for alg in algorithms:
            r = run_scenario(sid, alg,
                             save_anim=not args.no_anim,
                             verbose=not args.quiet)
            all_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    if all_results:
        print_metrics_table(all_results)
        save_metrics_csv(all_results)
        generate_all_plots(all_results)

    print("\n[main] Done. Results saved in output/ directory.")
    print("       Run with --compare    to compare A* and Greedy Best-First Search on the corridor scenario.")


if __name__ == "__main__":
    main()