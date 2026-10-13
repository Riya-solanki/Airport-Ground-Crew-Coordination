"""
simulation.py
=============
The discrete-time simulation loop that drives all ground-service agents,
detects conflicts, fires scenario events, and captures animation frames.

SIMULATION LOOP (per tick):
  1. CoordinatorAgent arbitrates requests (sends messages to agents).
  2. Fire any scheduled events (gate closure, emergency trigger).
  3. Step all agents (perceive -> decide -> act).
  4. Detect vertex conflicts (two agents at same cell).
  5. Detect swap conflicts (two agents trading positions).
  6. Capture visualizer frame.
  7. If all agents DONE or NO_PATH, stop.

CONFLICT DETECTION:
  A* and Greedy Best-First use the reservation table to coordinate movement.

REPLANNING:
  When a scenario event fires (e.g., gate closure), affected agents are set to IDLE.
  The simulation replans using the selected algorithm from current positions.
  The reservation table is also rebuilt to avoid stale entries.

FAILURE REASONS:
  NO_PATH               - 2D reachability check failed (static wall blocks all routes)
  TIME_HORIZON_EXCEEDED - route is reachable but space-time is saturated within MAX_TIME
"""

from __future__ import annotations
import time as _time
import numpy as np
from collections import defaultdict

from environment import AirportGrid
from agents import (GroundServiceAgent, VehicleStatus, CoordinatorAgent,
                    MessageType, CoordinationMessage)
from search.cooperative_astar import ReservationTable, plan_all_vehicles, cooperative_astar, MAX_TIME
from search.utils import is_reachable_2d
from search.greedy_best_first import greedy_best_first, plan_all_vehicles_greedy
from visualize import SimulationVisualizer
from metrics import ScenarioResult, collect_vehicle_metrics

MAX_STEPS = 120   # Safety cap: simulation terminates after this many ticks

# Alias for backward compatibility (Vehicle = GroundServiceAgent)
Vehicle = GroundServiceAgent


def detect_conflicts(agents: list[GroundServiceAgent]) -> int:
    """
    Count conflicts in the CURRENT time step.
    Vertex conflict: two agents at the same (r, c).
    Returns the total number of conflicting agent pairs.
    """
    pos_map: dict[tuple[int, int], list[str]] = defaultdict(list)
    for a in agents:
        if a.current_status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH):
            pos_map[a.current_position].append(a.agent_id)

    conflicts = sum(len(ids) - 1 for ids in pos_map.values() if len(ids) > 1)
    return conflicts


def _replan_affected(grid: AirportGrid, agents: list[GroundServiceAgent],
                     table: ReservationTable, current_time: int,
                     algorithm: str = "cooperative_astar"):
    """
    Replan all agents currently in IDLE state (set by replan()).
    Rebuilds the reservation table from scratch (cooperative, priority-ordered).
    """
    table.clear()
    # Re-reserve paths for agents already en-route or with valid paths
    for a in agents:
        if a.current_status not in (VehicleStatus.IDLE, VehicleStatus.DONE, VehicleStatus.NO_PATH):
            if a.spacetime_path:
                future = [(r, c, t) for r, c, t in a.spacetime_path if t >= current_time]
                for r, c, t in future:
                    table.vertex_reservations.add((r, c, t))
            elif a.route and a.path_index < len(a.route):
                rem = a.route[a.path_index:]
                table.reserve_path([a.current_position] + rem, start_time=current_time)

    # Plan the IDLE (replanning) agents
    idle_agents = [a for a in agents if a.current_status == VehicleStatus.IDLE]
    if algorithm == "greedy_best_first":
        plan_all_vehicles_greedy(grid, idle_agents, table, t_start=current_time)
    else:
        plan_all_vehicles(grid, idle_agents, table, t_start=current_time)

    # Check for time-horizon vs unreachable
    for a in idle_agents:
        if a.current_status == VehicleStatus.NO_PATH:
            if is_reachable_2d(grid, a.current_position, a.gate_pos):
                a.failure_reason = "TIME_HORIZON_EXCEEDED"
            else:
                a.failure_reason = "UNREACHABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
        grid:           AirportGrid,
        agents:         list[GroundServiceAgent],
        events:         dict,
        scenario_name:  str,
        algorithm:      str = "cooperative_astar",
        *,
        save_animation: bool = True,
        verbose:        bool = True,
        # Backward compatibility: also accept 'vehicles' kwarg
        vehicles:       list[GroundServiceAgent] | None = None,
) -> ScenarioResult:
    """
    Run one full simulation and return collected metrics.

    Parameters
    ----------
    algorithm : "cooperative_astar" | "greedy_best_first"
    """
    # Support both 'agents' and 'vehicles' parameter names
    if vehicles is not None:
        agents = vehicles

    event_log: list[str] = []
    table      = ReservationTable(grid.corridor_cells)
    coordinator = CoordinatorAgent()

    # Attach failure_reason attribute to agents
    for a in agents:
        if not hasattr(a, "failure_reason"):
            a.failure_reason = None

    if verbose:
        print(f"\n{'-'*60}")
        print(f"  Scenario  : {scenario_name}")
        print(f"  Algorithm : {algorithm}")
        print(f"  Agents    : {[a.agent_id for a in agents]}")
        print(f"{'-'*60}")

    # ── Initial planning ────────────────────────────────────────────────────
    t0_plan = _time.perf_counter()

    if algorithm == "cooperative_astar":
        plan_all_vehicles(grid, agents, table, t_start=0)

    elif algorithm == "greedy_best_first":
        plan_all_vehicles_greedy(grid, agents, table, t_start=0)

    # Post-plan: check for time-horizon vs unreachable on A*/Greedy Best-First
    for a in agents:
        if a.current_status == VehicleStatus.NO_PATH and a.failure_reason is None:
            if is_reachable_2d(grid, a.current_position, a.gate_pos):
                a.failure_reason = "TIME_HORIZON_EXCEEDED"
            else:
                a.failure_reason = "UNREACHABLE"

    plan_time_ms = (_time.perf_counter() - t0_plan) * 1000

    # ── CoordinatorAgent initial arbitration ─────────────────────────────────
    coordinator.arbitrate_requests(agents, current_time=0)

    # ── Visualizer setup ─────────────────────────────────────────────────────
    vis = SimulationVisualizer(grid, agents, scenario_name,
                               reservation_table=table)

    # ── Simulation tick loop ─────────────────────────────────────────────────
    prev_positions: dict[str, tuple[int, int]] = {a.agent_id: a.current_position for a in agents}
    total_conflicts = 0

    for step in range(MAX_STEPS):
        # 1. CoordinatorAgent arbitration at every step
        coordinator.arbitrate_requests(agents, current_time=step)

        # 2. Fire scheduled events
        if step in events:
            event_fn = events[step]
            event_fn(grid, agents, event_log)
            if verbose:
                msg = event_log[-1] if event_log else ''
                safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                print(f"  [t={step:03d}] EVENT fired: {safe_msg}")
            # Replan affected agents immediately
            if algorithm in ("cooperative_astar", "greedy_best_first"):
                _replan_affected(grid, agents, table, current_time=step, algorithm=algorithm)

        # 3. Step all agents using perceive -> decide -> act
        for a in agents:
            if a.current_status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH,
                                         VehicleStatus.IDLE):
                perception = a.perceive(grid, step)
                action     = a.decide(perception)
                a.act(action, step)

                # When agent reaches gate (outbound leg done), plan return path
                if a.current_status == VehicleStatus.RETURNING and not a.route:
                    _plan_return_leg(grid, a, table, step, algorithm)

                # Log service events to coordinator
                if a.current_status == VehicleStatus.SERVICING and a.service_done == 1:
                    coordinator.log_service_event(a, "started", step)
                    event_log.append(f"t={step}: {a.agent_id} SERVICE STARTED at {a.gate_name}")

        # 4. Detect vertex conflicts
        conflicts = detect_conflicts(agents)
        total_conflicts += conflicts
        for a in agents:
            if a.current_status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH):
                a.metrics.collisions += conflicts
        if conflicts > 0 and verbose:
            print(f"  [t={step:03d}] CONFLICT: {conflicts} pair(s)")
            event_log.append(f"t={step}: CONFLICT detected ({conflicts} pair(s))")

        # 5. Detect and log swap conflicts
        swap_confs = _detect_swap_conflicts(agents, prev_positions)
        if swap_confs > 0:
            total_conflicts += swap_confs
            event_log.append(f"t={step}: SWAP CONFLICT detected ({swap_confs} pairs)")
            if verbose:
                print(f"  [t={step:03d}] SWAP CONFLICT: {swap_confs} pairs")

        # 6. Capture frame
        res_snapshot = {(r, c, t) for r, c, t in table.vertex_reservations
                        if abs(t - step) <= 1}
        vis.capture_frame(
            step, event_log,
            vehicle_positions=[a.current_position for a in agents],
            reservation_snapshot=res_snapshot,
        )

        # 7. Log status periodically
        if verbose and step % 5 == 0:
            statuses = ", ".join(f"{a.agent_id}:{a.current_status.name[:3]}" for a in agents)
            print(f"  [t={step:03d}] {statuses}")

        prev_positions = {a.agent_id: a.current_position for a in agents}

        # 8. Check termination
        all_done = all(a.current_status in (VehicleStatus.DONE, VehicleStatus.NO_PATH)
                       for a in agents)
        if all_done:
            # Log failure reasons
            for a in agents:
                if a.current_status == VehicleStatus.NO_PATH:
                    reason = getattr(a, "failure_reason", "NO_PATH")
                    event_log.append(f"t={step}: {a.agent_id} FAILED ({reason})")
                    if verbose:
                        print(f"  [t={step:03d}] {a.agent_id} FAILED: {reason}")
            event_log.append(f"t={step}: All agents finished.")
            if verbose:
                print(f"  [t={step:03d}] All agents finished.")
            break

    # ── Finalize metrics ─────────────────────────────────────────────────────
    agg = collect_vehicle_metrics(agents)
    agg["num_conflicts"] = total_conflicts
    result = ScenarioResult(
        scenario  = scenario_name,
        algorithm = algorithm,
        **agg,
    )

    if verbose:
        print(f"  Makespan: {result.makespan}  | Conflicts: {result.num_conflicts}"
              f"  | Nodes: {result.nodes_expanded}  | RT: {result.runtime_ms:.1f}ms")

    # ── Save outputs ─────────────────────────────────────────────────────────
    if save_animation:
        vis.save_animation(f"{scenario_name}_{algorithm}")
    vis.save_final_frame(f"{scenario_name}_{algorithm}")
    vis.close()

    return result


def _plan_return_leg(grid: AirportGrid, agent: GroundServiceAgent,
                     table: ReservationTable, current_time: int,
                     algorithm: str):
    """Plan the return journey from gate back to depot."""
    if algorithm == "cooperative_astar":
        path_2d, st_path, expanded = cooperative_astar(
            grid, agent.current_position, agent.depot, table, t_start=current_time)
        agent.metrics.nodes_expanded += expanded
        if path_2d:
            agent.set_path(path_2d[1:], st_path)
            table.reserve_path(path_2d, current_time)
        else:
            agent.current_status = VehicleStatus.NO_PATH

    elif algorithm == "greedy_best_first":
        path_2d, st_path, expanded = greedy_best_first(
            grid, agent.current_position, agent.depot, table, t_start=current_time)
        agent.metrics.nodes_expanded += expanded
        if path_2d:
            agent.set_path(path_2d[1:], st_path)
            table.reserve_path(path_2d, current_time)
        else:
            agent.current_status = VehicleStatus.NO_PATH


def _detect_swap_conflicts(
        agents: list[GroundServiceAgent],
        prev_positions: dict[str, tuple[int, int]]
) -> int:
    """
    Count agent pairs that swapped positions in this step (swap conflict).
    A swap conflict: A was at P1->P2, B was at P2->P1 simultaneously.
    This is a head-on collision in a narrow corridor.
    """
    count = 0
    ids = [a.agent_id for a in agents
           if a.current_status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH)]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            a_prev = prev_positions.get(id_a)
            b_prev = prev_positions.get(id_b)
            a_curr = next((a.current_position for a in agents if a.agent_id == id_a), None)
            b_curr = next((a.current_position for a in agents if a.agent_id == id_b), None)
            if (a_prev and b_prev and a_curr and b_curr and
                    a_prev == b_curr and b_prev == a_curr and a_prev != b_prev):
                count += 1
    return count
