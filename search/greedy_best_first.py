"""Greedy Best-First Search in airport space-time (r, c, t).

Uses h(n) alone to order the frontier. The g/h/f values are still recorded
for inspection, but f=g+h does NOT determine expansion order in this algorithm.
Reservation and edge-swap checks are shared with the A* implementation so both
algorithms solve the same cooperative routing problem.
"""
from __future__ import annotations
import heapq
import time
from environment import AirportGrid
from search.cooperative_astar import ReservationTable, MAX_TIME, _reconstruct
from search.utils import manhattan, is_reachable_2d, compute_true_distance_map

Cell = tuple[int, int]
SpaceTimeState = tuple[int, int, int]


def greedy_best_first(
    grid: AirportGrid,
    start: Cell,
    goal: Cell,
    table: ReservationTable,
    t_start: int = 0,
    *,
    blocked_cells: set[Cell] | None = None,
    dwell_time: int = 0,
    trace_search: bool = False,
):
    """Return (path, space-time path, expanded count[, trace]).

    Frontier priority is h(n), not f(n). Ties are resolved by insertion order
    through a monotonically increasing counter for deterministic behavior.
    """
    if blocked_cells is None:
        blocked_cells = set()
    if not is_reachable_2d(grid, start, goal, blocked_cells):
        return ([], [], 0, []) if trace_search else ([], [], 0)

    gr, gc = goal
    dist_map = compute_true_distance_map(grid, goal, blocked_cells)
    h_start = dist_map.get(start, manhattan(start[0], start[1], gr, gc))
    start_state = (start[0], start[1], t_start)

    counter = 0
    # Greedy Best-First priority is h only. g is tracked for reporting/path cost.
    frontier: list[tuple[int, int, SpaceTimeState]] = [(h_start, counter, start_state)]
    came_from: dict[SpaceTimeState, SpaceTimeState | None] = {start_state: None}
    g_cost: dict[SpaceTimeState, int] = {start_state: 0}
    expanded = 0
    trace: list[dict] = []

    while frontier:
        _, _, state = heapq.heappop(frontier)
        r, c, t = state
        g = g_cost[state]
        h = dist_map.get((r, c), manhattan(r, c, gr, gc))
        f = g + h
        expanded += 1
        if trace_search and len(trace) < 500:
            trace.append({"state": state, "pos": (r, c), "time": t,
                          "g": g, "h": h, "f": f,
                          "parent": came_from.get(state),
                          "priority": h, "algorithm": "greedy_best_first"})

        if (r, c) == (gr, gc):
            if dwell_time <= 0 or table.is_stay_free(gr, gc, t, dwell_time):
                path, st_path = _reconstruct(came_from, state)
                return (path, st_path, expanded, trace) if trace_search else (path, st_path, expanded)

        if t >= MAX_TIME:
            continue

        for nr, nc in grid.neighbors(r, c) + [(r, c)]:  # four moves + WAIT
            if (nr, nc) in blocked_cells:
                continue
            nt = t + 1
            if not table.is_vertex_free(nr, nc, nt):
                continue
            if not table.is_corridor_free(nr, nc, nt):
                continue  # Entire narrow corridor is reserved by another vehicle
            if not table.is_edge_free(r, c, t, nr, nc, nt):
                continue
            new_state = (nr, nc, nt)
            new_g = g + 1
            if new_state in g_cost and g_cost[new_state] <= new_g:
                continue
            g_cost[new_state] = new_g
            came_from[new_state] = state
            child_h = dist_map.get((nr, nc), manhattan(nr, nc, gr, gc))
            counter += 1
            heapq.heappush(frontier, (child_h, counter, new_state))

    return ([], [], expanded, trace) if trace_search else ([], [], expanded)


def plan_all_vehicles_greedy(
    grid: AirportGrid,
    vehicles: list,
    table: ReservationTable,
    t_start: int = 0,
    *,
    blocked_cells: set[Cell] | None = None,
) -> None:
    """Prioritized multi-agent planning using Greedy Best-First Search."""
    from agents import VehicleStatus
    ordered = sorted(vehicles, key=lambda v: v.priority)
    for vehicle in ordered:
        gate_nm = getattr(vehicle, "gate_name", "")
        pos = getattr(vehicle, "current_position", getattr(vehicle, "pos", (0, 0)))
        gate_pos = getattr(vehicle, "gate_pos", (0, 0))
        leg = getattr(vehicle, "active_leg_to_gate", True)
        if not grid.gate_available.get(gate_nm, True):
            vehicle.current_status = VehicleStatus.NO_PATH
            continue
        dwell = getattr(vehicle, "service_duration", getattr(vehicle, "service_time", 3)) if leg else 0
        t0 = time.perf_counter()
        path, st_path, expanded = greedy_best_first(
            grid, pos, gate_pos, table, t_start=t_start,
            blocked_cells=blocked_cells, dwell_time=dwell)
        vehicle.metrics.nodes_expanded += expanded
        vehicle.metrics.runtime_ms += (time.perf_counter() - t0) * 1000
        if not path:
            vehicle.current_status = VehicleStatus.NO_PATH
        else:
            vehicle.set_path(path[1:], st_path)
            table.reserve_path(path, t_start, reserve_duration=dwell)
