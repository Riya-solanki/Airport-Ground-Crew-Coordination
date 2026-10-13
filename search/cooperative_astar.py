"""
search/cooperative_astar.py
============================
Cooperative A* with Prioritized Planning in (x, y, t) space-time.

ALGORITHM OVERVIEW:
  Cooperative A* (Silver 2005) extends standard A* by adding the time
  dimension to the state. The key insight is:

    Two vehicles conflict iff they occupy the same (r, c) at the same t.

  By planning vehicles ONE AT A TIME in priority order and recording each
  vehicle's (r, c, t) reservations in a shared table, later vehicles treat
  reserved cells as obstacles in space-time and route around them.

WHY SPACE-TIME:
  A 2-D grid cannot represent "vehicle A is here at t=3 but gone by t=5".
  Adding t as a 3rd dimension makes conflicts explicit: two states
  (r, c, t) and (r', c', t') conflict iff (r,c)=(r',c') AND t=t'.

RESERVATION TABLE:
  A set of (r, c, t) triples.  When vehicle v's path visits cell (r, c) at
  time step t, the planner inserts (r, c, t) BEFORE planning the next vehicle.
  Later vehicles see these as forbidden states.

EDGE CONFLICTS (swap conflicts):
  Vehicle A at (r1,c1)→(r2,c2) and B at (r2,c2)→(r1,c1) simultaneously
  causes a head-on collision even if vertex occupancy doesn't overlap by
  time-step.  We capture this by also reserving the EDGE (r,c,t)→(r',c',t+1).

PRIORITY ORDER:
  Lower priority number = plans first = takes preferred routes.
  Emergency vehicles are priority 0 — they always plan first.
  This is a greedy heuristic: globally optimal cooperative planning is NP-hard
  (equivalent to multi-agent path finding, MAPF).

WAIT ACTION:
  A vehicle can stay in place (r, c, t) → (r, c, t+1) with cost 1.
  This models a vehicle pausing in a taxiway to let a higher-priority vehicle
  pass through a corridor.  The wait action enables coordinated space-time routing.
"""

from __future__ import annotations
import heapq
from environment import AirportGrid
from search.utils import manhattan


# Type alias for clarity
Cell     = tuple[int, int]
SpaceTimeState = tuple[int, int, int]   # (row, col, time)

# Maximum time horizon: avoids infinite loops if no solution possible within T
MAX_TIME = 80   # Max time steps: 12x12 grid has ~144 cells, 80 steps is generous


class ReservationTable:
    """
    Shared (r, c, t) reservation table — the 'cooperation memory' of the system.

    WHY A SET (not a dict):
      We only need to know WHICH (r,c,t) are forbidden, not by whom.
      A set gives O(1) average lookup and O(1) insert, both critical since
      A* checks reservations for EVERY node it expands.

    EDGE CONFLICT STORAGE:
      We store directed edges as frozensets {(r,c,t), (r',c',t+1)} so that
      the swap conflict check is symmetric and cheap.
    """

    def __init__(self, corridor_cells=None):
        self.vertex_reservations: set[SpaceTimeState] = set()
        # Corridor resources are connected components of yellow CORRIDOR cells.
        # A resource is reserved at each time step that a vehicle occupies it.
        self.corridor_component_by_cell: dict[Cell, int] = {}
        self.corridor_reservations: set[tuple[int, int]] = set()
        self.configure_corridors(corridor_cells or set())
        # Edge reservations: directed moves (r, c, nr, nc, t) meaning "a
        # vehicle moves from (r,c) to (nr,nc) departing at time t (arriving
        # at t+1)". A swap conflict is any OTHER vehicle wanting to move
        # (nr,nc) -> (r,c) at that SAME t -- the reverse move, same t.
        #
        # WHY NOT frozenset({(r,c,t), (nr,nc,t+1)}):
        #   That representation bakes a different timestamp into each
        #   endpoint, so the forward edge frozenset({(4,7,2),(5,7,3)}) and
        #   the reverse edge frozenset({(5,7,2),(4,7,3)}) share no elements
        #   and can never compare equal -- so a genuine swap is never
        #   caught. Storing the directed move keyed by its departure time
        #   makes the reverse-at-same-t check exact.
        self.edge_reservations: set[tuple[int,int,int,int,int]] = set()

    def configure_corridors(self, corridor_cells):
        """Group orthogonally connected corridor cells into exclusive resources."""
        cells = set(corridor_cells or ())
        self.corridor_component_by_cell.clear()
        component = 0
        while cells:
            component += 1
            seed = cells.pop()
            stack = [seed]
            self.corridor_component_by_cell[seed] = component
            while stack:
                r, c = stack.pop()
                for nxt in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                    if nxt in cells:
                        cells.remove(nxt)
                        self.corridor_component_by_cell[nxt] = component
                        stack.append(nxt)

    def is_corridor_free(self, r: int, c: int, t: int) -> bool:
        """A corridor is exclusive at a time step, even across different cells."""
        component = self.corridor_component_by_cell.get((r, c))
        return component is None or (component, t) not in self.corridor_reservations

    def reserve_path(self, path: list[Cell], start_time: int = 0, reserve_duration: int = 0):
        """
        Record all (r, c, t) cells for a planned path starting at start_time.
        If reserve_duration > 0, the final cell (goal) is reserved for [final_t, final_t + reserve_duration].
        If reserve_duration == 0, it is reserved until MAX_TIME (e.g. permanent depot parking).
        """
        for i, (r, c) in enumerate(path):
            t = start_time + i
            self.vertex_reservations.add((r, c, t))
            component = self.corridor_component_by_cell.get((r, c))
            if component is not None:
                self.corridor_reservations.add((component, t))
            # Reserve directed edge: (r,c) -> (next cell), departing at t
            if i + 1 < len(path):
                nr, nc = path[i + 1]
                self.edge_reservations.add((r, c, nr, nc, t))

        # Goal reservation
        if path:
            gr, gc = path[-1]
            final_t = start_time + len(path) - 1
            if reserve_duration > 0:
                end_t = min(MAX_TIME, final_t + reserve_duration + 1)
                for t in range(final_t + 1, end_t):
                    self.vertex_reservations.add((gr, gc, t))
                    component = self.corridor_component_by_cell.get((gr, gc))
                    if component is not None:
                        self.corridor_reservations.add((component, t))
            else:
                for extra in range(1, MAX_TIME - final_t):
                    t = final_t + extra
                    self.vertex_reservations.add((gr, gc, t))
                    component = self.corridor_component_by_cell.get((gr, gc))
                    if component is not None:
                        self.corridor_reservations.add((component, t))

    def is_vertex_free(self, r: int, c: int, t: int) -> bool:
        return (r, c, t) not in self.vertex_reservations

    def is_edge_free(self, r1: int, c1: int, t1: int,
                     r2: int, c2: int, t2: int) -> bool:
        """
        Check for a swap (head-on) conflict for the move (r1,c1)@t1 ->
        (r2,c2)@t2. Blocked iff some other vehicle already reserved the
        REVERSE move (r2,c2) -> (r1,c1) departing at the SAME time t1.
        """
        return (r2, c2, r1, c1, t1) not in self.edge_reservations

    def is_stay_free(self, r: int, c: int, t_start: int, duration: int) -> bool:
        """Check if cell (r,c) is free for all ticks in [t_start, t_start + duration]."""
        for t in range(t_start, t_start + duration + 1):
            if (r, c, t) in self.vertex_reservations:
                return False
        return True

    def clear(self):
        self.vertex_reservations.clear()
        self.edge_reservations.clear()
        self.corridor_reservations.clear()



def cooperative_astar(
        grid:    AirportGrid,
        start:   Cell,
        goal:    Cell,
        table:   ReservationTable,
        t_start: int = 0,
        *,
        blocked_cells: set[Cell] | None = None,
        dwell_time: int = 0,
        trace_search: bool = False
) -> tuple[list[Cell], list[SpaceTimeState], int] | tuple[list[Cell], list[SpaceTimeState], int, list[dict]]:
    """
    A* in (row, col, time) space-time, respecting the shared reservation table.

    Parameters
    ----------
    dwell_time: int
        Number of additional time steps the vehicle must remain at the goal cell
        during service. The search will ONLY accept arriving at the goal at time `t`
        if the goal cell is completely free in the reservation table for all ticks
        [t, t + dwell_time].
    trace_search: bool
        If True, returns a 4-tuple including `search_trace`, a list of node inspection dicts
        {"state": (r,c,t), "g": g, "h": h, "f": f, "parent": parent_state} for the UI Search Inspector.
    """
    if blocked_cells is None:
        blocked_cells = set()

    # Cheap 2D reachability check prevents space-time expansion explosion
    from search.utils import is_reachable_2d, compute_true_distance_map
    if not is_reachable_2d(grid, start, goal, blocked_cells):
        return ([], [], 0, []) if trace_search else ([], [], 0)

    sr, sc = start
    gr, gc = goal
    # True-distance heuristic accounts for static obstacles
    dist_map = compute_true_distance_map(grid, goal, blocked_cells)
    h0 = dist_map.get(start, manhattan(sr, sc, gr, gc))

    # Priority queue: (f, g, (r, c, t))
    # f = g + h,  g = steps taken so far (= time elapsed since t_start)
    open_list: list[tuple[int, int, SpaceTimeState]] = [
        (h0, 0, (sr, sc, t_start))
    ]
    came_from: dict[SpaceTimeState, SpaceTimeState | None] = {
        (sr, sc, t_start): None
    }
    g_cost:    dict[SpaceTimeState, int] = {(sr, sc, t_start): 0}
    nodes_expanded = 0
    search_trace: list[dict] = []

    while open_list:
        f, g, state = heapq.heappop(open_list)
        nodes_expanded += 1

        r, c, t = state
        h = f - g

        if trace_search and len(search_trace) < 300:
            search_trace.append({
                "state": state,
                "pos": (r, c),
                "time": t,
                "g": g,
                "h": h,
                "f": f,
                "parent": came_from.get(state)
            })

        # Goal check: reached goal cell and goal remains free for entire dwell time!
        if (r, c) == (gr, gc):
            if dwell_time <= 0 or table.is_stay_free(gr, gc, t, dwell_time):
                path_2d, st_path = _reconstruct(came_from, state)
                if trace_search:
                    return path_2d, st_path, nodes_expanded, search_trace
                return path_2d, st_path, nodes_expanded

        if t >= MAX_TIME:
            continue    # Prune: time horizon exceeded

        # ── Generate successors ─────────────────────────────────────────
        # Actions: move in 4 directions + wait in place
        neighbors_xt = grid.neighbors(r, c) + [(r, c)]   # +wait action

        for nr, nc in neighbors_xt:
            if (nr, nc) in blocked_cells:
                continue

            nt     = t + 1
            new_g  = g + 1

            # Reservation table checks (cooperative constraint):
            if not table.is_vertex_free(nr, nc, nt):
                continue   # Another vehicle occupies this cell at this time
            if not table.is_corridor_free(nr, nc, nt):
                continue   # Another vehicle has exclusive use of this corridor
            if not table.is_edge_free(r, c, t, nr, nc, nt):
                continue   # Swap conflict: head-on in corridor

            new_state = (nr, nc, nt)
            if new_state in g_cost and g_cost[new_state] <= new_g:
                continue

            g_cost[new_state]    = new_g
            came_from[new_state] = state
            child_h = dist_map.get((nr, nc), manhattan(nr, nc, gr, gc))
            nf = new_g + child_h
            heapq.heappush(open_list, (nf, new_g, new_state))

    if trace_search:
        return [], [], nodes_expanded, search_trace
    return [], [], nodes_expanded   # No path found within time horizon




def _reconstruct(
        came_from: dict[SpaceTimeState, SpaceTimeState | None],
        goal_state: SpaceTimeState
) -> tuple[list[Cell], list[SpaceTimeState]]:
    """Reconstruct 2-D path and space-time path from parent pointers."""
    st_path: list[SpaceTimeState] = []
    node: SpaceTimeState | None   = goal_state

    while node is not None:
        st_path.append(node)
        node = came_from.get(node)

    st_path.reverse()
    path_2d = [(r, c) for r, c, _ in st_path]
    return path_2d, st_path


def plan_all_vehicles(
        grid:     AirportGrid,
        vehicles: list,
        table:    ReservationTable,
        t_start:  int = 0,
        *,
        blocked_cells: set[Cell] | None = None
) -> None:
    """
    Plan paths for ALL vehicles using Cooperative A* (prioritized planning).

    Vehicles are sorted by priority (ascending = highest first) and planned
    one at a time.  Each vehicle's path is immediately reserved so subsequent
    vehicles see it as forbidden space-time.

    WHY PRIORITY ORDER MATTERS:
      The order determines who gets the 'preferred' short path.  A vehicle
      planned later may need to wait or detour.  Emergency vehicles (priority 0)
      always plan first, minimising their travel time at the expense of others.
    """
    import time as _time

    # Sort vehicles/agents: lower priority number = plans first
    ordered = sorted(vehicles, key=lambda v: v.priority)

    for vehicle in ordered:
        gate_nm  = getattr(vehicle, 'gate_name',  getattr(vehicle, 'gate_name', ''))
        pos      = getattr(vehicle, 'current_position', getattr(vehicle, 'pos', (0, 0)))
        gate_pos = getattr(vehicle, 'gate_pos', (0, 0))
        leg      = getattr(vehicle, 'active_leg_to_gate', True)

        if not grid.gate_available.get(gate_nm, True):
            from agents import VehicleStatus
            vehicle.current_status = VehicleStatus.NO_PATH
            continue

        dwell = getattr(vehicle, 'service_duration', getattr(vehicle, 'service_time', 3)) if leg else 0
        t0 = _time.perf_counter()
        path_2d, st_path, expanded = cooperative_astar(
            grid, pos, gate_pos, table,
            t_start=t_start, blocked_cells=blocked_cells,
            dwell_time=dwell
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        vehicle.metrics.nodes_expanded += expanded
        vehicle.metrics.runtime_ms     += elapsed_ms

        vid = getattr(vehicle, 'agent_id', getattr(vehicle, 'vid', '?'))
        if not path_2d:
            from agents import VehicleStatus
            vehicle.current_status = VehicleStatus.NO_PATH
            print(f"  [WARN] {vid}: No path found to {gate_nm}!")
        else:
            vehicle.set_path(path_2d[1:], st_path)   # Exclude start cell
            table.reserve_path(path_2d, t_start, reserve_duration=dwell)