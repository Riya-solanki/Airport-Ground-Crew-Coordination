"""
environment.py
==============
Defines the Airport Grid Environment for the multi-agent simulation.

WHY THIS DESIGN:
- A 12x12 discrete grid is used so positions are integers, making A* search
  tractable and the state space finite (a requirement for completeness).
- Walls model buildings/immovable structures; the agent cannot enter them.
- 'Narrow corridor' cells are annotated separately so we can demonstrate the
  capacity-1 constraint (only one vehicle at a time) that causes conflicts.
- Gates and Depots are named landmarks; agents store their goal as a gate ID,
  allowing the scenario to update gate availability dynamically (Scenario 3 & 4).
"""

import numpy as np

# Cell type constants — using integers keeps grid arrays compact (numpy int8).
EMPTY    = 0
WALL     = 1
DEPOT    = 2
GATE     = 3
CORRIDOR = 4   # Narrow single-lane taxiway segment


class AirportGrid:
    """
    Represents the fully-observable, static topology of the airport.

    The grid is stored as a 2-D numpy array of cell-type codes.
    Dynamic state (vehicle positions, reservations) lives in the agents and
    the reservation table — NOT here — so this object is shareable and
    re-usable across search instances without mutation.
    """

    def __init__(self, rows: int = 12, cols: int = 12):
        self.rows = rows
        self.cols = cols
        # Primary grid: cell types
        self.grid = np.zeros((rows, cols), dtype=np.int8)

        # Named locations — populated by _build_layout()
        self.depots: dict[str, tuple[int,int]] = {}   # name -> (r, c)
        self.gates:  dict[str, tuple[int,int]] = {}   # name -> (r, c)
        self.corridor_cells: set[tuple[int,int]] = set()

        # Runtime flag: which gates are currently available
        self.gate_available: dict[str, bool] = {}

        self._build_layout()

    @classmethod
    def from_array(cls, cells) -> "AirportGrid":
        """
        Build a grid of ANY size from an explicit 2-D array of cell codes,
        bypassing __init__ (which always builds the hand-crafted 12x12 layout).

        WHY: the constructor's _build_layout() hardcodes coordinates such as
        row/col 11, so it cannot produce smaller grids. Tiny grids are needed
        for the search-tree demo, where the whole tree must fit in a report.

        All attributes that the rest of the code reads are initialised here,
        so the returned object works with astar(), print_search_tree(), etc.
        """
        arr = np.array(cells, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("cells must be a 2-D array of cell-type codes")

        obj = cls.__new__(cls)          # skip __init__ / _build_layout
        obj.rows, obj.cols = arr.shape
        obj.grid = arr
        obj.depots = {}
        obj.gates = {}
        obj.corridor_cells = {
            (int(r), int(c)) for r, c in zip(*np.where(arr == CORRIDOR))
        }
        obj.gate_available = {}
        return obj

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        """
        Hand-craft a 12×12 airport topology.

        Layout overview (row 0 = top):
          - Top-left quadrant: depot cluster
          - Right side: gates A–D
          - Two narrow corridors connect the depot area to the gates
          - Buildings / walls scattered to force routing choices
        """
        g = self.grid

        # ── Outer boundary walls (makes the search space bounded) ──────
        g[0, :]  = WALL
        g[11, :] = WALL
        g[:, 0]  = WALL
        g[:, 11] = WALL

        # ── Interior building blocks (force detours) ────────────────────
        # Central building block
        for r in range(4, 8):
            for c in range(4, 7):
                g[r, c] = WALL

        # Small obstacle near top-right
        for r in range(2, 4):
            for c in range(8, 10):
                g[r, c] = WALL

        # Small obstacle near bottom-left
        for r in range(8, 10):
            for c in range(2, 4):
                g[r, c] = WALL

        # ── Narrow single-lane corridors ────────────────────────────────
        # WHY corridors: they create capacity-1 bottlenecks that force
        # Cooperative A* to stagger vehicle entry times, demonstrating
        # the reservation table's power over independent planning.
        corridor_cells = [
            (3, 7), (4, 7), (5, 7),   # Vertical corridor — center-right
            (8, 5), (8, 6), (8, 7),   # Horizontal corridor — lower-mid
        ]
        for (r, c) in corridor_cells:
            if g[r, c] == EMPTY:       # Don't overwrite walls
                g[r, c] = CORRIDOR
                self.corridor_cells.add((r, c))

        # ── Depots (vehicle starting positions) ─────────────────────────
        depot_positions = {
            "fuel_depot":       (2, 2),
            "baggage_depot":    (2, 3),
            "catering_depot":   (3, 2),
            "maintenance_depot":(3, 3),
        }
        for name, (r, c) in depot_positions.items():
            g[r, c] = DEPOT
            self.depots[name] = (r, c)

        # ── Gates (aircraft service positions on the right side) ─────────
        gate_positions = {
            "gate_A": (2, 9),
            "gate_B": (5, 9),
            "gate_C": (8, 9),
            "gate_D": (9, 9),
        }
        for name, (r, c) in gate_positions.items():
            g[r, c] = GATE
            self.gates[name] = (r, c)
            self.gate_available[name] = True   # Default: all gates open

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def is_passable(self, r: int, c: int) -> bool:
        """Return True if a vehicle can enter cell (r, c)."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.grid[r, c] != WALL

    def is_corridor(self, r: int, c: int) -> bool:
        return (r, c) in self.corridor_cells

    def neighbors(self, r: int, c: int) -> list[tuple[int,int]]:
        """
        Return passable orthogonal neighbors of (r, c).
        The branching factor is at most 4 (move) + 1 (wait-in-place) = 5,
        but wait is handled by the search algorithm, not here.
        """
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        result = []
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if self.is_passable(nr, nc):
                result.append((nr, nc))
        return result

    def set_gate_available(self, gate_name: str, available: bool):
        """Dynamically toggle a gate — used in Scenarios 3 & 4."""
        self.gate_available[gate_name] = available

    def get_gate_position(self, gate_name: str) -> tuple[int,int] | None:
        return self.gates.get(gate_name)

    def __repr__(self):
        symbols = {EMPTY: '.', WALL: '#', DEPOT: 'R', GATE: 'G', CORRIDOR: '|'}
        lines = []
        for row in self.grid:
            lines.append(' '.join(symbols.get(c, '?') for c in row))
        return '\n'.join(lines)