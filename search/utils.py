"""Shared grid-search helpers used by A* and Greedy Best-First Search."""
from __future__ import annotations
from collections import deque
from environment import AirportGrid

Cell = tuple[int, int]

def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1-r2) + abs(c1-c2)

def compute_true_distance_map(grid: AirportGrid, goal: Cell, blocked_cells: set[Cell] | None = None) -> dict[Cell, int]:
    if blocked_cells is None:
        blocked_cells = set()
    dist_map = {goal: 0}
    queue = deque([goal])
    while queue:
        r,c = queue.popleft()
        for nr,nc in grid.neighbors(r,c):
            if (nr,nc) in blocked_cells or (nr,nc) in dist_map:
                continue
            dist_map[(nr,nc)] = dist_map[(r,c)] + 1
            queue.append((nr,nc))
    return dist_map

def is_reachable_2d(grid: AirportGrid, start: Cell, goal: Cell, blocked_cells: set[Cell] | None = None) -> bool:
    if start == goal:
        return True
    if blocked_cells is None:
        blocked_cells = set()
    if start in blocked_cells or goal in blocked_cells:
        return False
    if not grid.is_passable(*start) or not grid.is_passable(*goal):
        return False
    queue = deque([start])
    seen = {start}
    while queue:
        r,c = queue.popleft()
        for nr,nc in grid.neighbors(r,c):
            if (nr,nc) in blocked_cells or (nr,nc) in seen:
                continue
            if (nr,nc) == goal:
                return True
            seen.add((nr,nc))
            queue.append((nr,nc))
    return False
