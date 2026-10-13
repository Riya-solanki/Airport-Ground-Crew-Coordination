"""
Five airport ground-service coordination scenarios used by the case-study simulation.

1. Normal Operation
2. Corridor Conflict
3. Gate Unavailable
4. Emergency Priority
5. Unreachable Goal
"""
from __future__ import annotations
from environment import AirportGrid
from agents import (
    GroundServiceAgent, FuelAgent, BaggageAgent, CateringAgent, MaintenanceAgent,
    Vehicle, ServiceType, VehicleStatus
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create typed ground-service agents
# ─────────────────────────────────────────────────────────────────────────────

def _fuel(agent_id, depot_key, gate, grid, priority=1, emergency=False):
    return FuelAgent(
        agent_id=agent_id,
        service_type=ServiceType.FUEL,
        depot=grid.depots[depot_key],
        gate_name=gate,
        gate_pos=grid.gates[gate],
        priority=priority,
        is_emergency=emergency,
    )


def _baggage(agent_id, depot_key, gate, grid, priority=2):
    return BaggageAgent(
        agent_id=agent_id,
        service_type=ServiceType.BAGGAGE,
        depot=grid.depots[depot_key],
        gate_name=gate,
        gate_pos=grid.gates[gate],
        priority=priority,
    )


def _catering(agent_id, depot_key, gate, grid, priority=3):
    return CateringAgent(
        agent_id=agent_id,
        service_type=ServiceType.CATERING,
        depot=grid.depots[depot_key],
        gate_name=gate,
        gate_pos=grid.gates[gate],
        priority=priority,
    )


def _maintenance(agent_id, depot_key, gate, grid, priority=4):
    return MaintenanceAgent(
        agent_id=agent_id,
        service_type=ServiceType.MAINTENANCE,
        depot=grid.depots[depot_key],
        gate_name=gate,
        gate_pos=grid.gates[gate],
        priority=priority,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Normal Operation
# ─────────────────────────────────────────────────────────────────────────────




def scenario_normal(grid: AirportGrid):
    """
    Nominal operation: 4 ground-service vehicles navigate to their assigned gates
    concurrently. Cooperative A* ensures conflict-free space-time routing.
    """
    agents = [
        _fuel       ("FUEL01", "fuel_depot",        "gate_A", grid, priority=1),
        _baggage    ("BAG01",  "baggage_depot",      "gate_B", grid, priority=2),
        _catering   ("CAT01",  "catering_depot",     "gate_C", grid, priority=3),
        _maintenance("MAIN01", "maintenance_depot",  "gate_D", grid, priority=4),
    ]
    return agents, {}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Shared Corridor Conflict
# ─────────────────────────────────────────────────────────────────────────────


def scenario_corridor_conflict(grid: AirportGrid):
    """
    Two vehicles request the same single-lane taxiway corridor from opposite ends.
    Greedy Best-First uses h(n) to guide expansion; A* uses g(n)+h(n). Both use reservations and waits.
    Prioritized planning and tactical holds support multi-agent coordination.
    """
    f1 = FuelAgent(
        agent_id="FUEL01", service_type=ServiceType.FUEL,
        depot=(2, 7), gate_name="corridor_south_end", gate_pos=(9, 7),
        priority=1
    )
    f2 = BaggageAgent(
        agent_id="BAG01", service_type=ServiceType.BAGGAGE,
        depot=(6, 7), gate_name="corridor_north_end", gate_pos=(1, 7),
        priority=2
    )
    return [f1, f2], {}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Shared Gate Contention
# ─────────────────────────────────────────────────────────────────────────────


def scenario_gate_unavailable(grid: AirportGrid):
    """
    At t=8, Gate B is blocked due to a maintenance hold.
    Affected vehicle BAG01 receives a GATE_UNAVAILABLE message from the coordinator
    and diverts to Gate D. The other vehicles continue normally.
    """
    agents = [
        _fuel       ("FUEL01", "fuel_depot",       "gate_A", grid, priority=1),
        _baggage    ("BAG01",  "baggage_depot",     "gate_B", grid, priority=2),
        _catering   ("CAT01",  "catering_depot",   "gate_C", grid, priority=3),
    ]

    def close_gate_b(g, fs, log):
        g.set_gate_available("gate_B", False)
        log.append("t=8 EVENT: Gate B blocked by maintenance hold. "
                   "CoordinatorAgent issuing GATE_UNAVAILABLE to BAG01.")
        for f in fs:
            if f.gate_name == "gate_B":
                f.gate_name, f.gate_pos = "gate_D", g.gates["gate_D"]
                f.replan()
                log.append(f"  {f.agent_id} diverted to Gate D by coordinator.")

    return agents, {8: close_gate_b}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5: Emergency Priority
# ─────────────────────────────────────────────────────────────────────────────


def scenario_emergency(grid: AirportGrid):
    """
    At t=5, FUEL01 is upgraded to emergency priority (medical refuel).
    CoordinatorAgent issues EMERGENCY clearance, bumps other vehicles,
    and replans the entire fleet.
    """
    agents = [
        _fuel       ("FUEL01", "fuel_depot",       "gate_B", grid, priority=2),
        _baggage    ("BAG01",  "baggage_depot",     "gate_C", grid, priority=3),
        _catering   ("CAT01",  "catering_depot",   "gate_D", grid, priority=4),
        _maintenance("MAIN01", "maintenance_depot", "gate_A", grid, priority=1),
    ]

    def emergency_event(g, fs, log):
        log.append("t=5 EVENT: FUEL01 declared emergency fuel mission. "
                   "CoordinatorAgent issuing EMERGENCY clearance.")
        for f in fs:
            if f.agent_id == "FUEL01":
                f.priority    = 0
                f.is_emergency= True
                log.append("  FUEL01 priority set to 0 (EMERGENCY).")
            else:
                f.priority += 1
                log.append(f"  {f.agent_id} priority bumped to {f.priority}.")
            f.replan()

    return agents, {5: emergency_event}


def scenario_unreachable(grid: AirportGrid):
    """
    Gate A is physically isolated by walls on all four sides.
    The 2D reachability check immediately proves the goal is unreachable
    without wasting space-time search effort.
    Result: NO_PATH (UNREACHABLE — static obstacle blocks all routes).
    """
    # Surround gate_A at (2,9) with walls on all 4 sides
    gate_r, gate_c = grid.gates["gate_A"]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = gate_r + dr, gate_c + dc
        if 0 <= r < grid.rows and 0 <= c < grid.cols:
            grid.grid[r, c] = 1   # WALL

    agents = [
        _fuel   ("FUEL01", "fuel_depot",    "gate_A", grid, priority=1),
        _baggage("BAG01",  "baggage_depot", "gate_B", grid, priority=2),
    ]
    return agents, {}


SCENARIO_REGISTRY = {
    "1": ("normal_operation",   scenario_normal),
    "2": ("corridor_conflict",  scenario_corridor_conflict),
    "3": ("gate_unavailable",   scenario_gate_unavailable),
    "4": ("emergency_priority", scenario_emergency),
    "5": ("unreachable_goal",   scenario_unreachable),
}

SCENARIO_DESCRIPTIONS = {
    "1": ("Normal Operation",
           "Nominal ground-service routing: FUEL01, BAG01, CAT01 and MAIN01 navigate to gates A-D concurrently."),
    "2": ("Corridor Conflict",
           "FUEL01 and BAG01 request the same single-lane taxiway corridor from opposing directions. "
           "Cooperative A* coordinates their movement using reservations and wait actions."),
    "3": ("Gate Unavailable",
           "At t=8, Gate B is blocked by maintenance. BAG01 is diverted to Gate D and the fleet replans."),
    "4": ("Emergency Priority",
           "At t=5, FUEL01 is declared an emergency mission. The coordinator raises its priority and replans the fleet."),
    "5": ("Unreachable Goal",
           "Gate A is physically isolated by walls. The static reachability check detects that the goal cannot be reached."),
}
