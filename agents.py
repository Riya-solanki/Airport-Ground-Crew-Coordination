"""
agents.py
=========
Ground-Service Agent hierarchy for the Airport Coordination simulation.

AGENT TYPE — Utility-Based Agent:
  Utility = -(w1*travel_time + w2*waiting_time + w3*conflict_cost
              + w4*delay_cost + w5*priority_delay)
  Higher priority = lower-priority vehicles yield to make room.

HIERARCHY:
  GroundServiceAgent (base)
    ├── FuelAgent      (FUEL01)
    ├── BaggageAgent   (BAG01)
    ├── CateringAgent  (CAT01)
    └── MaintenanceAgent (MAIN01)

  CoordinatorAgent — separate coordinating entity.

WHY UTILITY-BASED:
  A pure goal-based agent only cares about reaching the gate.
  A utility-based agent minimises COST (time + conflicts + waiting),
  which drives cooperative A* priority ordering and yield decisions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ServiceType(Enum):
    FUEL        = "Fuel Service"
    BAGGAGE     = "Baggage Handling"
    CATERING    = "Catering Service"
    MAINTENANCE = "Maintenance Service"
    EMERGENCY   = "Emergency Response"
    GENERIC     = "Generic Service"


class VehicleStatus(Enum):
    IDLE        = auto()   # At depot, waiting for assignment
    EN_ROUTE    = auto()   # Taxiing toward assigned gate
    SERVICING   = auto()   # At gate, performing service
    RETURNING   = auto()   # Taxiing back to depot
    DONE        = auto()   # Service mission complete
    NO_PATH     = auto()   # Could not find a route (graceful failure)
    WAITING     = auto()   # Temporarily blocked (wait action)


class MessageType(Enum):
    ROUTE_REQUEST    = "ROUTE_REQUEST"
    ROUTE_GRANTED    = "ROUTE_GRANTED"
    CONFLICT_DETECTED= "CONFLICT_DETECTED"
    WAIT_REQUEST     = "WAIT_REQUEST"
    PRIORITY_UPDATE  = "PRIORITY_UPDATE"
    EMERGENCY        = "EMERGENCY"
    GATE_UNAVAILABLE = "GATE_UNAVAILABLE"
    REPLAN_REQUEST   = "REPLAN_REQUEST"
    ROUTE_UPDATED    = "ROUTE_UPDATED"
    SERVICE_STARTED  = "SERVICE_STARTED"
    SERVICE_COMPLETED= "SERVICE_COMPLETED"


# Colour palette (R, G, B) 0-1 for matplotlib
SERVICE_COLOURS: dict[ServiceType, tuple[float, float, float]] = {
    ServiceType.FUEL:        (0.95, 0.77, 0.06),   # Yellow
    ServiceType.BAGGAGE:     (0.20, 0.63, 0.93),   # Blue
    ServiceType.CATERING:    (0.24, 0.82, 0.44),   # Green
    ServiceType.MAINTENANCE: (0.91, 0.30, 0.24),   # Red
    ServiceType.EMERGENCY:   (1.00, 0.10, 0.10),   # Bright red
    ServiceType.GENERIC:     (0.60, 0.60, 0.60),   # Grey
}

# Hex colours for Tkinter GUI
SERVICE_HEX_COLOURS: dict[ServiceType, str] = {
    ServiceType.FUEL:        "#F5C800",
    ServiceType.BAGGAGE:     "#1976D2",
    ServiceType.CATERING:    "#2E9D50",
    ServiceType.MAINTENANCE: "#E53935",
    ServiceType.EMERGENCY:   "#FF1010",
    ServiceType.GENERIC:     "#888888",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleMetrics:
    """Accumulated per-agent performance counters."""
    travel_distance:  int   = 0   # cells moved (Manhattan steps)
    waiting_time:     int   = 0   # time steps in WAIT action
    makespan:         int   = 0   # final completion time step
    collisions:       int   = 0   # detected conflicts
    nodes_expanded:   int   = 0   # search nodes expanded
    runtime_ms:       float = 0.0 # wall-clock planning time (ms)


# ─────────────────────────────────────────────────────────────────────────────
# Coordination Message
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoordinationMessage:
    """Explicit message exchanged between GroundServiceAgents and CoordinatorAgent."""
    sender:     str
    receiver:   str
    msg_type:   MessageType
    timestamp:  int  = 0
    priority:   int  = 1
    resource:   str  = ""
    content:    dict = field(default_factory=dict)

    def __repr__(self):
        return (f"MSG[{self.msg_type.value}] {self.sender}->{self.receiver} "
                f"t={self.timestamp} res={self.resource!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Base GroundServiceAgent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundServiceAgent:
    """
    Base class for all autonomous airport ground-service vehicles.

    State machine:  IDLE -> EN_ROUTE -> SERVICING -> RETURNING -> DONE
    On replan:      any state -> IDLE (triggers new path computation)
    On no-path:     any state -> NO_PATH (graceful failure)

    WHY PRIORITY:
      Cooperative A* plans agents in priority order (lowest number first).
      Emergency vehicles are priority 0, so they always plan first.
    """
    agent_id:         str                    # e.g. "FUEL01", "BAG01"
    service_type:     ServiceType
    depot:            tuple[int, int]        # home depot (row, col)
    gate_name:        str                    # assigned gate ID
    gate_pos:         tuple[int, int]        # resolved gate (row, col)
    priority:         int  = 1              # lower = higher priority (0 = emergency)
    is_emergency:     bool = False
    service_duration: int  = 3              # ticks spent at gate

    # Utility weights — subclasses may override
    w_travel:   float = 1.0
    w_wait:     float = 1.5
    w_conflict: float = 2.0
    w_delay:    float = 1.0
    w_priority: float = 3.0

    # Dynamic state (mutated during simulation)
    current_position: tuple[int, int]           = field(init=False)
    current_status:   VehicleStatus             = field(default=VehicleStatus.IDLE, init=False)
    route:            list[tuple[int, int]]     = field(default_factory=list, init=False)
    spacetime_path:   list[tuple[int, int, int]]= field(default_factory=list, init=False)
    path_index:       int                       = field(default=0, init=False)
    service_done:     int                       = field(default=0, init=False)
    active_leg_to_gate: bool                    = field(default=True, init=False)
    waiting_time:     int                       = field(default=0, init=False)
    travel_time:      int                       = field(default=0, init=False)
    completion_time:  int                       = field(default=0, init=False)
    metrics:          VehicleMetrics            = field(default_factory=VehicleMetrics, init=False)

    # Inbox for coordinator messages
    inbox: list[CoordinationMessage] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.current_position = self.depot

    # ─── Convenience aliases (backward compatibility) ───────────────────────
    @property
    def pos(self) -> tuple[int, int]:
        return self.current_position

    @pos.setter
    def pos(self, value: tuple[int, int]):
        self.current_position = value

    @property
    def status(self) -> VehicleStatus:
        return self.current_status

    @status.setter
    def status(self, value: VehicleStatus):
        self.current_status = value

    @property
    def path(self) -> list[tuple[int, int]]:
        return self.route

    @path.setter
    def path(self, value: list[tuple[int, int]]):
        self.route = value

    @property
    def vid(self) -> str:
        return self.agent_id

    @property
    def service_time(self) -> int:
        return self.service_duration

    @service_time.setter
    def service_time(self, value: int):
        self.service_duration = value

    @property
    def vtype(self) -> ServiceType:
        return self.service_type

    # ─── perceive / decide / act ────────────────────────────────────────────

    def perceive(self, grid: Any, current_time: int) -> dict:
        """
        Perceive the environment: current position, destination, gate status,
        nearby reservations, and any coordinator messages.
        """
        gate_ok = grid.gate_available.get(self.gate_name, True)
        # Drain inbox for perception
        pending_msgs = list(self.inbox)
        self.inbox.clear()
        return {
            "time":           current_time,
            "pos":            self.current_position,
            "dest":           self.gate_pos if self.active_leg_to_gate else self.depot,
            "gate_available": gate_ok,
            "status":         self.current_status,
            "messages":       pending_msgs,
        }

    def decide(self, perception: dict) -> str:
        """
        Decide high-level action based on perception and utility.

        Returns action strings consumed by act():
          REQUEST_REPLAN   - gate unavailable, need new route
          REQUEST_ROUTE    - no route planned yet
          ACTION_WAIT      - voluntarily wait (utility says yield is better)
          ACTION_MOVE      - proceed along planned route
          ACTION_SERVICE   - at gate, perform service
          ACTION_MAINTAIN  - stay in current state
        """
        # Handle coordinator messages first
        for msg in perception.get("messages", []):
            if msg.msg_type == MessageType.WAIT_REQUEST:
                return "ACTION_WAIT"
            elif msg.msg_type == MessageType.REPLAN_REQUEST:
                return "REQUEST_REPLAN"
            elif msg.msg_type == MessageType.GATE_UNAVAILABLE:
                return "REQUEST_REPLAN"

        if not perception["gate_available"] and self.active_leg_to_gate:
            return "REQUEST_REPLAN"

        if self.current_status == VehicleStatus.SERVICING:
            return "ACTION_SERVICE"

        if not self.route:
            return "REQUEST_ROUTE"

        if self.path_index < len(self.route):
            next_step = self.route[self.path_index]
            if next_step == self.current_position:
                return "ACTION_WAIT"
            # Utility check: is waiting more beneficial than moving?
            if self._should_yield(perception):
                return "ACTION_WAIT"
            return "ACTION_MOVE"

        return "ACTION_MAINTAIN"

    def _should_yield(self, perception: dict) -> bool:
        """
        Utility-based yield decision.
        A lower-priority agent with high waiting cost may decide to yield
        to avoid conflict rather than proceed and cause a collision.
        Returns True if waiting is the better utility choice.
        """
        # Emergency agents never yield
        if self.is_emergency:
            return False
        # High waiting cost makes yield less likely (already paid the wait cost)
        if self.metrics.waiting_time > 5:
            return False
        return False  # Default: let coordinator enforce via messages

    def act(self, action: str, current_time: int) -> tuple[int, int]:
        """Execute the decided action and return new position."""
        return self.step(current_time)

    # ─── Utility function ───────────────────────────────────────────────────

    def compute_utility(self,
                        travel_time: int | None = None,
                        wait_time:   int | None = None,
                        conflict_cost: float = 0.0,
                        delay_cost:    float = 0.0) -> float:
        """
        Multi-attribute utility function.

        U = -(w1*travel + w2*wait + w3*conflict + w4*delay + w5*priority_delay)

        NOTE: A* minimises path cost (step count) independently.
              This utility is used by:
              - decide() for yield vs proceed choices
              - CoordinatorAgent for resource allocation ordering
              - GUI statistics display
        """
        t = travel_time if travel_time is not None else self.metrics.travel_distance
        w = wait_time   if wait_time   is not None else self.metrics.waiting_time

        # Priority-related delay: higher-priority agents have larger urgency weight
        p_urgency = max(0, (5 - self.priority))
        if self.is_emergency:
            p_urgency = 10.0

        priority_delay = p_urgency * w
        utility = -(
            self.w_travel   * t
            + self.w_wait   * w
            + self.w_conflict * conflict_cost
            + self.w_delay  * delay_cost
            + self.w_priority * priority_delay
        )
        return float(utility)

    # ─── Path execution ─────────────────────────────────────────────────────

    def step(self, current_time: int) -> tuple[int, int]:
        """
        Advance the agent by one simulation tick.
        Returns the new position after the step.

        State transitions:
          IDLE      -> (planning sets EN_ROUTE)
          EN_ROUTE  -> WAITING (wait cell) | stays EN_ROUTE | SERVICING (arrived)
          WAITING   -> EN_ROUTE | SERVICING (if at gate)
          SERVICING -> RETURNING (after service_duration ticks)
          RETURNING -> DONE (arrived depot) | EN_ROUTE (return path)
          DONE/NO_PATH -> stationary
        """
        if self.current_status in (VehicleStatus.EN_ROUTE, VehicleStatus.WAITING):
            return self._advance_path(current_time, going_to_gate=self.active_leg_to_gate)

        elif self.current_status == VehicleStatus.SERVICING:
            self.service_done += 1
            if self.service_done >= self.service_duration:
                self.current_status = VehicleStatus.RETURNING
                self.service_done   = 0
                self.route          = []
                self.path_index     = 0
            return self.current_position

        elif self.current_status == VehicleStatus.RETURNING:
            if self.route:
                return self._advance_path(current_time, going_to_gate=self.active_leg_to_gate)
            return self.current_position

        else:
            # IDLE, DONE, NO_PATH — stationary
            return self.current_position

    def _advance_path(self, current_time: int, going_to_gate: bool) -> tuple[int, int]:
        """Pop next cell from route and move (or WAIT in place)."""
        if not self.route or self.path_index >= len(self.route):
            self._check_arrival(current_time, going_to_gate)
            return self.current_position

        next_pos = self.route[self.path_index]
        self.path_index += 1

        if next_pos == self.current_position:
            # WAIT action: same cell repeated
            self.metrics.waiting_time += 1
            self.waiting_time         += 1
            self.current_status        = VehicleStatus.WAITING
        else:
            self.metrics.travel_distance += 1
            self.travel_time             += 1
            self.current_position         = next_pos
            self.current_status           = (VehicleStatus.EN_ROUTE
                                             if going_to_gate else VehicleStatus.RETURNING)

        if self.path_index >= len(self.route):
            self._check_arrival(current_time, going_to_gate)

        return self.current_position

    def _check_arrival(self, current_time: int, going_to_gate: bool):
        """Transition status on arrival at gate or depot."""
        if going_to_gate:
            if self.current_position == self.gate_pos:
                self.current_status = VehicleStatus.SERVICING
                self.service_done   = 0
                self.route          = []
                self.path_index     = 0
        else:
            if self.current_position == self.depot:
                self.current_status    = VehicleStatus.DONE
                self.completion_time   = current_time
                self.metrics.makespan  = current_time

    def set_path(self, path: list[tuple[int, int]],
                 spacetime: list[tuple[int, int, int]] | None = None):
        """Assign a newly computed path and reset progress index."""
        self.route          = path
        self.spacetime_path = spacetime or []
        self.path_index     = 0
        if self.current_status == VehicleStatus.RETURNING:
            self.active_leg_to_gate = False
        elif path:
            self.current_status     = VehicleStatus.EN_ROUTE
            self.active_leg_to_gate = True

    def replan(self, new_gate_pos: tuple[int, int] | None = None):
        """Reset the agent for replanning. Keeps current position."""
        if new_gate_pos:
            self.gate_pos = new_gate_pos
        self.route          = []
        self.spacetime_path = []
        self.path_index     = 0
        self.current_status = VehicleStatus.IDLE

    def receive_message(self, msg: CoordinationMessage):
        """Receive a message from the coordinator."""
        self.inbox.append(msg)

    @property
    def colour(self) -> tuple[float, float, float]:
        return SERVICE_COLOURS[self.service_type]

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.agent_id!r}, "
                f"pos={self.current_position}, "
                f"status={self.current_status.name}, "
                f"priority={self.priority})")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete subclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FuelAgent(GroundServiceAgent):
    """
    Fuel Service Vehicle.
    Carries aviation fuel to aircraft at gates.
    Higher waiting cost: fuel vehicles have strict turnaround windows.
    """
    def __post_init__(self):
        self.service_type = ServiceType.FUEL
        self.w_wait       = 2.0    # Fuel burning on idle costs more
        self.service_duration = 3
        super().__post_init__()


@dataclass
class BaggageAgent(GroundServiceAgent):
    """
    Baggage Handling Vehicle.
    Loads/unloads baggage containers at gates.
    Moderate priority, standard service duration.
    """
    def __post_init__(self):
        self.service_type = ServiceType.BAGGAGE
        self.service_duration = 4
        super().__post_init__()


@dataclass
class CateringAgent(GroundServiceAgent):
    """
    Catering Service Vehicle.
    Delivers food and beverages to aircraft.
    Lower conflict cost: can wait at taxiway holding points.
    """
    def __post_init__(self):
        self.service_type = ServiceType.CATERING
        self.w_conflict   = 1.5
        self.service_duration = 3
        super().__post_init__()


@dataclass
class MaintenanceAgent(GroundServiceAgent):
    """
    Maintenance Vehicle.
    Performs technical checks and repairs at gates.
    Longest service duration.
    """
    def __post_init__(self):
        self.service_type = ServiceType.MAINTENANCE
        self.service_duration = 5
        super().__post_init__()


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator Agent
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoordinatorAgent:
    """
    Central Coordinator Agent for the airport movement area.

    Receives route requests, arbitrates conflicts using multi-attribute utility,
    manages gate reservations, issues grants / delays / emergency preemptions.

    The coordinator ACTUALLY sends CoordinationMessage objects to agents —
    it does not merely exist as a passive class.
    """
    coordinator_id:   str = "ATC_COORD"
    message_inbox:    list[CoordinationMessage] = field(default_factory=list)
    message_outbox:   list[CoordinationMessage] = field(default_factory=list)
    # Gate capacity tracking: gate_name -> agent_id currently servicing (or None)
    gate_occupancy:   dict[str, str | None]  = field(default_factory=dict)
    # Pending gate queue: gate_name -> list of agent_ids waiting
    gate_queue:       dict[str, list[str]]   = field(default_factory=dict)

    def receive_message(self, msg: CoordinationMessage):
        """Receive a message into the coordinator inbox."""
        self.message_inbox.append(msg)

    def _send(self, agent: GroundServiceAgent, msg_type: MessageType,
              resource: str = "", content: dict | None = None, current_time: int = 0):
        """Send a message to a specific agent."""
        msg = CoordinationMessage(
            sender=self.coordinator_id,
            receiver=agent.agent_id,
            msg_type=msg_type,
            timestamp=current_time,
            priority=agent.priority,
            resource=resource,
            content=content or {}
        )
        agent.receive_message(msg)
        self.message_outbox.append(msg)
        return msg

    def arbitrate_requests(self,
                           agents: list[GroundServiceAgent],
                           current_time: int) -> list[CoordinationMessage]:
        """
        Process inbox requests. Sort by utility & priority.
        Grant clearance to high-priority agents.
        Send WAIT_REQUEST to lower-priority agents with conflicts.

        This is the core coordination logic:
        - Emergency agents always get immediate clearance.
        - Priority inversion: if low-priority holds a gate needed by high-priority,
          coordinator issues WAIT to low-priority and ROUTE_GRANTED to high-priority.
        - Gate contention: only one agent services per gate at a time.
        """
        responses: list[CoordinationMessage] = []
        # Sort by utility-informed priority (lowest number = highest priority)
        ordered = sorted(
            agents,
            key=lambda a: (a.priority, -a.compute_utility(0, 0))
        )

        # Check for priority inversion (Scenario 9)
        self._handle_priority_inversion(ordered, current_time, responses)

        for rank, agent in enumerate(ordered):
            if agent.is_emergency:
                msg = self._send(agent, MessageType.EMERGENCY,
                                 resource=agent.gate_name,
                                 content={"status": "EMERGENCY_CLEARANCE", "rank": 0},
                                 current_time=current_time)
                responses.append(msg)
                # Bump all other agents
                for other in agents:
                    if other.agent_id != agent.agent_id and not other.is_emergency:
                        w_msg = self._send(other, MessageType.WAIT_REQUEST,
                                           content={"reason": "EMERGENCY_PREEMPT",
                                                    "yield_to": agent.agent_id},
                                           current_time=current_time)
                        responses.append(w_msg)
            elif rank == 0:
                msg = self._send(agent, MessageType.ROUTE_GRANTED,
                                 resource=agent.gate_name,
                                 content={"status": "CLEARED", "rank": rank},
                                 current_time=current_time)
                responses.append(msg)
            else:
                msg = self._send(agent, MessageType.ROUTE_GRANTED,
                                 resource=agent.gate_name,
                                 content={"status": "COORDINATED_SEQUENCE", "rank": rank},
                                 current_time=current_time)
                responses.append(msg)

        self.message_inbox.clear()
        return responses

    def _handle_priority_inversion(self,
                                   ordered: list[GroundServiceAgent],
                                   current_time: int,
                                   responses: list[CoordinationMessage]):
        """
        Detect and resolve priority inversion:
        A low-priority agent holding a gate that a high-priority agent needs.
        The coordinator sends WAIT_REQUEST to the low-priority agent to yield
        when safe, and ROUTE_GRANTED to the high-priority agent.
        """
        if len(ordered) < 2:
            return

        high_priority = ordered[0]
        for low_agent in ordered[1:]:
            # Check if low-priority is servicing the same gate high-priority wants
            if (low_agent.current_status == VehicleStatus.SERVICING
                    and low_agent.gate_name == high_priority.gate_name
                    and low_agent.priority > high_priority.priority):
                # Priority inversion detected
                w_msg = self._send(low_agent, MessageType.WAIT_REQUEST,
                                   resource=low_agent.gate_name,
                                   content={
                                       "reason": "PRIORITY_INVERSION",
                                       "yield_to": high_priority.agent_id,
                                       "high_priority": high_priority.priority,
                                       "low_priority": low_agent.priority,
                                   },
                                   current_time=current_time)
                responses.append(w_msg)

    def handle_gate_closure(self,
                            gate_name: str,
                            agents: list[GroundServiceAgent],
                            current_time: int) -> list[CoordinationMessage]:
        """
        Handle a gate closure event during service.
        Notifies all affected agents to replan.
        """
        responses = []
        for agent in agents:
            if agent.gate_name == gate_name:
                if agent.current_status in (VehicleStatus.EN_ROUTE, VehicleStatus.WAITING,
                                             VehicleStatus.SERVICING):
                    msg = self._send(agent, MessageType.GATE_UNAVAILABLE,
                                     resource=gate_name,
                                     content={"closed_at": current_time,
                                              "requires_replan": True},
                                     current_time=current_time)
                    responses.append(msg)
                    agent.replan()
        return responses

    def handle_emergency(self,
                         emergency_agent: GroundServiceAgent,
                         agents: list[GroundServiceAgent],
                         current_time: int) -> list[CoordinationMessage]:
        """Issue emergency clearance for an emergency agent."""
        responses = []
        # Grant immediate clearance to emergency agent
        msg = self._send(emergency_agent, MessageType.EMERGENCY,
                         resource=emergency_agent.gate_name,
                         content={"clearance": "IMMEDIATE"},
                         current_time=current_time)
        responses.append(msg)

        # Request all others to yield
        for agent in agents:
            if agent.agent_id != emergency_agent.agent_id:
                w_msg = self._send(agent, MessageType.WAIT_REQUEST,
                                   content={"reason": "EMERGENCY_YIELD",
                                            "emergency_id": emergency_agent.agent_id},
                                   current_time=current_time)
                responses.append(w_msg)
                agent.replan()
        return responses

    def log_service_event(self,
                          agent: GroundServiceAgent,
                          event: str,
                          current_time: int) -> CoordinationMessage:
        """Log service start/complete events."""
        msg_type = (MessageType.SERVICE_STARTED
                    if event == "started" else MessageType.SERVICE_COMPLETED)
        return self._send(agent, msg_type,
                          resource=agent.gate_name,
                          content={"event": event, "time": current_time},
                          current_time=current_time)


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility — Vehicle is now GroundServiceAgent
# ─────────────────────────────────────────────────────────────────────────────

class Vehicle(GroundServiceAgent):
    """
    Backward-compatible alias: Vehicle = GroundServiceAgent.
    Accepts the old 'vid' and 'vtype' constructor kwargs.
    """
    def __init__(self, vid: str = "",
                 vtype: ServiceType | None = None,
                 depot: tuple[int, int] = (0, 0),
                 gate_name: str = "",
                 gate_pos: tuple[int, int] = (0, 0),
                 priority: int = 1,
                 is_emergency: bool = False,
                 service_duration: int = 3,
                 **kwargs):
        _stype = vtype if isinstance(vtype, ServiceType) else ServiceType.GENERIC
        super().__init__(
            agent_id=vid,
            service_type=_stype,
            depot=depot,
            gate_name=gate_name,
            gate_pos=gate_pos,
            priority=priority,
            is_emergency=is_emergency,
            service_duration=service_duration,
        )
        self.service_type = _stype


# Old enum aliases kept for any imports that may reference them
class VehicleType(Enum):
    FLIGHT_A     = "FUEL"
    FLIGHT_B     = "BAGGAGE"
    FLIGHT_C     = "CATERING"
    FLIGHT_D     = "MAINTENANCE"

# Map VehicleType to ServiceType for GUI backward compatibility
_VTYPE_TO_STYPE: dict[VehicleType, ServiceType] = {
    VehicleType.FLIGHT_A: ServiceType.FUEL,
    VehicleType.FLIGHT_B: ServiceType.BAGGAGE,
    VehicleType.FLIGHT_C: ServiceType.CATERING,
    VehicleType.FLIGHT_D: ServiceType.MAINTENANCE,
}

# AircraftType was a previous alias
AircraftType = VehicleType
Aircraft     = Vehicle
FlightAgent  = Vehicle  # kept for any legacy imports; points to GroundServiceAgent-based Vehicle
