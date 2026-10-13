# Airport Ground-Crew Coordination

**Course:** 23CSE401 Fundamentals of Artificial Intelligence

This project simulates autonomous airport ground-service vehicles coordinating movement on a shared 12x12 airport grid. Fuel, Baggage, Catering and Maintenance agents travel from depots to assigned gates, perform service, and return. The system demonstrates utility-based agents, a CoordinatorAgent, multi-agent coordination, search, reservation handling and dynamic replanning.

## Algorithms compared

- **A* (Cooperative)** — informed space-time search prioritizing f(n)=g(n)+h(n), using reservations, wait actions and conflict avoidance.
- **Greedy Best-First Search** — informed space-time search prioritizing h(n) alone, using the same reservation constraints. The Search Inspector still displays g, h and f=g+h for analysis, but Greedy Best-First chooses nodes by h only.

The GUI exposes only these two algorithms so the comparison remains focused.

## Five report scenarios

1. **Normal Operation** — four ground-service vehicles route to different gates.
2. **Corridor Conflict** — two vehicles approach a shared single-lane corridor from opposite directions.
3. **Gate Unavailable** — Gate B becomes unavailable and BAG01 is diverted to Gate D.
4. **Emergency Priority** — FUEL01 receives emergency priority and the fleet replans.
5. **Unreachable Goal** — Gate A is isolated by walls and the system reports no path.

Run all five scenarios with both algorithms:

```bash
python main.py --scenario all --algorithm all --no-anim
```

Compare the corridor-conflict case directly:

```bash
python main.py --compare --no-anim
```

Launch the GUI:

```bash
python main.py
```

## GUI

The desktop simulator provides:
- scenario selector for the five report scenarios
- A* (Cooperative) / Greedy Best-First selector
- step-by-step and auto-play simulation
- live ground-service agent status and route visualization
- Search Inspector with space-time `g`, `h`, `f` and parent values
- two dynamic-event controls: **Close Gate B** and **Emergency Priority**
- performance statistics including conflicts, replanning, expanded nodes and runtime

The reservation table remains part of both algorithms’ coordination logic, but the presentation dashboard does not expose a reservation time-slice slider. The ATC coordinator event-log panel and the experimental Add Flight control have also been removed to keep the interface focused on the five report scenarios.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+.


## Exclusive narrow-corridor reservations
Yellow corridor cells are grouped into orthogonally connected corridor resources. When a planned vehicle occupies any cell in a corridor at time t, that corridor resource is reserved at t. Another vehicle cannot enter a different cell of the same corridor at the same time, even if ordinary cell-level checks would permit it. The rule is enforced by both Cooperative A* and Greedy Best-First Search. Reservations are time-indexed, so following vehicles may enter after the prior vehicle clears the corridor.
