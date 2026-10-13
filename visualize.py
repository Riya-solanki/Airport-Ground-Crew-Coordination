"""
visualize.py
============
Matplotlib-based animation and static frame rendering for the simulation.

DESIGN:
  - Each agent is drawn as a filled circle with service-type colour.
  - Reserved cells (from Cooperative A* reservation table) shown as faint overlays.
  - A colour-coded legend, step counter, and scrolling event log are included.
  - Frames saved as animated GIF for reports.

VEHICLE LABELS:
  FUEL01 - Fuel Service       (Yellow)
  BAG01  - Baggage Handling   (Blue)
  CAT01  - Catering Service   (Green)
  MAIN01 - Maintenance        (Red)
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from environment import AirportGrid, EMPTY, WALL, DEPOT, GATE, CORRIDOR
from agents import GroundServiceAgent, VehicleStatus, ServiceType

OUTPUT_DIR = "output"


# ── Cell type colours ─────────────────────────────────────────────────────────
CELL_COLOURS = {
    EMPTY:    "#F8F9FA",
    WALL:     "#2D2D2D",
    DEPOT:    "#C8A2C8",
    GATE:     "#7EC8A0",
    CORRIDOR: "#FFD580",
}

# Vehicle marker colours by ServiceType
VEHICLE_MARKER_COLOURS = {
    ServiceType.FUEL:        "#F2C32E",   # golden
    ServiceType.BAGGAGE:     "#2196F3",   # blue
    ServiceType.CATERING:    "#4CAF50",   # green
    ServiceType.MAINTENANCE: "#E53935",   # red
    ServiceType.EMERGENCY:   "#FF1010",   # bright red
    ServiceType.GENERIC:     "#888888",   # grey
}

# Agent label abbreviations
AGENT_LABELS = {
    "FUEL01": "FU",
    "BAG01":  "BA",
    "CAT01":  "CA",
    "MAIN01": "MA",
    "FUEL02": "F2",
    "BAG02":  "B2",
}


def _get_vehicle_colour(agent: GroundServiceAgent) -> str:
    """Get colour for agent — works with both ServiceType and legacy VehicleType."""
    stype = getattr(agent, 'service_type', None)
    if stype is None:
        stype = getattr(agent, 'vtype', None)
    # Handle legacy VehicleType enum
    if stype is not None and hasattr(stype, 'value'):
        # Map legacy FLIGHT_A etc to ServiceType if needed
        val = stype.value
        if val in ("FUEL", "Fuel Service"):
            return VEHICLE_MARKER_COLOURS[ServiceType.FUEL]
        elif val in ("BAGGAGE", "Baggage Handling"):
            return VEHICLE_MARKER_COLOURS[ServiceType.BAGGAGE]
        elif val in ("CATERING", "Catering Service"):
            return VEHICLE_MARKER_COLOURS[ServiceType.CATERING]
        elif val in ("MAINTENANCE", "Maintenance Service"):
            return VEHICLE_MARKER_COLOURS[ServiceType.MAINTENANCE]
    if isinstance(stype, ServiceType):
        return VEHICLE_MARKER_COLOURS.get(stype, "#888888")
    return "#888888"


def _get_agent_id(agent: GroundServiceAgent) -> str:
    """Get agent_id (or vid for backward compat)."""
    return getattr(agent, 'agent_id', getattr(agent, 'vid', '?'))


def _get_label(agent: GroundServiceAgent) -> str:
    """Get display label for agent on canvas."""
    aid = _get_agent_id(agent)
    if agent.is_emergency:
        return "EM"
    return AGENT_LABELS.get(aid, aid[:2].upper())


class SimulationVisualizer:
    """
    Drives the matplotlib figure for a single scenario run.

    Usage:
        vis = SimulationVisualizer(grid, agents, scenario_name)
        vis.capture_frame(step, event_log, positions, reservations)
        vis.save_animation(filename)
        vis.save_final_frame(filename)
        vis.close()
    """

    def __init__(self, grid: AirportGrid, agents: list[GroundServiceAgent],
                 scenario_name: str = "scenario",
                 reservation_table=None):
        self.grid      = grid
        self.flights   = agents   # kept as 'flights' for internal references
        self.scenario  = scenario_name
        self.res_table = reservation_table

        self.fig, self.axes = plt.subplots(
            1, 2,
            figsize=(14, 7),
            gridspec_kw={"width_ratios": [3, 1]}
        )
        self.ax_grid = self.axes[0]
        self.ax_log  = self.axes[1]

        self.frames: list[dict] = []
        self._init_figure()

    def _init_figure(self):
        fig = self.fig
        fig.patch.set_facecolor("#1A1A2E")
        for ax in self.axes:
            ax.set_facecolor("#16213E")

        self.ax_grid.set_title(
            f"Airport Ground-Service Coordination\nScenario: {self.scenario}",
            color="white", fontsize=13, fontweight="bold", pad=10
        )
        self.ax_log.set_title("ATC Coordinator Log", color="white", fontsize=11,
                              fontweight="bold")
        self.ax_log.axis("off")
        self._draw_grid_base()
        self._draw_legend()

    def _draw_grid_base(self):
        """Draw the static cell background."""
        ax = self.ax_grid
        grid = self.grid
        rows, cols = grid.rows, grid.cols

        for r in range(rows):
            for c in range(cols):
                cell_type = grid.grid[r, c]
                colour    = CELL_COLOURS.get(int(cell_type), CELL_COLOURS[EMPTY])
                rect = mpatches.FancyBboxPatch(
                    (c, rows - 1 - r), 1, 1,
                    boxstyle="square,pad=0",
                    facecolor=colour, edgecolor="#888", linewidth=0.4, zorder=1
                )
                ax.add_patch(rect)

                if cell_type == DEPOT:
                    name = next((k for k, v in grid.depots.items() if v == (r, c)), "D")
                    ax.text(c + 0.5, rows - 1 - r + 0.5,
                            name[:4].upper(), ha="center", va="center",
                            fontsize=5, color="#333", zorder=3)
                elif cell_type == GATE:
                    name = next((k for k, v in grid.gates.items() if v == (r, c)), "G")
                    avail = self.grid.gate_available.get(name, True)
                    label_col = "#004D00" if avail else "#8B0000"
                    ax.text(c + 0.5, rows - 1 - r + 0.5,
                            name.replace("gate_", "").upper(), ha="center", va="center",
                            fontsize=7, color=label_col, fontweight="bold", zorder=3)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.grid(False)

        for x in range(cols + 1):
            ax.axvline(x, color="#444", linewidth=0.3, zorder=0)
        for y in range(rows + 1):
            ax.axhline(y, color="#444", linewidth=0.3, zorder=0)

    def _draw_legend(self):
        """Static legend on the right side."""
        ax = self.ax_log
        y = 0.97
        dy = 0.065

        def txt(text, col="#EEEEEE", bold=False, size=9):
            nonlocal y
            ax.text(0.05, y, text, transform=ax.transAxes,
                    color=col, fontsize=size,
                    fontweight="bold" if bold else "normal",
                    verticalalignment="top")
            y -= dy

        txt("LEGEND", bold=True, size=10, col="#FFD700")
        y -= 0.01

        agent_legend = [
            ("FUEL01", ServiceType.FUEL,        "Fuel Service"),
            ("BAG01",  ServiceType.BAGGAGE,      "Baggage Handling"),
            ("CAT01",  ServiceType.CATERING,     "Catering Service"),
            ("MAIN01", ServiceType.MAINTENANCE,  "Maintenance"),
        ]
        for aid, stype, label in agent_legend:
            col = VEHICLE_MARKER_COLOURS[stype]
            ax.add_patch(mpatches.Circle((0.08, y + 0.018), 0.022,
                                         color=col, transform=ax.transAxes, zorder=5))
            ax.text(0.15, y, f"{aid} - {label}", transform=ax.transAxes,
                    color="#EEEEEE", fontsize=7.5, verticalalignment="top")
            y -= dy

        y -= 0.01
        txt("CELL TYPES", bold=True, size=9, col="#FFD700")
        cell_labels = [
            ("#", "#2D2D2D", "Wall"),
            ("D", "#C8A2C8", "Depot"),
            ("G", "#7EC8A0", "Gate"),
            ("|", "#FFD580", "Narrow Corridor"),
        ]
        for sym, col, label in cell_labels:
            ax.text(0.05, y, sym, transform=ax.transAxes,
                    color=col, fontsize=10, verticalalignment="top", fontweight="bold")
            ax.text(0.15, y, label, transform=ax.transAxes,
                    color="#EEEEEE", fontsize=7.5, verticalalignment="top")
            y -= dy

        self._log_y_start = y - 0.02
        self._log_texts   = []

    def capture_frame(self, step: int, event_log: list[str],
                      vehicle_positions: list[tuple[int, int]],
                      reservation_snapshot: set | None = None):
        """Store a snapshot for later animation/frame rendering."""
        self.frames.append({
            "step":         step,
            "log":          list(event_log[-8:]),
            "positions":    list(vehicle_positions),
            "statuses":     [a.current_status for a in self.flights],
            "reservations": set(reservation_snapshot) if reservation_snapshot else set(),
        })

    def render_frame(self, frame_data: dict) -> plt.Figure:
        """Draw a single frame on the existing figure."""
        self._draw_grid_base()
        ax   = self.ax_grid
        rows = self.grid.rows

        step         = frame_data["step"]
        log          = frame_data["log"]
        pos_list     = frame_data["positions"]
        statuses     = frame_data["statuses"]
        reservations = frame_data.get("reservations", set())

        # Draw reservations (faint overlay)
        for (r, c, t) in reservations:
            if t == step or t == step + 1:
                alpha = 0.08 if t == step + 1 else 0.15
                rect = mpatches.Rectangle(
                    (c, rows - 1 - r), 1, 1,
                    facecolor="#FF6600", alpha=alpha, zorder=2
                )
                ax.add_patch(rect)

        # Draw agent circles with unique identifiers
        for i, agent in enumerate(self.flights):
            if i >= len(pos_list):
                continue
            r, c    = pos_list[i]
            status  = statuses[i]
            col     = _get_vehicle_colour(agent)
            aid     = _get_agent_id(agent)
            label   = _get_label(agent)

            # Map grid (row, col) to plot (x, y)
            cx = c + 0.5
            cy = rows - 1 - r + 0.5

            ring_col = {
                VehicleStatus.EN_ROUTE:   "#FFFFFF",
                VehicleStatus.SERVICING:  "#00FF88",
                VehicleStatus.RETURNING:  "#88AAFF",
                VehicleStatus.WAITING:    "#FF4444",
                VehicleStatus.DONE:       "#AAAAAA",
                VehicleStatus.NO_PATH:    "#FF0000",
                VehicleStatus.IDLE:       "#888888",
            }.get(status, "#FFFFFF")

            ax.add_patch(mpatches.Circle((cx, cy), 0.42, color=ring_col,
                                         zorder=5, alpha=0.7))
            ax.add_patch(mpatches.Circle((cx, cy), 0.33, color=col,
                                         zorder=6))

            # Unique agent label (FU, BA, CA, MA, EM for emergency)
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold", zorder=7)

            # Status badge
            badge = {
                VehicleStatus.WAITING:   "W",
                VehicleStatus.NO_PATH:   "X",
                VehicleStatus.SERVICING: "S",
                VehicleStatus.DONE:      "D",
            }.get(status, "")
            if badge:
                ax.text(cx + 0.25, cy + 0.25, badge, ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold", zorder=8)

        # Step counter and timeline
        ax.text(0.01, 0.99, f"t = {step}", transform=ax.transAxes,
                color="#FFD700", fontsize=12, fontweight="bold",
                verticalalignment="top", zorder=10)

        # Event log (right panel)
        ax_log = self.ax_log
        ax_log.cla()
        ax_log.axis("off")
        ax_log.set_facecolor("#16213E")
        ax_log.set_title("ATC Coordinator Log", color="white", fontsize=11,
                         fontweight="bold")
        self._draw_legend()

        y = self._log_y_start
        ax_log.text(0.05, y, "Events:", transform=ax_log.transAxes,
                    color="#FFD700", fontsize=9, fontweight="bold", va="top")
        y -= 0.06
        for entry in reversed(log[-6:]):
            safe = entry.encode('ascii', 'replace').decode('ascii')
            ax_log.text(0.05, y, f"- {safe}", transform=ax_log.transAxes,
                        color="#CCCCCC", fontsize=7, va="top", wrap=True)
            y -= 0.065

        # Agent status table
        y -= 0.04
        ax_log.text(0.05, y, "Agent Status:", transform=ax_log.transAxes,
                    color="#FFD700", fontsize=9, fontweight="bold", va="top")
        y -= 0.06
        for agent in self.flights:
            col = _get_vehicle_colour(agent)
            aid = _get_agent_id(agent)
            st  = agent.current_status.name.replace("_", " ")
            ax_log.add_patch(mpatches.Circle(
                (0.07, y + 0.02), 0.022, color=col,
                transform=ax_log.transAxes, zorder=5))
            ax_log.text(0.13, y, f"{aid}: {st}",
                        transform=ax_log.transAxes,
                        color="#EEEEEE", fontsize=7.5, va="top")
            y -= 0.055

        self.fig.canvas.draw_idle()
        return self.fig

    def save_animation(self, scenario_name: str, fps: int = 3):
        """Save all captured frames as an animated GIF."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not self.frames:
            return

        def _update(i):
            self.render_frame(self.frames[i])

        anim = FuncAnimation(self.fig, _update,
                             frames=len(self.frames),
                             interval=1000 // fps, blit=False)
        path = os.path.join(OUTPUT_DIR, f"animation_{scenario_name}.gif")
        try:
            anim.save(path, writer="pillow", fps=fps,
                      savefig_kwargs={"facecolor": "#1A1A2E"})
            print(f"  [vis] Animation saved -> {path}")
        except Exception as e:
            print(f"  [vis] Could not save GIF (pillow needed): {e}")

    def save_final_frame(self, scenario_name: str):
        """Save the last frame as a static PNG."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if self.frames:
            self.render_frame(self.frames[-1])
        path = os.path.join(OUTPUT_DIR, f"frame_{scenario_name}.png")
        self.fig.savefig(path, dpi=150, facecolor="#1A1A2E",
                         bbox_inches="tight")
        print(f"  [vis] Frame saved -> {path}")

    def close(self):
        plt.close(self.fig)


def draw_grid_static(grid: AirportGrid, title: str = "Airport Ground Layout") -> None:
    """
    Standalone: draw the airport grid as a single static image.
    Useful for sanity-checking the layout.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows, cols = grid.rows, grid.cols
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for r in range(rows):
        for c in range(cols):
            cell_type = grid.grid[r, c]
            colour    = CELL_COLOURS.get(int(cell_type), CELL_COLOURS[EMPTY])
            rect = mpatches.FancyBboxPatch(
                (c, rows - 1 - r), 1, 1,
                boxstyle="square,pad=0",
                facecolor=colour, edgecolor="#555", linewidth=0.5, zorder=1
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, rows - 1 - r + 0.5, f"{r},{c}",
                    ha="center", va="center", fontsize=5, color="#777", zorder=2)

    for name, (r, c) in grid.depots.items():
        ax.text(c + 0.5, rows - 1 - r + 0.5,
                name.replace("_depot", "")[:4].upper(),
                ha="center", va="center", fontsize=6,
                color="#333", fontweight="bold", zorder=3)
    for name, (r, c) in grid.gates.items():
        ax.text(c + 0.5, rows - 1 - r + 0.5,
                name.replace("gate_", "").upper(),
                ha="center", va="center", fontsize=8,
                color="#004D00", fontweight="bold", zorder=3)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    for x in range(cols + 1):
        ax.axvline(x, color="#333", linewidth=0.4)
    for y in range(rows + 1):
        ax.axhline(y, color="#333", linewidth=0.4)

    path = os.path.join(OUTPUT_DIR, "airport_layout.png")
    fig.savefig(path, dpi=150, facecolor="#1A1A2E", bbox_inches="tight")
    plt.close(fig)
    print(f"  [vis] Airport layout saved -> {path}")
