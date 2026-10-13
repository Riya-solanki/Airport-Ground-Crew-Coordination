"""

Tkinter GUI for Airport Ground-Crew Coordination: A Multi-Agent Routing System.



Features:

  - Airport Grid with live agent markers, labels (e.g. F1, F2, EM), and path overlays.

  - Route Visualization accurately mapped to the 12x12 airport topology without modulo wrap.

  - Search Inspector tab showing expanded space-time nodes with g, h, f values and parents.

  - Interactive live events for gate closure and emergency priority.

  - Presentation-focused dashboard with no reservation time-slice control or ATC event-log panel.

  - Multi-Speed Slider, Step-Back, Step-Forward controls.

  - Scenario description & Legend for presentation clarity.

"""

from __future__ import annotations



import time

import tkinter as tk

from tkinter import ttk



from scipy import stats



from environment import AirportGrid, EMPTY, WALL, DEPOT, GATE, CORRIDOR

from scenarios import (

    scenario_normal, scenario_corridor_conflict, scenario_gate_unavailable,

    scenario_emergency,

)

from agents import VehicleStatus, ServiceType, CoordinatorAgent

from search.cooperative_astar import ReservationTable, plan_all_vehicles, cooperative_astar

from search.greedy_best_first import greedy_best_first, plan_all_vehicles_greedy

from simulation import detect_conflicts, _replan_affected, _plan_return_leg, _detect_swap_conflicts





SCENARIOS = {

    "1. Normal Operation": ("normal_operation", scenario_normal),

    "2. Corridor Conflict": ("corridor_conflict", scenario_corridor_conflict),

    "3. Gate Unavailable": ("gate_unavailable", scenario_gate_unavailable),

    "4. Emergency Priority": ("emergency_vehicle", scenario_emergency),

}



SCENARIO_DESCRIPTIONS = {

    "1. Normal Operation": "Nominal ground-service routing: 4 vehicles start at runway/depots and navigate to gates A-D concurrently with conflict-free reservation scheduling.",

    "2. Corridor Conflict": "Two ground-service vehicles request the same single-lane taxiway corridor from opposite directions. Cooperative A* orders priority and inserts tactical holds.",

    "3. Gate Unavailable": "At t=8, Gate B is blocked due to maintenance. Affected vehicle BAG01 requests coordinator replanning to divert to Gate D.",

    "4. Emergency Priority": "At t=5, emergency vehicle FUEL01 requests priority clearance. Coordinator upgrades priority and replans the fleet.",

}



ALGORITHMS = ["A* (Cooperative)", "Greedy Best-First"]



AGENT_COLORS = {

    ServiceType.FUEL: "#E45756",        # Fuel*

    ServiceType.BAGGAGE: "#2F80ED",     # Baggage*

    ServiceType.CATERING: "#27AE60",    # Catering*

    ServiceType.MAINTENANCE: "#F2994A", # Maintenance*

    ServiceType.EMERGENCY: "#D64545",   # Emergency*

    ServiceType.GENERIC: "#7A869A",

}



AGENT_NAMES = {

    ServiceType.FUEL: "Fuel",

    ServiceType.BAGGAGE: "Baggage",

    ServiceType.CATERING: "Catering",

    ServiceType.MAINTENANCE: "Maintenance",

    ServiceType.EMERGENCY: "Emergency",

    ServiceType.GENERIC: "Ground Vehicle",

}



STATUS_COLORS = {

    VehicleStatus.IDLE: "#6B7280",

    VehicleStatus.EN_ROUTE: "#2563EB",

    VehicleStatus.SERVICING: "#059669",

    VehicleStatus.RETURNING: "#7C3AED",

    VehicleStatus.DONE: "#64748B",

    VehicleStatus.NO_PATH: "#DC2626",

    VehicleStatus.WAITING: "#D97706",

}



BG = "#EEF3F9"

PANEL = "#FFFFFF"

LINE = "#D7E0EC"

TEXT = "#172033"

MUTED = "#667085"

BLUE = "#2563EB"

DARK_BLUE = "#123B66"

RED = "#DC2626"

GREEN = "#16845B"

YELLOW = "#F2B84B"





class AirportGUI:

    def __init__(self, root: tk.Tk):

        self.root = root

        root.title("Airport Ground-Crew Coordination | Multi-Agent Routing System")


        root.geometry("1280x800")

        root.minsize(1000, 700)



        self.grid = None

        self.vehicles = []

        self.events = {}

        self.table = ReservationTable(self.grid.corridor_cells)

        self.coordinator = CoordinatorAgent()

        self.scenario_name = "normal_operation"

        self.algorithm = "cooperative_astar"

        self.step_no = 0

        self.total_conflicts = 0

        self.total_nodes = 0

        self.replans = 0

        self.event_log = []

        self.prev_positions = {}

        self.running = False

        self.after_id = None

        self.selected_agent = 0

        self.start_time = None

        self.speed_ms = 450

        self.history_snapshots: list[dict] = []

        self.search_traces: dict[str, list[dict]] = {}



        self._build_styles()

        self._build_layout()

        self.reset()



    # ------------------------------------------------------------------ UI*

    def _build_styles(self):

        style = ttk.Style()

        try:

            style.theme_use("clam")

        except tk.TclError:

            pass

        style.configure("TCombobox", fieldbackground="#FFFFFF", background="#FFFFFF",

                        foreground=TEXT, bordercolor=LINE, lightcolor="#FFFFFF", darkcolor="#FFFFFF")

        style.map("TCombobox", fieldbackground=[("readonly", "#FFFFFF")],

                  foreground=[("readonly", TEXT)])

        style.configure("Treeview", background="#FFFFFF", foreground=TEXT, fieldbackground="#FFFFFF",

                        rowheight=28, font=("Segoe UI", 9), borderwidth=0)

        style.configure("Treeview.Heading", background="#E8EEF8", foreground=DARK_BLUE,

                        font=("Segoe UI", 9, "bold"), relief="flat", padding=6)

        style.map("Treeview", background=[("selected", "#DCEBFF")],

                  foreground=[("selected", DARK_BLUE)])

        style.configure("TNotebook", background=BG, borderwidth=0)

        style.configure("TNotebook.Tab", background="#E8EEF8", foreground=DARK_BLUE,

                        padding=(12, 7), font=("Segoe UI", 9, "bold"))

        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")],

                  foreground=[("selected", BLUE)])



    def card(self, parent, title):

        outer = tk.Frame(parent, bg=PANEL, highlightbackground="#D7E0EC", highlightthickness=1)

        head = tk.Frame(outer, bg="#F7FAFF", height=34)

        head.pack(fill="x")

        tk.Frame(head, bg=BLUE, width=4).pack(side="left", fill="y")

        tk.Label(head, text=title, bg="#F7FAFF", fg=DARK_BLUE,

                 font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left", padx=10, pady=7)

        return outer



    def _build_layout(self):

        main = tk.Frame(self.root, bg=BG)

        main.pack(fill="both", expand=True, padx=10, pady=5)



        # Header with Title and Legend*

        header = tk.Frame(main, bg=DARK_BLUE, highlightbackground="#0D2E50", highlightthickness=1)

        header.pack(fill="x", pady=(0, 5))



        title_box = tk.Frame(header, bg=DARK_BLUE)

        title_box.pack(side="left", fill="x", expand=True, padx=14, pady=7)

        tk.Label(title_box, text="AIRPORT GROUND-CREW COORDINATION",

                 bg=DARK_BLUE, fg="#FFFFFF", font=("Segoe UI", 14, "bold")).pack(anchor="w")



        self.scenario_desc_var = tk.StringVar()

        tk.Label(title_box, textvariable=self.scenario_desc_var,

                 bg=DARK_BLUE, fg="#D8E7F7", font=("Segoe UI", 8), wraplength=700, justify="left").pack(anchor="w", pady=(2, 0))



        # Compact legend inside the header*

        legend_frame = tk.Frame(header, bg="#1B4D7A", padx=10, pady=7)

        legend_frame.pack(side="right", padx=10, pady=7)

        tk.Label(legend_frame, text="GRID LEGEND", bg="#1B4D7A", fg="#FFFFFF",

                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 3))

        legend_row = tk.Frame(legend_frame, bg="#1B4D7A")

        legend_row.pack(anchor="w")

        for sym, col, txt in [("■", "#374151", "Wall"), ("■", "#D9E9FF", "Depot"), ("■", "#DFF5E6", "Gate"), ("■", "#FFF0BF", "Corridor")]:

            tk.Label(legend_row, text=sym, fg=col, bg="#1B4D7A", font=("Segoe UI", 9, "bold")).pack(side="left")

            tk.Label(legend_row, text=f"{txt}  ", fg="#E7F0FA", bg="#1B4D7A", font=("Segoe UI", 8)).pack(side="left")



        # Body split*

        body = tk.Frame(main, bg=BG)

        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG, width=640)

        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(body, bg=BG, width=640)

        right.pack(side="left", fill="both", expand=True, padx=(8, 0))



        # Left Column: Airport grid*

        grid_card = self.card(left, "Airport Movement Area (12×12 Topology)")

        grid_card.pack(fill="both", expand=True, pady=(0, 6))

        self.canvas = tk.Canvas(grid_card, bg="#E9EEF3", highlightthickness=0)

        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas.bind("<Button-1>", self._canvas_click)

        self.canvas.bind("<Configure>", lambda e: self.render())



        # Right Column: Notebook Tabs (Agent Status & Route, Search Inspector)*

        nb_card = self.card(right, "Ground-Crew Tracking & Search Inspection")

        nb_card.pack(fill="both", expand=True, pady=(0, 6))



        self.notebook = ttk.Notebook(nb_card)

        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))



        # Tab 1: Agent Status & Route*

        tab_status = tk.Frame(self.notebook, bg=PANEL)

        self.notebook.add(tab_status, text="  Agent Status & Route  ")



        cols = ["Agent", "Callsign", "Position", "Destination", "Status", "Utility"]

        self.status_tree = ttk.Treeview(tab_status, columns=cols, show="headings", height=4)

        widths = [70, 95, 85, 100, 100, 85]

        for c, w in zip(cols, widths):

            self.status_tree.heading(c, text=c)

            self.status_tree.column(c, width=w, anchor="center")

        self.status_tree.pack(fill="x", padx=6, pady=(6, 6))

        self.status_tree.bind("<<TreeviewSelect>>", self._status_select)



        self.route_canvas = tk.Canvas(tab_status, bg="#FAFBFC", highlightthickness=1, highlightbackground=LINE)

        self.route_canvas.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        self.route_note = tk.StringVar(value="Select a ground-service vehicle above to inspect its active planned trajectory.")

        tk.Label(tab_status, textvariable=self.route_note, bg=PANEL, fg=MUTED,

                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=8, pady=(0, 4))



        # Tab 2: Search Inspector (Node Expansion)*

        tab_inspector = tk.Frame(self.notebook, bg=PANEL)

        self.notebook.add(tab_inspector, text="  Search Inspector (Space-Time Tree)  ")



        insp_top = tk.Frame(tab_inspector, bg=PANEL)

        insp_top.pack(fill="x", padx=8, pady=(8, 6))

        tk.Label(insp_top, text="Inspecting:", bg=PANEL, fg=DARK_BLUE,

                 font=("Segoe UI", 9, "bold")).pack(side="left")

        self.insp_agent_var = tk.StringVar(value="None")

        tk.Label(insp_top, textvariable=self.insp_agent_var, bg="#EAF2FF", fg=BLUE,

                 font=("Segoe UI", 9, "bold"), padx=8, pady=3).pack(side="left", padx=(8, 0))

        tk.Label(insp_top, text="  Search expansions (A* or Greedy Best-First)", bg=PANEL, fg=MUTED,

                 font=("Segoe UI", 8)).pack(side="left", padx=(10, 0))



        # Treeview for Search Nodes*

        insp_cols = ["Node (r,c,t)", "g (Cost)", "h (Heur)", "f = g+h", "Parent"]

        self.insp_tree = ttk.Treeview(tab_inspector, columns=insp_cols, show="headings", height=8)

        insp_widths = [110, 75, 75, 75, 110]

        for c, w in zip(insp_cols, insp_widths):

            self.insp_tree.heading(c, text=c)

            self.insp_tree.column(c, width=w, anchor="center")

        self.insp_tree.pack(fill="both", expand=True, padx=6, pady=(4, 6))



        # Bottom section: Controls + Performance Statistics*

        bottom = tk.Frame(main, bg=BG, height=140)

        bottom.pack(fill="x", pady=(6, 0))

        bottom.pack_propagate(False)

        # Controls Card — compact single-row dashboard to prevent vertical clipping
        controls = self.card(bottom, "Simulation Controls & Dynamic Events")
        controls.pack(side="left", fill="both", expand=True, padx=(0, 6))

        control_row = tk.Frame(controls, bg=PANEL)
        control_row.pack(fill="both", expand=True, padx=8, pady=5)

        self.step_btn = tk.Button(control_row, text="▶ Step", command=self.step,
                                  bg=GREEN, fg="white", activebackground="#117A3B",
                                  relief="flat", font=("Segoe UI", 8, "bold"), padx=7, pady=3)
        self.step_btn.pack(side="left", padx=(0, 2))

        self.step_back_btn = tk.Button(control_row, text="◀ Back", command=self.step_back,
                                       bg="#4B5563", fg="white", activebackground="#374151",
                                       relief="flat", font=("Segoe UI", 8, "bold"), padx=7, pady=3)
        self.step_back_btn.pack(side="left", padx=2)

        self.play_btn = tk.Button(control_row, text="▶▶ Play", command=self.toggle_autoplay,
                                  bg=BLUE, fg="white", activebackground="#0F57B5",
                                  relief="flat", font=("Segoe UI", 8, "bold"), padx=7, pady=3)
        self.play_btn.pack(side="left", padx=2)

        tk.Button(control_row, text="↻ Reset", command=self.reset,
                  bg=RED, fg="white", activebackground="#A91C1C",
                  relief="flat", font=("Segoe UI", 8, "bold"), padx=7, pady=3).pack(side="left", padx=2)

        tk.Label(control_row, text="Scenario", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(7, 2))
        self.scenario_var = tk.StringVar(value=list(SCENARIOS)[0])
        cb = ttk.Combobox(control_row, textvariable=self.scenario_var,
                          values=list(SCENARIOS), state="readonly", width=24)
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.reset())

        tk.Label(control_row, text="Algorithm", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(6, 2))
        self.algorithm_var = tk.StringVar(value="A* (Cooperative)")
        acb = ttk.Combobox(control_row, textvariable=self.algorithm_var,
                           values=ALGORITHMS, state="readonly", width=17)
        acb.pack(side="left", padx=2)
        acb.bind("<<ComboboxSelected>>", lambda e: self.reset())

        tk.Button(control_row, text="⚠ Gate B", command=self.inject_close_gate,
                  bg="#F59E0B", fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), padx=5, pady=3).pack(side="left", padx=(6, 2))
        tk.Button(control_row, text="⚡ Emergency", command=self.inject_emergency,
                  bg="#EF4444", fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), padx=5, pady=3).pack(side="left", padx=2)

        speed_box = tk.Frame(control_row, bg=PANEL)
        speed_box.pack(side="right", padx=(4, 0))
        tk.Label(speed_box, text="Speed", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 7)).pack(side="left")
        self.speed_slider = tk.Scale(speed_box, from_=100, to=1000, orient="horizontal",
                                     showvalue=0, command=self._on_speed_change,
                                     bg=PANEL, highlightthickness=0, length=55)
        self.speed_slider.set(450)
        self.speed_slider.pack(side="left")

        # Statistics Card*

        stats = self.card(bottom, "Performance Statistics")

        stats.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.stats_var = tk.StringVar()



        tk.Label(

            stats,

            textvariable=self.stats_var,

            bg=PANEL,

            fg=TEXT,

            justify="left",

            anchor="nw",

            font=("Consolas", 7),

            padx=10,

            pady=5

            ).pack(fill="both", expand=True)





    # --------------------------------------------------------------- State & Planning*

    def reset(self):

        self.stop_autoplay()

        scen_text = self.scenario_var.get()

        self.scenario_name, factory = SCENARIOS[scen_text]

        self.scenario_desc_var.set(SCENARIO_DESCRIPTIONS.get(scen_text, ""))

        self.grid = AirportGrid()

        self.vehicles, self.events = factory(self.grid)

        self.algorithm = self._algorithm_key()

        self.table = ReservationTable(self.grid.corridor_cells)

        self.coordinator = CoordinatorAgent()

        self.step_no = 0

        self.total_conflicts = 0

        self.total_nodes = 0

        self.replans = 0

        self.history_snapshots = []

        self.search_traces = {}

        self.event_log = [f"[00:00] Simulation initialized: {scen_text.split('. ', 1)[-1]}"]

        self.start_time = time.perf_counter()

        self.prev_positions = {v.vid: v.pos for v in self.vehicles}

        self.selected_agent = 0



        self._initial_plan()

        self._record_snapshot()

        self._refresh_all()

        self._append_log(f"[00:00] Algorithm selected: {self.algorithm_var.get()}")

        self._append_log("[00:00] Coordinator received slot requests from all ground-service agents.")



    def _algorithm_key(self):

        return {"A* (Cooperative)": "cooperative_astar", "Greedy Best-First": "greedy_best_first"}[self.algorithm_var.get()]



    def _initial_plan(self):

        t0 = time.perf_counter()
        search_fn = cooperative_astar if self.algorithm == "cooperative_astar" else greedy_best_first
        ordered = sorted(self.vehicles, key=lambda v: v.priority)

        for v in ordered:
            if not self.grid.gate_available.get(v.gate_name, True):
                v.status = VehicleStatus.NO_PATH
                continue

            path_2d, st_path, expanded, trace = search_fn(
                self.grid, v.pos, v.gate_pos, self.table, t_start=0,
                dwell_time=getattr(v, 'service_time', 3), trace_search=True
            )
            v.metrics.nodes_expanded += expanded
            self.search_traces[v.vid] = trace
            if path_2d:
                v.set_path(path_2d[1:], st_path)
                self.table.reserve_path(path_2d, 0, reserve_duration=getattr(v, 'service_time', 3))
            else:
                v.status = VehicleStatus.NO_PATH

        self.plan_ms = (time.perf_counter() - t0) * 1000
        self.total_nodes = sum(v.metrics.nodes_expanded for v in self.vehicles)


    def _record_snapshot(self):

        """Save a snapshot of all agent states to allow Step-Back."""

        snapshot = {

            "step_no": self.step_no,

            "total_conflicts": self.total_conflicts,

            "replans": self.replans,

            "event_log_len": len(self.event_log),

            "vehicles": [

                {

                    "vid": v.vid, "pos": v.pos, "status": v.status,

                    "path": list(v.path), "path_index": v.path_index,

                    "active_leg": v.active_leg_to_gate,

                    "service_done": v.service_done,

                    "metrics_dist": v.metrics.travel_distance,

                    "metrics_wait": v.metrics.waiting_time,

                }

                for v in self.vehicles

            ]

        }

        self.history_snapshots.append(snapshot)

        if len(self.history_snapshots) > 60:

            self.history_snapshots.pop(0)



    # --------------------------------------------------------------- Stepping & Controls*

    def step(self):

        if not self.vehicles or all(v.status in (VehicleStatus.DONE, VehicleStatus.NO_PATH) for v in self.vehicles):

            return



        self._record_snapshot()

        step = self.step_no



        # Scheduled events*

        if step in self.events:

            before = len(self.event_log)

            self.events[step](self.grid, self.vehicles, self.event_log)

            self.replans += sum(1 for v in self.vehicles if v.status == VehicleStatus.IDLE)

            self._append_new_events(before)

            if self.algorithm in ("cooperative_astar", "greedy_best_first"):

                _replan_affected(self.grid, self.vehicles, self.table, current_time=step, algorithm=self.algorithm)



        # Move agents*

        for v in self.vehicles:

            if v.status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH, VehicleStatus.IDLE):

                old = v.pos

                v.step(step)

                if v.pos != old:

                    self._append_log(f"[{step:02d}:{step:02d}] {v.vid} taxied to {v.pos}")

                if v.status == VehicleStatus.RETURNING and not v.path:

                    _plan_return_leg(self.grid, v, self.table, step, self.algorithm)



        conflicts = detect_conflicts(self.vehicles)

        swap = _detect_swap_conflicts(self.vehicles, self.prev_positions)

        if conflicts or swap:

            self.total_conflicts += conflicts + swap

            self._append_log(f"[{step:02d}:{step:02d}] Conflict detected: {conflicts + swap} pair(s)")

        self.prev_positions = {v.vid: v.pos for v in self.vehicles}



        self.step_no += 1

        self.total_nodes = sum(v.metrics.nodes_expanded for v in self.vehicles)

        if self.step_no % 5 == 0:

            self._append_log(f"[{step:02d}:{step:02d}] ATC Coordinator synchronized reservations.")



        if all(v.status in (VehicleStatus.DONE, VehicleStatus.NO_PATH) for v in self.vehicles):

            self._append_log(f"[{self.step_no:02d}:{self.step_no:02d}] All ground-service missions completed.")

            self.stop_autoplay()

        self._refresh_all()



        if self.running and self.vehicles:

            self.after_id = self.root.after(self.speed_ms, self.step)



    def step_back(self):

        """Restore the previous simulation step."""

        if not self.history_snapshots:

            return

        self.stop_autoplay()

        snap = self.history_snapshots.pop()

        self.step_no = snap["step_no"]

        self.total_conflicts = snap["total_conflicts"]

        self.replans = snap["replans"]

        if snap["event_log_len"] < len(self.event_log):

            self.event_log = self.event_log[:snap["event_log_len"]]



        for v_data, v in zip(snap["vehicles"], self.vehicles):

            v.pos = v_data["pos"]

            v.status = v_data["status"]

            v.path = list(v_data["path"])

            v.path_index = v_data["path_index"]

            v.active_leg_to_gate = v_data["active_leg"]

            v.service_done = v_data["service_done"]

            v.metrics.travel_distance = v_data["metrics_dist"]

            v.metrics.waiting_time = v_data["metrics_wait"]



        self._refresh_all()

        self._append_log(f"[STEP-BACK] Reverted to Step {self.step_no}.")



    def toggle_autoplay(self):

        if self.running:

            self.stop_autoplay()

        else:

            self.running = True

            self.play_btn.config(text="Ⅱ  Pause")

            self.step()



    def stop_autoplay(self):

        self.running = False

        self.play_btn.config(text="▶▶  Auto Play")

        if self.after_id:

            try:

                self.root.after_cancel(self.after_id)

            except tk.TclError:

                pass

            self.after_id = None



    def _on_speed_change(self, val):

        self.speed_ms = int(val)



    # --------------------------------------------------------------- Live Event Injections*

    def inject_close_gate(self):

        """Interactive live event: close Gate B and replan affected vehicles."""

        self.grid.set_gate_available("gate_B", False)

        self._append_log(f"[{self.step_no:02d}:{self.step_no:02d}] LIVE EVENT: Gate B closed by controller!")

        for v in self.vehicles:

            if v.gate_name == "gate_B" and v.status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH):

                v.gate_name, v.gate_pos = "gate_D", self.grid.gates["gate_D"]

                v.replan()

                self._append_log(f"[{self.step_no:02d}:{self.step_no:02d}] {v.vid} re-routed to Gate D.")

        if self.algorithm in ("cooperative_astar", "greedy_best_first"):

            _replan_affected(self.grid, self.vehicles, self.table, current_time=self.step_no, algorithm=self.algorithm)

        self.replans += 1

        self._refresh_all()



    def inject_emergency(self):

        """Interactive live event: upgrade an active ground-service vehicle to emergency priority."""

        candidates = [v for v in self.vehicles

                      if not v.is_emergency and v.status not in (VehicleStatus.DONE, VehicleStatus.NO_PATH)]

        if not candidates:

            return

        # Reuse an existing vehicle so the event demonstrates dynamic priority*

        # change without introducing an unrelated fifth vehicle type.*

        target = sorted(candidates, key=lambda v: v.priority)[0]

        target.is_emergency = True

        target.priority = 0

        target.replan()

        self._append_log(f"[{self.step_no:02d}:{self.step_no:02d}] LIVE EVENT: {target.vid} upgraded to emergency priority 0.")

        if self.algorithm in ("cooperative_astar", "greedy_best_first"):

            _replan_affected(self.grid, self.vehicles, self.table, current_time=self.step_no, algorithm=self.algorithm)

        self.replans += 1

        self._refresh_all()



    # --------------------------------------------------------------- Rendering*

    def render(self):

        if self.grid is None:

            return

        self.canvas.delete("all")

        w = max(self.canvas.winfo_width(), 400)

        h = max(self.canvas.winfo_height(), 300)

        cell = min((w - 28) / self.grid.cols, (h - 40) / self.grid.rows)

        ox = (w - cell * self.grid.cols) / 2

        oy = (h - cell * self.grid.rows) / 2 + 8



        # Draw grid cells*

        for r in range(self.grid.rows):

            for c in range(self.grid.cols):

                x0, y0 = ox + c * cell, oy + r * cell

                x1, y1 = x0 + cell, y0 + cell

                typ = int(self.grid.grid[r, c])

                fill = {EMPTY: "#F7F9FB", WALL: "#374151", DEPOT: "#D9E9FF", GATE: "#DFF5E6", CORRIDOR: "#FFF0BF"}.get(typ, "#F7F9FB")

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#D0D5DD")

                if typ == WALL:

                    self.canvas.create_rectangle(x0+3, y0+3, x1-3, y1-3, fill="#30343B", outline="")



        # Depots and gates text*

        for name, (r, c) in self.grid.depots.items():

            x = ox + (c + .5) * cell

            y = oy + (r + .5) * cell

            self.canvas.create_text(x, y, text=name.split("_")[0].upper()[:4], fill=DARK_BLUE,

                                    font=("Segoe UI", max(6, int(cell*0.16)), "bold"))

        for name, (r, c) in self.grid.gates.items():

            x = ox + (c + .5) * cell

            y = oy + (r + .5) * cell

            avail = self.grid.gate_available.get(name, True)

            self.canvas.create_text(x, y, text=name.replace("gate_", "G").upper(),

                                    fill=GREEN if avail else RED,

                                    font=("Segoe UI", max(6, int(cell*0.18)), "bold"))



        # Planned trajectories*

        for idx, v in enumerate(self.vehicles):

            if not v.path:

                continue

            col = AGENT_COLORS.get(v.vtype, "#6B7280")

            pts = [v.pos] + v.path[v.path_index:]

            for a, b in zip(pts, pts[1:]):

                x1, y1 = ox + (a[1]+.5)*cell, oy + (a[0]+.5)*cell

                x2, y2 = ox + (b[1]+.5)*cell, oy + (b[0]+.5)*cell

                self.canvas.create_line(x1, y1, x2, y2, fill=col, width=3 if idx == self.selected_agent else 2,

                                        dash=(5, 3) if idx != self.selected_agent else None)



        # Agent Tokens with Distinct Identifiers*

        for idx, v in enumerate(self.vehicles):

            r, c = v.pos

            cx, cy = ox + (c+.5)*cell, oy + (r+.5)*cell

            col = AGENT_COLORS.get(v.vtype, "#6B7280")

            rad = cell * .32

            self.canvas.create_oval(cx-rad, cy-rad, cx+rad, cy+rad, fill=col,

                                    outline="#FFFFFF" if idx == self.selected_agent else "#1E293B",

                                    width=3 if idx == self.selected_agent else 1.5)

            # Distinct label: EM for emergency, or e.g. F1, F2, F3*

            label = "EM" if v.is_emergency else (v.vid[0] + v.vid[-1] if len(v.vid) > 1 else v.vid)

            self.canvas.create_text(cx, cy, text=label, fill="white",

                                    font=("Segoe UI", max(7, int(cell*.20)), "bold"))



        self.canvas.create_text(10, 12, anchor="nw", text=f"Time Step: t = {self.step_no}",

                                fill=DARK_BLUE, font=("Segoe UI", 11, "bold"))



    def draw_route(self):

        """Draw selected agent route on 12x12 grid without modulo wrap."""

        self.route_canvas.delete("all")

        if not self.vehicles:

            return

        if self.selected_agent >= len(self.vehicles):

            self.selected_agent = 0

        v = self.vehicles[self.selected_agent]



        w = max(self.route_canvas.winfo_width(), 360)

        h = max(self.route_canvas.winfo_height(), 200)

        rows, cols = self.grid.rows, self.grid.cols

        cell = min((w - 30) / cols, (h - 30) / rows)

        ox, oy = (w - cell * cols) / 2, (h - cell * rows) / 2 + 5



        # Background grid dots*

        for r in range(rows):

            for c in range(cols):

                self.route_canvas.create_oval(ox+c*cell+cell*.38, oy+r*cell+cell*.38,

                                              ox+c*cell+cell*.62, oy+r*cell+cell*.62,

                                              fill="#CBD5E1", outline="")



        # Draw real trajectory*

        pts = v.path[v.path_index:] if v.path else []

        if pts:

            route = [v.pos] + pts

            for a, b in zip(route, route[1:]):

                ar, ac = a[0], a[1]

                br, bc = b[0], b[1]

                self.route_canvas.create_line(ox+(ac+.5)*cell, oy+(ar+.5)*cell,

                                              ox+(bc+.5)*cell, oy+(br+.5)*cell,

                                              fill=AGENT_COLORS.get(v.vtype, "#2563EB"), width=3.5)



        # Current Agent position*

        r, c = v.pos[0], v.pos[1]

        self.route_canvas.create_oval(ox+(c+.25)*cell, oy+(r+.25)*cell,

                                      ox+(c+.75)*cell, oy+(r+.75)*cell,

                                      fill=AGENT_COLORS.get(v.vtype, "#2563EB"), outline="#1E293B", width=2)



        # Destination Target*

        target = v.gate_pos if v.active_leg_to_gate else v.depot

        gr, gc = target[0], target[1]

        self.route_canvas.create_rectangle(ox+(gc+.2)*cell, oy+(gr+.2)*cell,

                                           ox+(gc+.8)*cell, oy+(gr+.8)*cell,

                                           fill="#E6F4EA", outline=GREEN, width=2)

        dest_name = v.gate_name.replace("gate_", "Gate ").upper() if v.active_leg_to_gate else "DEPOT"

        self.route_canvas.create_text(ox+8, oy-2, anchor="nw", text=f"{v.vid} -> {dest_name}",

                                      fill=DARK_BLUE, font=("Segoe UI", 9, "bold"))

        u = v.compute_utility()

        self.route_note.set(f"{v.vid} ({AGENT_NAMES.get(v.vtype, 'Ground Vehicle')}): {v.status.name.replace('_',' ').title()} | Priority={v.priority} | Remaining={max(0, len(v.path)-v.path_index)} steps | Utility={u:.1f}")



    def _refresh_inspector(self):

        """Populate Search Inspector with expanded nodes for the selected ground-service vehicle."""

        for item in self.insp_tree.get_children():

            self.insp_tree.delete(item)

        if not self.vehicles:

            return

        if self.selected_agent >= len(self.vehicles):

            self.selected_agent = 0

        v = self.vehicles[self.selected_agent]

        self.insp_agent_var.set(f"{v.vid} (P={v.priority})")



        trace = self.search_traces.get(v.vid, [])

        for node in trace[:120]:

            state_str = f"({node['pos'][0]},{node['pos'][1]},t={node['time']})"

            parent_str = f"({node['parent'][0]},{node['parent'][1]},t={node['parent'][2]})" if node['parent'] else "ROOT"

            self.insp_tree.insert("", "end", values=(

                state_str,

                node['g'],

                f"{node['h']:.1f}" if isinstance(node['h'], float) else node['h'],

                f"{node['f']:.1f}" if isinstance(node['f'], float) else node['f'],

                parent_str

            ))



    def _refresh_all(self):

        self.render()

        self.draw_route()

        self._refresh_inspector()

        self._refresh_status()

        self._refresh_stats()



    def _refresh_status(self):

        for item in self.status_tree.get_children():

            self.status_tree.delete(item)

        for idx, v in enumerate(self.vehicles):

            u = v.compute_utility()

            dest_str = v.gate_name.replace("gate_", "Gate ").upper() if v.active_leg_to_gate else "DEPOT"

            vals = (v.vid, AGENT_NAMES.get(v.vtype, "Ground Vehicle"), str(v.pos), dest_str,

                    v.status.name.replace("_", " ").title(), f"{u:.1f}")

            iid = self.status_tree.insert("", "end", values=vals)

            if idx == self.selected_agent:

                self.status_tree.selection_set(iid)



    def _refresh_stats(self):

        waiting = sum(v.metrics.waiting_time for v in self.vehicles)

        travel = sum(v.metrics.travel_distance for v in self.vehicles)

        runtime = sum(v.metrics.runtime_ms for v in self.vehicles) + self.plan_ms

        self.stats_var.set(

            f"Time Step:             t = {self.step_no}\n"

            f"Total Travel Distance: {travel} cells\n"

            f"Total Waiting Penalty: {waiting} steps\n"

            f"Conflicts Detected:    {self.total_conflicts}\n"

            f"Replanning Events:     {self.replans}\n"

            f"Total Nodes Expanded:  {self.total_nodes}\n"

            f"Planning Runtime:      {runtime:.2f} ms"

        )



    def _refresh_log(self):

        # Event history is retained internally for simulation state/step-back,*

        # but the presentation dashboard intentionally does not display an ATC log panel.*

        return



    def _append_log(self, line):

        self.event_log.append(line)

        self._refresh_log()



    def _append_new_events(self, before):

        for line in self.event_log[before:]:

            self._append_log(f"[{self.step_no:02d}:{self.step_no:02d}] {line}")



    def _status_select(self, event=None):

        sel = self.status_tree.selection()

        if not sel:

            return

        item = self.status_tree.item(sel[0])

        vid = item["values"][0]

        for i, v in enumerate(self.vehicles):

            if v.vid == vid:

                self.selected_agent = i

                self.render()

                self.draw_route()

                self._refresh_inspector()

                break



    def _canvas_click(self, event):

        if not self.vehicles:

            return

        w = max(self.canvas.winfo_width(), 400)

        h = max(self.canvas.winfo_height(), 300)

        cell = min((w - 28) / self.grid.cols, (h - 40) / self.grid.rows)

        ox = (w - cell * self.grid.cols) / 2

        oy = (h - cell * self.grid.rows) / 2 + 8

        c = int((event.x - ox) / cell)

        r = int((event.y - oy) / cell)

        for i, v in enumerate(self.vehicles):

            if v.pos == (r, c):

                self.selected_agent = i

                self._refresh_all()

                break





def launch():

    root = tk.Tk()

    AirportGUI(root)

    root.mainloop()





if __name__ == "__main__":

    launch()
