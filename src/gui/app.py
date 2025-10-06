from __future__ import annotations
import os
import sys
import threading
import subprocess
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

# Python scripts
SCRIPT_DATA          = ROOT / "src" / "00_data_processing.py"
SCRIPT_WINDOWS       = ROOT / "src" / "ml" / "01_make_windows.py"
SCRIPT_TRAIN_GRU     = ROOT / "src" / "ml" / "02_train_gru.py"
SCRIPT_TRAIN_LSTM    = ROOT / "src" / "ml" / "02_train_lstm.py"   # optional
SCRIPT_FLOW_TO_TIME  = ROOT / "src" / "ml" / "03_flow_to_time.py"
SCRIPT_CREATE_GRAPH  = ROOT / "src" / "routing" / "05_create_graph.py"  # placeholder

# RouteFinder executable (published single-file exe)
SCRIPT_ROUTE_FINDER_EXE = ROOT / "src" / "routing" / "RouteFinder.exe"

# Expected outputs that unlock subsequent steps
EXPECTED_AFTER_1 = ART / "processed_data.csv"
EXPECTED_AFTER_2 = ART / "window_meta.csv"
EXPECTED_GRU_OUT  = ART / "gru_predictions.csv"
EXPECTED_LSTM_OUT = ART / "lstm_predictions.csv"   # adjust if different
EXPECTED_AFTER_4 = ART / "flow_speed_times.csv"

def must_exist(p: Path, label: str):
    if not p.exists():
        messagebox.showerror("Missing file", f"{label} not found:\n{p}")
        return False
    return True

class Runner:
    """
    Run a process in the background (Python script or any executable),
    stream stdout/stderr to a Text widget, then call on_complete(exit_code).
    """
    def __init__(self, root: tk.Tk, text_widget: tk.Text):
        self.root = root
        self.text = text_widget
        self.proc = None
        self.thread = None
        self.queue = queue.Queue()
        self._stop = False
        self.on_complete = None  # callback set by caller

    def append(self, msg: str):
        self.text.configure(state="normal")
        self.text.insert("end", msg)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _pump_stdout(self, pipe, prefix=""):
        try:
            for line in iter(pipe.readline, b""):
                if self._stop:
                    break
                try:
                    s = line.decode(errors="replace")
                except Exception:
                    s = str(line)
                self.queue.put(prefix + s)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _run(self, cmd, env=None, cwd: Path | None = None):
        try:
            if cwd:
                self.append(f"\n$ (cwd={cwd}) {' '.join(cmd)}\n")
            else:
                self.append(f"\n$ {' '.join(cmd)}\n")
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env or os.environ.copy(),
            )
            t_out = threading.Thread(target=self._pump_stdout, args=(self.proc.stdout, ""), daemon=True)
            t_err = threading.Thread(target=self._pump_stdout, args=(self.proc.stderr, "[ERR] "), daemon=True)
            t_out.start(); t_err.start()

            while True:
                try:
                    chunk = self.queue.get(timeout=0.1)
                    self.append(chunk)
                except queue.Empty:
                    pass
                ret = self.proc.poll()
                if ret is not None:
                    # Drain anything left
                    while True:
                        try:
                            chunk = self.queue.get_nowait()
                            self.append(chunk)
                        except queue.Empty:
                            break
                    if ret == 0:
                        self.append("\n[OK] Finished.\n")
                    else:
                        self.append(f"\n[FAIL] Exit code {ret}\n")
                    if self.on_complete:
                        self.root.after(0, self.on_complete, ret)
                    break
        except Exception as e:
            self.append(f"\n[FAIL] Exception: {e}\n")
            if self.on_complete:
                self.root.after(0, self.on_complete, 1)

    def run_script(self, script_path: Path, extra_env: dict | None = None):
        """Run a Python script with the current interpreter."""
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running. Please wait.")
            return
        if not must_exist(script_path, "Script"):
            return
        cmd = [sys.executable, str(script_path)]
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        self.thread = threading.Thread(target=self._run, args=(cmd, env, None), daemon=True)
        self.thread.start()

    def run_cmd(self, cmd: list[str], extra_env: dict | None = None, cwd: Path | None = None):
        """Run any command (e.g., RouteFinder.exe) and stream output."""
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running. Please wait.")
            return
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        self.thread = threading.Thread(target=self._run, args=(cmd, env, cwd), daemon=True)
        self.thread.start()

def make_gui():
    root = tk.Tk()
    root.title("TBRGS – COS30019 Assignment 2B")

    # ---- Layout: 2 columns (left log, right sidebar) ----
    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Left log
    log = tk.Text(main, height=24, wrap="word", state="disabled")
    log.grid(row=0, column=0, sticky="nsew", padx=(0,10))
    main.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)
    yscroll = ttk.Scrollbar(main, command=log.yview)
    log.configure(yscrollcommand=yscroll.set)
    yscroll.grid(row=0, column=1, sticky="ns")

    # Right sidebar (top: steps; middle: statuses; bottom: inputs + route button)
    side = ttk.Frame(main)
    side.grid(row=0, column=2, sticky="nsew")
    side.rowconfigure(0, weight=0)   # steps block
    side.rowconfigure(1, weight=1)   # status block
    side.rowconfigure(2, weight=0)   # bottom inputs

    # ========== TOP: Step buttons (keep layout) ==========
    steps = ttk.Frame(side)
    steps.grid(row=0, column=0, sticky="ew")
    for c in range(2):
        steps.columnconfigure(c, weight=1)

    btn1 = ttk.Button(steps, text="1) Data Processing")
    btn2 = ttk.Button(steps, text="2) Make Windows")

    # Step 3 row = button + model dropdown
    step3_row = ttk.Frame(steps)
    step3_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
    step3_row.columnconfigure(0, weight=1)
    step3_row.columnconfigure(1, weight=0)

    btn3 = ttk.Button(step3_row, text="3) Train selected model")
    model_var = tk.StringVar(value="GRU")
    model_combo = ttk.Combobox(step3_row, textvariable=model_var, state="readonly",
                               values=["GRU", "LSTM", "Custom (disabled)"], width=20)

    btn4 = ttk.Button(steps, text="4) Flow → Time")
    btn5 = ttk.Button(steps, text="5) Create Graph")   # placeholder

    btn1.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,4))
    btn2.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
    btn3.grid(row=0, column=0, sticky="ew", in_=step3_row, padx=(0, 8))
    model_combo.grid(row=0, column=1, sticky="e", in_=step3_row)
    btn4.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
    btn5.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4,0))

    # ========== MIDDLE: Artifact status lines ==========
    status = ttk.LabelFrame(side, text="Status")
    status.grid(row=1, column=0, sticky="nsew", pady=(10, 8))
    status.columnconfigure(0, weight=1)

    s1_var = tk.StringVar()
    s2_var = tk.StringVar()
    s3_var = tk.StringVar()
    s4_var = tk.StringVar()

    s1 = ttk.Label(status, textvariable=s1_var, anchor="w")
    s2 = ttk.Label(status, textvariable=s2_var, anchor="w")
    s3 = ttk.Label(status, textvariable=s3_var, anchor="w")
    s4 = ttk.Label(status, textvariable=s4_var, anchor="w")

    s1.grid(row=0, column=0, sticky="w", padx=6, pady=2)
    s2.grid(row=1, column=0, sticky="w", padx=6, pady=2)
    s3.grid(row=2, column=0, sticky="w", padx=6, pady=2)
    s4.grid(row=3, column=0, sticky="w", padx=6, pady=2)

    # ========== BOTTOM: Origin/Destination + Algorithm + Route Finder ==========
    bottom = ttk.Frame(side)
    bottom.grid(row=2, column=0, sticky="sew", pady=(6,0))
    bottom.columnconfigure(0, weight=1)
    bottom.columnconfigure(1, weight=1)

    ttk.Label(bottom, text="Origin HF:").grid(row=0, column=0, sticky="w", pady=(0,4))
    origin_var = tk.StringVar()
    ent_origin = ttk.Entry(bottom, textvariable=origin_var, width=18)
    ent_origin.grid(row=0, column=1, sticky="e", pady=(0,4))

    ttk.Label(bottom, text="Destination HF:").grid(row=1, column=0, sticky="w", pady=(0,8))
    dest_var = tk.StringVar()
    ent_dest = ttk.Entry(bottom, textvariable=dest_var, width=18)
    ent_dest.grid(row=1, column=1, sticky="e", pady=(0,8))

    ttk.Label(bottom, text="Route algorithm:").grid(row=2, column=0, sticky="w", pady=(0,6))
    algo_var = tk.StringVar(value="A* Search")
    algo_combo = ttk.Combobox(
        bottom, textvariable=algo_var, state="readonly", width=26,
        values=[
            "Depth-First Search (DFS)",
            "Breadth-First Search (BFS)",
            "Greedy Best-First Search (GBFS)",
            "A* Search",
            "Dijkstra's",
            "Iterative Deepening A* (IDA*)"
        ]
    )
    algo_combo.grid(row=2, column=1, sticky="e", pady=(0,6))

    # Route Finder button (green)
    btn6 = tk.Button(
        bottom,
        text="6) Launch Route Finder",
        bg="#28a745", fg="white",
        activebackground="#218838", activeforeground="white",
        relief="raised", padx=12, pady=6
    )
    btn6.grid(row=3, column=0, columnspan=2, sticky="ew")

    # Footer path hint
    footer = ttk.Label(side, text=f"Artifacts: {ART}", foreground="#666")
    footer.grid(row=3, column=0, sticky="w", pady=(8,0))

    runner = Runner(root, log)

    # ------------ helpers ------------
    def env_from_inputs():
        return {
            "TBRGS_ORIGIN_HF": origin_var.get().strip(),
            "TBRGS_DEST_HF": dest_var.get().strip(),
            "TBRGS_ROUTE_ALGO": algo_var.get(),
            "TBRGS_MODEL": model_var.get(),
        }

    def set_btn_state(widget, enabled: bool):
        try:
            widget.configure(state=("normal" if enabled else "disabled"))
        except Exception:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                pass

    def mark(path: Path, label: str) -> str:
        return f"{'✅' if path.exists() else '❌'} {label} – {path.name}"

    def refresh_status_labels():
        s1_var.set(mark(EXPECTED_AFTER_1, "After Step 1"))
        s2_var.set(mark(EXPECTED_AFTER_2, "After Step 2"))
        have_gru = EXPECTED_GRU_OUT.exists()
        have_lstm = EXPECTED_LSTM_OUT.exists()
        s3_label = "After Step 3 (predictions)"
        if have_gru and have_lstm:
            s3_var.set(f"✅ {s3_label} – {EXPECTED_GRU_OUT.name} / {EXPECTED_LSTM_OUT.name}")
        elif have_gru:
            s3_var.set(f"✅ {s3_label} – {EXPECTED_GRU_OUT.name}")
        elif have_lstm:
            s3_var.set(f"✅ {s3_label} – {EXPECTED_LSTM_OUT.name}")
        else:
            s3_var.set(f"❌ {s3_label} – {EXPECTED_GRU_OUT.name} or {EXPECTED_LSTM_OUT.name}")
        s4_var.set(mark(EXPECTED_AFTER_4, "After Step 4"))

    def check_outputs_and_gate():
        """Enable/disable buttons based on files present; called at start and after each step."""
        refresh_status_labels()

        # Step 1 always available
        set_btn_state(btn1, True)

        # Step 2 requires processed_data.csv
        step1_ok = EXPECTED_AFTER_1.exists()
        set_btn_state(btn2, step1_ok)

        # Step 3 requires window_meta.csv
        step2_ok = EXPECTED_AFTER_2.exists()
        set_btn_state(btn3, step2_ok)
        model_combo.configure(state=("readonly" if step2_ok else "disabled"))

        # Step 4 requires either GRU or LSTM predictions
        step3_ok = EXPECTED_GRU_OUT.exists() or EXPECTED_LSTM_OUT.exists()
        set_btn_state(btn4, step3_ok)

        # Step 5 enabled once Step 4 done (placeholder)
        step4_ok = EXPECTED_AFTER_4.exists()
        set_btn_state(btn5, step4_ok)

        # Inputs & algorithm dropdown only make sense once routing is possible
        ent_origin.configure(state=("normal" if step4_ok else "disabled"))
        ent_dest.configure(state=("normal" if step4_ok else "disabled"))
        algo_combo.configure(state=("readonly" if step4_ok else "disabled"))

        # RouteFinder button requires flow_speed_times.csv AND the exe AND Pathfinder.txt somewhere
        route_ok = step4_ok and SCRIPT_ROUTE_FINDER_EXE.exists()
        set_btn_state(btn6, route_ok)

    # call once at launch
    check_outputs_and_gate()

    # Map label -> algorithm code RouteFinder.exe expects (e.g., 'bfs')
    def map_algo_label_to_positional(label: str) -> str:
        m = {
            "Depth-First Search (DFS)": "dfs",
            "Breadth-First Search (BFS)": "bfs",
            "Greedy Best-First Search (GBFS)": "gbfs",
            "A* Search": "astar",
            "Dijkstra's": "dsa",
            "Iterative Deepening A* (IDA*)": "idastar",
        }
        return m.get(label, "astar")

    # ------------ button handlers ------------
    def on_complete_refresh(exit_code: int):
        check_outputs_and_gate()

    def run_data():
        if not must_exist(SCRIPT_DATA, "Data Processing Script"):
            return
        runner.on_complete = on_complete_refresh
        runner.run_script(SCRIPT_DATA, extra_env=env_from_inputs())

    def run_windows():
        if not must_exist(SCRIPT_WINDOWS, "Make Windows Script"):
            return
        runner.on_complete = on_complete_refresh
        runner.run_script(SCRIPT_WINDOWS, extra_env=env_from_inputs())

    def run_train_selected():
        if not must_exist(SCRIPT_WINDOWS, "Make Windows Script (Step 2)"):
            return
        model = model_var.get()
        if model == "Custom (disabled)":
            messagebox.showinfo("Not available", "Custom model is not available yet.")
            return
        if model == "GRU":
            if not must_exist(SCRIPT_TRAIN_GRU, "GRU Training Script"):
                return
            script = SCRIPT_TRAIN_GRU
        elif model == "LSTM":
            if not SCRIPT_TRAIN_LSTM.exists():
                messagebox.showwarning("Missing script", f"{SCRIPT_TRAIN_LSTM} not found. Select GRU or add the LSTM script.")
                return
            script = SCRIPT_TRAIN_LSTM
        else:
            messagebox.showwarning("Unknown model", f"Unsupported model: {model}")
            return

        runner.on_complete = on_complete_refresh
        runner.run_script(script, extra_env=env_from_inputs())

    def run_flow_to_time():
        if not (EXPECTED_GRU_OUT.exists() or EXPECTED_LSTM_OUT.exists()):
            messagebox.showwarning("Missing predictions", "Run Step 3 first (GRU or LSTM) to produce predictions.")
            return
        if not must_exist(SCRIPT_FLOW_TO_TIME, "Flow→Time Script"):
            return
        runner.on_complete = on_complete_refresh
        runner.run_script(SCRIPT_FLOW_TO_TIME, extra_env=env_from_inputs())

    def run_create_graph():
        messagebox.showinfo(
            "Create Graph (Step 5)",
            "Placeholder – script not implemented yet.\n\n"
            f"Expected path:\n{SCRIPT_CREATE_GRAPH}\n\n"
            "When ready, it should consume flow_speed_times.csv and export artifacts/Pathfinder.txt."
        )
        check_outputs_and_gate()

    def run_route_finder():
        origin = origin_var.get().strip()
        dest = dest_var.get().strip()
        algo_label = algo_var.get()

        if not origin or not dest:
            messagebox.showwarning("Missing inputs", "Please enter Origin and Destination HF IDs.")
            return
        if not EXPECTED_AFTER_4.exists():
            messagebox.showwarning("Missing data", "Need flow_speed_times.csv first (run Step 4).")
            return
        if not SCRIPT_ROUTE_FINDER_EXE.exists():
            messagebox.showwarning("Missing executable", f"Route finder EXE not found:\n{SCRIPT_ROUTE_FINDER_EXE}")
            return

        # Ensure Pathfinder.txt is present in the EXE folder
        exe_dir = SCRIPT_ROUTE_FINDER_EXE.parent
        graph_filename = "Pathfinder.txt"  # case-insensitive on Windows
        graph_in_exe = exe_dir / graph_filename
        if not graph_in_exe.exists():
            alt = ART / graph_filename
            if alt.exists():
                try:
                    import shutil
                    shutil.copy2(alt, graph_in_exe)
                    runner.append(f"[INFO] Copied {alt} → {graph_in_exe}\n")
                except Exception as e:
                    runner.append(f"[WARN] Could not copy {alt} to {graph_in_exe}: {e}\n")
            else:
                messagebox.showwarning("Missing graph file",
                                       f"{graph_filename} not found in {exe_dir} or {ART}.\n"
                                       "Create it first (Step 5).")
                return

        # Map dropdown label to short code (e.g., "bfs")
        algo_code = map_algo_label_to_positional(algo_label)

        # Minimal positional command, matching the working .bat:
        #   RouteFinder.exe Pathfinder.txt bfs
        cmd = [str(SCRIPT_ROUTE_FINDER_EXE), graph_filename, algo_code]

        runner.on_complete = on_complete_refresh
        # Run with cwd set to the EXE folder so the relative graph filename resolves as in the .bat
        runner.run_cmd(cmd, extra_env=env_from_inputs(), cwd=exe_dir)

    btn1.configure(command=run_data)
    btn2.configure(command=run_windows)
    btn3.configure(command=run_train_selected)
    btn4.configure(command=run_flow_to_time)
    btn5.configure(command=run_create_graph)
    btn6.configure(command=run_route_finder)

    # keep gating in sync when model selection changes
    def on_model_change(_evt=None):
        check_outputs_and_gate()
    model_combo.bind("<<ComboboxSelected>>", on_model_change)

    return root

if __name__ == "__main__":
    app = make_gui()
    app.minsize(1100, 600)
    app.mainloop()
