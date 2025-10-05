# src/gui/app.py
# Minimal Tkinter GUI with:
# - Left: live log console
# - Right: controls (origin/dest + buttons)
# - Step gating: buttons 2+ are disabled until required files exist
# - Re-checks outputs after each run and enables the next step
#
# NOTE: Steps 5 & 6 are placeholders (message dialog only).
# Adjust EXPECTED_* paths if your filenames differ.

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

# Scripts
SCRIPT_DATA          = ROOT / "src" / "00_data_processing_v2.py"
SCRIPT_WINDOWS       = ROOT / "src" / "ml" / "01_make_windows_copy.py"
SCRIPT_TRAIN_GRU     = ROOT / "src" / "ml" / "02_train_gru_copy.py"
SCRIPT_FLOW_TO_TIME  = ROOT / "src" / "ml" / "03_flow_to_time.py"
SCRIPT_CREATE_GRAPH  = ROOT / "src" / "routing" / "05_create_graph.py"    # placeholder
SCRIPT_ROUTE_FINDER  = ROOT / "src" / "routing" / "06_route_finder.py"    # placeholder

# Expected outputs that unlock subsequent steps
EXPECTED_AFTER_1 = ART / "features_hourly.csv"
EXPECTED_AFTER_2 = ART / "windows_train.npz"
EXPECTED_AFTER_3 = ART / "gru_predictions.csv"
EXPECTED_AFTER_4 = ART / "flow_speed_times.csv"

def must_exist(p: Path, label: str):
    if not p.exists():
        messagebox.showerror("Missing file", f"{label} not found:\n{p}")
        return False
    return True

class Runner:
    """Run a script in background, stream stdout/stderr to a Text widget, then call on_complete(exit_code)."""
    def __init__(self, root: tk.Tk, text_widget: tk.Text):
        self.root = root
        self.text = text_widget
        self.proc = None
        self.thread = None
        self.queue = queue.Queue()
        self._stop = False
        self.on_complete = None  # set by caller

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

    def _run(self, cmd, env=None):
        try:
            self.append(f"\n$ {' '.join(cmd)}\n")
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env or os.environ.copy(),
            )
            t_out = threading.Thread(target=self._pump_stdout, args=(self.proc.stdout, ""), daemon=True)
            t_err = threading.Thread(target=self._pump_stdout, args=(self.proc.stderr, "[ERR] "), daemon=True)
            t_out.start(); t_err.start()

            # Event loop: pull text & check completion
            while True:
                try:
                    chunk = self.queue.get(timeout=0.1)
                    self.append(chunk)
                except queue.Empty:
                    pass
                ret = self.proc.poll()
                if ret is not None:
                    # Drain remaining messages
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
                    # Notify UI thread when done
                    if self.on_complete:
                        self.root.after(0, self.on_complete, ret)
                    break
        except Exception as e:
            self.append(f"\n[FAIL] Exception: {e}\n")
            if self.on_complete:
                self.root.after(0, self.on_complete, 1)

    def run_script(self, script_path: Path, extra_env: dict | None = None):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running. Please wait.")
            return
        if not must_exist(script_path, "Script"):
            return

        cmd = [sys.executable, str(script_path)]
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        self.thread = threading.Thread(target=self._run, args=(cmd, env), daemon=True)
        self.thread.start()

def make_gui():
    root = tk.Tk()
    root.title("TBRGS – Simple GUI")

    # ---- Layout: 2 columns ----
    # Left: log (grows)
    # Right: sidebar controls
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

    # Right control panel
    side = ttk.Frame(main)
    side.grid(row=0, column=2, sticky="n")

    # Style: green Route Finder button
    style = ttk.Style()
    # Try a green style; not all ttk themes honor background; still improves on most platforms
    style.configure("Green.TButton", foreground="white", background="#2e7d32")
    style.map("Green.TButton",
              background=[("active", "#1b5e20"), ("!disabled", "#2e7d32")],
              foreground=[("disabled", "#dddddd"), ("!disabled", "white")])

    # Origin / Destination inputs on the right
    ttk.Label(side, text="Origin HF:").grid(row=0, column=0, sticky="w", pady=(0,4))
    origin_var = tk.StringVar()
    ttk.Entry(side, textvariable=origin_var, width=18).grid(row=0, column=1, sticky="w", pady=(0,4))

    ttk.Label(side, text="Destination HF:").grid(row=1, column=0, sticky="w", pady=(0,8))
    dest_var = tk.StringVar()
    ttk.Entry(side, textvariable=dest_var, width=18).grid(row=1, column=1, sticky="w", pady=(0,8))

    # Buttons (stacked on the right)
    btn1 = ttk.Button(side, text="1) Data Processing")
    btn2 = ttk.Button(side, text="2) Make Windows")
    btn3 = ttk.Button(side, text="3) Train GRU")
    btn4 = ttk.Button(side, text="4) Flow → Time")
    btn5 = ttk.Button(side, text="5) Create Graph")       # placeholder
    btn6 = ttk.Button(side, text="6) Route Finder", style="Green.TButton")  # green, placeholder

    btn1.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0,4))
    btn2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
    btn3.grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)
    btn4.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
    btn5.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4,4))
    btn6.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8,0))

    footer = ttk.Label(side, text=f"Artifacts: {ART}", foreground="#666")
    footer.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10,0))

    runner = Runner(root, log)

    def env_from_inputs():
        return {
            "TBRGS_ORIGIN_HF": origin_var.get().strip(),
            "TBRGS_DEST_HF": dest_var.get().strip(),
        }

    # --- gating / file checks ---
    def set_btn_state(widget: ttk.Button, enabled: bool):
        widget.state(["!disabled"] if enabled else ["disabled"])

    def check_outputs_and_gate():
        """Called on startup and after each successful step to enable next buttons."""
        # Step 1 is always available (assuming raw data present); we keep it enabled
        set_btn_state(btn1, True)

        # Step 2 requires features_hourly.csv
        step1_ok = EXPECTED_AFTER_1.exists()
        set_btn_state(btn2, step1_ok)

        # Step 3 requires windows_train.npz
        step2_ok = EXPECTED_AFTER_2.exists()
        set_btn_state(btn3, step2_ok)

        # Step 4 requires gru_predictions.csv
        step3_ok = EXPECTED_AFTER_3.exists()
        set_btn_state(btn4, step3_ok)

        # Steps 5 & 6 (placeholders) enabled only after flow_speed_times.csv exists
        step4_ok = EXPECTED_AFTER_4.exists()
        set_btn_state(btn5, step4_ok)
        set_btn_state(btn6, step4_ok)

    # call once at launch
    check_outputs_and_gate()

    # --- button handlers ---
    def on_complete_refresh(exit_code: int):
        # After any step completes, refresh gating.
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

    def run_gru():
        if not must_exist(SCRIPT_TRAIN_GRU, "GRU Training Script"):
            return
        runner.on_complete = on_complete_refresh
        runner.run_script(SCRIPT_TRAIN_GRU, extra_env=env_from_inputs())

    def run_flow_to_time():
        if not must_exist(SCRIPT_FLOW_TO_TIME, "Flow→Time Script"):
            return
        runner.on_complete = on_complete_refresh
        runner.run_script(SCRIPT_FLOW_TO_TIME, extra_env=env_from_inputs())

    def run_create_graph():
        messagebox.showinfo(
            "Create Graph (Step 5)",
            "Placeholder – script not implemented yet.\n\n"
            f"Expected path:\n{SCRIPT_CREATE_GRAPH}\n\n"
            "When ready, it should consume flow_speed_times.csv and export your graph/edges."
        )
        # Still refresh gating (no-op now)
        check_outputs_and_gate()

    def run_route_finder():
        messagebox.showinfo(
            "Route Finder (Step 6)",
            "Placeholder – script not implemented yet.\n\n"
            f"Expected path:\n{SCRIPT_ROUTE_FINDER}\n\n"
            "It will load the graph + time profiles and compute top-k routes."
        )
        check_outputs_and_gate()

    btn1.configure(command=run_data)
    btn2.configure(command=run_windows)
    btn3.configure(command=run_gru)
    btn4.configure(command=run_flow_to_time)
    btn5.configure(command=run_create_graph)
    btn6.configure(command=run_route_finder)

    return root

if __name__ == "__main__":
    app = make_gui()
    app.minsize(1050, 560)
    app.mainloop()
