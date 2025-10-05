# src/gui/app.py
# Minimal Tkinter GUI to run your pipeline scripts and capture logs.

from __future__ import annotations
import os
import sys
import threading
import subprocess
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# Resolve project root: two parents up from this file
ROOT = Path(__file__).resolve().parents[2]

# Scripts (adjust names if yours differ)
SCRIPT_DATA          = ROOT / "src" / "00_data_processing_v2.py"
SCRIPT_WINDOWS       = ROOT / "src" / "ml" / "01_make_windows_copy.py"
SCRIPT_TRAIN_GRU     = ROOT / "src" / "ml" / "02_train_gru_copy.py"
SCRIPT_FLOW_TO_TIME  = ROOT / "src" / "ml" / "03_flow_to_time.py"

ARTIFACTS_DIR        = ROOT / "artifacts"

# Helper to check script existence
def must_exist(p: Path, label: str):
    if not p.exists():
        messagebox.showerror("Missing file", f"{label} not found:\n{p}")
        return False
    return True

class Runner:
    """
    Runs a Python script in a background thread and streams output into a Tk text widget.
    """
    def __init__(self, text_widget: tk.Text):
        self.text = text_widget
        self.proc = None
        self.thread = None
        self.queue = queue.Queue()
        self._stop = False

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
            pipe.close()

    def _run(self, cmd):
        try:
            self.append(f"\n$ {' '.join(cmd)}\n")
            # Launch process
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Threads to read stdout/stderr
            t_out = threading.Thread(target=self._pump_stdout, args=(self.proc.stdout, ""), daemon=True)
            t_err = threading.Thread(target=self._pump_stdout, args=(self.proc.stderr, "[ERR] "), daemon=True)
            t_out.start(); t_err.start()

            # Poll output queue and push to text
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
                    break
        except Exception as e:
            self.append(f"\n[FAIL] Exception: {e}\n")

    def run_script(self, script_path: Path, extra_env: dict | None = None):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running. Please wait.")
            return

        if not must_exist(script_path, "Script"):
            return

        # Use current Python interpreter (inside your venv)
        cmd = [sys.executable, str(script_path)]

        # Optional: pass origin/destination via env vars so scripts can read them later if needed
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        # Start thread
        self.thread = threading.Thread(target=self._run, args=(cmd,), daemon=True)
        self.thread.start()

def make_gui():
    root = tk.Tk()
    root.title("TBRGS – Simple GUI")

    # Main frame
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Origin / Destination
    ttk.Label(frame, text="Origin HF (e.g., 249):").grid(row=0, column=0, sticky="w")
    origin_var = tk.StringVar()
    origin_entry = ttk.Entry(frame, textvariable=origin_var, width=20)
    origin_entry.grid(row=0, column=1, sticky="w", padx=(5, 10))

    ttk.Label(frame, text="Destination HF:").grid(row=0, column=2, sticky="w")
    dest_var = tk.StringVar()
    dest_entry = ttk.Entry(frame, textvariable=dest_var, width=20)
    dest_entry.grid(row=0, column=3, sticky="w", padx=(5, 0))

    # Buttons row
    btns = ttk.Frame(frame)
    btns.grid(row=1, column=0, columnspan=4, pady=(8, 8), sticky="w")

    # Log output
    log = tk.Text(frame, height=22, wrap="word", state="disabled")
    log.grid(row=2, column=0, columnspan=4, sticky="nsew")
    frame.rowconfigure(2, weight=1)
    yscroll = ttk.Scrollbar(frame, command=log.yview)
    log.configure(yscrollcommand=yscroll.set)
    yscroll.grid(row=2, column=4, sticky="ns")

    runner = Runner(log)

    def env_from_inputs():
        # You can later read these in your scripts via os.getenv("TBRGS_ORIGIN_HF")
        return {
            "TBRGS_ORIGIN_HF": origin_var.get().strip(),
            "TBRGS_DEST_HF": dest_var.get().strip(),
        }

    # Buttons
    def run_data():
        if must_exist(SCRIPT_DATA, "Data Processing Script"):
            runner.run_script(SCRIPT_DATA, extra_env=env_from_inputs())

    def run_windows():
        if must_exist(SCRIPT_WINDOWS, "Make Windows Script"):
            runner.run_script(SCRIPT_WINDOWS, extra_env=env_from_inputs())

    def run_gru():
        if must_exist(SCRIPT_TRAIN_GRU, "GRU Training Script"):
            runner.run_script(SCRIPT_TRAIN_GRU, extra_env=env_from_inputs())

    def run_flow_to_time():
        if must_exist(SCRIPT_FLOW_TO_TIME, "Flow→Time Script"):
            runner.run_script(SCRIPT_FLOW_TO_TIME, extra_env=env_from_inputs())

    def run_all():
        # Simple chained run (sequential). For real chaining, you could await thread completion,
        # but this keeps it super simple: it prompts the user to click the next step after each completes.
        if not all([
            must_exist(SCRIPT_DATA, "Data Processing"),
            must_exist(SCRIPT_WINDOWS, "Make Windows"),
            must_exist(SCRIPT_TRAIN_GRU, "Train GRU"),
            must_exist(SCRIPT_FLOW_TO_TIME, "Flow→Time")
        ]):
            return
        messagebox.showinfo("Run All", "This will run each step one-by-one. Click each button as the previous finishes.\n\n(Keeping it simple so the UI doesn’t get complex.)")

    ttk.Button(btns, text="1) Data Processing", command=run_data).grid(row=0, column=0, padx=(0,6))
    ttk.Button(btns, text="2) Make Windows", command=run_windows).grid(row=0, column=1, padx=6)
    ttk.Button(btns, text="3) Train GRU", command=run_gru).grid(row=0, column=2, padx=6)
    ttk.Button(btns, text="4) Flow → Time", command=run_flow_to_time).grid(row=0, column=3, padx=6)
    ttk.Button(btns, text="Run All (manual)", command=run_all).grid(row=0, column=4, padx=(12,0))

    # Footer
    footer = ttk.Label(frame, text=f"Artifacts: {ARTIFACTS_DIR}", foreground="#666")
    footer.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6,0))

    return root

if __name__ == "__main__":
    app = make_gui()
    app.minsize(900, 500)
    app.mainloop()
