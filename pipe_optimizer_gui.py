#!/usr/bin/env python3
"""
Pipe Optimizer GUI - Lightweight Tkinter Frontend
==================================================

A minimal GUI wrapper for the V5 pipe optimizer.
Works on Mac and Windows with just Python + a few pip packages.

Requirements:
    pip install openpyxl pulp

Usage:
    python3 pipe_optimizer_gui.py
"""

import os
import sys
import stat
import threading
import queue
from pathlib import Path

# Tkinter is in stdlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def fix_cbc_permissions():
    """
    Fix CBC solver executable permissions in bundled app.
    PyInstaller sometimes strips execute permissions from binaries.
    """
    try:
        import pulp
        pulp_dir = Path(pulp.__file__).parent
        solver_dirs = [
            pulp_dir / "solverdir" / "cbc" / "osx" / "i64" / "cbc",
            pulp_dir / "solverdir" / "cbc" / "linux" / "i64" / "cbc",
            pulp_dir / "solverdir" / "cbc" / "linux" / "i32" / "cbc",
            pulp_dir / "solverdir" / "cbc" / "linux" / "arm64" / "cbc",
        ]
        for solver_path in solver_dirs:
            if solver_path.exists():
                # Add execute permission
                current_mode = solver_path.stat().st_mode
                if not (current_mode & stat.S_IXUSR):
                    solver_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass  # Non-critical - will fail later with better error message


# Fix CBC permissions before importing pulp
fix_cbc_permissions()

# Check for required packages
MISSING_PACKAGES = []
try:
    import pandas as pd
except ImportError:
    MISSING_PACKAGES.append("pandas")

try:
    import numpy as np
except ImportError:
    MISSING_PACKAGES.append("numpy")

try:
    from pulp import LpProblem
except ImportError:
    MISSING_PACKAGES.append("pulp")

try:
    import openpyxl
except ImportError:
    MISSING_PACKAGES.append("openpyxl")

try:
    import psutil
except ImportError:
    MISSING_PACKAGES.append("psutil")

if MISSING_PACKAGES:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Packages",
        f"Please install required packages:\n\npip install {' '.join(MISSING_PACKAGES)}"
    )
    sys.exit(1)

# Import the optimizer (must be in same directory or bundled)
try:
    from pipe_optimizer_v5_safe import (
        load_pipe_data,
        SymmetryAwareSafeSolver,
        MemoryMonitor,
        export_to_excel,
        detect_system_capabilities,
        SystemCapabilities
    )
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Optimizer",
        "Could not find pipe_optimizer_v5_safe.py\n\n"
        "Make sure it's in the same folder as this GUI."
    )
    sys.exit(1)


class PipeOptimizerGUI:
    """Minimal GUI for the pipe optimizer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pipe Pile Optimizer")
        self.root.geometry("600x520")
        self.root.resizable(True, True)

        # Detect system capabilities for adaptive settings
        self.sys_caps = detect_system_capabilities()

        # State
        self.selected_file = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()

        self.setup_ui()

        # Start polling for progress updates
        self.poll_progress()

    def setup_ui(self):
        """Build the UI components."""
        # Main container with padding
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill='both', expand=True)

        # Title
        title = ttk.Label(main, text="Pipe Pile Optimizer", font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 5))

        # --- System Info (adaptive resource detection) ---
        sys_info = f"System: {self.sys_caps.platform} | RAM: {self.sys_caps.available_memory_gb:.0f}GB available | Cores: {self.sys_caps.cpu_cores}"
        sys_label = ttk.Label(main, text=sys_info, foreground='gray', font=('Helvetica', 9))
        sys_label.pack(pady=(0, 10))

        # --- File Selection ---
        file_frame = ttk.LabelFrame(main, text="Input File", padding="5")
        file_frame.pack(fill='x', pady=5)

        self.file_label = ttk.Label(file_frame, text="No file selected", foreground='gray')
        self.file_label.pack(side='left', fill='x', expand=True)

        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side='right')

        # --- Parameters ---
        param_frame = ttk.LabelFrame(main, text="Parameters", padding="5")
        param_frame.pack(fill='x', pady=5)

        # Target length
        ttk.Label(param_frame, text="Target pile length (ft):").grid(row=0, column=0, sticky='w', padx=5)
        self.target_var = tk.StringVar(value="100.0")
        target_entry = ttk.Entry(param_frame, textvariable=self.target_var, width=10)
        target_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Max waste
        ttk.Label(param_frame, text="Max waste per pile (ft):").grid(row=1, column=0, sticky='w', padx=5)
        self.waste_var = tk.StringVar(value="5.0")
        waste_entry = ttk.Entry(param_frame, textvariable=self.waste_var, width=10)
        waste_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        # Tip
        tip = ttk.Label(param_frame, text="Tip: Lower waste = faster solve, but may find fewer piles",
                        foreground='gray', font=('Helvetica', 9))
        tip.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=(5, 0))

        # --- Progress ---
        progress_frame = ttk.LabelFrame(main, text="Progress", padding="5")
        progress_frame.pack(fill='x', pady=5)

        self.progress = ttk.Progressbar(progress_frame, length=400, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)

        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()

        # --- Buttons ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Run Optimizer", command=self.run_optimizer)
        self.run_btn.pack(side='left', padx=5)

        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_optimizer, state='disabled')
        self.cancel_btn.pack(side='left', padx=5)

        # --- Results ---
        results_frame = ttk.LabelFrame(main, text="Results", padding="5")
        results_frame.pack(fill='both', expand=True, pady=5)

        self.results_text = tk.Text(results_frame, height=8, width=60, state='disabled',
                                     font=('Courier', 10))
        self.results_text.pack(fill='both', expand=True)

        # --- Open Output Button (hidden initially) ---
        self.output_file = None
        self.open_btn = ttk.Button(main, text="Open Output File", command=self.open_output)
        # Don't pack yet - will show after successful run

    def browse_file(self):
        """Open file dialog to select input file."""
        path = filedialog.askopenfilename(
            title="Select Pipe Data File",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.selected_file = path
            # Show just filename, not full path
            filename = Path(path).name
            self.file_label.config(text=filename, foreground='black')

    def validate_inputs(self) -> tuple:
        """Validate inputs, return (valid, error_message)."""
        if not self.selected_file:
            return False, "Please select an input file first."

        if not Path(self.selected_file).exists():
            return False, "Selected file no longer exists."

        try:
            target = float(self.target_var.get())
            if target <= 0:
                return False, "Target length must be positive."
        except ValueError:
            return False, "Target length must be a number."

        try:
            waste = float(self.waste_var.get())
            if waste < 0:
                return False, "Max waste cannot be negative."
        except ValueError:
            return False, "Max waste must be a number."

        return True, ""

    def run_optimizer(self):
        """Start the optimization in a background thread."""
        # Validate
        valid, error = self.validate_inputs()
        if not valid:
            messagebox.showerror("Invalid Input", error)
            return

        # Check output file
        input_path = Path(self.selected_file)
        output_path = input_path.parent / f"{input_path.stem}_OPTIMIZED.xlsx"
        if output_path.exists():
            if not messagebox.askyesno("File Exists",
                    f"{output_path.name} already exists.\n\nOverwrite?"):
                return

        self.output_file = str(output_path)

        # Clear previous results
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.config(state='disabled')
        self.open_btn.pack_forget()

        # Update UI state
        self.run_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')
        self.stop_event.clear()
        self.progress.start(10)  # Indeterminate animation

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._optimizer_worker, daemon=True)
        self.worker_thread.start()

    def _optimizer_worker(self):
        """Worker thread that runs the optimizer."""
        try:
            target = float(self.target_var.get())
            waste = float(self.waste_var.get())

            # Load data
            self.progress_queue.put(("status", "Loading data..."))
            pipe_data = load_pipe_data(self.selected_file)

            if self.stop_event.is_set():
                self.progress_queue.put(("cancelled", None))
                return

            self.progress_queue.put(("status", f"Loaded {pipe_data.valid_count} pipes"))

            # Initialize solver with adaptive memory limits
            caps = self.sys_caps
            memory = MemoryMonitor(
                soft_limit_gb=caps.recommended_soft_limit_gb,
                hard_limit_gb=caps.recommended_hard_limit_gb
            )
            solver = SymmetryAwareSafeSolver(
                pipe_data.lengths,
                target_length=target,
                max_waste=waste,
                memory_monitor=memory
            )

            if self.stop_event.is_set():
                self.progress_queue.put(("cancelled", None))
                return

            # Generate patterns
            self.progress_queue.put(("status", "Generating patterns..."))
            patterns = solver.generate_patterns()

            if self.stop_event.is_set():
                self.progress_queue.put(("cancelled", None))
                return

            self.progress_queue.put(("status", f"Found {len(patterns):,} patterns. Solving ({caps.recommended_threads} threads)..."))

            # Solve with adaptive thread count
            solution, status, solve_time = solver.solve_ilp(
                patterns,
                time_limit=1800,
                threads=caps.recommended_threads
            )

            if self.stop_event.is_set():
                self.progress_queue.put(("cancelled", None))
                return

            if solution is None or len(solution) == 0:
                self.progress_queue.put(("error", f"No solution found. Status: {status}"))
                return

            # Export results
            self.progress_queue.put(("status", "Saving results..."))
            export_to_excel(solver, solution, self.output_file)

            # Build results summary
            total_piles = len(solution)
            total_waste = sum(p['waste'] for p in solution)
            theoretical_max = solver.theoretical_max
            efficiency = total_piles / theoretical_max * 100 if theoretical_max else 0

            from collections import Counter
            weld_counts = Counter(p['num_welds'] for p in solution)

            results = f"""OPTIMIZATION COMPLETE

Piles created: {total_piles} / {theoretical_max} ({efficiency:.1f}%)
Total waste: {total_waste:.1f} ft ({total_waste/total_piles:.2f} avg)
Solve time: {solve_time:.1f} seconds
Status: {status}

Weld distribution:
"""
            for welds in sorted(weld_counts.keys()):
                count = weld_counts[welds]
                results += f"  {welds} weld(s): {count} piles\n"

            results += f"\nOutput saved to:\n{self.output_file}"

            self.progress_queue.put(("done", results))

        except Exception as e:
            self.progress_queue.put(("error", str(e)))

    def cancel_optimizer(self):
        """Signal the worker thread to stop."""
        self.stop_event.set()
        self.status_label.config(text="Cancelling...")

    def poll_progress(self):
        """Check the queue for updates from worker thread."""
        try:
            while True:
                msg_type, value = self.progress_queue.get_nowait()

                if msg_type == "status":
                    self.status_label.config(text=value)

                elif msg_type == "done":
                    self.progress.stop()
                    self.status_label.config(text="Complete!")
                    self.run_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    self.results_text.config(state='normal')
                    self.results_text.insert('end', value)
                    self.results_text.config(state='disabled')
                    self.open_btn.pack(pady=5)

                elif msg_type == "error":
                    self.progress.stop()
                    self.status_label.config(text="Error")
                    self.run_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    messagebox.showerror("Error", value)

                elif msg_type == "cancelled":
                    self.progress.stop()
                    self.status_label.config(text="Cancelled")
                    self.run_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')

        except queue.Empty:
            pass

        # Poll again in 100ms
        self.root.after(100, self.poll_progress)

    def open_output(self):
        """Open the output file in the default application."""
        if self.output_file and Path(self.output_file).exists():
            import subprocess
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.output_file])
            elif sys.platform == 'win32':  # Windows
                os.startfile(self.output_file)
            else:  # Linux
                subprocess.run(['xdg-open', self.output_file])

    def run(self):
        """Start the application."""
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        """Clean up on window close."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.worker_thread.join(timeout=1.0)
        self.root.destroy()


def main():
    app = PipeOptimizerGUI()
    app.run()


if __name__ == "__main__":
    # Required for Windows PyInstaller to prevent infinite spawn loops
    import multiprocessing
    multiprocessing.freeze_support()
    main()
