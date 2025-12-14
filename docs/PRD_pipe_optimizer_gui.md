# PRD: Lightweight Pipe Optimizer GUI

## Problem Statement

The V5 pipe optimizer is powerful but requires command-line usage. Non-technical users need a simple GUI to:
1. Select their spreadsheet file
2. Configure basic parameters
3. See progress while it runs
4. Get results without touching a terminal

## Goals

1. **Zero-dependency GUI** - Use only Python standard library (Tkinter)
2. **Cross-platform** - Works on Mac and Windows without modification
3. **Single file** - One `.py` file that bundles everything
4. **Barebones but powerful** - File picker, progress bar, results display

## Non-Goals

- Fancy styling or themes
- Web interface
- Installation wizard
- Auto-updates

## Technical Design

### Stack Choice: Tkinter

| Option | Pros | Cons |
|--------|------|------|
| **Tkinter** | Built-in, no install, cross-platform | Dated look |
| PySimpleGUI | Cleaner API | Extra dependency |
| Streamlit | Modern UI | Requires browser, heavy |
| Electron | Native feel | Massive overhead |

**Decision: Tkinter** - Zero dependencies, ships with Python on Mac/Windows.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Pipe Optimizer GUI                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │  [Browse...]  pipe_data.xlsx                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Target Length: [100.0] ft    Max Waste: [5.0] ft      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ████████████████░░░░░░░░░░  67%                │   │
│  │  Solving ILP (234/264 piles found)...           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│              [ Run Optimizer ]                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Results:                                        │   │
│  │  ✓ 264 piles (100% of theoretical max)          │   │
│  │  ✓ 43.3' total waste (0.16' average)            │   │
│  │  ✓ Saved to: pipe_optimization_V5_SAFE.xlsx     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Components

#### 1. File Selector
```python
from tkinter import filedialog

def select_file():
    path = filedialog.askopenfilename(
        title="Select Pipe Data",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    return path
```

#### 2. Parameter Inputs
- Target length (default: 100.0)
- Max waste (default: 5.0)
- Output filename (auto-generated from input)

#### 3. Progress Display
```python
# Run optimizer in separate thread to keep UI responsive
import threading

def run_optimization():
    thread = threading.Thread(target=_run_optimizer_thread)
    thread.start()

def _run_optimizer_thread():
    # Update progress bar via queue
    progress_queue.put(("status", "Loading data..."))
    progress_queue.put(("progress", 10))
    # ... run optimizer phases
```

#### 4. Results Panel
- Piles created vs theoretical max
- Total waste
- Output file location
- Open output button

### File Structure

```
pipes/
├── pipe_optimizer_gui.py      # Single-file GUI application
├── pipe_optimizer_v5_safe.py  # Core optimizer (imported)
└── ...
```

### Threading Model

```
┌─────────────────┐     ┌─────────────────┐
│   Main Thread   │     │  Worker Thread  │
│   (Tkinter UI)  │     │  (Optimizer)    │
├─────────────────┤     ├─────────────────┤
│ - Handle clicks │     │ - Load data     │
│ - Update UI     │◄────│ - Generate pats │
│ - Read queue    │     │ - Solve ILP     │
│                 │     │ - Write results │
└─────────────────┘     └─────────────────┘
        ▲                       │
        └───── Queue ───────────┘
```

### Implementation Details

#### Main Window Setup
```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue

class PipeOptimizerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pipe Pile Optimizer")
        self.root.geometry("500x400")

        self.progress_queue = queue.Queue()
        self.setup_ui()

    def setup_ui(self):
        # File selection
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(file_frame, text="Browse...",
                   command=self.browse_file).pack(side='left')
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side='left', padx=10)

        # Parameters
        param_frame = ttk.Frame(self.root)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Target (ft):").pack(side='left')
        self.target_var = tk.StringVar(value="100.0")
        ttk.Entry(param_frame, textvariable=self.target_var,
                  width=8).pack(side='left', padx=5)

        ttk.Label(param_frame, text="Max Waste (ft):").pack(side='left')
        self.waste_var = tk.StringVar(value="5.0")
        ttk.Entry(param_frame, textvariable=self.waste_var,
                  width=8).pack(side='left', padx=5)

        # Progress
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=10)

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack()

        # Run button
        self.run_btn = ttk.Button(self.root, text="Run Optimizer",
                                   command=self.run_optimizer)
        self.run_btn.pack(pady=10)

        # Results
        self.results_text = tk.Text(self.root, height=8, width=55)
        self.results_text.pack(padx=10, pady=5)

    def run(self):
        self.root.mainloop()
```

#### Progress Updates from Worker Thread
```python
def update_progress(self):
    """Called periodically to check queue and update UI"""
    try:
        while True:
            msg_type, value = self.progress_queue.get_nowait()
            if msg_type == "progress":
                self.progress['value'] = value
            elif msg_type == "status":
                self.status_label.config(text=value)
            elif msg_type == "result":
                self.results_text.insert('end', value + '\n')
            elif msg_type == "done":
                self.run_btn.config(state='normal')
                return
    except queue.Empty:
        pass

    # Check again in 100ms
    self.root.after(100, self.update_progress)
```

### Error Handling

| Error | User Message |
|-------|--------------|
| File not found | "Could not find the selected file" |
| Invalid format | "Unsupported file format. Use Excel or CSV" |
| No length column | "Could not find pipe length column. Expected: length, size, etc." |
| No valid data | "No valid pipe lengths found in file" |
| Solver timeout | "Solver timed out. Try increasing max waste or reducing data size" |
| Memory error | "Not enough memory. Try closing other applications" |

### Packaging Options

#### Option A: Single .py file (Recommended for now)
```bash
# User just runs:
python3 pipe_optimizer_gui.py
```

#### Option B: PyInstaller executable (Future)
```bash
# Creates standalone .app or .exe
pyinstaller --onefile --windowed pipe_optimizer_gui.py
```

### Success Criteria

1. **Works on Mac**: Double-click or `python3 pipe_optimizer_gui.py`
2. **Works on Windows**: Double-click or `python pipe_optimizer_gui.py`
3. **No pip installs** required (except openpyxl for Excel output)
4. **Complete run** in under 5 minutes for typical datasets
5. **Clear errors** if something goes wrong

### Dependencies

Required (usually pre-installed):
- Python 3.8+
- tkinter (included with Python on Mac/Windows)

Optional (for full functionality):
- openpyxl (Excel read/write)
- psutil (memory monitoring)
- pulp (solver - required)

### Installation Instructions for Users

```
PIPE OPTIMIZER - Quick Start
============================

1. Make sure Python 3 is installed
   - Mac: Usually pre-installed, or `brew install python`
   - Windows: Download from python.org

2. Install required packages (one time):
   pip3 install openpyxl pulp psutil

3. Run the application:
   python3 pipe_optimizer_gui.py

4. Click "Browse" to select your Excel/CSV file
5. Adjust target length and max waste if needed
6. Click "Run Optimizer"
7. Results saved to same folder as input file
```

### Open Questions

1. Should we bundle into a standalone .app/.exe? (Requires PyInstaller, larger file)
2. Should we add a "column selector" dropdown after file load?
3. Should we show a preview of the data before running?

## Timeline

Immediate implementation - single Python file with Tkinter.
