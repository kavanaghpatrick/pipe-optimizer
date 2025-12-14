# PRD: Pipe Optimizer V4

## Overview

Upgrade the pipe pile optimizer from a hardcoded script to a flexible, production-ready CLI tool that accepts any spreadsheet format and provides comprehensive output options.

**Base**: `pipe_optimizer_v3.py` (bug-fixed ILP solver achieving 98.9% efficiency)

---

## Goals

1. **Input Flexibility**: Accept any CSV/Excel file with automatic column detection
2. **Parameterization**: All constraints configurable via CLI arguments
3. **Robustness**: Comprehensive error handling with actionable feedback
4. **Output Options**: Excel + JSON export, better console visualization
5. **Performance**: Parallel pattern generation for large datasets

## Non-Goals

- GUI/Streamlit interface (separate future project)
- Real-time streaming output
- Database integration
- Cloud deployment

---

## Functional Requirements

### FR1: Universal Input Loader

| Requirement | Details |
|-------------|---------|
| FR1.1 | Support file formats: `.csv`, `.tsv`, `.xlsx`, `.xls` |
| FR1.2 | Auto-detect length column by name: `length`, `len`, `pipe_length`, `size`, `l` (case-insensitive) |
| FR1.3 | Fallback: use first numeric column if no name match |
| FR1.4 | Handle files with or without headers |
| FR1.5 | Support custom delimiters for CSV (`,`, `;`, `\t`) |
| FR1.6 | Filter out non-positive values and NaN |

**Implementation**:
```python
def load_pipe_data(file_path: str, column_name: str = None,
                   delimiter: str = ',', has_header: bool = True) -> np.ndarray:
```

### FR2: CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input`, `-i` | str | `pipe_lengths_clean.csv` | Input file path |
| `--output`, `-o` | str | `pipe_optimization_V4.xlsx` | Output Excel file |
| `--column`, `-c` | str | `None` (auto-detect) | Length column name |
| `--delimiter` | str | `,` | CSV delimiter |
| `--has-header` | flag | `True` | File has header row |
| `--target`, `-t` | float | `100.0` | Target pile length (ft) |
| `--waste`, `-w` | float | `20.0` | Max waste per pile (ft) |
| `--precision`, `-p` | int | `1` | Decimal places for rounding |
| `--threads` | int | `14` | Solver threads |
| `--time-limit` | int | `900` | Solver time limit (seconds) |
| `--gap` | float | `0.005` | Stop if optimality gap < this (0.5%) |
| `--json` | flag | `False` | Also export JSON output |
| `--quiet`, `-q` | flag | `False` | Suppress progress output |

**Usage Examples**:
```bash
# Default (backward compatible with V3)
python3 pipe_optimizer_v4.py

# Custom input file with auto-detection
python3 pipe_optimizer_v4.py -i my_pipes.xlsx --has-header

# Full customization
python3 pipe_optimizer_v4.py -i data.csv -c "Pipe Length" -t 120 -w 15 -o results.xlsx --json

# Quick run with relaxed optimality
python3 pipe_optimizer_v4.py -i data.csv --gap 0.01 --time-limit 300
```

### FR3: Input Validation

| Validation | Error Message |
|------------|---------------|
| File not found | `ERROR: Input file '{path}' not found` |
| Unsupported format | `ERROR: Unsupported file format '{ext}'. Use .csv, .tsv, .xlsx, or .xls` |
| Column not found | `ERROR: Column '{name}' not found. Available: {columns}` |
| No numeric columns | `ERROR: No numeric columns found in file` |
| No valid lengths | `ERROR: No valid pipe lengths found (all NaN or <= 0)` |
| Target <= 0 | `ERROR: Target length must be positive` |
| Waste < 0 | `ERROR: Max waste cannot be negative` |
| No pipes meet criteria | `WARNING: No pipes can form valid piles. Check target + waste range.` |

### FR4: Enhanced Error Handling

| Scenario | Behavior |
|----------|----------|
| No valid patterns generated | Exit with actionable message: "Increase max_waste or check input data" |
| Solver infeasible | Exit with message: "Problem is infeasible. Relax constraints." |
| Solver timeout | Continue with best solution found, mark as "TIMEOUT (not optimal)" |
| Solver finds optimal | Mark as "OPTIMAL (guaranteed)" |

### FR5: Output Enhancements

#### FR5.1: Excel Output (3 sheets)
- **Summary**: Metrics, comparison with baseline, solver status
- **Pile Details**: Pile #, Segment, Pipe Index, Lengths, Total, Waste, Welds
- **Unused Pipes**: Index, Length

#### FR5.2: JSON Output (optional, `--json` flag)
```json
{
  "metadata": {
    "version": "V4",
    "input_file": "data.csv",
    "target_length": 100.0,
    "max_waste": 20.0,
    "solve_time_seconds": 45.2,
    "status": "OPTIMAL"
  },
  "summary": {
    "total_piles": 261,
    "theoretical_max": 264,
    "efficiency_pct": 98.9,
    "pipes_used": 747,
    "pipes_unused": 11,
    "total_waste_ft": 152.3
  },
  "piles": [...],
  "unused_pipes": [...]
}
```

#### FR5.3: Console Visualization
- ASCII histogram of waste distribution
- Progress bars with ETA
- Clear section headers

### FR6: Performance Optimization

| Optimization | Details |
|--------------|---------|
| FR6.1 | Parallel 3-pipe pattern generation using `multiprocessing.Pool` |
| FR6.2 | Early solver termination when gap < threshold (default 0.5%) |
| FR6.3 | Configurable thread count for CBC solver |

---

## Technical Design

### Architecture

```
pipe_optimizer_v4.py
├── load_pipe_data()           # Universal input loader
├── parse_arguments()          # CLI argument parsing
├── validate_inputs()          # Input validation
├── print_histogram()          # ASCII histogram
├── class SymmetryAwareSolverV4
│   ├── __init__()             # With validation
│   ├── generate_patterns()    # Parallelized
│   ├── _generate_3pipe_chunk() # Worker function
│   └── solve_ilp()            # With gap tolerance
├── export_results()           # Excel + JSON export
└── main()                     # Orchestration
```

### Dependencies

```
numpy
pandas
pulp
openpyxl
argparse (stdlib)
multiprocessing (stdlib)
json (stdlib)
```

### Backward Compatibility

Running `python3 pipe_optimizer_v4.py` with no arguments produces identical behavior to V3:
- Reads `pipe_lengths_clean.csv`
- Uses target=100, waste=20, precision=1
- Outputs `pipe_optimization_V4.xlsx`

---

## Test Cases

| Test | Input | Expected |
|------|-------|----------|
| TC1: Default run | No args | Same results as V3 (261 piles) |
| TC2: Excel input | `--input test.xlsx` | Loads and processes correctly |
| TC3: Auto-detect column | CSV with "Length" header | Finds and uses "Length" column |
| TC4: Custom target | `--target 120` | Creates 120' piles |
| TC5: Invalid file | `--input nonexistent.csv` | Clear error message |
| TC6: No valid patterns | Very small pipes only | Actionable error message |
| TC7: JSON output | `--json` | Creates both .xlsx and .json |
| TC8: Large dataset | 10k+ pipes | Completes with parallel speedup |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Backward compatibility | 100% (same results as V3 with defaults) |
| New file formats supported | CSV, TSV, XLSX, XLS |
| Error scenarios handled | 8+ distinct cases with clear messages |
| Performance (758 pipes) | < 60 seconds total |
| Code quality | Type hints on public functions |

---

## Out of Scope (Future)

- Config file support (JSON/YAML)
- Multiple output formats (PDF report)
- Cost analysis integration
- Web API / Streamlit UI
- Database persistence
