# Pipe Pile Optimizer

[![Build](https://github.com/kavanaghpatrick/pipe-optimizer/actions/workflows/build-release.yml/badge.svg)](https://github.com/kavanaghpatrick/pipe-optimizer/actions)
[![Release](https://img.shields.io/github/v/release/kavanaghpatrick/pipe-optimizer)](https://github.com/kavanaghpatrick/pipe-optimizer/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Mac%20%7C%20Windows-lightgrey)](https://github.com/kavanaghpatrick/pipe-optimizer/releases)

**Optimize pipe pile combinations to minimize waste.** A standalone desktop application that finds the mathematically optimal way to combine pipe segments into piles of a target length.

---

## Download

### Latest Release

| Platform | Download | Size |
|----------|----------|------|
| **Mac** | [Pipe_Optimizer_Mac.zip](https://github.com/kavanaghpatrick/pipe-optimizer/releases/latest/download/Pipe_Optimizer_Mac.zip) | ~140 MB |
| **Windows** | [Pipe_Optimizer_Windows.zip](https://github.com/kavanaghpatrick/pipe-optimizer/releases/latest/download/Pipe_Optimizer_Windows.zip) | ~150 MB |

### Installation

**Mac:**
1. Download and unzip `Pipe_Optimizer_Mac.zip`
2. Drag `Pipe Optimizer.app` to your Applications folder
3. First launch: Right-click → Open (required for unsigned apps)
4. Double-click to run thereafter

**Windows:**
1. Download and unzip `Pipe_Optimizer_Windows.zip`
2. Run `PipeOptimizer.exe`
3. Windows may show a SmartScreen warning - click "More info" → "Run anyway"

---

## Features

- **Optimal Solutions** - Uses Integer Linear Programming (ILP) to find mathematically proven optimal combinations
- **Fast** - Symmetry-aware algorithm reduces 758 pipes to 273 types, solving in minutes not hours
- **Memory Safe** - Monitors system resources and adapts to your computer's RAM/CPU
- **Cross-Platform** - Native apps for Mac and Windows, no Python required
- **Simple** - Just select your file, set parameters, and click Run

---

## How to Use

### 1. Prepare Your Data

Create an Excel (`.xlsx`) or CSV file with a column containing pipe lengths:

| length |
|--------|
| 45.2   |
| 32.8   |
| 51.4   |
| 28.9   |
| ...    |

The app auto-detects columns named: `length`, `len`, `pipe_length`, `size`, `cut_length`, etc.

### 2. Run the Optimizer

1. Click **Browse** to select your file
2. Set **Target pile length** (default: 100 ft)
3. Set **Max waste per pile** (default: 5 ft) - lower = faster but may find fewer piles
4. Click **Run Optimizer**

### 3. Get Results

Results are saved to `<yourfile>_OPTIMIZED.xlsx` with three sheets:
- **Summary** - Total piles, efficiency, waste statistics
- **Pile Details** - Each pile with pipe assignments
- **Unused Pipes** - Pipes that couldn't be used

---

## Example Results

From 758 pipe segments (17-52 ft each):
- **264 piles created** (100% of theoretical maximum)
- **Total waste: 43 ft** (0.16 ft average per pile)
- **Solve time: 2.7 minutes**

---

## Algorithm

The optimizer uses a three-phase approach:

1. **Symmetry Reduction** - Groups identical pipe lengths (758 pipes → 273 types)
2. **Pattern Generation** - Finds all valid 2-pipe and 3-pipe combinations within waste tolerance
3. **ILP Optimization** - Solves for maximum piles using the CBC solver

This achieves provably optimal solutions while being memory-efficient.

---

## Building from Source

### Requirements
- Python 3.8+
- Dependencies: `pip install pandas numpy openpyxl pulp psutil pyinstaller`

### Build Commands

**Mac:**
```bash
./build_app.sh
# Output: dist/Pipe Optimizer.app
```

**Windows:**
```cmd
build_windows.bat
# Output: dist\PipeOptimizer\PipeOptimizer.exe
```

### Project Structure

```
pipe-optimizer/
├── pipe_optimizer_gui.py      # GUI application (Tkinter)
├── pipe_optimizer_v5_safe.py  # Core optimization engine
├── PipeOptimizer.spec         # PyInstaller configuration
├── build_app.sh               # Mac build script
├── build_windows.bat          # Windows build script
├── data/                      # Sample data files
├── docs/                      # Documentation
└── archive/                   # Old versions (reference only)
```

---

## Technical Details

- **Solver**: CBC (Coin-or Branch and Cut) via PuLP
- **GUI**: Tkinter (cross-platform, no dependencies)
- **Memory Safety**: psutil monitoring with configurable limits
- **File Safety**: Atomic writes prevent corruption on crash

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Mac: "App is damaged" | Right-click → Open, or: `xattr -cr "Pipe Optimizer.app"` |
| Windows: SmartScreen blocks | Click "More info" → "Run anyway" |
| Solver not found | Rebuild with `./build_app.sh` to rebundle CBC |
| Out of memory | Reduce "Max waste" parameter (try 3-5 ft) |

---

## License

MIT License - feel free to use, modify, and distribute.

---

## Contributing

Issues and pull requests welcome at [github.com/kavanaghpatrick/pipe-optimizer](https://github.com/kavanaghpatrick/pipe-optimizer)
