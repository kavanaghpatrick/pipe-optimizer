# Pipe Pile Optimizer

A standalone desktop application for optimizing pipe pile combinations. Finds the optimal way to combine pipe segments into piles of a target length, minimizing waste.

## Features

- **Optimal Solutions**: Uses Integer Linear Programming (ILP) to find mathematically optimal combinations
- **Memory Safe**: Monitors system resources and adapts to available RAM/CPU
- **Cross-Platform**: Works on Mac and Windows
- **Easy to Use**: Simple GUI - just select your file and click Run

## Download

Download the latest release for your platform:
- **Mac**: `Pipe_Optimizer_Mac.zip` - Unzip and drag to Applications
- **Windows**: `Pipe_Optimizer_Windows.zip` - Unzip and run `PipeOptimizer.exe`

## Usage

1. **Select File**: Click "Browse" to select your Excel (.xlsx) or CSV file containing pipe lengths
2. **Set Parameters**:
   - Target pile length (default: 100 ft)
   - Max waste per pile (default: 5 ft)
3. **Run**: Click "Run Optimizer" and wait for results
4. **Output**: Results saved to `<yourfile>_OPTIMIZED.xlsx`

## Input File Format

Your input file should have a column containing pipe lengths. The optimizer auto-detects columns named:
- `length`, `len`, `pipe_length`, `size`, `cut_length`, etc.

Example:
| length |
|--------|
| 45.2   |
| 32.8   |
| 51.4   |
| ...    |

## Building from Source

### Requirements
- Python 3.8+
- Dependencies: `pip install pandas numpy openpyxl pulp psutil pyinstaller`

### Build Mac App
```bash
./build_app.sh
```

### Build Windows Executable
```bash
build_windows.bat
```

Or use GitHub Actions - push a version tag to auto-build both platforms:
```bash
git tag v1.0.0
git push --tags
```

## How It Works

1. **Symmetry Reduction**: Groups identical pipe lengths (758 pipes → 273 types)
2. **Pattern Generation**: Finds all valid 2-pipe and 3-pipe combinations
3. **ILP Optimization**: Solves for maximum piles with minimum waste
4. **Adaptive Resources**: Auto-detects system RAM/CPU and adjusts accordingly

## License

MIT License
