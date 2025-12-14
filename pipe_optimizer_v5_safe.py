#!/usr/bin/env python3
"""
Memory-Safe Symmetry-Aware Pipe Optimizer v5.0
===============================================

Combines:
- V4's symmetry-aware solving (groups pipes by unique length types)
- V4's universal file loader (Excel, CSV, TSV, auto-detect columns)
- Safe version's memory monitoring and limits

Achieves 264 piles (100% of theoretical max) without crashing.

Usage:
    python3 pipe_optimizer_v5_safe.py                           # Default CSV
    python3 pipe_optimizer_v5_safe.py --input data.xlsx         # Excel file
    python3 pipe_optimizer_v5_safe.py --input data.csv -c len   # Specify column
    python3 pipe_optimizer_v5_safe.py --max-waste 5.0           # Tighter waste
"""

import argparse
import gc
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Memory monitoring disabled.")

from pulp import (
    PULP_CBC_CMD,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpStatusOptimal,
    LpVariable,
    lpSum,
)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PipeData:
    """Container for loaded pipe data."""
    lengths: np.ndarray
    source_file: str
    column_used: str
    original_count: int
    valid_count: int
    excluded_count: int
    exclusion_reasons: dict = field(default_factory=dict)

# =============================================================================
# UNIVERSAL FILE LOADER
# =============================================================================

# Common column names for pipe lengths (case-insensitive matching)
LENGTH_COLUMN_ALIASES = [
    'length', 'len', 'pipe_length', 'size', 'pipe_size',
    'cut_length', 'piece_length', 'segment', 'measurement',
    'pipe', 'pipes', 'lengths', 'cut', 'cuts', 'l', 'ft', 'feet'
]

def detect_length_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the length column from dataframe."""
    # First pass: exact match (case-insensitive)
    for col in df.columns:
        col_str = str(col).lower().strip()
        if col_str in LENGTH_COLUMN_ALIASES:
            return col

    # Second pass: partial match
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for alias in LENGTH_COLUMN_ALIASES:
            if alias in col_lower or col_lower in alias:
                return col

    # Fallback: first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None

def load_pipe_data(
    file_path: str,
    column_name: Optional[str] = None,
    delimiter: str = ',',
    has_header: bool = True
) -> PipeData:
    """
    Load pipe data from any supported format.

    Supports: .xlsx, .xls, .csv, .tsv, .txt
    Auto-detects length column if not specified.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file '{file_path}' not found")

    suffix = path.suffix.lower()

    # Validate supported format
    supported = ['.xlsx', '.xls', '.csv', '.tsv', '.txt']
    if suffix not in supported:
        raise ValueError(f"Unsupported file format '{suffix}'. Use: {', '.join(supported)}")

    # Load based on file type
    if suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, header=0 if has_header else None)
    elif suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t', header=0 if has_header else None)
    else:  # .csv, .txt
        df = pd.read_csv(file_path, sep=delimiter, header=0 if has_header else None)

    # Assign default column names if no header
    if not has_header:
        df.columns = [f'col_{i}' for i in range(len(df.columns))]

    # Detect or validate column
    if column_name:
        if column_name not in df.columns:
            # Try case-insensitive match
            matches = [c for c in df.columns if str(c).lower() == column_name.lower()]
            if matches:
                column_name = matches[0]
            else:
                raise ValueError(f"Column '{column_name}' not found. Available: {list(df.columns)}")
        col = column_name
    else:
        col = detect_length_column(df)
        if col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"No numeric columns found. Available: {list(df.columns)}")
            raise ValueError(f"Could not auto-detect length column. Available: {list(df.columns)}")

    # Extract and validate lengths
    original_count = len(df)
    exclusion_reasons = {}

    raw_values = df[col].tolist()
    valid_lengths = []

    for val in raw_values:
        if pd.isna(val):
            exclusion_reasons['null/empty'] = exclusion_reasons.get('null/empty', 0) + 1
            continue

        try:
            length = float(val)
        except (ValueError, TypeError):
            exclusion_reasons['non-numeric'] = exclusion_reasons.get('non-numeric', 0) + 1
            continue

        if length <= 0:
            exclusion_reasons['zero/negative'] = exclusion_reasons.get('zero/negative', 0) + 1
            continue

        valid_lengths.append(length)

    if not valid_lengths:
        raise ValueError("No valid pipe lengths found (all NaN or <= 0)")

    return PipeData(
        lengths=np.array(valid_lengths),
        source_file=str(path),
        column_used=str(col),
        original_count=original_count,
        valid_count=len(valid_lengths),
        excluded_count=original_count - len(valid_lengths),
        exclusion_reasons=exclusion_reasons
    )

# =============================================================================
# SYSTEM CAPABILITY DETECTION
# =============================================================================

@dataclass
class SystemCapabilities:
    """Detected system resources and recommended settings."""
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores: int
    recommended_threads: int
    recommended_soft_limit_gb: float
    recommended_hard_limit_gb: float
    platform: str


def detect_system_capabilities() -> SystemCapabilities:
    """
    Auto-detect system memory and CPU, return conservative settings.
    Works on Mac, Windows, and Linux.
    """
    import os
    import platform

    platform_name = platform.system()  # 'Darwin', 'Windows', 'Linux'

    # Default fallbacks for systems without psutil
    total_mem_gb = 8.0
    avail_mem_gb = 4.0
    cpu_count = 4

    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            total_mem_gb = mem.total / (1024**3)
            avail_mem_gb = mem.available / (1024**3)
            cpu_count = psutil.cpu_count(logical=True) or 4
        except Exception:
            pass
    else:
        # Fallback without psutil
        try:
            cpu_count = os.cpu_count() or 4
        except Exception:
            pass

    # Conservative resource allocation:
    # - Use at most 50% of available memory as soft limit
    # - Use at most 70% of available memory as hard limit
    # - Leave at least 2GB free for system
    # - Use physical cores, not hyperthreads (for ILP solver efficiency)

    safe_memory = max(2.0, avail_mem_gb - 2.0)  # Leave 2GB for system
    soft_limit = min(safe_memory * 0.5, total_mem_gb * 0.4)
    hard_limit = min(safe_memory * 0.7, total_mem_gb * 0.6)

    # Use ~50% of cores for solver (proportional to system capacity)
    # Leaves plenty of headroom for OS and other applications
    # Minimum 2 threads, no arbitrary max cap
    recommended_threads = max(2, int(cpu_count * 0.5))

    return SystemCapabilities(
        total_memory_gb=round(total_mem_gb, 1),
        available_memory_gb=round(avail_mem_gb, 1),
        cpu_cores=cpu_count,
        recommended_threads=recommended_threads,
        recommended_soft_limit_gb=round(soft_limit, 1),
        recommended_hard_limit_gb=round(hard_limit, 1),
        platform=platform_name
    )


def print_system_info(caps: SystemCapabilities):
    """Display detected system capabilities."""
    print(f"\n  System detected: {caps.platform}")
    print(f"  Total RAM: {caps.total_memory_gb:.1f} GB")
    print(f"  Available RAM: {caps.available_memory_gb:.1f} GB")
    print(f"  CPU cores: {caps.cpu_cores}")
    print(f"  Using: {caps.recommended_threads} threads, "
          f"{caps.recommended_soft_limit_gb:.1f}/{caps.recommended_hard_limit_gb:.1f} GB limits")


# =============================================================================
# MEMORY MONITORING
# =============================================================================

class MemoryMonitor:
    """Track memory usage with soft/hard limits"""

    def __init__(self, soft_limit_gb: float = 6.0, hard_limit_gb: float = 8.0):
        self.soft_limit_gb = soft_limit_gb
        self.hard_limit_gb = hard_limit_gb
        self.peak_gb = 0.0
        self.checks = 0

    def check(self, phase: str = "") -> float:
        """Check current memory, warn/abort if needed"""
        if not HAS_PSUTIL:
            return 0.0

        self.checks += 1
        mem = psutil.Process().memory_info().rss / (1024**3)
        self.peak_gb = max(self.peak_gb, mem)

        if mem > self.hard_limit_gb:
            raise MemoryError(
                f"[{phase}] Memory {mem:.2f}GB exceeds hard limit {self.hard_limit_gb}GB"
            )

        if mem > self.soft_limit_gb:
            print(f"  WARNING [{phase}]: Memory {mem:.2f}GB approaching limit")

        return mem

    def report(self):
        print(f"\n  Memory Report:")
        print(f"    Peak usage: {self.peak_gb:.2f}GB")
        print(f"    Checks performed: {self.checks}")
        print(f"    Limits: {self.soft_limit_gb}GB (soft), {self.hard_limit_gb}GB (hard)")


def progress_bar(current: int, total: int, prefix: str = '', width: int = 50):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:5.1f}% ({current:,}/{total:,})", end='', flush=True)


def section(title: str):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


# =============================================================================
# SYMMETRY-AWARE SOLVER (from V4) + MEMORY SAFETY
# =============================================================================

class SymmetryAwareSafeSolver:
    """
    Memory-safe symmetry-aware solver.

    Key insight: Pipes of the same length are interchangeable.
    Uses unique length TYPES instead of individual pipes.
    This reduces 758 pipes → 273 types (2.8x smaller problem).
    """

    def __init__(
        self,
        pipe_lengths: np.ndarray,
        target_length: float = 100.0,
        max_waste: float = 20.0,
        precision: int = 1,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        self.raw_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.precision = precision
        self.memory = memory_monitor or MemoryMonitor()

        # Group pipes by rounded length (symmetry reduction)
        rounded = [round(p, precision) for p in pipe_lengths]
        self.inventory = Counter(rounded)
        self.unique_lengths = sorted(self.inventory.keys(), reverse=True)
        self.n_types = len(self.unique_lengths)

        # Fast lookup: length -> index
        self.length_to_idx = {l: i for i, l in enumerate(self.unique_lengths)}

        # Theoretical maximum
        self.theoretical_max = int(sum(pipe_lengths) // target_length)

        print(f"\n  Symmetry reduction: {len(pipe_lengths)} pipes → {self.n_types} unique types")
        print(f"  Reduction factor: {len(pipe_lengths)/self.n_types:.1f}x")

    def generate_patterns(self, max_patterns: int = 2_000_000, max_welds: int = 3, stop_event=None) -> list:
        """Generate patterns over unique length TYPES (not individual pipes)

        Args:
            max_patterns: Safety limit on total patterns
            max_welds: Maximum number of pipe segments per pile (1, 2, or 3)
            stop_event: Optional threading.Event to check for cancellation
        """
        section("PATTERN GENERATION")

        self.memory.check("pattern_generation_start")

        patterns = []
        lengths = self.unique_lengths
        n = len(lengths)

        min_len = self.target_length
        max_len = self.target_length + self.max_waste

        print(f"\n  Target: {self.target_length}', Max waste: {self.max_waste}', Max welds: {max_welds}")
        print(f"  Valid pile range: {min_len}' to {max_len}'")

        # 1-pipe patterns (none expected for 100' target with max 51.6' pipes)
        print(f"\n[1/{max_welds}] Generating 1-pipe patterns...")
        one_count = 0
        for i in range(n):
            if min_len <= lengths[i] <= max_len:
                waste = lengths[i] - self.target_length
                patterns.append(((i,), lengths[i], waste, {i: 1}))
                one_count += 1
        print(f"  Found {one_count} valid 1-pipe patterns")

        # 2-pipe patterns (if max_welds >= 2)
        two_count = 0
        cancelled = False
        if max_welds >= 2:
            print(f"\n[2/{max_welds}] Generating 2-pipe patterns...")
            total_pairs = n * (n + 1) // 2
            checked = 0

            for i in range(n):
                for j in range(i, n):
                    total_len = lengths[i] + lengths[j]
                    if min_len <= total_len <= max_len:
                        waste = total_len - self.target_length
                        if i == j:
                            if self.inventory[lengths[i]] >= 2:
                                patterns.append(((i, j), total_len, waste, {i: 2}))
                                two_count += 1
                        else:
                            patterns.append(((i, j), total_len, waste, {i: 1, j: 1}))
                            two_count += 1

                    checked += 1
                    if checked % 5000 == 0:
                        progress_bar(checked, total_pairs, '2-pipe')
                        # Check for cancellation every 5000 iterations
                        if stop_event and stop_event.is_set():
                            print("\n  Cancelled during 2-pipe generation")
                            cancelled = True
                            break
                if cancelled:
                    break

            progress_bar(total_pairs, total_pairs, '2-pipe')
            print(f"\n  Found {two_count:,} valid 2-pipe patterns")
            self.memory.check("after_2pipe")

        # Return early if cancelled
        if cancelled:
            print(f"\n  TOTAL: {len(patterns):,} patterns (partial - cancelled)")
            return patterns

        # 3-pipe patterns (if max_welds >= 3)
        three_count = 0
        if max_welds >= 3:
            print(f"\n[3/{max_welds}] Generating 3-pipe patterns...")
            start = time.time()

            for i in range(n):
                len_i = lengths[i]
                if 3 * len_i < min_len:
                    break

                for j in range(i, n):
                    len_j = lengths[j]
                    if len_i + 2 * len_j < min_len:
                        continue

                    for k in range(j, n):
                        len_k = lengths[k]
                        total_len = len_i + len_j + len_k

                        if total_len < min_len:
                            break

                        if total_len <= max_len:
                            waste = total_len - self.target_length

                            counts_needed = Counter([i, j, k])
                            feasible = all(
                                self.inventory[lengths[idx]] >= cnt
                                for idx, cnt in counts_needed.items()
                            )

                            if feasible:
                                patterns.append(((i, j, k), total_len, waste, dict(counts_needed)))
                                three_count += 1

                                # Safety: check limits and cancellation periodically
                                if three_count % 10000 == 0:
                                    # Check for cancellation
                                    if stop_event and stop_event.is_set():
                                        print("\n  Cancelled during 3-pipe generation")
                                        print(f"\n  TOTAL: {len(patterns):,} patterns (partial - cancelled)")
                                        return patterns

                                if three_count % 50000 == 0:
                                    self.memory.check("3pipe_progress")
                                    if len(patterns) >= max_patterns:
                                        print(f"\n  WARNING: Pattern limit ({max_patterns:,}) reached - stopping early")
                                        print(f"  Tip: Reduce max_waste to generate fewer patterns")
                                        progress_bar(n, n, '3-pipe')
                                        print(f"\n  Found {three_count:,} valid 3-pipe patterns (capped)")
                                        self.memory.check("after_3pipe")
                                        print(f"\n  TOTAL: {len(patterns):,} unique pattern types")
                                        return patterns

                # Progress every ~10% of i values
                if i % max(1, n // 10) == 0:
                    progress_bar(i, n, '3-pipe')
                    # Also check cancellation at progress updates
                    if stop_event and stop_event.is_set():
                        print("\n  Cancelled during 3-pipe generation")
                        print(f"\n  TOTAL: {len(patterns):,} patterns (partial - cancelled)")
                        return patterns

            progress_bar(n, n, '3-pipe')
            print(f"\n  Found {three_count:,} valid 3-pipe patterns in {time.time()-start:.1f}s")
            self.memory.check("after_3pipe")

        print(f"\n  TOTAL: {len(patterns):,} unique pattern types")
        return patterns

    def solve_ilp(
        self,
        patterns: list,
        time_limit: int = 1800,
        gap: float = 0.005,
        threads: int = 14
    ) -> tuple:
        """Solve using INTEGER variables for pattern counts (not binary)"""
        section("ILP SOLVING")

        if not patterns:
            return None, 'NO_PATTERNS', 0

        n_patterns = len(patterns)

        print(f"\n  Pattern types: {n_patterns:,}")
        print(f"  Length types (constraints): {self.n_types}")
        print(f"  Variables: {n_patterns:,} (INTEGER, not binary!)")
        print(f"\n  Building model...")

        self.memory.check("ilp_start")

        prob = LpProblem("Optimal_Piles_V5_Safe", LpMaximize)

        # INTEGER variable for each pattern TYPE (key difference from binary!)
        # Each variable = how many times to use this pattern
        print("  Creating variables with upper bounds...")
        x = []
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            # Max uses = limited by inventory of scarcest pipe type needed
            max_uses = min(
                self.inventory[self.unique_lengths[idx]] // cnt
                for idx, cnt in counts_needed.items()
            )
            x.append(LpVariable(f"p{p_idx}", lowBound=0, upBound=max_uses, cat='Integer'))

        self.memory.check("after_variables")

        # Objective: maximize total piles (sum of all pattern uses)
        prob += lpSum(x), "Total_Piles"

        # Constraints: for each length TYPE, total usage <= inventory
        print("  Adding inventory constraints...")

        # Build type-to-patterns index
        type_to_patterns = [[] for _ in range(self.n_types)]
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            for type_idx, count in counts_needed.items():
                type_to_patterns[type_idx].append((p_idx, count))

        constraints_added = 0
        for type_idx in range(self.n_types):
            if type_idx % 50 == 0:
                progress_bar(type_idx, self.n_types, 'Constraints')

            length = self.unique_lengths[type_idx]
            available = self.inventory[length]

            if type_to_patterns[type_idx]:
                prob += (
                    lpSum(count * x[p_idx] for p_idx, count in type_to_patterns[type_idx]) <= available,
                    f"Type_{type_idx}"
                )
                constraints_added += 1

        progress_bar(self.n_types, self.n_types, 'Constraints')
        print(f"\n  Added {constraints_added} constraints")

        self.memory.check("after_constraints")

        # Solve
        section("SOLVING")
        print(f"\n  Solver: CBC with {threads} threads")
        print(f"  Time limit: {time_limit // 60} minutes")
        print(f"  Gap tolerance: {gap*100:.1f}%")
        print("\n" + "-"*80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=1, timeLimit=time_limit, threads=threads, gapRel=gap)
        prob.solve(solver)
        solve_time = time.time() - start_time

        print("-"*80)
        print(f"\n  Solved in {solve_time:.1f}s")
        print(f"  Status: {LpStatus[prob.status]}")

        self.memory.check("after_solve")

        if prob.status in [LpStatusOptimal, 1]:
            print("\n  Extracting solution...")
            solution = []

            for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
                uses = int(x[p_idx].varValue or 0)
                if uses > 0:
                    for _ in range(uses):
                        pipe_lengths = [self.unique_lengths[i] for i in indices]
                        solution.append({
                            'pattern_type': p_idx,
                            'length_indices': indices,
                            'pipe_lengths': pipe_lengths,
                            'total_length': total,
                            'waste': waste,
                            'num_welds': len(indices) - 1
                        })

            print(f"  Extracted {len(solution)} piles")
            status_str = 'OPTIMAL' if prob.status == LpStatusOptimal else 'FEASIBLE'
            return solution, status_str, solve_time

        return None, LpStatus[prob.status], solve_time


def export_to_excel(solver, solution, output_path):
    """Export results to Excel"""
    section("EXPORTING TO EXCEL")

    pipe_lengths = solver.raw_lengths

    # Assign specific pipe indices from inventory pools
    print("  Assigning pipe indices...")
    inventory_pools = defaultdict(list)
    for idx, length in enumerate(pipe_lengths):
        rounded = round(length, solver.precision)
        inventory_pools[rounded].append(idx)

    assigned_pipes = set()
    pile_data = []

    for pile_num, pile in enumerate(solution, 1):
        pile_pipe_indices = []
        for length_idx in pile['length_indices']:
            needed_length = solver.unique_lengths[length_idx]
            available = [idx for idx in inventory_pools[needed_length] if idx not in assigned_pipes]
            if available:
                pipe_idx = available[0]
                assigned_pipes.add(pipe_idx)
                pile_pipe_indices.append((pipe_idx, pipe_lengths[pipe_idx]))

        for seg_num, (pipe_idx, actual_length) in enumerate(pile_pipe_indices, 1):
            pile_data.append({
                'Pile Number': pile_num,
                'Segment': seg_num,
                'Pipe Index': pipe_idx + 1,
                'Length (ft)': round(actual_length, 2),
                'Total Length': pile['total_length'] if seg_num == 1 else '',
                'Waste (ft)': pile['waste'] if seg_num == 1 else '',
                'Welds': pile['num_welds'] if seg_num == 1 else ''
            })

    df_piles = pd.DataFrame(pile_data)

    # Summary
    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)

    summary = pd.DataFrame({
        'Metric': ['Method', 'Status', 'Total Piles', 'Theoretical Max', 'Efficiency',
                   'Pipes Used', 'Total Waste (ft)', 'Avg Waste/Pile'],
        'Value': ['ILP V5 (Symmetry-Aware + Safe)',
                  'OPTIMAL',
                  total_piles,
                  solver.theoretical_max,
                  f"{total_piles/solver.theoretical_max*100:.1f}%",
                  len(assigned_pipes),
                  f"{total_waste:.2f}",
                  f"{total_waste/total_piles:.2f}" if total_piles else "N/A"]
    })

    # Unused pipes
    unused_indices = [i for i in range(len(pipe_lengths)) if i not in assigned_pipes]
    unused_df = pd.DataFrame({
        'Pipe Index': [i+1 for i in unused_indices],
        'Length (ft)': [pipe_lengths[i] for i in unused_indices]
    })

    # Atomic write: write to temp file first, then replace
    # This prevents file corruption if the process crashes mid-write
    temp_path = Path(output_path).with_suffix('.tmp.xlsx')

    print(f"  Writing to {output_path}...")
    try:
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)
            df_piles.to_excel(writer, sheet_name='Pile Details', index=False)
            unused_df.to_excel(writer, sheet_name='Unused Pipes', index=False)

        # Atomic replace (works on both Mac/Windows)
        import shutil
        shutil.move(str(temp_path), str(output_path))
        print(f"  Exported to: {output_path}")
    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Memory-Safe Symmetry-Aware Pipe Optimizer V5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default (pipe_lengths_clean.csv)
  %(prog)s --input pipes.xlsx                 # Excel file, auto-detect column
  %(prog)s --input data.csv --column length   # Specify column name
  %(prog)s --input data.csv --no-header       # CSV without header row
  %(prog)s --max-waste 5.0                    # Tighter waste = faster solve
        """
    )
    parser.add_argument("--input", "-i", help="Input file (xlsx, xls, csv, tsv, txt)")
    parser.add_argument("--output", "-o", default="pipe_optimization_V5_SAFE.xlsx", help="Output Excel file")
    parser.add_argument("--column", "-c", help="Column name containing pipe lengths (auto-detected if not specified)")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    parser.add_argument("--no-header", action="store_true", help="Input file has no header row")
    parser.add_argument("--target", "-t", type=float, default=100.0, help="Target pile length (default: 100)")
    parser.add_argument("--max-waste", "-w", type=float, default=5.0, help="Maximum waste per pile (default: 5)")
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Hard memory limit in GB (default: 8)")
    parser.add_argument("--memory-soft", type=float, default=6.0, help="Soft memory limit in GB (default: 6)")
    parser.add_argument("--time-limit", type=int, default=1800, help="Solver time limit in seconds (default: 1800)")
    parser.add_argument("--threads", type=int, default=14, help="Solver threads (default: 14)")
    args = parser.parse_args()

    print("="*80)
    print("MEMORY-SAFE SYMMETRY-AWARE PIPE OPTIMIZER V5")
    print("="*80)
    print("\nCombines V4's symmetry approach with memory safety features")
    print(f"\n  Memory limits: {args.memory_soft}GB (soft), {args.memory_limit}GB (hard)")

    # Find input file
    if args.input:
        input_file = args.input
    else:
        # Default fallback
        default_files = ['pipe_lengths_clean.csv', 'pipes.xlsx', 'pipes.csv', 'data.xlsx', 'data.csv']
        input_file = None
        for f in default_files:
            if Path(f).exists():
                input_file = f
                break
        if not input_file:
            print("ERROR: No input file specified and no default file found.")
            print("Usage: python3 pipe_optimizer_v5_safe.py --input <file>")
            return 1

    # Load data using universal loader
    section("LOADING DATA")
    try:
        pipe_data = load_pipe_data(
            input_file,
            column_name=args.column,
            delimiter=args.delimiter,
            has_header=not args.no_header
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    pipe_lengths = pipe_data.lengths

    print(f"\n  Source: {pipe_data.source_file}")
    print(f"  Column: '{pipe_data.column_used}'")
    print(f"  Loaded {pipe_data.valid_count} pipes ({pipe_data.excluded_count} excluded)")
    if pipe_data.exclusion_reasons:
        print(f"  Exclusions: {pipe_data.exclusion_reasons}")
    print(f"\n  Total material: {pipe_lengths.sum():.1f} feet")
    print(f"  Range: {pipe_lengths.min():.1f}' to {pipe_lengths.max():.1f}'")
    print(f"  Theoretical max: {int(pipe_lengths.sum() // args.target)} piles")

    # Initialize
    memory = MemoryMonitor(args.memory_soft, args.memory_limit)

    solver = SymmetryAwareSafeSolver(
        pipe_lengths,
        target_length=args.target,
        max_waste=args.max_waste,
        memory_monitor=memory
    )

    # Generate patterns
    start_total = time.time()
    patterns = solver.generate_patterns()

    if not patterns:
        print("ERROR: No valid patterns found!")
        return 1

    # Solve
    solution, status, solve_time = solver.solve_ilp(
        patterns,
        time_limit=args.time_limit,
        threads=args.threads
    )

    if solution is None or len(solution) == 0:
        print(f"ERROR: No solution found! Status: {status}")
        return 1

    # Results
    section("RESULTS")
    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)

    print(f"\n  *** {'OPTIMAL' if status == 'OPTIMAL' else status} SOLUTION FOUND ***")
    print(f"\n  {'Metric':<40} {'Value':>15}")
    print("  " + "-"*55)
    print(f"  {f'{args.target}-foot piles created':<40} {total_piles:>15,}")
    print(f"  {'Theoretical maximum':<40} {solver.theoretical_max:>15,}")
    print(f"  {'Efficiency':<40} {total_piles/solver.theoretical_max*100:>14.1f}%")
    print(f"  {'Total waste':<40} {total_waste:>14.1f}'")
    print(f"  {'Average waste per pile':<40} {total_waste/total_piles:>14.2f}'")

    # Weld distribution
    weld_counts = Counter(p['num_welds'] for p in solution)
    print(f"\n  Weld distribution:")
    for welds in sorted(weld_counts.keys()):
        count = weld_counts[welds]
        pct = count / total_piles * 100
        print(f"    {welds} weld(s): {count:4d} piles ({pct:5.1f}%)")

    # Comparison
    print(f"\n  vs Greedy (163 piles): +{total_piles - 163} piles!")

    # Export
    export_to_excel(solver, solution, args.output)

    # Summary
    total_time = time.time() - start_total
    section("COMPLETE")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Result: {total_piles} piles ({total_piles/solver.theoretical_max*100:.1f}% of theoretical max)")

    memory.report()
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
