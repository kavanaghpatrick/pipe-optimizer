#!/usr/bin/env python3
"""
Pipe Pile Optimizer V4 - Production-Ready CLI Tool
===================================================
Combines shorter pipes into target-length piles using ILP optimization.

Features over V3:
- Universal input loader (CSV, TSV, Excel)
- CLI arguments for all parameters
- Auto-detect length column
- Comprehensive error handling
- JSON export option
- ASCII histogram visualization

Usage:
    python3 pipe_optimizer_v4.py                              # Defaults (backward compat)
    python3 pipe_optimizer_v4.py --input pipes.xlsx           # Auto-detect column
    python3 pipe_optimizer_v4.py --input data.csv -c length   # Specify column
    python3 pipe_optimizer_v4.py --input data.xlsx --json     # JSON output
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpStatusOptimal,
    LpVariable,
    lpSum,
)

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

@dataclass
class PipeData:
    """Container for loaded pipe data."""
    lengths: np.ndarray
    source_file: str
    column_used: str
    original_count: int
    valid_count: int
    excluded_count: int
    exclusion_reasons: dict[str, int] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Complete optimization results."""
    success: bool
    piles: list[dict]
    total_piles: int
    theoretical_max: int
    efficiency: float
    pipes_used: int
    pipes_unused: int
    total_waste: float
    solve_time: float
    solver_status: str
    pile_assignments: list  # For export
    unused_indices: list    # For export
    error_message: Optional[str] = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(msg: str, end: str = '\n', quiet: bool = False) -> None:
    """Print with immediate flush."""
    if not quiet:
        print(msg, end=end, flush=True)

def progress_bar(current: int, total: int, prefix: str = '', width: int = 50, quiet: bool = False) -> None:
    """Print a progress bar with immediate flush."""
    if quiet:
        return
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '\u2588' * filled + '\u2591' * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:5.1f}% ({current:,}/{total:,})", end='', flush=True)

def section(title: str, quiet: bool = False) -> None:
    """Print a section header."""
    if not quiet:
        log(f"\n{'='*80}")
        log(title)
        log('='*80)

# ============================================================================
# UNIVERSAL INPUT LOADER
# ============================================================================

# Common column names for pipe lengths (case-insensitive matching)
LENGTH_COLUMN_ALIASES = [
    'length', 'len', 'pipe_length', 'size', 'pipe_size',
    'cut_length', 'piece_length', 'segment', 'measurement',
    'pipe', 'pipes', 'lengths', 'cut', 'cuts', 'l'
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
    has_header: bool = True,
    quiet: bool = False
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
    elif suffix in ['.csv', '.txt']:
        df = pd.read_csv(file_path, sep=delimiter, header=0 if has_header else None)
    else:
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
            # Check if there are any numeric columns at all
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"No numeric columns found in file. Available: {list(df.columns)}")
            raise ValueError(f"Could not auto-detect length column. Available: {list(df.columns)}")

    # Extract and validate lengths
    original_count = len(df)
    exclusion_reasons: dict[str, int] = {}

    raw_values = df[col].tolist()
    valid_lengths: list[float] = []

    for val in raw_values:
        # Skip NaN/None
        if pd.isna(val):
            exclusion_reasons['null/empty'] = exclusion_reasons.get('null/empty', 0) + 1
            continue

        # Convert to float
        try:
            length = float(val)
        except (ValueError, TypeError):
            exclusion_reasons['non-numeric'] = exclusion_reasons.get('non-numeric', 0) + 1
            continue

        # Validate positive
        if length <= 0:
            exclusion_reasons['zero/negative'] = exclusion_reasons.get('zero/negative', 0) + 1
            continue

        valid_lengths.append(length)

    if not valid_lengths:
        raise ValueError(f"No valid pipe lengths found (all NaN or <= 0)")

    return PipeData(
        lengths=np.array(valid_lengths),
        source_file=str(path),
        column_used=str(col),
        original_count=original_count,
        valid_count=len(valid_lengths),
        excluded_count=original_count - len(valid_lengths),
        exclusion_reasons=exclusion_reasons
    )

# ============================================================================
# SYMMETRY-AWARE SOLVER V4
# ============================================================================

class SymmetryAwareSolverV4:
    """
    V4: Production-ready solver with CLI support.

    Exploits symmetry: pipes of the same length are interchangeable.
    Uses unique length TYPES instead of individual pipes.
    """

    def __init__(
        self,
        pipe_lengths: np.ndarray,
        target_length: float = 100.0,
        max_waste: float = 20.0,
        precision: int = 1,
        quiet: bool = False
    ):
        self.raw_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.precision = precision
        self.quiet = quiet

        # Group pipes by rounded length
        rounded = [round(p, precision) for p in pipe_lengths]
        self.inventory = Counter(rounded)
        self.unique_lengths = sorted(self.inventory.keys(), reverse=True)
        self.n_types = len(self.unique_lengths)

        # Create fast lookup: length -> index
        self.length_to_idx = {l: i for i, l in enumerate(self.unique_lengths)}

        # Theoretical maximum (floor division)
        self.theoretical_max = int(sum(pipe_lengths) // target_length)

    def generate_patterns(self) -> list:
        """Generate patterns over unique length TYPES (not individual pipes)."""
        if not self.quiet:
            section("PATTERN GENERATION", self.quiet)
            log(f"\n  Unique length types: {self.n_types} (vs {len(self.raw_lengths)} raw pipes)")
            log(f"  Symmetry reduction: {len(self.raw_lengths)/self.n_types:.1f}x")
            log(f"  Target: {self.target_length}', Max waste: {self.max_waste}'")
            log(f"  Valid pile range: {self.target_length}' to {self.target_length + self.max_waste}' (INCLUSIVE)")

        patterns = []
        lengths = self.unique_lengths
        n = len(lengths)

        min_len = self.target_length
        max_len = self.target_length + self.max_waste

        # 1-pipe patterns
        if not self.quiet:
            log("\n[1/3] Generating 1-pipe patterns...")
        start = time.time()
        one_count = 0

        for i in range(n):
            length = lengths[i]
            if min_len <= length <= max_len:
                waste = length - self.target_length
                patterns.append(((i,), length, waste, {i: 1}))
                one_count += 1
                if not self.quiet:
                    log(f"    Found: {length}' pipe (waste: {waste:.1f}')")

        if not self.quiet:
            log(f"  Found {one_count} valid 1-pipe patterns in {time.time()-start:.2f}s")

        # 2-pipe patterns
        if not self.quiet:
            log("\n[2/3] Generating 2-pipe patterns...")
        start = time.time()
        two_count = 0
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
                if checked % 1000 == 0:
                    progress_bar(checked, total_pairs, '2-pipe', quiet=self.quiet)

        if not self.quiet:
            progress_bar(total_pairs, total_pairs, '2-pipe', quiet=self.quiet)
            log(f"\n  Found {two_count:,} valid 2-pipe patterns in {time.time()-start:.1f}s")

        # 3-pipe patterns
        if not self.quiet:
            log("\n[3/3] Generating 3-pipe patterns...")
        start = time.time()
        three_count = 0
        total_est = n * (n + 1) * (n + 2) // 6
        checked = 0

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

                    checked += 1
                    if checked % 10000 == 0:
                        progress_bar(min(checked, total_est), total_est, '3-pipe', quiet=self.quiet)

        if not self.quiet:
            progress_bar(total_est, total_est, '3-pipe', quiet=self.quiet)
            log(f"\n  Found {three_count:,} valid 3-pipe patterns in {time.time()-start:.1f}s")
            log(f"\n  TOTAL: {len(patterns):,} unique pattern types")

        return patterns

    def solve_ilp(
        self,
        patterns: list,
        time_limit: int = 900,
        gap: float = 0.005,
        threads: int = 14
    ) -> tuple:
        """Solve using integer variables for pattern counts."""
        if not self.quiet:
            section("ILP SOLVING", self.quiet)

        if not patterns:
            return None, 'NO_PATTERNS', 0

        n_patterns = len(patterns)
        if not self.quiet:
            log(f"\n  Pattern types: {n_patterns:,}")
            log(f"  Length types: {self.n_types}")
            log(f"  Variables: {n_patterns:,} (integer)")
            log(f"  Building model...")

        prob = LpProblem("Optimal_Piles_V4", LpMaximize)

        # Integer variable for each pattern TYPE
        x = []
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            max_uses = min(
                self.inventory[self.unique_lengths[idx]] // cnt
                for idx, cnt in counts_needed.items()
            )
            x.append(LpVariable(f"p{p_idx}", lowBound=0, upBound=max_uses, cat='Integer'))

        # Objective: maximize total piles
        prob += lpSum(x), "Total_Piles"

        # Constraints: for each length TYPE, total usage <= inventory
        if not self.quiet:
            log("  Adding inventory constraints...")

        type_to_patterns = [[] for _ in range(self.n_types)]
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            for type_idx, count in counts_needed.items():
                type_to_patterns[type_idx].append((p_idx, count))

        for type_idx in range(self.n_types):
            if type_idx % 50 == 0:
                progress_bar(type_idx, self.n_types, 'Constraints', quiet=self.quiet)

            length = self.unique_lengths[type_idx]
            available = self.inventory[length]

            if type_to_patterns[type_idx]:
                prob += (
                    lpSum(count * x[p_idx] for p_idx, count in type_to_patterns[type_idx]) <= available,
                    f"Type_{type_idx}_{length}"
                )

        if not self.quiet:
            progress_bar(self.n_types, self.n_types, 'Constraints', quiet=self.quiet)
            log("")
            section("SOLVING", self.quiet)
            log(f"\n  Solver: CBC with {threads} threads")
            log(f"  Time limit: {time_limit // 60} minutes")
            log(f"  Gap tolerance: {gap*100:.1f}%")
            log("\n" + "-"*80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=0 if self.quiet else 1, timeLimit=time_limit, threads=threads, gapRel=gap)
        prob.solve(solver)
        solve_time = time.time() - start_time

        if not self.quiet:
            log("-"*80)
            log(f"\n  Solved in {solve_time:.1f}s")
            log(f"  Status: {LpStatus[prob.status]}")

        if prob.status in [LpStatusOptimal, 1]:  # OPTIMAL or NOT_SOLVED (but has solution)
            if not self.quiet:
                log("\n  Extracting solution...")
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

            if not self.quiet:
                log(f"  Extracted {len(solution)} piles")

            status_str = 'OPTIMAL' if prob.status == LpStatusOptimal else 'FEASIBLE'
            return solution, status_str, solve_time

        return None, LpStatus[prob.status], solve_time

# ============================================================================
# OUTPUT FORMATTERS
# ============================================================================

def print_ascii_histogram(piles: list, target_length: float, quiet: bool = False) -> None:
    """Print ASCII histogram of waste distribution."""
    if quiet or not piles:
        return

    log("\n" + "="*60)
    log("WASTE DISTRIBUTION HISTOGRAM")
    log("="*60)

    # Group by waste ranges
    waste_bins = [0, 2, 5, 10, 15, 20, float('inf')]
    bin_counts = [0] * (len(waste_bins) - 1)

    for p in piles:
        for i in range(len(waste_bins) - 1):
            if waste_bins[i] <= p['waste'] < waste_bins[i + 1]:
                bin_counts[i] += 1
                break

    max_count = max(bin_counts) if bin_counts else 1
    width = 40

    for i in range(len(waste_bins) - 1):
        lower = waste_bins[i]
        upper = waste_bins[i + 1]
        count = bin_counts[i]
        bar_len = int(count / max_count * width) if max_count > 0 else 0

        if upper == float('inf'):
            label = f"{lower:4.0f}+ ft "
        else:
            label = f"{lower:4.0f}-{upper:<3.0f}ft"

        bar = "#" * bar_len
        print(f"  {label} | {bar} ({count})")

def process_results(
    solver: SymmetryAwareSolverV4,
    solution: list,
    status: str,
    solve_time: float,
    pipe_data: PipeData,
    show_histogram: bool = False
) -> OptimizationResult:
    """Process and display results."""
    quiet = solver.quiet
    pipe_lengths = solver.raw_lengths

    if solution is None or len(solution) == 0:
        return OptimizationResult(
            success=False,
            piles=[],
            total_piles=0,
            theoretical_max=solver.theoretical_max,
            efficiency=0,
            pipes_used=0,
            pipes_unused=len(pipe_lengths),
            total_waste=0,
            solve_time=solve_time,
            solver_status=status,
            pile_assignments=[],
            unused_indices=list(range(len(pipe_lengths))),
            error_message=f"Solver failed with status: {status}"
        )

    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)

    # Count pipe usage by type
    type_usage = Counter()
    for pile in solution:
        for idx in pile['length_indices']:
            type_usage[idx] += 1

    used_pipes = sum(type_usage.values())
    unused_pipes = len(pipe_lengths) - used_pipes

    efficiency = total_piles / solver.theoretical_max * 100 if solver.theoretical_max > 0 else 0

    if not quiet:
        section("RESULTS", quiet)
        if status == 'OPTIMAL':
            log("\n  *** OPTIMAL SOLUTION FOUND (GUARANTEED) ***")
        else:
            log(f"\n  Best solution found ({status})")

        log(f"\n  {'Metric':<40} {'Value':>15}")
        log("  " + "-"*55)
        log(f"  {f'{solver.target_length}-foot piles created':<40} {total_piles:>15,}")
        log(f"  {'Theoretical maximum':<40} {solver.theoretical_max:>15,}")
        log(f"  {'Efficiency vs theoretical':<40} {efficiency:>14.1f}%")
        log(f"  {'Total waste':<40} {total_waste:>14.1f}'")
        log(f"  {'Average waste per pile':<40} {total_waste/total_piles if total_piles else 0:>14.2f}'")
        log(f"  {'Pipes used':<40} {used_pipes:>15,}")
        log(f"  {'Pipes unused':<40} {unused_pipes:>15,}")
        log(f"  {'Solve time':<40} {solve_time:>14.1f}s")

        # Weld distribution
        section("WELD DISTRIBUTION", quiet)
        weld_counts = Counter(p['num_welds'] for p in solution)
        for welds in sorted(weld_counts.keys()):
            count = weld_counts[welds]
            pct = count / total_piles * 100
            log(f"  {welds} weld(s): {count:4d} piles ({pct:5.1f}%)")

        # Sample piles
        section("SAMPLE PILES (first 15)", quiet)
        for i, pile in enumerate(solution[:15], 1):
            segs = " + ".join([f"{l:.1f}'" for l in pile['pipe_lengths']])
            log(f"  {i:3d}: {segs} = {pile['total_length']:.1f}' (waste: {pile['waste']:.1f}')")

    # Assign specific pipe indices
    if not quiet:
        section("ASSIGNING PIPE INDICES", quiet)
        log("\n  Mapping solution to specific pipes from inventory...")

    inventory_pools = defaultdict(list)
    for idx, length in enumerate(pipe_lengths):
        rounded = round(length, solver.precision)
        inventory_pools[rounded].append(idx)

    assigned_pipes = set()
    pile_assignments = []

    for pile in solution:
        pile_pipe_indices = []
        for length_idx in pile['length_indices']:
            needed_length = solver.unique_lengths[length_idx]
            available = [idx for idx in inventory_pools[needed_length] if idx not in assigned_pipes]
            if available:
                pipe_idx = available[0]
                assigned_pipes.add(pipe_idx)
                pile_pipe_indices.append((pipe_idx, pipe_lengths[pipe_idx]))
            else:
                pile_pipe_indices.append((None, needed_length))
        pile_assignments.append(pile_pipe_indices)

    unused_indices = [i for i in range(len(pipe_lengths)) if i not in assigned_pipes]

    if not quiet:
        log(f"  Assigned {len(assigned_pipes)} unique pipe indices")
        log(f"  Unused pipes: {len(unused_indices)}")

    if show_histogram:
        print_ascii_histogram(solution, solver.target_length, quiet)

    return OptimizationResult(
        success=True,
        piles=solution,
        total_piles=total_piles,
        theoretical_max=solver.theoretical_max,
        efficiency=efficiency,
        pipes_used=used_pipes,
        pipes_unused=unused_pipes,
        total_waste=total_waste,
        solve_time=solve_time,
        solver_status=status,
        pile_assignments=pile_assignments,
        unused_indices=unused_indices
    )

def export_to_excel(
    result: OptimizationResult,
    pipe_data: PipeData,
    solver: SymmetryAwareSolverV4,
    output_path: str
) -> None:
    """Export results to Excel with multiple sheets."""
    if not result.success:
        log(f"  Skipping export - no solution found")
        return

    section("EXPORTING TO EXCEL", solver.quiet)

    pipe_lengths = solver.raw_lengths

    # Build pile details
    pile_data = []
    for pile_num, (pile, assignments) in enumerate(zip(result.piles, result.pile_assignments), 1):
        for seg_num, (pipe_idx, actual_length) in enumerate(assignments, 1):
            pile_data.append({
                'Pile Number': pile_num,
                'Segment': seg_num,
                'Pipe Index': pipe_idx if pipe_idx is not None else 'ERROR',
                'Actual Length (ft)': round(actual_length, 2),
                'Rounded Length (ft)': round(actual_length, 1),
                'Total Length': pile['total_length'] if seg_num == 1 else '',
                'Waste (ft)': pile['waste'] if seg_num == 1 else '',
                'Welds': pile['num_welds'] if seg_num == 1 else ''
            })

    df_piles = pd.DataFrame(pile_data)

    # Summary sheet
    summary = pd.DataFrame({
        'Metric': [
            'Method', 'Status', 'Source File', 'Column Used',
            'Total Piles', 'Theoretical Max', 'Efficiency',
            'Pipes Used', 'Pipes Unused', 'Total Waste (ft)', 'Avg Waste/Pile',
            'Solve Time (s)', 'Unique Pipe Indices'
        ],
        'Value': [
            'ILP V4',
            result.solver_status,
            pipe_data.source_file,
            pipe_data.column_used,
            result.total_piles,
            result.theoretical_max,
            f"{result.efficiency:.1f}%",
            result.pipes_used,
            result.pipes_unused,
            f"{result.total_waste:.2f}",
            f"{result.total_waste/result.total_piles if result.total_piles else 0:.2f}",
            f"{result.solve_time:.1f}",
            f"{result.pipes_used} (no duplicates)"
        ]
    })

    # Unused pipes sheet
    unused_df = pd.DataFrame({
        'Pipe Index': result.unused_indices,
        'Length (ft)': [pipe_lengths[i] for i in result.unused_indices]
    })

    if not solver.quiet:
        log(f"  Writing to {output_path}...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        df_piles.to_excel(writer, sheet_name='Pile Details', index=False)
        unused_df.to_excel(writer, sheet_name='Unused Pipes', index=False)

    if not solver.quiet:
        log(f"\n  Exported to: {output_path}")

def export_to_json(
    result: OptimizationResult,
    pipe_data: PipeData,
    solver: SymmetryAwareSolverV4,
    output_path: str
) -> None:
    """Export results to JSON for programmatic consumption."""
    output = {
        'metadata': {
            'version': 'V4',
            'input_file': pipe_data.source_file,
            'column_used': pipe_data.column_used,
            'target_length': solver.target_length,
            'max_waste': solver.max_waste,
            'solve_time_seconds': round(result.solve_time, 1),
            'status': result.solver_status
        },
        'summary': {
            'total_piles': result.total_piles,
            'theoretical_max': result.theoretical_max,
            'efficiency_pct': round(result.efficiency, 1),
            'pipes_used': result.pipes_used,
            'pipes_unused': result.pipes_unused,
            'total_waste_ft': round(result.total_waste, 2)
        },
        'piles': [
            {
                'pile_number': i + 1,
                'segments': [
                    {'pipe_index': idx, 'length': length}
                    for idx, length in assignments
                ],
                'total_length': pile['total_length'],
                'waste': pile['waste'],
                'welds': pile['num_welds']
            }
            for i, (pile, assignments) in enumerate(zip(result.piles, result.pile_assignments))
        ],
        'unused_pipes': [
            {'index': idx, 'length': solver.raw_lengths[idx]}
            for idx in result.unused_indices
        ]
    }

    if result.error_message:
        output['error'] = result.error_message

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    if not solver.quiet:
        log(f"\n  JSON exported to: {output_path}")

# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pipe Pile Optimizer V4 - Combine shorter pipes into target-length piles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default (backward compatible with V3)
  %(prog)s --input pipes.xlsx                 # Auto-detect column
  %(prog)s --input data.csv --column length   # Specify column
  %(prog)s --input data.xlsx -t 120 -w 15     # Custom target and waste
  %(prog)s --input data.xlsx --json           # Also export JSON
        """
    )

    parser.add_argument(
        '--input', '-i',
        help='Input file (xlsx, xls, csv, tsv). Default: pipe_lengths_clean.csv'
    )
    parser.add_argument(
        '--column', '-c',
        help='Column name containing pipe lengths (auto-detected if not specified)'
    )
    parser.add_argument(
        '--delimiter',
        default=',',
        help='CSV delimiter. Default: ,'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Input file has no header row'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output Excel file path. Default: pipe_optimization_V4.xlsx'
    )
    parser.add_argument(
        '--target', '-t',
        type=float,
        default=100.0,
        help='Target pile length in feet. Default: 100'
    )
    parser.add_argument(
        '--waste', '-w',
        type=float,
        default=20.0,
        help='Maximum acceptable waste per pile in feet. Default: 20'
    )
    parser.add_argument(
        '--precision', '-p',
        type=int,
        default=1,
        help='Decimal places for rounding lengths. Default: 1'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=14,
        help='Number of solver threads. Default: 14'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=900,
        help='Maximum solver time in seconds. Default: 900 (15 min)'
    )
    parser.add_argument(
        '--gap',
        type=float,
        default=0.005,
        help='Stop if optimality gap < this ratio. Default: 0.005 (0.5%%)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Also export results as JSON'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Output only JSON to stdout (no Excel, no console output)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--histogram',
        action='store_true',
        help='Show ASCII waste distribution histogram'
    )

    return parser.parse_args()

def main() -> int:
    """Main entry point."""
    args = parse_args()

    quiet = args.quiet or args.json_only

    # Validate inputs
    if args.target <= 0:
        print("ERROR: Target length must be positive")
        return 1
    if args.waste < 0:
        print("ERROR: Max waste cannot be negative")
        return 1

    # Determine input file
    if args.input:
        input_file = args.input
    else:
        # Default fallback for backward compatibility with V3
        default_files = ['pipe_lengths_clean.csv', 'pipes.xlsx', 'pipes.csv', 'test_pipes.csv']
        input_file = None
        for f in default_files:
            if Path(f).exists():
                input_file = f
                break
        if not input_file:
            print("ERROR: No input file specified and no default file found.")
            print("Usage: python3 pipe_optimizer_v4.py --input <file>")
            return 1

    # Load data
    try:
        if not quiet:
            section("PIPE PILE OPTIMIZER V4", quiet)
            log("\n  Loading data...")

        pipe_data = load_pipe_data(
            input_file,
            column_name=args.column,
            delimiter=args.delimiter,
            has_header=not args.no_header,
            quiet=quiet
        )

        if not quiet:
            log(f"  Source: {pipe_data.source_file}")
            log(f"  Column: '{pipe_data.column_used}'")
            log(f"  Loaded {pipe_data.valid_count} pipes ({pipe_data.excluded_count} excluded)")
            if pipe_data.exclusion_reasons:
                log(f"  Exclusions: {pipe_data.exclusion_reasons}")

            total_material = pipe_data.lengths.sum()
            theoretical_max = int(total_material // args.target)
            log(f"\n  Total material: {total_material:.1f} feet")
            log(f"  Range: {pipe_data.lengths.min():.1f}' to {pipe_data.lengths.max():.1f}'")
            log(f"  Theoretical max: {total_material:.1f} / {args.target} = {theoretical_max} piles")

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize solver
    solver = SymmetryAwareSolverV4(
        pipe_data.lengths,
        target_length=args.target,
        max_waste=args.waste,
        precision=args.precision,
        quiet=quiet
    )

    if not quiet:
        section("SOLVER CONFIGURATION", quiet)
        log(f"\n  Target pile length: {solver.target_length}'")
        log(f"  Max waste allowed: {solver.max_waste}'")
        log(f"  Valid range: {solver.target_length}' to {solver.target_length + solver.max_waste}' (inclusive)")
        log(f"  Precision: {solver.precision} decimal place(s)")
        log(f"\n  Raw pipes: {len(solver.raw_lengths)}")
        log(f"  Unique length types: {solver.n_types}")
        log(f"  Symmetry factor: {len(solver.raw_lengths)/solver.n_types:.1f}x")

    # Generate patterns
    start_total = time.time()
    patterns = solver.generate_patterns()

    if not patterns:
        print("ERROR: No valid patterns found. Try increasing --waste value or check input data.")
        return 1

    # Solve
    solution, status, solve_time = solver.solve_ilp(
        patterns,
        time_limit=args.time_limit,
        gap=args.gap,
        threads=args.threads
    )

    # Process results
    result = process_results(
        solver, solution, status, solve_time, pipe_data,
        show_histogram=args.histogram
    )

    if not result.success:
        print(f"ERROR: {result.error_message}")
        return 1

    # Output
    if args.json_only:
        # JSON to stdout
        output = {
            'success': result.success,
            'total_piles': result.total_piles,
            'theoretical_max': result.theoretical_max,
            'efficiency_pct': round(result.efficiency, 1),
            'pipes_used': result.pipes_used,
            'pipes_unused': result.pipes_unused,
            'total_waste_ft': round(result.total_waste, 2),
            'status': result.solver_status
        }
        print(json.dumps(output, indent=2))
    else:
        # Excel export
        output_file = args.output or 'pipe_optimization_V4.xlsx'
        export_to_excel(result, pipe_data, solver, output_file)

        # JSON export (optional)
        if args.json:
            json_file = Path(output_file).stem + '.json'
            export_to_json(result, pipe_data, solver, json_file)

        total_time = time.time() - start_total
        section("COMPLETE", quiet)
        log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        log(f"\n  Final result: {result.total_piles} piles out of {result.theoretical_max} theoretical max ({result.efficiency:.1f}%)")
        log("="*80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
