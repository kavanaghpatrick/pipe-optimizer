#!/usr/bin/env python3
"""
Memory-Safe M4 MacBook Pipe Pile Optimizer v2.0
================================================

Fixes from crash analysis (Grok + Gemini review):
1. Worker initializer pattern - no data copying on macOS spawn
2. Pattern limits with adaptive waste filtering
3. Pre-built reverse index for O(1) constraint lookup (fixes O(N*P) bug)
4. Memory monitoring throughout with configurable limits
5. Proper pool cleanup with context managers

Usage:
    python3 pipe_optimizer_m4_safe.py
    python3 pipe_optimizer_m4_safe.py --max-waste 5.0 --max-patterns 1000000
    python3 pipe_optimizer_m4_safe.py --memory-limit 6.0 --memory-verbose
"""

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip3 install psutil")

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
# GLOBAL CONFIGURATION
# =============================================================================

# Memory limits (GB) - configurable via CLI
DEFAULT_MEMORY_SOFT_LIMIT_GB = 5.0  # Warning threshold
DEFAULT_MEMORY_HARD_LIMIT_GB = 7.0  # Abort threshold

# Pattern limits - conservative based on reviewer feedback
DEFAULT_MAX_PATTERNS_PER_CHUNK = 50_000   # Per-worker cap
DEFAULT_MAX_TOTAL_PATTERNS = 1_000_000    # Total cap (was 2M, reduced for safety)

# Adaptive waste levels (progressively tighter)
WASTE_LEVELS = [10.0, 7.0, 5.0, 3.0, 2.0]

# =============================================================================
# GLOBAL WORKER STATE (for macOS spawn - avoids data copying)
# =============================================================================

_worker_sorted_indices: Optional[list] = None
_worker_sorted_lengths: Optional[list] = None
_worker_target: float = 100.0
_worker_max_waste: float = 10.0
_worker_max_patterns: int = 50_000


def init_worker(sorted_indices: list, sorted_lengths: list,
                target: float, max_waste: float, max_patterns: int):
    """Initialize worker with shared read-only data (called once per worker)"""
    global _worker_sorted_indices, _worker_sorted_lengths
    global _worker_target, _worker_max_waste, _worker_max_patterns
    _worker_sorted_indices = sorted_indices
    _worker_sorted_lengths = sorted_lengths
    _worker_target = target
    _worker_max_waste = max_waste
    _worker_max_patterns = max_patterns


def generate_chunk_worker(args: tuple) -> list:
    """Worker function - uses globals to avoid data copying"""
    chunk_id, chunk_start, chunk_end, all_indices, progress_dict = args

    # Access globals (set by init_worker)
    sorted_indices = _worker_sorted_indices
    sorted_lengths = _worker_sorted_lengths
    target = _worker_target
    max_waste = _worker_max_waste
    max_patterns = _worker_max_patterns

    patterns = []
    n = len(sorted_lengths)

    for idx_num, idx in enumerate(range(chunk_start, chunk_end)):
        # Safety cap per chunk
        if len(patterns) >= max_patterns:
            break

        i = all_indices[idx]
        len_i = sorted_lengths[i]

        for j in range(i + 1, n - 1):
            if len(patterns) >= max_patterns:
                break

            len_j = sorted_lengths[j]

            # Early stop: if max possible sum < target
            if len_i + len_j + sorted_lengths[j + 1] < target:
                break

            for k in range(j + 1, n):
                len_k = sorted_lengths[k]
                total = len_i + len_j + len_k

                # Early stop: remaining k are smaller
                if total < target:
                    break

                waste = total - target
                if waste <= max_waste:
                    patterns.append((
                        [sorted_indices[i], sorted_indices[j], sorted_indices[k]],
                        total,
                        waste
                    ))

                    if len(patterns) >= max_patterns:
                        break

        # Update progress
        if progress_dict is not None:
            progress_dict[chunk_id] = idx_num + 1

    return patterns


# =============================================================================
# MEMORY MONITORING
# =============================================================================

class MemoryMonitor:
    """Memory monitoring with configurable limits"""

    def __init__(self, soft_limit_gb: float, hard_limit_gb: float, verbose: bool = False):
        self.soft_limit_gb = soft_limit_gb
        self.hard_limit_gb = hard_limit_gb
        self.verbose = verbose
        self.peak_gb = 0.0
        self.checks = 0

    def get_memory_gb(self) -> float:
        """Get current process memory in GB"""
        if not HAS_PSUTIL:
            return 0.0
        return psutil.Process().memory_info().rss / (1024**3)

    def get_system_available_gb(self) -> float:
        """Get available system memory in GB"""
        if not HAS_PSUTIL:
            return float('inf')
        return psutil.virtual_memory().available / (1024**3)

    def check(self, phase: str) -> float:
        """Check memory and raise if hard limit exceeded"""
        mem = self.get_memory_gb()
        self.checks += 1
        self.peak_gb = max(self.peak_gb, mem)

        if self.verbose and self.checks % 10 == 0:
            print(f"  [Memory] {phase}: {mem:.2f}GB (peak: {self.peak_gb:.2f}GB)")

        if mem > self.hard_limit_gb:
            raise MemoryError(
                f"[{phase}] Memory {mem:.2f}GB exceeds hard limit {self.hard_limit_gb}GB. "
                f"Peak was {self.peak_gb:.2f}GB. Aborting to prevent crash."
            )

        if mem > self.soft_limit_gb:
            print(f"  WARNING [{phase}]: Memory {mem:.2f}GB approaching limit "
                  f"({self.soft_limit_gb}GB soft, {self.hard_limit_gb}GB hard)")

        return mem

    def report(self):
        """Print memory usage report"""
        print(f"\n  Memory Report:")
        print(f"    Peak usage: {self.peak_gb:.2f}GB")
        print(f"    Checks performed: {self.checks}")
        print(f"    Limits: {self.soft_limit_gb}GB (soft), {self.hard_limit_gb}GB (hard)")


# =============================================================================
# PROGRESS BAR
# =============================================================================

def progress_bar(current: int, total: int, prefix: str = '', width: int = 50):
    """Print a progress bar"""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '\u2588' * filled + '\u2591' * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:5.1f}% ({current:,}/{total:,})", end='', flush=True)


def section(title: str):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


# =============================================================================
# MEMORY-SAFE SOLVER
# =============================================================================

class MemorySafeSolver:
    """
    Memory-safe pipe pile optimizer with:
    - Worker initializer (no data copying)
    - Pattern limits
    - Pre-built reverse index for constraints
    - Memory monitoring throughout
    """

    def __init__(
        self,
        pipe_lengths: np.ndarray,
        target_length: float = 100.0,
        max_waste: float = 10.0,
        max_patterns_per_chunk: int = DEFAULT_MAX_PATTERNS_PER_CHUNK,
        max_total_patterns: int = DEFAULT_MAX_TOTAL_PATTERNS,
        memory_monitor: Optional[MemoryMonitor] = None,
        n_cores: Optional[int] = None
    ):
        self.pipe_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.max_patterns_per_chunk = max_patterns_per_chunk
        self.max_total_patterns = max_total_patterns
        self.memory = memory_monitor or MemoryMonitor(
            DEFAULT_MEMORY_SOFT_LIMIT_GB,
            DEFAULT_MEMORY_HARD_LIMIT_GB
        )
        self.n_pipes = len(pipe_lengths)

        # Auto-detect cores, leave 2 for system
        if n_cores is None:
            self.n_cores = max(1, mp.cpu_count() - 2)
        else:
            self.n_cores = n_cores

    def generate_two_pipe_patterns(self) -> list:
        """Generate all 2-pipe patterns with waste filter"""
        patterns = []
        n = self.n_pipes

        print("\n[1/2] Generating 2-pipe patterns...")
        count = 0
        total = n * (n - 1) // 2
        last_report = 0

        for i in range(n):
            for j in range(i + 1, n):
                total_len = self.pipe_lengths[i] + self.pipe_lengths[j]
                if total_len >= self.target_length:
                    waste = total_len - self.target_length
                    if waste <= self.max_waste:
                        patterns.append(([i, j], total_len, waste))

                        # Check pattern limit
                        if len(patterns) >= self.max_total_patterns // 2:
                            print(f"\n  Hit 2-pipe pattern limit ({len(patterns):,})")
                            return patterns

                count += 1
                if count - last_report >= total // 100 or count == total:
                    progress_bar(count, total, '2-pipe')
                    last_report = count

        print()
        print(f"  Found {len(patterns):,} valid 2-pipe patterns")
        self.memory.check("2-pipe patterns")
        return patterns

    def generate_three_pipe_patterns_safe(self) -> list:
        """Generate 3-pipe patterns with memory-safe multiprocessing"""
        print(f"\n[2/2] Generating 3-pipe patterns using {self.n_cores} cores...")
        print(f"  Max waste: {self.max_waste}', Pattern limit: {self.max_total_patterns:,}")

        # Sort by length descending for early stopping
        pipes = sorted(enumerate(self.pipe_lengths), key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in pipes]
        sorted_lengths = [length for _, length in pipes]

        # Find max starting position
        i_max = -1
        for i in range(len(sorted_lengths) - 2):
            max_sum = sorted_lengths[i] + sorted_lengths[i + 1] + sorted_lengths[i + 2]
            if max_sum < self.target_length:
                break
            i_max = i
        else:
            i_max = len(sorted_lengths) - 3

        if i_max < 0:
            print("  No valid 3-pipe patterns possible")
            return []

        print(f"  Max starting position: {i_max}")
        all_indices = list(range(i_max + 1))

        # Create chunks (fewer than before - just 2x cores)
        n_chunks = self.n_cores * 2
        chunk_size = max(1, len(all_indices) // n_chunks)

        # Shared progress tracking
        manager = mp.Manager()
        progress_dict = manager.dict()

        chunks = []
        chunk_sizes = []
        for chunk_id, c in enumerate(range(0, len(all_indices), chunk_size)):
            end = min(c + chunk_size, len(all_indices))
            progress_dict[chunk_id] = 0
            chunk_sizes.append(end - c)
            # Note: only indices in args, not data (data via init_worker)
            chunks.append((chunk_id, c, end, all_indices, progress_dict))

        print(f"  Processing {len(all_indices)} positions in {len(chunks)} chunks...")

        self.memory.check("Before pool")
        start_time = time.time()

        # Use context manager + initializer + maxtasksperchild for memory safety
        patterns = []
        with mp.Pool(
            processes=self.n_cores,
            initializer=init_worker,
            initargs=(sorted_indices, sorted_lengths, self.target_length,
                     self.max_waste, self.max_patterns_per_chunk),
            maxtasksperchild=1  # Restart workers to prevent leaks
        ) as pool:

            # Use imap_unordered for streaming results
            total_work = sum(chunk_sizes)
            for result in pool.imap_unordered(generate_chunk_worker, chunks):
                patterns.extend(result)

                # Progress
                completed = sum(progress_dict.values())
                progress_bar(completed, total_work, '3-pipe')

                # Memory check
                self.memory.check("3-pipe aggregation")

                # Pattern cap
                if len(patterns) >= self.max_total_patterns:
                    print(f"\n  Hit total pattern limit at {len(patterns):,}")
                    break

        # Cleanup
        manager.shutdown()
        progress_bar(total_work, total_work, '3-pipe')
        print()

        elapsed = time.time() - start_time
        print(f"  Found {len(patterns):,} valid 3-pipe patterns in {elapsed:.1f}s")

        # Truncate if over limit
        if len(patterns) > self.max_total_patterns:
            patterns = patterns[:self.max_total_patterns]
            print(f"  Truncated to {len(patterns):,} patterns")

        self.memory.check("After 3-pipe generation")
        return patterns

    def generate_patterns_adaptive(self) -> list:
        """Generate patterns with adaptive waste filtering"""
        section("ADAPTIVE PATTERN GENERATION")

        for max_waste in WASTE_LEVELS:
            print(f"\nTrying max_waste = {max_waste}'...")
            self.max_waste = max_waste

            # Generate patterns
            patterns = []
            patterns.extend(self.generate_two_pipe_patterns())
            patterns.extend(self.generate_three_pipe_patterns_safe())

            total = len(patterns)
            print(f"\nTotal patterns: {total:,}")

            if total <= self.max_total_patterns:
                print(f"SUCCESS: {total:,} patterns fits within {self.max_total_patterns:,} limit")
                return patterns

            print(f"Too many patterns ({total:,} > {self.max_total_patterns:,}), tightening waste filter...")
            del patterns
            gc.collect()
            self.memory.check("After GC")

        raise ValueError(
            f"Cannot generate safe number of patterns even with max_waste={WASTE_LEVELS[-1]}'. "
            f"Consider reducing --max-patterns or using a smaller dataset."
        )

    def solve_ilp(self, patterns: list) -> tuple:
        """
        Solve ILP with pre-built reverse index (fixes O(N*P) bug).

        Original streaming approach was O(N * P) = billions of operations.
        This uses O(P) index build + O(N) constraint add = much faster.
        """
        n_patterns = len(patterns)

        section("ILP SOLVING")
        print(f"\n  Pipes: {self.n_pipes:,}")
        print(f"  Patterns: {n_patterns:,}")
        print(f"  Variables: {n_patterns:,} (binary)")

        # Warning instead of hard error - pattern generation already caps
        if n_patterns > self.max_total_patterns * 1.5:
            raise ValueError(f"Too many patterns ({n_patterns:,}) - would exceed memory")
        elif n_patterns > self.max_total_patterns:
            print(f"  WARNING: {n_patterns:,} patterns slightly exceeds target {self.max_total_patterns:,}")

        self.memory.check("ILP setup")

        # Build model
        print("\n  Building ILP model...")
        prob = LpProblem("Optimal_Piles_Safe", LpMaximize)

        # Create variables
        print("  Creating variables...")
        x = [LpVariable(f"x{p}", cat='Binary') for p in range(n_patterns)]
        prob += lpSum(x), "Total_Piles"

        self.memory.check("After variables")

        # BUILD REVERSE INDEX (critical fix from Gemini review)
        # This is O(P) instead of O(N*P) for constraint generation
        print("  Building pipe-to-pattern index...")
        pipe_to_patterns = defaultdict(list)
        for p_idx, (pipe_indices, _, _) in enumerate(patterns):
            for pipe_idx in pipe_indices:
                pipe_to_patterns[pipe_idx].append(p_idx)

            if p_idx % 100_000 == 0:
                progress_bar(p_idx, n_patterns, 'Index')

        progress_bar(n_patterns, n_patterns, 'Index')
        print()
        self.memory.check("After index build")

        # Add constraints using index (O(N) not O(N*P))
        print("  Adding constraints...")
        constraints_added = 0
        for pipe_idx in range(self.n_pipes):
            pattern_indices = pipe_to_patterns.get(pipe_idx, [])
            if pattern_indices:
                prob += lpSum(x[p] for p in pattern_indices) <= 1, f"Pipe_{pipe_idx}"
                constraints_added += 1

            if pipe_idx % 100 == 0:
                progress_bar(pipe_idx, self.n_pipes, 'Constraints')

            # Check memory every 500 constraints
            if pipe_idx % 500 == 0:
                self.memory.check(f"Constraint {pipe_idx}")

        progress_bar(self.n_pipes, self.n_pipes, 'Constraints')
        print()
        print(f"  Added {constraints_added} constraints")

        # Free index memory
        del pipe_to_patterns
        gc.collect()
        self.memory.check("Before solve")

        # Solve
        section("SOLVING")
        print(f"\n  Solver: CBC with {self.n_cores} threads")
        print(f"  Time limit: 30 minutes")
        print("-" * 80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=1, timeLimit=1800, threads=self.n_cores)
        prob.solve(solver)
        solve_time = time.time() - start_time

        print("-" * 80)
        print(f"\n  Solved in {solve_time:.1f}s")
        print(f"  Status: {LpStatus[prob.status]}")

        self.memory.check("After solve")

        # Extract solution
        if prob.status in [LpStatusOptimal, 1]:
            print("  Extracting solution...")
            solution = []
            for p in range(n_patterns):
                if x[p].varValue and x[p].varValue > 0.5:
                    pipe_indices, total_length, waste = patterns[p]
                    solution.append({
                        'pipe_indices': pipe_indices,
                        'pipe_lengths': [self.pipe_lengths[i] for i in pipe_indices],
                        'total_length': total_length,
                        'waste': waste,
                        'num_welds': len(pipe_indices) - 1
                    })
            print(f"  Extracted {len(solution)} piles")
            return solution, prob.status, solve_time

        return None, prob.status, solve_time


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def export_results(pipe_lengths: np.ndarray, solution: list, status: int,
                   solve_time: float, output_file: str, max_waste_used: float):
    """Export results to Excel"""
    n_pipes = len(pipe_lengths)
    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)
    avg_waste = total_waste / total_piles if total_piles > 0 else 0

    used_pipes = set()
    for pile in solution:
        used_pipes.update(pile['pipe_indices'])

    unused = set(range(n_pipes)) - used_pipes

    section("RESULTS")

    if status == LpStatusOptimal:
        print("\n  *** OPTIMAL SOLUTION FOUND ***")
    else:
        print(f"\n  Best solution found ({LpStatus[status]})")

    print(f"\n  {'Metric':<40} {'Value':>15}")
    print("  " + "-"*55)
    print(f"  {'100-foot piles created':<40} {total_piles:>15,}")
    print(f"  {'Total waste':<40} {total_waste:>14.1f}'")
    print(f"  {'Average waste per pile':<40} {avg_waste:>14.2f}'")
    print(f"  {'Pipes used':<40} {len(used_pipes):>15,}")
    print(f"  {'Pipes unused':<40} {len(unused):>15,}")
    print(f"  {'Max waste filter used':<40} {max_waste_used:>14.1f}'")
    print(f"  {'Solve time':<40} {solve_time:>14.1f}s")

    # Comparison with greedy
    print(f"\n  vs Greedy (163 piles):")
    if total_piles > 163:
        print(f"    +{total_piles - 163} piles improvement!")
    elif total_piles == 163:
        print(f"    Same as greedy")
    else:
        print(f"    {total_piles - 163} piles (may need looser waste filter)")

    # Weld distribution
    weld_counts = defaultdict(int)
    for pile in solution:
        weld_counts[pile['num_welds']] += 1

    print(f"\n  Weld distribution:")
    for welds in sorted(weld_counts.keys()):
        count = weld_counts[welds]
        print(f"    {welds} weld(s): {count} piles ({count/total_piles*100:.1f}%)")

    # Export to Excel
    section("EXPORTING TO EXCEL")

    pile_data = []
    for pile_num, pile in enumerate(solution, 1):
        for seg_num, (idx, length) in enumerate(zip(pile['pipe_indices'], pile['pipe_lengths']), 1):
            pile_data.append({
                'Pile Number': pile_num,
                'Segment': seg_num,
                'Pipe Index': idx + 1,
                'Length (ft)': length,
                'Total Length': pile['total_length'] if seg_num == 1 else '',
                'Waste (ft)': pile['waste'] if seg_num == 1 else '',
                'Welds': pile['num_welds'] if seg_num == 1 else ''
            })

    df_piles = pd.DataFrame(pile_data)

    summary = pd.DataFrame({
        'Metric': ['Method', 'Status', 'Total Piles', 'Total Waste', 'Avg Waste',
                   'Pipes Used', 'Pipes Unused', 'vs Greedy', 'Max Waste Filter', 'Solve Time'],
        'Value': ['ILP (Memory-Safe v2.0)',
                  'OPTIMAL' if status == LpStatusOptimal else LpStatus[status],
                  total_piles, f"{total_waste:.2f}", f"{avg_waste:.2f}",
                  len(used_pipes), len(unused),
                  f"+{total_piles - 163}" if total_piles > 163 else str(total_piles - 163),
                  f"{max_waste_used}'",
                  f"{solve_time:.1f}s"]
    })

    unused_df = pd.DataFrame({
        'Pipe Index': [idx + 1 for idx in sorted(unused)],
        'Length (ft)': [pipe_lengths[idx] for idx in sorted(unused)]
    })

    print(f"  Writing to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        df_piles.to_excel(writer, sheet_name='Pile Details', index=False)
        unused_df.to_excel(writer, sheet_name='Unused Pipes', index=False)

    print(f"  Exported to: {output_file}")

    # Sample piles
    print(f"\n  Sample piles (first 10):")
    for i, pile in enumerate(solution[:10], 1):
        segs = " + ".join([f"{l:.1f}'" for l in pile['pipe_lengths']])
        print(f"    {i}: {segs} = {pile['total_length']:.1f}' (waste: {pile['waste']:.1f}')")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Memory-Safe Pipe Optimizer v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', '-i', default='pipe_lengths_clean.csv',
                        help='Input CSV file')
    parser.add_argument('--output', '-o', default='pipe_optimization_M4_SAFE.xlsx',
                        help='Output Excel file')
    parser.add_argument('--target', '-t', type=float, default=100.0,
                        help='Target pile length (default: 100)')
    parser.add_argument('--max-waste', '-w', type=float, default=None,
                        help='Max waste (default: adaptive from 10 down to 2)')
    parser.add_argument('--max-patterns', type=int, default=DEFAULT_MAX_TOTAL_PATTERNS,
                        help=f'Max total patterns (default: {DEFAULT_MAX_TOTAL_PATTERNS:,})')
    parser.add_argument('--max-patterns-chunk', type=int, default=DEFAULT_MAX_PATTERNS_PER_CHUNK,
                        help=f'Max patterns per chunk (default: {DEFAULT_MAX_PATTERNS_PER_CHUNK:,})')
    parser.add_argument('--memory-limit', type=float, default=DEFAULT_MEMORY_HARD_LIMIT_GB,
                        help=f'Hard memory limit in GB (default: {DEFAULT_MEMORY_HARD_LIMIT_GB})')
    parser.add_argument('--memory-soft', type=float, default=DEFAULT_MEMORY_SOFT_LIMIT_GB,
                        help=f'Soft memory limit in GB (default: {DEFAULT_MEMORY_SOFT_LIMIT_GB})')
    parser.add_argument('--memory-verbose', action='store_true',
                        help='Verbose memory monitoring')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of cores (default: auto-detect minus 2)')

    return parser.parse_args()


def main():
    args = parse_args()

    section("MEMORY-SAFE PIPE OPTIMIZER v2.0")
    print("\nFixes applied from Grok + Gemini crash analysis:")
    print("  - Worker initializer (no data copying on macOS)")
    print("  - Pattern limits with adaptive waste filtering")
    print("  - Pre-built reverse index (O(P) not O(N*P))")
    print("  - Memory monitoring throughout")
    print("  - Proper pool cleanup with context managers")

    # Initialize memory monitor
    memory = MemoryMonitor(
        soft_limit_gb=args.memory_soft,
        hard_limit_gb=args.memory_limit,
        verbose=args.memory_verbose
    )

    print(f"\n  Memory limits: {args.memory_soft}GB (soft), {args.memory_limit}GB (hard)")
    print(f"  Pattern limits: {args.max_patterns:,} total, {args.max_patterns_chunk:,} per chunk")

    # Load data
    section("LOADING DATA")
    try:
        pipe_lengths = np.loadtxt(args.input, delimiter=',')
    except Exception as e:
        print(f"\nERROR: Could not load {args.input}: {e}")
        sys.exit(1)

    print(f"\n  Loaded {len(pipe_lengths)} pipes from {args.input}")
    print(f"  Total material: {pipe_lengths.sum():.1f} feet")
    print(f"  Range: {pipe_lengths.min():.1f}' to {pipe_lengths.max():.1f}'")

    memory.check("After data load")

    # Initialize solver
    solver = MemorySafeSolver(
        pipe_lengths=pipe_lengths,
        target_length=args.target,
        max_waste=args.max_waste or WASTE_LEVELS[0],
        max_patterns_per_chunk=args.max_patterns_chunk,
        max_total_patterns=args.max_patterns,
        memory_monitor=memory,
        n_cores=args.cores
    )

    print(f"\n  Using {solver.n_cores} cores")

    # Generate patterns (adaptive or fixed)
    start_time = time.time()

    if args.max_waste:
        # Fixed waste filter
        patterns = []
        patterns.extend(solver.generate_two_pipe_patterns())
        patterns.extend(solver.generate_three_pipe_patterns_safe())
        max_waste_used = args.max_waste
    else:
        # Adaptive waste filtering
        patterns = solver.generate_patterns_adaptive()
        max_waste_used = solver.max_waste

    pattern_time = time.time() - start_time

    print(f"\n  Total patterns: {len(patterns):,}")
    print(f"  Pattern generation time: {pattern_time:.1f}s")

    two_pipe = sum(1 for p in patterns if len(p[0]) == 2)
    three_pipe = sum(1 for p in patterns if len(p[0]) == 3)
    print(f"  2-pipe: {two_pipe:,}, 3-pipe: {three_pipe:,}")

    if not patterns:
        print("\nERROR: No valid patterns found!")
        sys.exit(1)

    # Solve
    solution, status, solve_time = solver.solve_ilp(patterns)

    if solution is None or len(solution) == 0:
        print(f"\nERROR: No solution found! Status: {LpStatus[status]}")
        sys.exit(1)

    # Export
    export_results(pipe_lengths, solution, status, solve_time, args.output, max_waste_used)

    # Final report
    total_time = time.time() - start_time
    section("COMPLETE")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Result: {len(solution)} piles")
    memory.report()
    print("="*80)


if __name__ == "__main__":
    # Ensure spawn method on macOS
    mp.set_start_method('spawn', force=True)
    main()
