#!/usr/bin/env python3
"""
Pipe Pile Optimizer V2 - Symmetry-Aware Formulation
====================================================
Key optimizations based on Gemini audit:
1. UNBUFFERED OUTPUT - real-time visibility of all steps
2. SYMMETRY-AWARE - group pipes by unique lengths
3. INTEGER VARIABLES - "how many of pattern type X" vs binary per-pipe
4. REDUCED PATTERN SPACE - ~1M patterns vs 28.5M

This should solve in seconds instead of 30+ minutes.
"""

import numpy as np
import pandas as pd
from pulp import *
from collections import Counter
import time
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg, end='\n'):
    """Print with immediate flush"""
    print(msg, end=end, flush=True)

def progress_bar(current, total, prefix='', width=50):
    """Print a progress bar with immediate flush"""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:5.1f}% ({current:,}/{total:,})", end='', flush=True)

def section(title):
    """Print a section header"""
    log(f"\n{'='*80}")
    log(title)
    log('='*80)


class SymmetryAwareSolver:
    """
    Exploits the fact that pipes of the same length are interchangeable.
    Instead of 758 unique pipe variables, we work with ~273 unique LENGTH types.
    """

    def __init__(self, pipe_lengths, target_length=100.0, max_waste=20.0, precision=1):
        self.raw_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.precision = precision

        # Group pipes by rounded length
        rounded = [round(p, precision) for p in pipe_lengths]
        self.inventory = Counter(rounded)
        self.unique_lengths = sorted(self.inventory.keys(), reverse=True)
        self.n_types = len(self.unique_lengths)

        # Create fast lookup: length -> index
        self.length_to_idx = {l: i for i, l in enumerate(self.unique_lengths)}

    def generate_patterns(self):
        """Generate patterns over unique length TYPES (not individual pipes)"""
        section("PATTERN GENERATION (Symmetry-Aware)")

        log(f"\n  Unique length types: {self.n_types} (vs {len(self.raw_lengths)} raw pipes)")
        log(f"  Symmetry reduction: {len(self.raw_lengths)/self.n_types:.1f}x")
        log(f"  Target: {self.target_length}', Max waste: {self.max_waste}'")

        patterns = []
        lengths = self.unique_lengths
        n = len(lengths)

        # 2-pipe patterns
        log("\n[1/2] Generating 2-type patterns...")
        start = time.time()
        two_count = 0
        total_pairs = n * (n + 1) // 2  # Including same-type pairs
        checked = 0

        for i in range(n):
            for j in range(i, n):  # j >= i to include same-type pairs (e.g., two 50.3' pipes)
                total_len = lengths[i] + lengths[j]
                if self.target_length <= total_len < self.target_length + self.max_waste:
                    waste = total_len - self.target_length
                    # Pattern: (length_indices, total, waste, multiplicity_requirement)
                    # multiplicity_requirement: how many of each type needed
                    if i == j:
                        # Need 2 of the same type
                        if self.inventory[lengths[i]] >= 2:
                            patterns.append(((i, j), total_len, waste, {i: 2}))
                            two_count += 1
                    else:
                        # Need 1 of each type
                        patterns.append(((i, j), total_len, waste, {i: 1, j: 1}))
                        two_count += 1

                checked += 1
                if checked % 1000 == 0:
                    progress_bar(checked, total_pairs, '2-type')

        progress_bar(total_pairs, total_pairs, '2-type')
        log(f"\n  Found {two_count:,} valid 2-type patterns in {time.time()-start:.1f}s")

        # 3-pipe patterns
        log("\n[2/2] Generating 3-type patterns...")
        start = time.time()
        three_count = 0

        # Estimate total - roughly n^3/6 for combinations with repetition
        total_est = n * (n + 1) * (n + 2) // 6
        checked = 0

        for i in range(n):
            len_i = lengths[i]
            # Early termination: if 3 of the longest remaining can't reach target, stop
            if 3 * len_i < self.target_length:
                break

            for j in range(i, n):
                len_j = lengths[j]
                # Early termination for j
                if len_i + 2 * len_j < self.target_length:
                    continue
                if len_i + len_j + lengths[j] > self.target_length + self.max_waste:
                    continue

                for k in range(j, n):
                    len_k = lengths[k]
                    total_len = len_i + len_j + len_k

                    # Early termination: lengths are sorted descending
                    if total_len < self.target_length:
                        break

                    if total_len < self.target_length + self.max_waste:
                        waste = total_len - self.target_length

                        # Check inventory feasibility
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
                        progress_bar(min(checked, total_est), total_est, '3-type')

        progress_bar(total_est, total_est, '3-type')
        log(f"\n  Found {three_count:,} valid 3-type patterns in {time.time()-start:.1f}s")

        log(f"\n  TOTAL: {len(patterns):,} unique pattern types")
        log(f"  (vs ~28.5M in naive formulation - {28580646/len(patterns):.0f}x reduction!)")

        return patterns

    def solve_ilp(self, patterns):
        """Solve using integer variables for pattern counts"""
        section("ILP SOLVING (Integer Variables)")

        n_patterns = len(patterns)
        log(f"\n  Pattern types: {n_patterns:,}")
        log(f"  Length types: {self.n_types}")
        log(f"  Variables: {n_patterns:,} (integer)")
        log(f"  Constraints: {self.n_types}")

        log("\n  Building model...")
        prob = LpProblem("Optimal_Piles_V2", LpMaximize)

        # Integer variable for each pattern TYPE: how many times to use it
        log("  Creating variables...", end='')
        sys.stdout.flush()

        # Calculate max possible uses per pattern based on inventory
        x = []
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            max_uses = min(
                self.inventory[self.unique_lengths[idx]] // cnt
                for idx, cnt in counts_needed.items()
            )
            x.append(LpVariable(f"p{p_idx}", lowBound=0, upBound=max_uses, cat='Integer'))

        log(" done")

        # Objective: maximize total piles (sum of pattern uses)
        log("  Setting objective...", end='')
        sys.stdout.flush()
        prob += lpSum(x), "Total_Piles"
        log(" done")

        # Constraints: for each length TYPE, total usage <= inventory
        log("  Adding inventory constraints...")

        # Build index: which patterns use each length type
        type_to_patterns = [[] for _ in range(self.n_types)]
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            for type_idx, count in counts_needed.items():
                type_to_patterns[type_idx].append((p_idx, count))

        for type_idx in range(self.n_types):
            if type_idx % 50 == 0:
                progress_bar(type_idx, self.n_types, 'Constraints')

            length = self.unique_lengths[type_idx]
            available = self.inventory[length]

            if type_to_patterns[type_idx]:
                prob += (
                    lpSum(count * x[p_idx] for p_idx, count in type_to_patterns[type_idx]) <= available,
                    f"Type_{type_idx}_{length}"
                )

        progress_bar(self.n_types, self.n_types, 'Constraints')
        log("")

        # Solve
        section("SOLVING")
        log("\n  Solver: CBC with 14 threads")
        log("  Time limit: 10 minutes")
        log("\n" + "-"*80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=1, timeLimit=600, threads=14)
        prob.solve(solver)
        solve_time = time.time() - start_time

        log("-"*80)
        log(f"\n  Solved in {solve_time:.1f}s")
        log(f"  Status: {LpStatus[prob.status]}")

        if prob.status in [LpStatusOptimal, LpStatusNotSolved]:
            # Extract solution
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

            log(f"  Extracted {len(solution)} piles")
            return solution, prob.status, solve_time

        return None, prob.status, solve_time


def export_results(solver, solution, status, solve_time):
    """Export and display results"""
    section("RESULTS")

    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)
    avg_waste = total_waste / total_piles if total_piles > 0 else 0

    # Count pipe usage by type
    type_usage = Counter()
    for pile in solution:
        for idx in pile['length_indices']:
            type_usage[idx] += 1

    used_pipes = sum(type_usage.values())
    unused_pipes = len(solver.raw_lengths) - used_pipes

    if status == LpStatusOptimal:
        log("\n  *** OPTIMAL SOLUTION FOUND (GUARANTEED) ***")
    else:
        log(f"\n  Best solution found ({LpStatus[status]})")

    log(f"\n  {'Metric':<40} {'Value':>15}")
    log("  " + "-"*55)
    log(f"  {'100-foot piles created':<40} {total_piles:>15,}")
    log(f"  {'Total waste':<40} {total_waste:>14.1f}'")
    log(f"  {'Average waste per pile':<40} {avg_waste:>14.2f}'")
    log(f"  {'Pipes used':<40} {used_pipes:>15,}")
    log(f"  {'Pipes unused':<40} {unused_pipes:>15,}")
    log(f"  {'Solve time':<40} {solve_time:>14.1f}s")

    section("COMPARISON WITH GREEDY")
    log(f"\n  Greedy baseline: 163 piles")
    log(f"  This solution:   {total_piles} piles")

    if total_piles > 163:
        improvement = total_piles - 163
        pct = improvement / 163 * 100
        log(f"  Improvement:     +{improvement} piles (+{pct:.1f}%)")
        log("\n  *** WE BEAT GREEDY! ***")
    elif total_piles == 163:
        log("  Result: Greedy was already optimal")
    else:
        log(f"  Difference: {total_piles - 163} piles")

    # Weld distribution
    section("WELD DISTRIBUTION")
    weld_counts = Counter(p['num_welds'] for p in solution)
    for welds in sorted(weld_counts.keys()):
        count = weld_counts[welds]
        pct = count / total_piles * 100
        log(f"  {welds} weld(s): {count:4d} piles ({pct:5.1f}%)")

    # Sample piles
    section("SAMPLE PILES (first 20)")
    for i, pile in enumerate(solution[:20], 1):
        segs = " + ".join([f"{l:.1f}'" for l in pile['pipe_lengths']])
        log(f"  {i:3d}: {segs} = {pile['total_length']:.1f}' (waste: {pile['waste']:.1f}')")

    # Export to Excel
    section("EXPORTING TO EXCEL")

    log("  Building dataframes...")

    pile_data = []
    for pile_num, pile in enumerate(solution, 1):
        for seg_num, length in enumerate(pile['pipe_lengths'], 1):
            pile_data.append({
                'Pile Number': pile_num,
                'Segment': seg_num,
                'Length (ft)': length,
                'Total Length': pile['total_length'] if seg_num == 1 else '',
                'Waste (ft)': pile['waste'] if seg_num == 1 else '',
                'Welds': pile['num_welds'] if seg_num == 1 else ''
            })

    df_piles = pd.DataFrame(pile_data)

    summary = {
        'Metric': ['Method', 'Status', 'Total Piles', 'Waste (ft)', 'Avg Waste',
                   'Pipes Used', 'Pipes Unused', 'vs Greedy', 'Solve Time (s)'],
        'Value': ['ILP V2 (Symmetry-Aware)',
                  'OPTIMAL' if status == LpStatusOptimal else LpStatus[status],
                  total_piles, f"{total_waste:.2f}", f"{avg_waste:.2f}",
                  used_pipes, unused_pipes,
                  f"+{total_piles - 163}" if total_piles > 163 else "Same" if total_piles == 163 else f"{total_piles - 163}",
                  f"{solve_time:.1f}"]
    }
    df_summary = pd.DataFrame(summary)

    output_file = 'pipe_optimization_V2_OPTIMAL.xlsx'
    log(f"  Writing to {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_piles.to_excel(writer, sheet_name='Pile Details', index=False)

    log(f"\n  Exported to: {output_file}")


def main():
    section("PIPE PILE OPTIMIZER V2")
    log("\nSymmetry-Aware Formulation with Unbuffered Output")
    log("Based on Gemini performance audit recommendations")

    section("LOADING DATA")

    try:
        pipe_lengths = np.loadtxt('pipe_lengths_clean.csv', delimiter=',')
    except FileNotFoundError:
        log("ERROR: pipe_lengths_clean.csv not found")
        sys.exit(1)

    log(f"\n  Loaded {len(pipe_lengths)} pipes")
    log(f"  Total material: {pipe_lengths.sum():.1f} feet")
    log(f"  Range: {pipe_lengths.min():.1f}' to {pipe_lengths.max():.1f}'")

    section("INITIALIZING SOLVER")

    solver = SymmetryAwareSolver(pipe_lengths, target_length=100.0, max_waste=20.0, precision=1)

    log(f"\n  Target pile length: {solver.target_length}'")
    log(f"  Max waste allowed: {solver.max_waste}'")
    log(f"  Precision: {solver.precision} decimal place(s)")
    log(f"\n  Raw pipes: {len(solver.raw_lengths)}")
    log(f"  Unique length types: {solver.n_types}")
    log(f"  Symmetry factor: {len(solver.raw_lengths)/solver.n_types:.1f}x")

    log("\n  Top 10 inventory:")
    for length, count in sorted(solver.inventory.items(), key=lambda x: -x[1])[:10]:
        log(f"    {length}': {count} pipes")

    # Generate patterns
    start_total = time.time()
    patterns = solver.generate_patterns()

    # Solve
    solution, status, solve_time = solver.solve_ilp(patterns)

    if solution is None or len(solution) == 0:
        log(f"\nERROR: No solution found! Status: {LpStatus[status]}")
        sys.exit(1)

    # Export
    export_results(solver, solution, status, solve_time)

    total_time = time.time() - start_total
    section("COMPLETE")
    log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    log("="*80)


if __name__ == "__main__":
    main()
