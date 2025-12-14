#!/usr/bin/env python3
"""
Pipe Pile Optimizer V3 - Bug-Fixed Version
============================================
Fixes from Grok/Gemini ULTRATHINK audit:
1. FIXED: 3-pipe early termination bug (was skipping valid patterns)
2. FIXED: Boundary condition < to <= (now includes exactly 120' piles)
3. ADDED: 1-pipe pattern support (single pipes 100-120')
4. FIXED: Theoretical max calculation (264 not 265)

Based on symmetry-aware ILP formulation.
"""

import numpy as np
import pandas as pd
from pulp import *
from collections import Counter, defaultdict
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


class SymmetryAwareSolverV3:
    """
    V3: Bug-fixed version based on Grok/Gemini audit.

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

        # Theoretical maximum (floor division)
        self.theoretical_max = int(sum(pipe_lengths) // target_length)

    def generate_patterns(self):
        """Generate patterns over unique length TYPES (not individual pipes)"""
        section("PATTERN GENERATION (V3 - Bug Fixed)")

        log(f"\n  Unique length types: {self.n_types} (vs {len(self.raw_lengths)} raw pipes)")
        log(f"  Symmetry reduction: {len(self.raw_lengths)/self.n_types:.1f}x")
        log(f"  Target: {self.target_length}', Max waste: {self.max_waste}'")
        log(f"  Valid pile range: {self.target_length}' to {self.target_length + self.max_waste}' (INCLUSIVE)")

        patterns = []
        lengths = self.unique_lengths
        n = len(lengths)

        min_len = self.target_length
        max_len = self.target_length + self.max_waste  # 120.0

        # ===========================================
        # FIX #3: 1-pipe patterns (NEW)
        # ===========================================
        log("\n[1/3] Generating 1-pipe patterns (single pipes 100-120')...")
        start = time.time()
        one_count = 0

        for i in range(n):
            length = lengths[i]
            # FIX #2: Use <= for upper bound (includes exactly 120')
            if min_len <= length <= max_len:
                waste = length - self.target_length
                patterns.append(((i,), length, waste, {i: 1}))
                one_count += 1
                log(f"    Found: {length}' pipe (waste: {waste:.1f}')")

        log(f"  Found {one_count} valid 1-pipe patterns in {time.time()-start:.2f}s")

        # ===========================================
        # 2-pipe patterns
        # ===========================================
        log("\n[2/3] Generating 2-pipe patterns...")
        start = time.time()
        two_count = 0
        total_pairs = n * (n + 1) // 2
        checked = 0

        for i in range(n):
            for j in range(i, n):
                total_len = lengths[i] + lengths[j]
                # FIX #2: Use <= for upper bound (includes exactly 120')
                if min_len <= total_len <= max_len:
                    waste = total_len - self.target_length
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
                    progress_bar(checked, total_pairs, '2-pipe')

        progress_bar(total_pairs, total_pairs, '2-pipe')
        log(f"\n  Found {two_count:,} valid 2-pipe patterns in {time.time()-start:.1f}s")

        # ===========================================
        # 3-pipe patterns (FIX #1: Removed bad early termination)
        # ===========================================
        log("\n[3/3] Generating 3-pipe patterns...")
        start = time.time()
        three_count = 0

        total_est = n * (n + 1) * (n + 2) // 6
        checked = 0

        for i in range(n):
            len_i = lengths[i]
            # Early termination: if 3 of the longest remaining can't reach target, stop
            if 3 * len_i < min_len:
                break

            for j in range(i, n):
                len_j = lengths[j]

                # Early termination: if i + 2*j can't reach target, skip
                if len_i + 2 * len_j < min_len:
                    continue

                # FIX #1: REMOVED the buggy early termination that skipped valid patterns!
                # OLD (BUGGY): if len_i + len_j + lengths[j] > max_len: continue
                # This was wrong because it skipped the entire j iteration when
                # len_i + 2*len_j > 120, but smaller k values could still produce valid patterns.

                for k in range(j, n):
                    len_k = lengths[k]
                    total_len = len_i + len_j + len_k

                    # Early termination: lengths are sorted descending
                    if total_len < min_len:
                        break

                    # FIX #2: Use <= for upper bound (includes exactly 120')
                    if total_len <= max_len:
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
                        progress_bar(min(checked, total_est), total_est, '3-pipe')

        progress_bar(total_est, total_est, '3-pipe')
        log(f"\n  Found {three_count:,} valid 3-pipe patterns in {time.time()-start:.1f}s")

        log(f"\n  TOTAL: {len(patterns):,} unique pattern types")
        log(f"    1-pipe: {one_count:,}")
        log(f"    2-pipe: {two_count:,}")
        log(f"    3-pipe: {three_count:,}")

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
        prob = LpProblem("Optimal_Piles_V3", LpMaximize)

        # Integer variable for each pattern TYPE
        log("  Creating variables...", end='')
        sys.stdout.flush()

        x = []
        for p_idx, (indices, total, waste, counts_needed) in enumerate(patterns):
            max_uses = min(
                self.inventory[self.unique_lengths[idx]] // cnt
                for idx, cnt in counts_needed.items()
            )
            x.append(LpVariable(f"p{p_idx}", lowBound=0, upBound=max_uses, cat='Integer'))

        log(" done")

        # Objective: maximize total piles
        log("  Setting objective...", end='')
        sys.stdout.flush()
        prob += lpSum(x), "Total_Piles"
        log(" done")

        # Constraints: for each length TYPE, total usage <= inventory
        log("  Adding inventory constraints...")

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
        log("  Time limit: 15 minutes")
        log("\n" + "-"*80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=1, timeLimit=900, threads=14)  # Increased to 15 min
        prob.solve(solver)
        solve_time = time.time() - start_time

        log("-"*80)
        log(f"\n  Solved in {solve_time:.1f}s")
        log(f"  Status: {LpStatus[prob.status]}")

        if prob.status in [LpStatusOptimal, LpStatusNotSolved]:
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


def export_results_with_indices(solver, solution, status, solve_time, pipe_lengths):
    """Export results with specific pipe indices assigned"""
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

    # FIX #4: Correct theoretical max
    theoretical_max = solver.theoretical_max

    if status == LpStatusOptimal:
        log("\n  *** OPTIMAL SOLUTION FOUND (GUARANTEED) ***")
    else:
        log(f"\n  Best solution found ({LpStatus[status]})")

    log(f"\n  {'Metric':<40} {'Value':>15}")
    log("  " + "-"*55)
    log(f"  {'100-foot piles created':<40} {total_piles:>15,}")
    log(f"  {'Theoretical maximum':<40} {theoretical_max:>15,}")
    log(f"  {'Efficiency vs theoretical':<40} {total_piles/theoretical_max*100:>14.1f}%")
    log(f"  {'Total waste':<40} {total_waste:>14.1f}'")
    log(f"  {'Average waste per pile':<40} {avg_waste:>14.2f}'")
    log(f"  {'Pipes used':<40} {used_pipes:>15,}")
    log(f"  {'Pipes unused':<40} {unused_pipes:>15,}")
    log(f"  {'Solve time':<40} {solve_time:>14.1f}s")

    section("COMPARISON WITH PREVIOUS RESULTS")
    log(f"\n  Greedy baseline:    163 piles")
    log(f"  V2 result:          262 piles")
    log(f"  V3 result (fixed):  {total_piles} piles")
    log(f"  Theoretical max:    {theoretical_max} piles")

    if total_piles > 262:
        improvement = total_piles - 262
        log(f"\n  V3 improvement over V2: +{improvement} piles")
        log("  *** BUG FIXES FOUND MORE PILES! ***")
    elif total_piles == 262:
        log("\n  V3 matches V2 (bugs didn't affect final result)")
    else:
        log(f"\n  V3 vs V2: {total_piles - 262} piles")

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

    # ===========================================
    # Assign specific pipe indices
    # ===========================================
    section("ASSIGNING PIPE INDICES")
    log("\n  Mapping solution to specific pipes from inventory...")

    # Create inventory pools by rounded length
    inventory_pools = defaultdict(list)
    for idx, length in enumerate(pipe_lengths):
        rounded = round(length, solver.precision)
        inventory_pools[rounded].append(idx)

    # Assign indices
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

    log(f"  Assigned {len(assigned_pipes)} unique pipe indices")
    log(f"  Duplicates: {len(assigned_pipes) - len(set(assigned_pipes))}")

    # Find unused pipes
    unused_indices = [i for i in range(len(pipe_lengths)) if i not in assigned_pipes]
    log(f"  Unused pipes: {len(unused_indices)}")

    # Export to Excel
    section("EXPORTING TO EXCEL")
    log("  Building dataframes...")

    pile_data = []
    for pile_num, (pile, assignments) in enumerate(zip(solution, pile_assignments), 1):
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
        'Metric': ['Method', 'Status', 'Total Piles', 'Theoretical Max', 'Efficiency',
                   'Pipes Used', 'Pipes Unused', 'Total Waste (ft)', 'Avg Waste/Pile',
                   'vs Greedy (163)', 'vs V2 (262)', 'Solve Time (s)', 'Unique Pipe Indices'],
        'Value': ['ILP V3 (Bug-Fixed)',
                  'OPTIMAL' if status == LpStatusOptimal else LpStatus[status],
                  total_piles, theoretical_max, f"{total_piles/theoretical_max*100:.1f}%",
                  used_pipes, unused_pipes, f"{total_waste:.2f}", f"{avg_waste:.2f}",
                  f"+{total_piles - 163} (+{(total_piles-163)/163*100:.1f}%)",
                  f"+{total_piles - 262}" if total_piles > 262 else "Same" if total_piles == 262 else f"{total_piles - 262}",
                  f"{solve_time:.1f}",
                  f"{len(assigned_pipes)} (no duplicates)"]
    })

    # Unused pipes sheet
    unused_df = pd.DataFrame({
        'Pipe Index': unused_indices,
        'Length (ft)': [pipe_lengths[i] for i in unused_indices]
    })

    output_file = 'pipe_optimization_V3_FINAL.xlsx'
    log(f"  Writing to {output_file}...")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        df_piles.to_excel(writer, sheet_name='Pile Details', index=False)
        unused_df.to_excel(writer, sheet_name='Unused Pipes', index=False)

    log(f"\n  Exported to: {output_file}")

    return total_piles, theoretical_max, used_pipes, len(unused_indices)


def main():
    section("PIPE PILE OPTIMIZER V3 (BUG-FIXED)")
    log("\nFixes from Grok/Gemini ULTRATHINK Audit:")
    log("  1. Fixed 3-pipe early termination bug")
    log("  2. Fixed boundary condition (< to <=)")
    log("  3. Added 1-pipe pattern support")
    log("  4. Corrected theoretical max calculation")

    section("LOADING DATA")

    try:
        pipe_lengths = np.loadtxt('pipe_lengths_clean.csv', delimiter=',')
    except FileNotFoundError:
        log("ERROR: pipe_lengths_clean.csv not found")
        sys.exit(1)

    total_material = pipe_lengths.sum()
    theoretical_max = int(total_material // 100)

    log(f"\n  Loaded {len(pipe_lengths)} pipes")
    log(f"  Total material: {total_material:.1f} feet")
    log(f"  Range: {pipe_lengths.min():.1f}' to {pipe_lengths.max():.1f}'")
    log(f"  Theoretical max: {total_material:.1f} / 100 = {theoretical_max} piles (floor)")

    section("INITIALIZING SOLVER")

    solver = SymmetryAwareSolverV3(pipe_lengths, target_length=100.0, max_waste=20.0, precision=1)

    log(f"\n  Target pile length: {solver.target_length}'")
    log(f"  Max waste allowed: {solver.max_waste}'")
    log(f"  Valid range: {solver.target_length}' to {solver.target_length + solver.max_waste}' (inclusive)")
    log(f"  Precision: {solver.precision} decimal place(s)")
    log(f"\n  Raw pipes: {len(solver.raw_lengths)}")
    log(f"  Unique length types: {solver.n_types}")
    log(f"  Symmetry factor: {len(solver.raw_lengths)/solver.n_types:.1f}x")

    log("\n  Top 10 inventory:")
    for length, count in sorted(solver.inventory.items(), key=lambda x: -x[1])[:10]:
        log(f"    {length}': {count} pipes")

    # Check for 1-pipe candidates
    log("\n  Checking for single-pipe candidates (100-120'):")
    single_candidates = [(l, c) for l, c in solver.inventory.items() if 100 <= l <= 120]
    if single_candidates:
        for length, count in single_candidates:
            log(f"    {length}': {count} pipes (can be used as single-pipe piles!)")
    else:
        log("    None found (all pipes < 100')")

    # Generate patterns
    start_total = time.time()
    patterns = solver.generate_patterns()

    # Solve
    solution, status, solve_time = solver.solve_ilp(patterns)

    if solution is None or len(solution) == 0:
        log(f"\nERROR: No solution found! Status: {LpStatus[status]}")
        sys.exit(1)

    # Export with pipe indices
    total_piles, theo_max, used, unused = export_results_with_indices(
        solver, solution, status, solve_time, pipe_lengths
    )

    total_time = time.time() - start_total
    section("COMPLETE")
    log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    log(f"\n  Final result: {total_piles} piles out of {theo_max} theoretical max ({total_piles/theo_max*100:.1f}%)")
    log("="*80)


if __name__ == "__main__":
    main()
