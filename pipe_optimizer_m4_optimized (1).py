#!/usr/bin/env python3
"""
M4 MacBook Optimized Pipe Pile Solver
Designed for 14-core M4 with unified memory architecture

FIXED VERSION - Corrected 3-pipe pattern generation bug
WITH PROGRESS REPORTING
"""

import numpy as np
import pandas as pd
from pulp import *
import multiprocessing as mp
from multiprocessing import Manager
import time
import sys

def progress_bar(current, total, prefix='', width=50):
    """Print a progress bar"""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:5.1f}% ({current:,}/{total:,})", end='', flush=True)

class M4OptimizedSolver:
    def __init__(self, pipe_lengths, target_length=100.0, max_waste=20.0):
        self.pipe_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.n_pipes = len(pipe_lengths)

    def generate_two_pipe_patterns(self):
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
                    if waste < self.max_waste:
                        patterns.append(([i, j], total_len, waste))

                count += 1
                # Update progress every 1%
                if count - last_report >= total // 100 or count == total:
                    progress_bar(count, total, '2-pipe')
                    last_report = count

        print()  # New line after progress bar
        print(f"  ✓ Found {len(patterns):,} valid 2-pipe patterns")
        return patterns

    def _generate_three_pipe_chunk(self, args):
        """Worker function for parallel 3-pipe generation - FIXED VERSION"""
        chunk_id, chunk_start, chunk_end, all_indices, sorted_indices, sorted_lengths, target, max_waste, progress_dict = args
        patterns = []
        total_in_chunk = chunk_end - chunk_start

        for idx_num, idx in enumerate(range(chunk_start, chunk_end)):
            i = all_indices[idx]  # i is POSITION in sorted list (0 to i_max)
            len_i = sorted_lengths[i]

            for j in range(i + 1, len(sorted_lengths) - 1):
                len_j = sorted_lengths[j]
                # Early stop: max possible for this j
                if len_i + len_j + sorted_lengths[j + 1] < target:
                    break

                for k in range(j + 1, len(sorted_lengths)):
                    len_k = sorted_lengths[k]
                    # Early stop: remaining k are smaller
                    if len_i + len_j + len_k < target:
                        break

                    total = len_i + len_j + len_k
                    waste = total - target
                    if waste < max_waste:
                        orig_i = sorted_indices[i]
                        orig_j = sorted_indices[j]
                        orig_k = sorted_indices[k]
                        patterns.append(([orig_i, orig_j, orig_k], total, waste))

            # Update progress
            if progress_dict is not None:
                progress_dict[chunk_id] = idx_num + 1

        return patterns

    def generate_three_pipe_patterns_parallel(self, n_cores=14):
        """Generate 3-pipe patterns using all cores - FIXED VERSION"""
        print(f"\n[2/2] Generating 3-pipe patterns using {n_cores} cores...")
        print(f"  Strategy: Sorted descending with early stopping and waste < {self.max_waste}' filter")

        # Sort by length descending
        pipes = sorted(enumerate(self.pipe_lengths), key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in pipes]
        sorted_lengths = [length for _, length in pipes]

        # Find max i where sum >= target is possible
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

        print(f"  Max starting position: {i_max} (out of {len(sorted_lengths) - 3})")
        all_indices = list(range(i_max + 1))  # 0 to i_max
        total_positions = len(all_indices)

        # Divide into chunks
        n_chunks = n_cores * 4
        chunk_size = max(1, len(all_indices) // n_chunks)

        # Use manager for shared progress tracking
        manager = Manager()
        progress_dict = manager.dict()

        chunks = []
        chunk_sizes = []
        for chunk_id, c in enumerate(range(0, len(all_indices), chunk_size)):
            end = min(c + chunk_size, len(all_indices))
            progress_dict[chunk_id] = 0
            chunk_sizes.append(end - c)
            chunks.append((chunk_id, c, end, all_indices, sorted_indices, sorted_lengths,
                           self.target_length, self.max_waste, progress_dict))

        print(f"  Processing {total_positions} positions in {len(chunks)} chunks...")
        print(f"  Starting parallel generation...")

        start_time = time.time()

        # Start async processing
        pool = mp.Pool(processes=n_cores)
        result_async = pool.map_async(self._generate_three_pipe_chunk, chunks)

        # Monitor progress while waiting
        total_work = sum(chunk_sizes)
        while not result_async.ready():
            completed = sum(progress_dict.values())
            progress_bar(completed, total_work, '3-pipe')
            time.sleep(0.5)

        # Final progress update
        progress_bar(total_work, total_work, '3-pipe')
        print()

        results = result_async.get()
        pool.close()
        pool.join()

        # Combine results
        patterns = []
        for result in results:
            patterns.extend(result)

        elapsed = time.time() - start_time
        print(f"  ✓ Found {len(patterns):,} valid 3-pipe patterns in {elapsed:.1f}s")
        return patterns

    def solve_ilp(self, patterns):
        """Solve using ILP with optimized constraint building"""
        n_patterns = len(patterns)

        print(f"\n{'='*80}")
        print("ILP SOLVING")
        print(f"{'='*80}")
        print(f"\nProblem size:")
        print(f"  Pipes: {self.n_pipes:,}")
        print(f"  Patterns: {n_patterns:,}")
        print(f"  Variables: {n_patterns:,} (binary)")
        print(f"  Constraints: {self.n_pipes:,}")

        # Build model
        print("\nBuilding ILP model...")
        prob = LpProblem("Optimal_Piles", LpMaximize)

        # Binary variable for each pattern
        print("  Creating variables...")
        x = [LpVariable(f"p{p}", cat='Binary') for p in range(n_patterns)]

        # Objective: maximize number of piles
        print("  Setting objective...")
        prob += lpSum(x), "Total_Piles"

        # OPTIMIZED: Build pipe-to-pattern index with progress
        print("  Building pipe-to-pattern index...")
        pipe_to_patterns = [[] for _ in range(self.n_pipes)]
        for p in range(n_patterns):
            for idx in patterns[p][0]:
                pipe_to_patterns[idx].append(p)
            if p % 100000 == 0:
                progress_bar(p, n_patterns, 'Index')
        progress_bar(n_patterns, n_patterns, 'Index')
        print()

        # Add constraints with progress
        print("  Adding constraints...")
        constraints_added = 0
        for i in range(self.n_pipes):
            if pipe_to_patterns[i]:
                prob += lpSum(x[p] for p in pipe_to_patterns[i]) <= 1, f"Pipe_{i}"
                constraints_added += 1
            if i % 100 == 0:
                progress_bar(i, self.n_pipes, 'Constraints')
        progress_bar(self.n_pipes, self.n_pipes, 'Constraints')
        print()
        print(f"  ✓ Added {constraints_added} constraints")

        # Solve with CBC
        print("\n" + "="*80)
        print("SOLVING (CBC with 14 threads, 30 min limit)")
        print("="*80)
        print("\nCBC solver output below:")
        print("-"*80)

        start_time = time.time()
        solver = PULP_CBC_CMD(msg=1, timeLimit=1800, threads=14)
        prob.solve(solver)
        solve_time = time.time() - start_time

        print("-"*80)
        print(f"\n✓ Solved in {solve_time:.1f}s")
        print(f"  Status: {LpStatus[prob.status]}")

        if prob.status in [LpStatusOptimal, LpStatusNotSolved]:
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
            print(f"  ✓ Extracted {len(solution)} piles from solution")
            return solution, prob.status, solve_time

        return None, prob.status, solve_time


def export_results(pipe_lengths, solution, status, solve_time, output_file):
    """Export results to Excel"""
    n_pipes = len(pipe_lengths)
    total_piles = len(solution)
    total_waste = sum(p['waste'] for p in solution)
    total_used = sum(p['total_length'] for p in solution)
    avg_waste = total_waste / total_piles if total_piles > 0 else 0

    used_pipes = set()
    for pile in solution:
        used_pipes.update(pile['pipe_indices'])

    unused = set(range(n_pipes)) - used_pipes
    unused_length = sum(pipe_lengths[i] for i in unused)
    utilization = (total_piles * 100.0) / total_used * 100 if total_used > 0 else 0

    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    if status == LpStatusOptimal:
        print("\n✓✓✓ OPTIMAL SOLUTION FOUND (GUARANTEED) ✓✓✓")
    else:
        print(f"\n✓ BEST SOLUTION FOUND ({LpStatus[status]})")

    print(f"\n{'Metric':<45} {'Value':>20}")
    print("-" * 80)
    print(f"{'100-foot piles created':<45} {total_piles:>20,}")
    print(f"{'Total waste':<45} {total_waste:>19.1f}'")
    print(f"{'Average waste per pile':<45} {avg_waste:>19.2f}'")
    print(f"{'Material utilization':<45} {utilization:>19.1f}%")
    print(f"{'Pipes used':<45} {len(used_pipes):>20,}")
    print(f"{'Pipes unused':<45} {len(unused):>20,}")
    print(f"{'Solve time':<45} {solve_time:>19.1f}s")

    print(f"\n{'='*80}")
    print("COMPARISON WITH GREEDY SOLUTION")
    print(f"{'='*80}")
    print(f"Greedy solution:  163 piles")
    print(f"This solution:    {total_piles} piles")

    if total_piles > 163:
        improvement = total_piles - 163
        pct = improvement / 163 * 100
        print(f"Improvement:      +{improvement} piles (+{pct:.1f}%)")
        print("\n✓✓✓ WE BEAT THE GREEDY SOLUTION! ✓✓✓")
        print("The mathematical reviewers were correct - greedy was suboptimal!")
    elif total_piles == 163:
        print("Improvement:      None")
        print("\n✓ Greedy solution was already optimal!")
    else:
        print(f"Difference:       {total_piles - 163} piles")
        print("\nNote: Pattern filtering may have been too aggressive")

    # Weld distribution
    weld_counts = {}
    for pile in solution:
        w = pile['num_welds']
        weld_counts[w] = weld_counts.get(w, 0) + 1

    print(f"\n{'='*80}")
    print("WELD DISTRIBUTION")
    print(f"{'='*80}")
    for welds in sorted(weld_counts.keys()):
        count = weld_counts[welds]
        pct = count / total_piles * 100
        print(f"  {welds} weld(s): {count:4d} piles ({pct:5.1f}%)")

    # Export to Excel
    print(f"\n{'='*80}")
    print("EXPORTING TO EXCEL")
    print(f"{'='*80}")

    print("  Building pile data...")
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

    print("  Building summary...")
    summary = {
        'Metric': ['Method', 'Status', 'Total Piles', 'Waste (ft)', 'Avg Waste',
                   'Utilization %', 'Pipes Used', 'Pipes Unused', 'vs Greedy', 'Solve Time (s)'],
        'Value': ['ILP (M4-Optimized)',
                  'OPTIMAL' if status == LpStatusOptimal else LpStatus[status],
                  total_piles, f"{total_waste:.2f}", f"{avg_waste:.2f}",
                  f"{utilization:.2f}", len(used_pipes), len(unused),
                  f"+{total_piles - 163}" if total_piles > 163 else "Same" if total_piles == 163 else f"{total_piles - 163}",
                  f"{solve_time:.1f}"]
    }
    df_summary = pd.DataFrame(summary)

    print("  Building unused pipes list...")
    unused_data = [{'Pipe Index': idx + 1, 'Length (ft)': pipe_lengths[idx]}
                   for idx in sorted(unused)]
    df_unused = pd.DataFrame(unused_data)

    print("  Writing Excel file...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_piles.to_excel(writer, sheet_name='Pile Details', index=False)
        df_unused.to_excel(writer, sheet_name='Unused Pipes', index=False)

    print(f"\n✓ Results exported to: {output_file}")

    # Sample piles
    print(f"\n{'='*80}")
    print("SAMPLE PILES (first 20)")
    print(f"{'='*80}")
    for i, pile in enumerate(solution[:20], 1):
        segs = " + ".join([f"{l:.1f}'" for l in pile['pipe_lengths']])
        print(f"  {i:3d}: {segs} = {pile['total_length']:.1f}' "
              f"(waste: {pile['waste']:.1f}', welds: {pile['num_welds']})")


def main():
    print("="*80)
    print("M4 MACBOOK OPTIMIZED PIPE PILE SOLVER")
    print("="*80)
    print("\nDesigned for 14-core M4 with unified memory")
    print("Will use all 14 cores for parallel pattern generation")
    print("\n*** FIXED VERSION - Corrected 3-pipe pattern generation bug ***")
    print("*** WITH PROGRESS REPORTING ***")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    try:
        pipe_lengths = np.loadtxt('pipe_lengths_clean.csv', delimiter=',')
    except:
        print("\nERROR: Could not find pipe_lengths_clean.csv")
        print("Please ensure the file is in the current directory")
        sys.exit(1)

    print(f"\n  ✓ Loaded {len(pipe_lengths)} pipes")
    print(f"  ✓ Total material: {pipe_lengths.sum():.1f} feet")
    print(f"  ✓ Pipe range: {pipe_lengths.min():.1f}' to {pipe_lengths.max():.1f}'")

    # Show distribution
    print(f"\n  Length distribution:")
    ranges = [(17, 25), (25, 35), (35, 45), (45, 52)]
    for low, high in ranges:
        count = sum(1 for p in pipe_lengths if low <= p < high)
        print(f"    {low}'-{high}': {count} pipes")

    # Initialize solver
    print("\n" + "="*80)
    print("INITIALIZING SOLVER")
    print("="*80)
    solver = M4OptimizedSolver(pipe_lengths, target_length=100.0, max_waste=20.0)
    print(f"\n  Target pile length: {solver.target_length}'")
    print(f"  Max waste allowed: {solver.max_waste}'")
    print(f"  Max welds per pile: 2 (i.e., 2 or 3 pipe segments)")

    # Generate patterns
    print(f"\n{'='*80}")
    print("PATTERN GENERATION")
    print(f"{'='*80}")

    start_total = time.time()

    patterns = []
    patterns.extend(solver.generate_two_pipe_patterns())
    patterns.extend(solver.generate_three_pipe_patterns_parallel(n_cores=14))

    pattern_time = time.time() - start_total
    print(f"\n✓ Total patterns generated: {len(patterns):,} in {pattern_time:.1f}s")

    # Pattern stats
    two_pipe = sum(1 for p in patterns if len(p[0]) == 2)
    three_pipe = sum(1 for p in patterns if len(p[0]) == 3)
    print(f"  2-pipe patterns: {two_pipe:,}")
    print(f"  3-pipe patterns: {three_pipe:,}")

    if len(patterns) == 0:
        print("\nERROR: No valid patterns found!")
        sys.exit(1)

    # Solve ILP
    solution, status, solve_time = solver.solve_ilp(patterns)

    if solution is None or len(solution) == 0:
        print(f"\nERROR: No solution found! Status: {LpStatus[status]}")
        sys.exit(1)

    # Export results
    output_file = 'pipe_optimization_M4_OPTIMAL.xlsx'
    export_results(pipe_lengths, solution, status, solve_time, output_file)

    total_time = time.time() - start_total
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE - Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
