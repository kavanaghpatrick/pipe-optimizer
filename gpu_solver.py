#!/usr/bin/env python3
"""
GPU-accelerated pipe optimizer using Metal compute via forge-ffi.

Provides:
- Metal kernel-based pattern generation (2-pipe and 3-pipe combinations)
- Greedy solver: sort patterns by waste, single-pass inventory assignment
- LP-guided solver: LP relaxation + floor + greedy mop-up (stub)
- GPUPipeOptimizer class with same interface as SymmetryAwareSafeSolver
"""

import math
import time
from collections import Counter

import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

# forge_bindings.py will be created by another teammate
try:
    from forge_bindings import ForgeContext, ForgeBuffer, ForgeKernel, ForgePipeline, ForgeError
    HAS_FORGE = True
except ImportError:
    HAS_FORGE = False
    # Define stub so module can be imported without forge_bindings
    class ForgeError(Exception):
        pass

# =============================================================================
# METAL KERNEL SOURCE STRINGS
# =============================================================================

GEN_PATTERNS_3_MSL = """\
#include <metal_stdlib>
using namespace metal;

kernel void gen_patterns_3(
    device const float* lengths [[buffer(0)]],
    device const float* params  [[buffer(1)]],
    device uint*  out_i         [[buffer(2)]],
    device uint*  out_j         [[buffer(3)]],
    device uint*  out_k         [[buffer(4)]],
    device float* out_waste     [[buffer(5)]],
    device atomic_uint* counter [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = uint(params[0]);
    float target = params[1];
    float max_waste = params[2];

    // Decode flat tid to (i, j, k)
    uint i = tid / (n * n);
    uint j = (tid / n) % n;
    uint k = tid % n;

    // Upper triangle only: i <= j <= k
    if (i > j || j > k) return;
    // Bounds check
    if (i >= n || j >= n || k >= n) return;

    float sum = lengths[i] + lengths[j] + lengths[k];
    float waste = sum - target;

    // Range check: waste must be in [0, max_waste]
    if (waste < 0.0f || waste > max_waste) return;

    // Inventory feasibility check (4 duplicate cases)
    float inv_i = params[3 + i];
    float inv_j = params[3 + j];
    float inv_k = params[3 + k];

    if (i == j && j == k) {
        // All same type: need 3
        if (inv_i < 3.0f) return;
    } else if (i == j) {
        // First two same: need 2 of i, 1 of k
        if (inv_i < 2.0f || inv_k < 1.0f) return;
    } else if (j == k) {
        // Last two same: need 1 of i, 2 of j
        if (inv_i < 1.0f || inv_j < 2.0f) return;
    } else {
        // All different: need 1 each
        if (inv_i < 1.0f || inv_j < 1.0f || inv_k < 1.0f) return;
    }

    // Atomically claim output slot
    uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    out_i[idx] = i;
    out_j[idx] = j;
    out_k[idx] = k;
    out_waste[idx] = waste;
}
"""

GEN_PATTERNS_2_MSL = """\
#include <metal_stdlib>
using namespace metal;

kernel void gen_patterns_2(
    device const float* lengths [[buffer(0)]],
    device const float* params  [[buffer(1)]],
    device uint*  out_i         [[buffer(2)]],
    device uint*  out_j         [[buffer(3)]],
    device float* out_waste     [[buffer(4)]],
    device atomic_uint* counter [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = uint(params[0]);
    float target = params[1];
    float max_waste = params[2];

    // Decode flat tid to (i, j)
    uint i = tid / n;
    uint j = tid % n;

    // Upper triangle only: i <= j
    if (i > j) return;
    // Bounds check
    if (i >= n || j >= n) return;

    float sum = lengths[i] + lengths[j];
    float waste = sum - target;

    // Range check: waste must be in [0, max_waste]
    if (waste < 0.0f || waste > max_waste) return;

    // Inventory feasibility check (2 duplicate cases)
    float inv_i = params[3 + i];
    float inv_j = params[3 + j];

    if (i == j) {
        // Same type: need 2
        if (inv_i < 2.0f) return;
    } else {
        // Different types: need 1 each
        if (inv_i < 1.0f || inv_j < 1.0f) return;
    }

    // Atomically claim output slot
    uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    out_i[idx] = i;
    out_j[idx] = j;
    out_waste[idx] = waste;
}
"""

# =============================================================================
# GPU PATTERN GENERATOR
# =============================================================================

MAX_PATTERNS = 4_000_000  # Pre-allocated output buffer size


class GPUPatternGenerator:
    """Generate pipe combination patterns using Metal GPU kernels."""

    def __init__(self, ctx, unique_lengths, inventory, target_length, max_waste):
        """
        Args:
            ctx: ForgeContext instance
            unique_lengths: sorted list of unique pipe lengths (descending)
            inventory: dict mapping length -> available count
            target_length: target pile length (e.g. 100.0)
            max_waste: maximum allowed waste per pile
        """
        self.ctx = ctx
        self.unique_lengths = unique_lengths
        self.inventory = inventory
        self.target_length = target_length
        self.max_waste = max_waste
        self.n = len(unique_lengths)

        # Compile both kernels
        self.kernel_3 = ForgeKernel(ctx, GEN_PATTERNS_3_MSL, "gen_patterns_3")
        self.kernel_2 = ForgeKernel(ctx, GEN_PATTERNS_2_MSL, "gen_patterns_2")

    def _build_params(self):
        """Build params buffer: [n, target, max_waste, inv_0, ..., inv_n-1]."""
        params = np.zeros(3 + self.n, dtype=np.float32)
        params[0] = float(self.n)
        params[1] = float(self.target_length)
        params[2] = float(self.max_waste)
        for idx, length in enumerate(self.unique_lengths):
            params[3 + idx] = float(self.inventory[length])
        return params

    def _build_lengths_array(self):
        """Build float32 array of unique lengths."""
        return np.array(self.unique_lengths, dtype=np.float32)

    def _generate_1pipe(self):
        """Generate 1-pipe patterns using numpy (no GPU needed)."""
        lengths_arr = np.array(self.unique_lengths)
        min_len = self.target_length
        max_len = self.target_length + self.max_waste
        mask = (lengths_arr >= min_len) & (lengths_arr <= max_len)
        patterns = []
        for i in np.where(mask)[0]:
            waste = self.unique_lengths[i] - self.target_length
            patterns.append(((i,), self.unique_lengths[i], waste, {i: 1}))
        return patterns

    def _generate_2pipe(self):
        """Generate 2-pipe patterns via GPU kernel."""
        lengths_buf = ForgeBuffer.from_numpy(self.ctx, self._build_lengths_array())
        params_buf = ForgeBuffer.from_numpy(self.ctx, self._build_params())

        # Allocate output buffers
        out_i = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'u32')
        out_j = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'u32')
        out_waste = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'f32')
        counter = ForgeBuffer.alloc(self.ctx, 1, 'u32')

        thread_count = self.n * self.n

        pip = ForgePipeline(self.ctx)
        pip.dispatch_1d(
            self.kernel_2,
            [lengths_buf, params_buf, out_i, out_j, out_waste, counter],
            thread_count,
        )
        gpu_ms = pip.execute()

        # Read results
        count = int(counter.to_numpy()[0])
        if count == 0:
            return [], gpu_ms

        i_arr = out_i.to_numpy()[:count]
        j_arr = out_j.to_numpy()[:count]
        w_arr = out_waste.to_numpy()[:count]

        # Convert to v5 pattern format
        patterns = []
        for p in range(count):
            i_idx = int(i_arr[p])
            j_idx = int(j_arr[p])
            waste = float(w_arr[p])
            total = self.target_length + waste
            if i_idx == j_idx:
                counts = {i_idx: 2}
            else:
                counts = {i_idx: 1, j_idx: 1}
            patterns.append(((i_idx, j_idx), total, waste, counts))

        return patterns, gpu_ms

    def _generate_3pipe(self):
        """Generate 3-pipe patterns via GPU kernel."""
        lengths_buf = ForgeBuffer.from_numpy(self.ctx, self._build_lengths_array())
        params_buf = ForgeBuffer.from_numpy(self.ctx, self._build_params())

        # Allocate output buffers
        out_i = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'u32')
        out_j = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'u32')
        out_k = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'u32')
        out_waste = ForgeBuffer.alloc(self.ctx, MAX_PATTERNS, 'f32')
        counter = ForgeBuffer.alloc(self.ctx, 1, 'u32')

        thread_count = self.n * self.n * self.n

        pip = ForgePipeline(self.ctx)
        pip.dispatch_1d(
            self.kernel_3,
            [lengths_buf, params_buf, out_i, out_j, out_k, out_waste, counter],
            thread_count,
        )
        gpu_ms = pip.execute()

        # Read results
        count = int(counter.to_numpy()[0])
        if count == 0:
            return [], gpu_ms

        i_arr = out_i.to_numpy()[:count]
        j_arr = out_j.to_numpy()[:count]
        k_arr = out_k.to_numpy()[:count]
        w_arr = out_waste.to_numpy()[:count]

        # Convert to v5 pattern format
        patterns = []
        for p in range(count):
            i_idx = int(i_arr[p])
            j_idx = int(j_arr[p])
            k_idx = int(k_arr[p])
            waste = float(w_arr[p])
            total = self.target_length + waste

            # Build counts dict (4 cases)
            if i_idx == j_idx and j_idx == k_idx:
                counts = {i_idx: 3}
            elif i_idx == j_idx:
                counts = {i_idx: 2, k_idx: 1}
            elif j_idx == k_idx:
                counts = {i_idx: 1, j_idx: 2}
            else:
                counts = {i_idx: 1, j_idx: 1, k_idx: 1}

            patterns.append(((i_idx, j_idx, k_idx), total, waste, counts))

        return patterns, gpu_ms

    def generate(self, max_welds=3):
        """Generate all valid patterns up to max_welds pipe segments.

        Args:
            max_welds: Maximum number of pipe segments per pile (1, 2, or 3)

        Returns:
            List of pattern tuples: (indices_tuple, total_length, waste, counts_dict)
        """
        all_patterns = []
        total_gpu_ms = 0.0

        # 1-pipe patterns (CPU, trivial)
        patterns_1 = self._generate_1pipe()
        all_patterns.extend(patterns_1)

        # 2-pipe patterns (GPU)
        if max_welds >= 2:
            patterns_2, gpu_ms = self._generate_2pipe()
            all_patterns.extend(patterns_2)
            total_gpu_ms += gpu_ms

        # 3-pipe patterns (GPU)
        if max_welds >= 3:
            patterns_3, gpu_ms = self._generate_3pipe()
            all_patterns.extend(patterns_3)
            total_gpu_ms += gpu_ms

        return all_patterns


# =============================================================================
# GREEDY SOLVER
# =============================================================================

def greedy_solve(patterns, unique_lengths, inventory, stop_event=None):
    """Greedy solver: sort patterns by waste ascending, single-pass assignment.

    Args:
        patterns: list of (indices_tuple, total_length, waste, counts_dict)
        unique_lengths: sorted list of unique pipe lengths
        inventory: dict mapping length -> available count
        stop_event: optional threading.Event to check for cancellation

    Returns:
        (solution_list, 'GREEDY', elapsed_time)
        where solution_list contains v5-compatible dicts.
    """
    start = time.time()

    if not patterns:
        return [], 'GREEDY', 0.0

    # Sort by waste ascending (best patterns first)
    waste_arr = np.array([p[2] for p in patterns], dtype=np.float32)
    sorted_indices = np.argsort(waste_arr)

    # Working copy of inventory
    remaining = dict(inventory)

    solution = []
    for sort_idx in sorted_indices:
        if stop_event and stop_event.is_set():
            break

        indices, total, waste, counts = patterns[sort_idx]

        # How many times can we use this pattern?
        max_uses = min(
            remaining.get(unique_lengths[idx], 0) // cnt
            for idx, cnt in counts.items()
        )

        if max_uses <= 0:
            continue

        # Assign max_uses copies
        for idx, cnt in counts.items():
            remaining[unique_lengths[idx]] -= cnt * max_uses

        # Add to solution (one dict per pile, matching v5 format)
        for _ in range(max_uses):
            pipe_lengths = [unique_lengths[i] for i in indices]
            solution.append({
                'pattern_type': sort_idx,
                'length_indices': indices,
                'pipe_lengths': pipe_lengths,
                'total_length': total,
                'waste': waste,
                'num_welds': len(indices) - 1,
            })

    elapsed = time.time() - start
    return solution, 'GREEDY', elapsed


# =============================================================================
# LP-GUIDED SOLVER
# =============================================================================

def lp_guided_solve(patterns, unique_lengths, inventory, stop_event=None):
    """LP relaxation + floor + greedy mop-up solver.

    Solves an LP relaxation (continuous variables instead of integer),
    floors the fractional solutions, then uses greedy to assign remaining
    inventory. Near-optimal results in seconds instead of minutes.

    Args:
        patterns: list of (indices_tuple, total_length, waste, counts_dict)
        unique_lengths: sorted list of unique pipe lengths
        inventory: dict mapping length -> available count
        stop_event: optional threading.Event for cancellation

    Returns:
        (solution_list, 'LP_GUIDED', elapsed_time)
    """
    start = time.time()

    if not patterns:
        return [], 'LP_GUIDED', 0.0

    n_patterns = len(patterns)
    n_types = len(unique_lengths)

    # Build LP model (identical to ILP but with Continuous variables)
    prob = LpProblem("LP_Guided_Piles", LpMaximize)

    x = []
    for p_idx, (indices, total, waste, counts) in enumerate(patterns):
        max_uses = min(
            inventory.get(unique_lengths[idx], 0) // cnt
            for idx, cnt in counts.items()
        )
        x.append(LpVariable(f"p{p_idx}", lowBound=0, upBound=max_uses, cat='Continuous'))

    # Objective: maximize total piles
    prob += lpSum(x), "Total_Piles"

    # Constraints: for each length type, total usage <= inventory
    type_to_patterns = [[] for _ in range(n_types)]
    for p_idx, (indices, total, waste, counts) in enumerate(patterns):
        for type_idx, count in counts.items():
            type_to_patterns[type_idx].append((p_idx, count))

    for type_idx in range(n_types):
        length = unique_lengths[type_idx]
        available = inventory[length]
        if type_to_patterns[type_idx]:
            prob += (
                lpSum(count * x[p_idx] for p_idx, count in type_to_patterns[type_idx]) <= available,
                f"Type_{type_idx}"
            )

    # Solve LP relaxation (very fast)
    solver = PULP_CBC_CMD(msg=0, threads=4)
    prob.solve(solver)

    # Floor all variable values
    remaining = dict(inventory)
    solution = []

    for p_idx, (indices, total, waste, counts) in enumerate(patterns):
        if stop_event and stop_event.is_set():
            break

        val = x[p_idx].varValue or 0.0
        floor_uses = int(math.floor(val))
        if floor_uses <= 0:
            continue

        # Check feasibility against remaining inventory
        actual_uses = min(
            floor_uses,
            *(remaining.get(unique_lengths[idx], 0) // cnt for idx, cnt in counts.items())
        )
        if actual_uses <= 0:
            continue

        # Assign
        for idx, cnt in counts.items():
            remaining[unique_lengths[idx]] -= cnt * actual_uses

        pipe_lengths = [unique_lengths[i] for i in indices]
        for _ in range(actual_uses):
            solution.append({
                'pattern_type': p_idx,
                'length_indices': indices,
                'pipe_lengths': pipe_lengths,
                'total_length': total,
                'waste': waste,
                'num_welds': len(indices) - 1,
            })

    # Greedy mop-up on remaining inventory
    mop_up_solution, _, _ = greedy_solve(patterns, unique_lengths, remaining, stop_event)
    solution.extend(mop_up_solution)

    elapsed = time.time() - start
    return solution, 'LP_GUIDED', elapsed


# =============================================================================
# GPU PIPE OPTIMIZER
# =============================================================================

class GPUPipeOptimizer:
    """GPU-accelerated pipe optimizer. Same interface as SymmetryAwareSafeSolver.

    Exposes: raw_lengths, precision, unique_lengths, inventory, n_types,
             theoretical_max, length_to_idx -- all attributes used by export_to_excel().
    """

    def __init__(self, pipe_lengths, target_length=100.0, max_waste=20.0, precision=1):
        self.raw_lengths = pipe_lengths
        self.target_length = target_length
        self.max_waste = max_waste
        self.precision = precision

        # Symmetry reduction (identical to v5)
        rounded = [round(p, precision) for p in pipe_lengths]
        self.inventory = Counter(rounded)
        self.unique_lengths = sorted(self.inventory.keys(), reverse=True)
        self.n_types = len(self.unique_lengths)

        # Fast lookup: length -> index
        self.length_to_idx = {l: i for i, l in enumerate(self.unique_lengths)}

        # Theoretical maximum
        self.theoretical_max = int(sum(pipe_lengths) // target_length)

        # GPU init (may fail -- sets self.gpu_available = False)
        self.gpu_available = False
        self.ctx = None
        if HAS_FORGE:
            try:
                self.ctx = ForgeContext()
                self.gpu_available = True
            except (ForgeError, OSError):
                self.gpu_available = False

    def generate_patterns(self, max_welds=3, stop_event=None):
        """Generate patterns via GPU (or CPU fallback).

        Args:
            max_welds: Maximum pipe segments per pile (1, 2, or 3)
            stop_event: Optional threading.Event for cancellation

        Returns:
            List of pattern tuples: (indices_tuple, total_length, waste, counts_dict)
        """
        if self.gpu_available:
            return self._generate_patterns_gpu(max_welds, stop_event)
        else:
            return self._generate_patterns_cpu(max_welds, stop_event)

    def _generate_patterns_gpu(self, max_welds, stop_event=None):
        """Generate patterns using GPU kernels."""
        gen = GPUPatternGenerator(
            self.ctx,
            self.unique_lengths,
            self.inventory,
            self.target_length,
            self.max_waste,
        )
        return gen.generate(max_welds=max_welds)

    def _generate_patterns_cpu(self, max_welds, stop_event=None):
        """CPU fallback for pattern generation (mirrors v5 logic)."""
        patterns = []
        lengths = self.unique_lengths
        n = len(lengths)
        min_len = self.target_length
        max_len = self.target_length + self.max_waste

        # 1-pipe patterns
        for i in range(n):
            if stop_event and stop_event.is_set():
                return patterns
            if min_len <= lengths[i] <= max_len:
                waste = lengths[i] - self.target_length
                patterns.append(((i,), lengths[i], waste, {i: 1}))

        # 2-pipe patterns
        if max_welds >= 2:
            for i in range(n):
                if stop_event and stop_event.is_set():
                    return patterns
                for j in range(i, n):
                    total_len = lengths[i] + lengths[j]
                    if min_len <= total_len <= max_len:
                        waste = total_len - self.target_length
                        if i == j:
                            if self.inventory[lengths[i]] >= 2:
                                patterns.append(((i, j), total_len, waste, {i: 2}))
                        else:
                            patterns.append(((i, j), total_len, waste, {i: 1, j: 1}))

        # 3-pipe patterns
        if max_welds >= 3:
            for i in range(n):
                if stop_event and stop_event.is_set():
                    return patterns
                for j in range(i, n):
                    for k in range(j, n):
                        total_len = lengths[i] + lengths[j] + lengths[k]
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

        return patterns

    def solve_greedy(self, patterns):
        """Greedy solve: sort by waste, single-pass assignment.

        Returns:
            (solution_list, 'GREEDY', solve_time)
        """
        return greedy_solve(patterns, self.unique_lengths, self.inventory)

    def solve_lp_guided(self, patterns):
        """LP relaxation + floor + greedy mop-up.

        Returns:
            (solution_list, 'LP_GUIDED', solve_time)
        """
        return lp_guided_solve(patterns, self.unique_lengths, self.inventory)
