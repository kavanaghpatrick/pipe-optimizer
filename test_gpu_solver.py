#!/usr/bin/env python3
"""
Tests for GPUPipeOptimizer (gpu_solver.py).

Validates pattern generation, inventory integrity, solver quality,
GPU vs CPU speed, and CPU fallback behavior.
"""

import unittest
import time
from collections import Counter
from pipe_optimizer_v5_safe import load_pipe_data, SymmetryAwareSafeSolver
from gpu_solver import GPUPipeOptimizer


class TestGPUSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_pipe_data('data/sample_pipe_data.csv')
        cls.target = 100.0
        cls.waste = 5.0
        cls.precision = 1
        cls.optimal_piles = 264

    def test_pattern_counts(self):
        """GPU and CPU generate similar pattern counts."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)
        gpu_patterns = opt.generate_patterns(3)

        opt.gpu_available = False
        cpu_patterns = opt.generate_patterns(3)

        # Allow small discrepancy due to float precision in GPU
        ratio = len(gpu_patterns) / len(cpu_patterns)
        self.assertAlmostEqual(ratio, 1.0, delta=0.02,
            msg=f"GPU ({len(gpu_patterns)}) vs CPU ({len(cpu_patterns)}) pattern count mismatch")

    def test_inventory_integrity(self):
        """No pipe type over-allocated in greedy solution."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)
        patterns = opt.generate_patterns(3)
        solution, _, _ = opt.solve_greedy(patterns)

        # Sum up usage per type
        usage = Counter()
        for pile in solution:
            for i, length in zip(pile['length_indices'], pile['pipe_lengths']):
                usage[length] += 1

        # Check no type exceeds inventory
        for length, count in usage.items():
            self.assertLessEqual(count, opt.inventory[length],
                msg=f"Over-allocated length {length}: used {count}, have {opt.inventory[length]}")

    def test_greedy_quality(self):
        """Greedy achieves >= 90% of optimal."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)
        patterns = opt.generate_patterns(3)
        solution, status, _ = opt.solve_greedy(patterns)

        self.assertEqual(status, 'GREEDY')
        self.assertGreaterEqual(len(solution), int(0.90 * self.optimal_piles),
            msg=f"Greedy {len(solution)} piles < 90% of {self.optimal_piles}")

    def test_lp_guided_quality(self):
        """LP-guided achieves >= 99% of optimal."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)
        patterns = opt.generate_patterns(3)
        solution, status, _ = opt.solve_lp_guided(patterns)

        self.assertEqual(status, 'LP_GUIDED')
        self.assertGreaterEqual(len(solution), int(0.99 * self.optimal_piles),
            msg=f"LP-guided {len(solution)} piles < 99% of {self.optimal_piles}")

    def test_gpu_speed(self):
        """GPU pattern generation faster than CPU."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)

        if not opt.gpu_available:
            self.skipTest("GPU not available")

        t0 = time.time()
        opt.generate_patterns(3)
        gpu_time = time.time() - t0

        opt.gpu_available = False
        t0 = time.time()
        opt.generate_patterns(3)
        cpu_time = time.time() - t0

        print(f"\n  GPU: {gpu_time:.3f}s, CPU: {cpu_time:.3f}s, Speedup: {cpu_time/gpu_time:.1f}x")
        self.assertLess(gpu_time, cpu_time, "GPU should be faster than CPU")

    def test_fallback(self):
        """CPU fallback works when GPU unavailable."""
        opt = GPUPipeOptimizer(self.data.lengths, self.target, self.waste, self.precision)
        opt.gpu_available = False
        patterns = opt.generate_patterns(3)

        self.assertGreater(len(patterns), 0)
        solution, _, _ = opt.solve_greedy(patterns)
        self.assertGreater(len(solution), 0)


if __name__ == '__main__':
    unittest.main()
