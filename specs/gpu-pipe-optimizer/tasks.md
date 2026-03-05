---
spec: gpu-pipe-optimizer
phase: tasks
total_tasks: 14
created: 2026-03-05
generated: auto
---

# Tasks: gpu-pipe-optimizer

## Phase 1: Make It Work (POC)

Focus: Validate GPU pattern gen + greedy solver works end-to-end. Skip LP-guided, skip tests, accept hardcoded paths.

- [x] 1.1 Build forge-ffi dylib (prerequisite)
  - **Do**: Run `cd ~/gpu_kernel/metal-forge-compute && cargo build -p forge-ffi --release`. Verify `target/release/libforge_ffi.dylib` exists. Copy dylib to `/Users/patrickkavanagh/pipes/` for easy discovery.
  - **Files**: `libforge_ffi.dylib`
  - **Done when**: `libforge_ffi.dylib` exists in pipes directory and `file` command shows Mach-O dylib
  - **Verify**: `file /Users/patrickkavanagh/pipes/libforge_ffi.dylib` outputs "Mach-O 64-bit dynamically linked shared library arm64"
  - **Commit**: `build: compile forge-ffi dylib and copy to project`
  - _Requirements: FR-1_
  - _Design: Component A_

- [ ] 1.2 Create forge_bindings.py -- ctypes wrapper
  - **Do**: Create `/Users/patrickkavanagh/pipes/forge_bindings.py` with: (1) `_find_dylib()` function searching `./`, `~/gpu_kernel/.../target/release/`, `_MEIPASS`, `DYLD_LIBRARY_PATH`; (2) `ForgeContext` class wrapping `forge_context_create/destroy` with `__del__` + context manager; (3) `ForgeBuffer` class with `from_numpy()`, `alloc()`, `to_numpy()` zero-copy, `__del__`; (4) `ForgeKernel` class wrapping `forge_kernel_compile` + `__del__`; (5) `ForgePipeline` class with `dispatch_1d()` and `execute()` returning gpu_time_ms; (6) `ForgeDtype` enum matching C enum values (U32=0, I32=1, F32=2). Map numpy dtypes to ForgeDtype. Use `ctypes.CDLL`, `ctypes.c_void_p`, `ctypes.POINTER`. Set `argtypes` and `restype` for all FFI functions. Raise `ForgeError` with message from `forge_last_error_message()` on any failure.
  - **Files**: `forge_bindings.py`
  - **Done when**: `python3 -c "from forge_bindings import ForgeContext; ctx = ForgeContext(); print('GPU OK')"` prints "GPU OK"
  - **Verify**: `python3 -c "from forge_bindings import ForgeContext, ForgeBuffer; import numpy as np; ctx = ForgeContext(); b = ForgeBuffer.from_numpy(ctx, np.array([1.0, 2.0, 3.0], dtype=np.float32)); print(b.to_numpy())"`
  - **Commit**: `feat(gpu): add forge_bindings.py ctypes wrapper for forge-ffi`
  - _Requirements: FR-1, FR-12_
  - _Design: Component A_

- [x] 1.3 Create gpu_solver.py -- Metal kernel source strings
  - **Do**: Create `/Users/patrickkavanagh/pipes/gpu_solver.py`. Add module-level string constants `GEN_PATTERNS_3_MSL` and `GEN_PATTERNS_2_MSL` containing the Metal kernel source. Follow the kernel design from design.md exactly: `device const float* lengths`, `device const float* params`, output arrays `device uint* out_i/j/k`, `device float* out_waste`, `device atomic_uint* counter`. The 3-pipe kernel decodes `tid` to `(i,j,k)`, checks `i<=j<=k`, computes sum, checks range, checks inventory (4 duplicate cases), atomically writes output. The 2-pipe kernel is similar with n^2 threads and `(i,j)` decode, `i<=j`, 2 duplicate cases.
  - **Files**: `gpu_solver.py`
  - **Done when**: `python3 -c "from gpu_solver import GEN_PATTERNS_3_MSL, GEN_PATTERNS_2_MSL; print(len(GEN_PATTERNS_3_MSL), len(GEN_PATTERNS_2_MSL))"` prints two numbers > 100
  - **Verify**: `python3 -c "from gpu_solver import GEN_PATTERNS_3_MSL, GEN_PATTERNS_2_MSL; assert len(GEN_PATTERNS_3_MSL) > 100; assert len(GEN_PATTERNS_2_MSL) > 100; print('Kernel strings OK')"`
  - **Commit**: `feat(gpu): add Metal kernel source strings for pattern generation`
  - _Requirements: FR-2_
  - _Design: Metal Kernel Design_

- [x] 1.4 Create gpu_solver.py -- GPUPatternGenerator class
  - **Do**: Add `GPUPatternGenerator` class to `gpu_solver.py`. Constructor takes `ForgeContext`, `unique_lengths` (list), `inventory` (dict), `target_length`, `max_waste`. Compiles both kernels. Method `generate(max_welds=3)` does: (1) Build params buffer `[n, target, max_waste, inv_0, ..., inv_n-1]` as float32 array. (2) Upload lengths and params as ForgeBuffers. (3) Allocate output buffers: `out_i`, `out_j`, `out_k` (uint32, 4M each), `out_waste` (float32, 4M), `counter` (uint32, 1 element, zeroed). (4) If max_welds >= 3: dispatch gen_patterns_3 with thread_count=n^3, execute, read counter. (5) If max_welds >= 2: dispatch gen_patterns_2 similarly with separate counter. (6) 1-pipe: numpy mask (CPU). (7) Convert GPU output arrays to pattern tuples: `((i, j, k), total_len, waste, {i: c1, j: c2, ...})` matching v5 format. (8) Return combined pattern list.
  - **Files**: `gpu_solver.py`
  - **Done when**: Can generate patterns from sample data
  - **Verify**: `python3 -c "from gpu_solver import GPUPatternGenerator; print('GPUPatternGenerator class OK')"`
  - **Commit**: `feat(gpu): add GPUPatternGenerator class with Metal kernel dispatch`
  - _Requirements: FR-3_
  - _Design: Component B, Data Flow_

- [x] 1.5 Create gpu_solver.py -- greedy_solve function
  - **Do**: Add `greedy_solve(patterns, unique_lengths, inventory)` function to `gpu_solver.py`. Takes pattern list (same format as v5), unique_lengths list, inventory Counter. (1) Extract waste values into numpy array. (2) `sorted_indices = np.argsort(waste_array)`. (3) Copy inventory to `remaining = dict(inventory)`. (4) For each pattern index in sorted order: compute `max_uses = min(remaining[unique_lengths[idx]] // cnt for idx, cnt in pattern.counts.items())`. If max_uses > 0: create `max_uses` solution dicts matching v5 format, subtract from remaining. (5) Return `(solution_list, 'GREEDY', elapsed_time)`.
  - **Files**: `gpu_solver.py`
  - **Done when**: Greedy solver returns a valid solution list from GPU-generated patterns
  - **Verify**: `python3 -c "from gpu_solver import greedy_solve; print('greedy_solve function OK')"`
  - **Commit**: `feat(gpu): add greedy_solve function for fast solving`
  - _Requirements: FR-4_
  - _Design: Component B, Data Flow_

- [x] 1.6 Create gpu_solver.py -- GPUPipeOptimizer class
  - **Do**: Add `GPUPipeOptimizer` class to `gpu_solver.py`. Same interface as `SymmetryAwareSafeSolver`: `__init__(pipe_lengths, target_length, max_waste, precision)` does symmetry reduction (Counter, unique_lengths, n_types, theoretical_max). Stores `raw_lengths`, `precision`, `unique_lengths`, `inventory`, `n_types`, `theoretical_max` as attributes (needed by `export_to_excel()`). Creates `ForgeContext` in try/except, sets `gpu_available`. Method `generate_patterns(max_welds=3, stop_event=None)` calls `GPUPatternGenerator` if GPU available, else falls back to CPU loop (copy from v5 `generate_patterns`). Methods `solve_greedy(patterns)` and `solve_lp_guided(patterns)` delegate to module functions.
  - **Files**: `gpu_solver.py`
  - **Done when**: `GPUPipeOptimizer` can load data, generate patterns, solve greedy, and produce export-compatible solution
  - **Verify**: `python3 -c "from gpu_solver import GPUPipeOptimizer; print('GPUPipeOptimizer class OK')"`
  - **Commit**: `feat(gpu): add GPUPipeOptimizer class with full solver interface`
  - _Requirements: FR-6, FR-11_
  - _Design: GPUPipeOptimizer Class Design_

- [x] 1.7 POC Checkpoint -- end-to-end GPU pattern gen + greedy solve
  - **Do**: Verify the full pipeline works: load sample data, GPU pattern gen, greedy solve, export to Excel. Compare GPU pattern count vs CPU pattern count. Verify output Excel opens and has correct sheets.
  - **Done when**: GPU path produces valid results on `data/sample_pipe_data.csv` with target=100, waste=5
  - **Verify**: `python3 -c "from gpu_solver import GPUPipeOptimizer; from pipe_optimizer_v5_safe import load_pipe_data; d = load_pipe_data('data/sample_pipe_data.csv'); opt = GPUPipeOptimizer(d.lengths, 100.0, 5.0, 1); p = opt.generate_patterns(3); s, st, t = opt.solve_greedy(p); print(f'{len(s)} piles'); assert len(s) >= 250"`
  - **Commit**: `feat(gpu): complete POC for GPU pattern generation and greedy solver`

## Phase 2: Refactoring + LP-Guided Solver

After POC validated, add LP-guided solver and clean up code.

- [x] 2.1 Add lp_guided_solve function
  - **Do**: Add `lp_guided_solve(patterns, unique_lengths, inventory)` to `gpu_solver.py`. (1) Build PuLP LP model identical to `solve_ilp` but with `cat='Continuous'` (not Integer). (2) Solve LP relaxation (fast, ~seconds). (3) Floor all variable values: `floor_uses = int(math.floor(x[i].varValue or 0))`. (4) Assign floored patterns, subtract from inventory copy. (5) Run `greedy_solve()` on remaining inventory with remaining patterns. (6) Combine solutions. (7) Return `(combined_solution, 'LP_GUIDED', elapsed_time)`. Wire into `GPUPipeOptimizer.solve_lp_guided()`.
  - **Files**: `gpu_solver.py`
  - **Done when**: LP-guided solver returns solution with pile count >= 99% of ILP optimal
  - **Verify**: `python3 -c "from gpu_solver import GPUPipeOptimizer; from pipe_optimizer_v5_safe import load_pipe_data; d = load_pipe_data('data/sample_pipe_data.csv'); opt = GPUPipeOptimizer(d.lengths, 100.0, 5.0, 1); p = opt.generate_patterns(3); s, st, t = opt.solve_lp_guided(p); print(f'{len(s)} piles in {t:.2f}s ({st})')"`
  - **Commit**: `feat(gpu): add LP-guided rounding solver`
  - _Requirements: FR-5_
  - _Design: Component B, LP-Guided Solve Flow_

- [x] 2.2 Add CPU fallback for pattern generation
  - **Do**: In `GPUPipeOptimizer._generate_patterns_cpu()`, implement CPU fallback that mirrors `SymmetryAwareSafeSolver.generate_patterns()` logic. This ensures the optimizer works on non-Apple-Silicon machines. Add proper `gpu_available` checks and try/except around all GPU operations in `_generate_patterns_gpu()`.
  - **Files**: `gpu_solver.py`
  - **Done when**: Setting `gpu_available=False` manually still produces correct patterns via CPU path
  - **Verify**: `python3 -c "from gpu_solver import GPUPipeOptimizer; from pipe_optimizer_v5_safe import load_pipe_data; d = load_pipe_data('data/sample_pipe_data.csv'); opt = GPUPipeOptimizer(d.lengths, 100.0, 5.0, 1); opt.gpu_available = False; p = opt.generate_patterns(3); print(f'{len(p)} CPU patterns')"`
  - **Commit**: `feat(gpu): add CPU fallback for pattern generation`
  - _Requirements: FR-9_
  - _Design: Error Handling_

- [x] 2.3 Modify pipe_optimizer_gui.py -- add solver mode dropdown + worker routing
  - **Do**: (1) Add `solver_var` StringVar and combobox at row 5 in `adv_frame` with values `('ILP (Optimal)', 'GPU Greedy (Fast)', 'GPU + LP Rounding (Balanced)')`, default "ILP (Optimal)". Shift tip label to row 6. (2) Add `solver_mode` to params dict in `run_optimizer()`. (3) In `_optimizer_worker()`, check `solver_mode`: if ILP, run existing path unchanged; if GPU Greedy or GPU+LP, import `GPUPipeOptimizer`, create instance, generate patterns, call `solve_greedy` or `solve_lp_guided`. Wrap GPU path in try/except, fallback to ILP with status message. (4) Update results summary string to include solver mode used. (5) Use `gpu_solver` instance as `solver` for `export_to_excel()`.
  - **Files**: `pipe_optimizer_gui.py`
  - **Done when**: GUI launches with solver mode dropdown, all 3 modes selectable
  - **Verify**: `python3 -c "import pipe_optimizer_gui; print('GUI import OK')"`
  - **Commit**: `feat(gui): add solver mode dropdown with GPU Greedy and LP Rounding options`
  - _Requirements: FR-7, FR-8_
  - _Design: Component C, GUI Integration Design_

## Phase 3: Testing

- [x] 3.1 Create test_gpu_solver.py -- all validation tests
  - **Do**: Create `/Users/patrickkavanagh/pipes/test_gpu_solver.py` with 6 test functions using `unittest`: (1) `test_pattern_counts`: Load sample data, generate patterns with both CPU (SymmetryAwareSafeSolver) and GPU (GPUPipeOptimizer). Assert pattern counts are equal. (2) `test_inventory_integrity`: Run greedy solve, verify no pipe type over-allocated by summing pattern usage per type vs inventory. (3) `test_greedy_quality`: Run greedy, compare pile count to known ILP optimal (264). Assert `piles >= 0.95 * 264`. (4) `test_lp_guided_quality`: Run LP-guided, assert `piles >= 0.99 * 264`. (5) `test_gpu_speed`: Time GPU vs CPU pattern gen, assert GPU faster. (6) `test_fallback`: Set `gpu_available=False` on GPUPipeOptimizer, verify `generate_patterns()` still works. Use `data/sample_pipe_data.csv` as test data (758 pipes, target=100, waste=5).
  - **Files**: `test_gpu_solver.py`
  - **Done when**: `python3 -m pytest test_gpu_solver.py -v` or `python3 test_gpu_solver.py` runs all 6 tests
  - **Verify**: `cd /Users/patrickkavanagh/pipes && python3 -m pytest test_gpu_solver.py -v`
  - **Commit**: `test(gpu): add validation tests for GPU solver`
  - _Requirements: FR-10_
  - _Design: Component B_

## Phase 4: Quality Gates

- [x] 4.1 Local quality check
  - **Do**: Run all quality checks: (1) `python3 -m pytest test_gpu_solver.py -v` -- all pass. (2) `python3 -c "from forge_bindings import *; from gpu_solver import *"` -- imports clean. (3) Launch GUI, run with each solver mode on sample data, verify Excel output. (4) Verify GPU pattern count matches CPU pattern count exactly.
  - **Verify**: `python3 -c "from forge_bindings import ForgeContext; from gpu_solver import GPUPipeOptimizer; print('imports OK')"`
  - **Done when**: All 6 tests pass, GUI works with all 3 modes, Excel export valid
  - **Commit**: `fix(gpu): address any issues found in quality check` (if needed)

- [ ] 4.2 Create PR and verify CI
  - **Do**: Push branch, create PR with `gh pr create`. PR title: "feat: GPU-accelerated pattern generation + greedy/LP solvers". Include benchmark comparison (GPU vs CPU pattern gen time, greedy vs ILP pile count).
  - **Verify**: `echo "PR created"`
  - **Done when**: PR created, CI passes, ready for review

## Notes

- **POC shortcuts taken**: LP-guided solver deferred to Phase 2. CPU fallback deferred to Phase 2. Tests deferred to Phase 3.
- **Production TODOs**: Add progress callbacks during GPU dispatch for GUI progress bar. Consider batch dispatch for very large n (>500 types). Profile GPU memory usage for large datasets.
- **Prerequisite**: forge-ffi dylib must be built before any GPU code runs. Task 1.1 handles this.
- **Quality baseline**: 264 piles is the known ILP optimal for `data/sample_pipe_data.csv` with target=100, waste=5.
