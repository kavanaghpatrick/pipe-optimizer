---
spec: gpu-pipe-optimizer
phase: requirements
created: 2026-03-05
generated: auto
---

# Requirements: gpu-pipe-optimizer

## Summary

Add GPU-accelerated pattern generation via forge-ffi Metal kernels and CPU greedy/LP-guided solvers as faster alternatives to the ILP solver. Provide GUI integration with solver mode selection and graceful fallback when GPU is unavailable.

## User Stories

### US-1: GPU Pattern Generation

As an optimizer user, I want pattern generation to complete in milliseconds instead of minutes so that I can iterate on parameters quickly.

**Acceptance Criteria**:
- AC-1.1: GPU generates identical pattern counts to CPU for 2-pipe and 3-pipe patterns (within 0 tolerance)
- AC-1.2: GPU pattern generation for n=273 types completes in <1 second (vs 1-5 min CPU)
- AC-1.3: All generated patterns satisfy `target <= sum <= target + max_waste`
- AC-1.4: Inventory feasibility is checked (no pattern requires more pipes of a type than available)

### US-2: Fast Greedy Solver

As an optimizer user, I want a fast solver mode that finds a good solution in seconds so that I don't need to wait 30 minutes for optimal results.

**Acceptance Criteria**:
- AC-2.1: Greedy solver runs in <1 second for up to 1M patterns
- AC-2.2: Greedy solution achieves >=95% of ILP optimal pile count on reference dataset (758 pipes, target=100)
- AC-2.3: No pipe is over-allocated (inventory constraints respected)
- AC-2.4: Solution format is identical to ILP output (`export_to_excel()` compatible)

### US-3: LP-Guided Rounding Solver

As an optimizer user, I want a balanced solver mode that provides near-optimal results faster than ILP so that I get high quality without long waits.

**Acceptance Criteria**:
- AC-3.1: LP-guided solver completes in <60 seconds for reference dataset
- AC-3.2: LP-guided solution achieves >=99% of ILP optimal pile count
- AC-3.3: Inventory constraints respected
- AC-3.4: Solution format compatible with `export_to_excel()`

### US-4: GUI Solver Mode Selection

As a GUI user, I want to choose between ILP, GPU Greedy, and GPU+LP Rounding solver modes so that I can trade off speed vs quality.

**Acceptance Criteria**:
- AC-4.1: "Solver mode" dropdown appears in Advanced Parameters section
- AC-4.2: Three options: "ILP (Optimal)", "GPU Greedy (Fast)", "GPU + LP Rounding (Balanced)"
- AC-4.3: Default is "ILP (Optimal)" (backward compatible)
- AC-4.4: Results display shows which solver mode was used

### US-5: Graceful Fallback

As a user on a non-Apple-Silicon machine, I want the optimizer to fall back to CPU automatically so that I can still use the tool.

**Acceptance Criteria**:
- AC-5.1: If libforge_ffi.dylib not found, pattern gen falls back to CPU (existing Python loops)
- AC-5.2: If GPU kernel compilation fails, pattern gen falls back to CPU
- AC-5.3: If GPU solver selected but GPU unavailable, falls back to ILP solver with user notification
- AC-5.4: Fallback does not crash or require user intervention

### US-6: Forge FFI Bindings

As a developer, I want a clean Python ctypes wrapper for forge-ffi so that GPU operations are easy to use from Python.

**Acceptance Criteria**:
- AC-6.1: `ForgeContext` class wraps `forge_context_create/destroy` with RAII (`__del__` + `__enter__/__exit__`)
- AC-6.2: `ForgeBuffer` class wraps buffer lifecycle with `to_numpy()` zero-copy view
- AC-6.3: `ForgeKernel` wraps `forge_kernel_compile` with error propagation
- AC-6.4: `ForgePipeline` wraps `pipeline_create`, `dispatch_1d`, `execute_timed`
- AC-6.5: Dylib resolution searches: `./`, `~/gpu_kernel/.../target/release/`, `_MEIPASS`, `DYLD_LIBRARY_PATH`

### US-7: Validation Tests

As a developer, I want automated tests that verify GPU solver correctness so that regressions are caught.

**Acceptance Criteria**:
- AC-7.1: `test_pattern_counts` -- GPU and CPU produce identical pattern counts
- AC-7.2: `test_inventory_integrity` -- no pipe over-allocated in greedy solution
- AC-7.3: `test_greedy_quality` -- greedy achieves >=95% of optimal
- AC-7.4: `test_lp_guided_quality` -- LP-guided achieves >=99% of optimal
- AC-7.5: `test_gpu_speed` -- GPU pattern gen faster than CPU
- AC-7.6: `test_fallback` -- CPU fallback works when GPU unavailable

### US-8: Solution Format Compatibility

As a user, I want GPU solver output to work with the existing Excel export so that my workflow is unchanged.

**Acceptance Criteria**:
- AC-8.1: Solution dict has keys: `pattern_type`, `length_indices`, `pipe_lengths`, `total_length`, `waste`, `num_welds`
- AC-8.2: `export_to_excel(solver, solution, path)` works with GPU solver instance
- AC-8.3: Output Excel has Summary, Pile Details, Unused Pipes sheets (same as ILP output)

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | `forge_bindings.py` provides `ForgeContext`, `ForgeBuffer`, `ForgeKernel`, `ForgePipeline` Python wrappers for libforge_ffi.dylib | Must | US-6 |
| FR-2 | `gpu_solver.py` contains Metal kernel source strings for `gen_patterns_3` (n^3 threads) and `gen_patterns_2` (n^2 threads) | Must | US-1 |
| FR-3 | `GPUPatternGenerator` class generates patterns via GPU kernel dispatch with atomic output counter | Must | US-1 |
| FR-4 | `greedy_solve()` sorts patterns by waste, single-pass assignment respecting inventory | Must | US-2 |
| FR-5 | `lp_guided_solve()` uses LP relaxation + floor + greedy mop-up | Should | US-3 |
| FR-6 | `GPUPipeOptimizer` class provides same interface as `SymmetryAwareSafeSolver` | Must | US-8 |
| FR-7 | GUI adds solver mode dropdown with 3 options in Advanced Parameters | Must | US-4 |
| FR-8 | GUI worker thread routes to GPU solver or ILP based on selected mode | Must | US-4 |
| FR-9 | All GPU paths have try/except fallback to CPU/ILP equivalents | Must | US-5 |
| FR-10 | `test_gpu_solver.py` validates pattern counts, inventory, quality, speed, fallback | Must | US-7 |
| FR-11 | GPU solver solution dict format matches ILP solver exactly | Must | US-8 |
| FR-12 | Dylib resolution searches multiple paths including PyInstaller `_MEIPASS` | Should | US-6 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | GPU pattern generation completes in <1s for n=273 types | Performance |
| NFR-2 | Greedy solver completes in <1s for up to 1M patterns | Performance |
| NFR-3 | GPU buffer allocation stays under 256MB total | Memory |
| NFR-4 | No data races in Metal kernel (atomic counter for output index) | Correctness |
| NFR-5 | CPU fallback adds <5s overhead vs direct CPU path | Performance |
| NFR-6 | forge_bindings.py has no dependencies beyond ctypes + numpy | Portability |

## Out of Scope

- CUDA/OpenCL GPU backends (Metal only via forge-ffi)
- GPU-accelerated ILP solving (ILP stays on CPU via PuLP/CBC)
- Multi-GPU dispatch (single device only)
- Pattern generation for >3 pipes per pile (max_welds capped at 3)
- Windows/Linux GPU support (forge-ffi is macOS-only)

## Dependencies

- `libforge_ffi.dylib` must be pre-built via `cargo build -p forge-ffi --release`
- Apple Silicon Mac required for GPU path (Intel Macs fall back to CPU)
- Existing `pipe_optimizer_v5_safe.py` unchanged (used as fallback + quality baseline)
- PuLP/CBC already installed (used by LP-guided mode for LP relaxation)
