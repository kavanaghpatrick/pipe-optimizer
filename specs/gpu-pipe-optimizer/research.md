---
spec: gpu-pipe-optimizer
phase: research
created: 2026-03-05
generated: auto
---

# Research: gpu-pipe-optimizer

## Executive Summary

The current pipe optimizer spends 1-5 min in O(n^3) Python pattern generation and 2-30 min in CBC ILP solving. By offloading pattern generation to Metal GPU via forge-ffi (20.3M threads for n=273 types) and replacing ILP with a greedy/LP-guided CPU solver, total solve time drops from 3-35 min to ~1-30 seconds. Feasibility is HIGH -- forge-ffi's custom kernel API matches the required buffer binding pattern exactly.

## Codebase Analysis

### Existing Patterns

- **Symmetry reduction** (`pipe_optimizer_v5_safe.py:325-505`): Groups 758 pipes into 273 unique length types via `Counter(rounded)`. Pattern generation iterates `i <= j <= k` (upper triangle). This reduction is preserved in GPU approach.
- **Solution dict format** (`pipe_optimizer_v5_safe.py:597-615`): Each pile is `{'pattern_type', 'length_indices', 'pipe_lengths', 'total_length', 'waste', 'num_welds'}`. GPU solver must emit identical format for `export_to_excel()` compatibility.
- **GUI worker pattern** (`pipe_optimizer_gui.py:386-498`): Background `threading.Thread` with `stop_event` cancellation, progress via `queue.Queue`, main-thread polling at 100ms. GPU solver integrates identically.
- **File loader** (`pipe_optimizer_v5_safe.py:97-185`): `load_pipe_data()` returns `PipeData` with `.lengths` ndarray. Shared by both ILP and GPU paths.

### Dependencies

| Dependency | Status | Usage |
|-----------|--------|-------|
| `libforge_ffi.dylib` | Must build (`cargo build -p forge-ffi --release`) | GPU context, buffers, custom kernel dispatch |
| `numpy` | Already installed | Array ops, inventory tracking, argsort |
| `PuLP/CBC` | Already installed | LP relaxation (continuous) for LP-guided mode |
| `ctypes` | stdlib | Python-to-C FFI bridge |

### Constraints

- **Apple Silicon only**: Metal GPU requires macOS with Apple Silicon (M1+). forge-ffi has no CUDA path.
- **UMA memory**: forge-ffi uses `device const float*` for input buffers, `device float*` for output. All zero-copy via UMA shared memory.
- **Atomic counter**: Metal `atomic_fetch_add_explicit` for output index. Single atomic counter buffer needed per kernel dispatch.
- **Max threadgroup size**: Metal limit is 1024 threads/group. forge-ffi handles threadgroup sizing internally via `dispatch_1d`.
- **Pattern output buffer**: Must pre-allocate worst-case size. For n=273, 3-pipe upper triangle = C(273+2,3) = ~3.4M max patterns. Allocate 4M entries.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | forge-ffi custom kernel API proven (sigmoid example). Pattern gen is embarrassingly parallel. |
| Effort Estimate | **M** (3-5 days) | 4 new files, 1 modification. Most complexity in Metal kernel + ctypes wrapper. |
| Risk Level | **Low** | CPU fallback ensures functionality even without GPU. Greedy solver is straightforward. |
| GPU Thread Count | **Feasible** | 20.3M threads (273^3) well within M-series GPU capability (100M+ threads common). |
| Memory | **Low risk** | 4M patterns * 4 floats * 4 bytes = ~64MB GPU buffer. UMA shares with system RAM. |

## Key Technical Findings

### Pattern Generation Parallelism

Each thread `tid` in 0..n^3 maps to `(i, j, k)`:
```
i = tid / (n * n)
j = (tid / n) % n
k = tid % n
```
Skip if `i > j` or `j > k` (upper triangle). ~83% of threads exit early, but GPU handles this efficiently. Net valid work: ~3.4M pattern candidates checked, ~500K-1M valid patterns output.

### Greedy Solver Quality

Literature and plan analysis suggests greedy (sort by waste, single-pass) achieves 95%+ of ILP optimal for bin-packing-like problems with many pattern options. For 758 pipes / 273 types with 500K+ patterns, greedy coverage is excellent.

### LP-Guided Rounding Quality

LP relaxation (continuous variables, not integer) solves in seconds via CBC. Floor fractional values, then greedy mop-up of remainder. Expected 99%+ of ILP optimal based on bin-packing LP gap analysis.

### forge-ffi API Fit

The `forge_kernel_compile` + `forge_pipeline_dispatch_1d` API is a direct match:
1. Compile MSL source string at runtime
2. Bind input/output buffers to `[[buffer(N)]]`
3. Dispatch with `thread_count = n * n * n`
4. Read results via `forge_buffer_contents()` (zero-copy UMA pointer)

No gaps in API coverage. The custom_kernel.c example confirms the exact pattern needed.

## Recommendations

1. Build forge-ffi dylib as prerequisite -- verify it links on target machine before writing Python code
2. Start with 2-pipe kernel (simpler, n^2 = 74K threads) as POC, then extend to 3-pipe
3. Pre-allocate output buffers conservatively (4M entries) -- Metal has no dynamic allocation
4. Implement CPU fallback in pattern generation (pure numpy) for non-Apple-Silicon machines
5. Keep ILP solver unchanged as reference baseline for quality validation tests
