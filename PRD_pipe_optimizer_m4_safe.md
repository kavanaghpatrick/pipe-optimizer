# PRD: Memory-Safe M4 Pipe Optimizer v2.0

## Problem Statement

The current `pipe_optimizer_m4_optimized.py` crashes the entire computer due to memory exhaustion. Root cause analysis by Grok and Gemini identified:

1. **Combinatorial explosion**: 3-pipe pattern generation creates billions of combinations (O(N³))
2. **Data duplication**: macOS `spawn` copies full arrays to each of 56 worker chunks
3. **ILP memory explosion**: Millions of PuLP LpVariable objects consume 3-5GB+
4. **No safety limits**: No caps on patterns, no memory monitoring

## Goals

1. **Primary**: Prevent computer crashes - never exceed 8GB RAM usage
2. **Secondary**: Maintain optimization quality (still beat greedy's 163 piles)
3. **Tertiary**: Improve performance through smarter memory management

## Non-Goals

- Changing the core ILP algorithm
- Supporting different solvers (stick with CBC)
- Adding new optimization features

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory-Safe Pipe Optimizer                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Memory Monitor (psutil)                                      │
│     - Check before each phase                                    │
│     - Hard limit: 8GB, soft limit: 6GB                          │
├─────────────────────────────────────────────────────────────────┤
│  2. Pattern Generator (Memory-Efficient)                         │
│     - Worker initializer (no data copying)                       │
│     - Per-chunk pattern limits                                   │
│     - Adaptive waste filtering                                   │
├─────────────────────────────────────────────────────────────────┤
│  3. Pattern Aggregator                                           │
│     - Streaming aggregation (not bulk)                           │
│     - Total pattern cap: 2M                                      │
│     - Priority: lower waste patterns first                       │
├─────────────────────────────────────────────────────────────────┤
│  4. ILP Solver (Memory-Bounded)                                  │
│     - Pattern count validation before solving                    │
│     - Streaming constraint generation                            │
│     - Memory check during solve                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 1. Memory Monitoring System

```python
import psutil

MEMORY_SOFT_LIMIT_GB = 6.0
MEMORY_HARD_LIMIT_GB = 8.0

def get_memory_gb():
    """Get current process memory in GB"""
    return psutil.Process().memory_info().rss / (1024**3)

def check_memory(phase: str, soft_action="warn", hard_action="abort"):
    """Check memory and take action if limits exceeded"""
    mem = get_memory_gb()
    if mem > MEMORY_HARD_LIMIT_GB:
        raise MemoryError(f"[{phase}] Memory {mem:.1f}GB exceeds hard limit {MEMORY_HARD_LIMIT_GB}GB")
    if mem > MEMORY_SOFT_LIMIT_GB:
        print(f"WARNING [{phase}]: Memory {mem:.1f}GB approaching limit")
    return mem
```

#### 2. Worker Initializer Pattern (Eliminate Data Copying)

```python
# Global variables for worker processes
_worker_sorted_indices = None
_worker_sorted_lengths = None
_worker_target = None
_worker_max_waste = None

def init_worker(sorted_indices, sorted_lengths, target, max_waste):
    """Initialize worker with shared read-only data"""
    global _worker_sorted_indices, _worker_sorted_lengths
    global _worker_target, _worker_max_waste
    _worker_sorted_indices = sorted_indices
    _worker_sorted_lengths = sorted_lengths
    _worker_target = target
    _worker_max_waste = max_waste

def generate_chunk(args):
    """Worker function - uses globals, no data in args"""
    chunk_id, chunk_start, chunk_end, all_indices, progress_dict = args
    # Access globals instead of unpacking from args
    sorted_indices = _worker_sorted_indices
    sorted_lengths = _worker_sorted_lengths
    target = _worker_target
    max_waste = _worker_max_waste
    # ... rest of generation logic
```

#### 3. Pattern Limits

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MAX_PATTERNS_PER_CHUNK | 100,000 | Prevents any single worker from exploding |
| MAX_TOTAL_PATTERNS | 2,000,000 | ~400MB for pattern list, safe for ILP |
| INITIAL_MAX_WASTE | 10.0 | Tighter than 20.0, reduces combinations |
| FALLBACK_MAX_WASTE | 5.0 | If still too many patterns, tighten further |

#### 4. Adaptive Waste Filtering

```python
def generate_patterns_adaptive(self, target_max_patterns=2_000_000):
    """Generate patterns with adaptive waste filtering"""
    waste_levels = [10.0, 7.0, 5.0, 3.0]  # Progressively tighter

    for max_waste in waste_levels:
        self.max_waste = max_waste
        print(f"Trying max_waste={max_waste}'...")

        patterns = self._generate_all_patterns()

        if len(patterns) <= target_max_patterns:
            print(f"Success: {len(patterns):,} patterns with max_waste={max_waste}'")
            return patterns

        print(f"Too many patterns ({len(patterns):,}), tightening waste filter...")
        del patterns  # Free memory
        gc.collect()

    raise ValueError("Cannot generate safe number of patterns even with tightest filter")
```

#### 5. Pool Management with Context Manager

```python
def generate_three_pipe_patterns_safe(self, n_cores=14):
    """Memory-safe 3-pipe generation"""
    # Prepare data
    pipes = sorted(enumerate(self.pipe_lengths), key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in pipes]
    sorted_lengths = [length for _, length in pipes]

    # Use context manager for proper cleanup
    with mp.Pool(
        processes=n_cores,
        initializer=init_worker,
        initargs=(sorted_indices, sorted_lengths, self.target_length, self.max_waste)
    ) as pool:
        # Create lightweight chunks (no data, just indices)
        chunks = self._create_chunks(n_cores)

        # Process with memory monitoring
        results = []
        for result in pool.imap_unordered(generate_chunk, chunks):
            results.extend(result)
            check_memory("3-pipe generation")

            if len(results) > MAX_TOTAL_PATTERNS:
                print(f"Hit pattern limit at {len(results):,}")
                break

    return results[:MAX_TOTAL_PATTERNS]
```

#### 6. Streaming Constraint Generation

```python
def solve_ilp_safe(self, patterns):
    """Memory-efficient ILP solving"""
    n_patterns = len(patterns)

    # Safety check
    if n_patterns > 2_000_000:
        raise ValueError(f"Too many patterns ({n_patterns:,}) for safe ILP solving")

    check_memory("ILP setup")

    prob = LpProblem("Optimal_Piles_Safe", LpMaximize)

    # Create variables
    x = [LpVariable(f"x{p}", cat='Binary') for p in range(n_patterns)]
    prob += lpSum(x), "Total_Piles"

    check_memory("After variables")

    # Stream constraints (don't build full index first)
    print("Adding constraints (streaming)...")
    for i in range(self.n_pipes):
        relevant = [p for p in range(n_patterns) if i in patterns[p][0]]
        if relevant:
            prob += lpSum(x[p] for p in relevant) <= 1, f"Pipe_{i}"

        if i % 100 == 0:
            check_memory(f"Constraint {i}")

    # Solve
    check_memory("Before solve")
    solver = PULP_CBC_CMD(msg=1, timeLimit=1800, threads=n_cores)
    prob.solve(solver)

    return self._extract_solution(prob, x, patterns)
```

### File Structure

```
pipes/
├── pipe_optimizer_m4_safe.py     # New memory-safe version
├── pipe_lengths_clean.csv         # Input data
└── pipe_optimization_M4_SAFE.xlsx # Output
```

### CLI Interface

```bash
# Basic usage (defaults to safe limits)
python3 pipe_optimizer_m4_safe.py

# Custom limits
python3 pipe_optimizer_m4_safe.py --max-waste 5.0 --max-patterns 1000000

# Memory monitoring verbosity
python3 pipe_optimizer_m4_safe.py --memory-verbose
```

### Success Criteria

1. **No crashes**: Complete runs without memory exhaustion on 16GB M4 MacBook
2. **Quality**: Produce >= 163 piles (match or beat greedy)
3. **Performance**: Complete in < 30 minutes
4. **Monitoring**: Clear memory usage reporting throughout

### Testing Plan

1. **Unit test**: Memory monitor functions
2. **Integration test**: Run with small dataset (100 pipes)
3. **Stress test**: Run with full dataset, monitor with Activity Monitor
4. **Regression test**: Compare solution quality to previous runs

### Rollback Plan

If new version produces worse solutions:
1. Increase MAX_TOTAL_PATTERNS gradually
2. Loosen waste filters
3. Fall back to original with `ulimit -v` memory cap

## Timeline

1. Implement memory monitoring - immediate
2. Implement worker initializer - immediate
3. Implement pattern limits - immediate
4. Integration and testing - immediate
5. Deploy and validate - immediate

## Open Questions

1. Should we support column generation for very large datasets? (Future enhancement)
2. Should we add checkpointing for long runs? (Future enhancement)
