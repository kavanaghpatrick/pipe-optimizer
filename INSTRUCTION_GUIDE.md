# Pipe Pile Optimization: Excel Output Guide

## Overview

The file `pipe_optimization_V2_VERIFIED.xlsx` contains the complete, verified cutting plan for assembling 262 optimal 100-foot pipe piles from your inventory. Each pipe is assigned a unique index to prove no duplicate usage.

---

## File Contents

The Excel workbook contains three sheets:

### Sheet 1: Summary

High-level results and metrics:

| Metric | Description |
|--------|-------------|
| Method | ILP V2 (Symmetry-Aware) |
| Status | OPTIMAL (Verified) - mathematically proven best, no duplicates |
| Total Piles | 262 piles that can be built |
| Pipes Used | 750 pipes consumed |
| Pipes Unused | 8 pipes that couldn't be used |
| Total Waste (ft) | Total excess length across all piles |
| Avg Waste/Pile | Average excess per pile |
| vs Greedy (163) | Improvement over simple greedy algorithm |
| Unique Pipe Indices | Confirms 750 unique pipes, no duplicates |

### Sheet 2: Pile Details

Complete construction guide with one row per pipe segment:

| Column | Description |
|--------|-------------|
| **Pile Number** | Unique ID for each pile (1-262) |
| **Segment** | Segment position within the pile (1, 2, or 3) |
| **Pipe Index** | **Unique identifier (0-757) for the specific pipe to use** |
| **Actual Length (ft)** | Exact length of this pipe from inventory |
| **Rounded Length (ft)** | Length rounded to 0.1' (used in optimization) |
| **Total Length** | Combined length of all segments (first row only) |
| **Waste (ft)** | Excess over 100' (first row only) |
| **Welds** | Number of welds needed (first row only) |

### Sheet 3: Unused Pipes

Lists the 8 pipes that are not used in the solution:

| Column | Description |
|--------|-------------|
| **Pipe Index** | The specific pipe identifier (0-757) |
| **Length (ft)** | The length of the unused pipe |

---

## Verifying No Duplicate Pipe Usage

To confirm each pipe is used at most once:

1. Open `pipe_optimization_V2_VERIFIED.xlsx`
2. Go to **Pile Details** sheet
3. Select the **Pipe Index** column
4. Sort A-Z (ascending)
5. Scroll through - **each number 0-757 should appear at most once**
6. Cross-reference with **Unused Pipes** sheet for the 8 missing indices

### Expected Result
- 750 unique pipe indices in Pile Details
- 8 pipe indices in Unused Pipes
- Total: 758 (matches inventory)
- No index appears twice

---

## How to Use the Cutting Plan

### Step 1: Label Your Pipe Inventory

Before starting, label each physical pipe with its index (0-757):

1. Open your original inventory file (`pipe_lengths_clean.csv`)
2. Row 1 = Pipe Index 0, Row 2 = Pipe Index 1, etc.
3. Physically tag or mark each pipe with its index number

**Example:**
```
Index 0:  First pipe in CSV  -> Tag as "P0"
Index 1:  Second pipe in CSV -> Tag as "P1"
...
Index 757: Last pipe in CSV  -> Tag as "P757"
```

### Step 2: Read the Pile Details Sheet

Each pile spans 2-3 rows in the spreadsheet. Example:

| Pile Number | Segment | Pipe Index | Actual Length | Rounded Length | Total | Waste | Welds |
|-------------|---------|------------|---------------|----------------|-------|-------|-------|
| 1 | 1 | 757 | 51.6 | 51.6 | 100.1 | 0.1 | 1 |
| 1 | 2 | 617 | 48.5 | 48.5 | | | |
| 2 | 1 | 756 | 51.1 | 51.1 | 100.0 | 0.0 | 1 |
| 2 | 2 | 621 | 48.9 | 48.9 | | | |

**Reading Pile 1:**
- Get pipe #757 (51.6') and pipe #617 (48.5')
- Weld them together
- Total: 100.1', Waste: 0.1', 1 weld

### Step 3: Assembly Workflow

**Option A: Sequential Assembly**
1. Start at Pile 1
2. Locate pipes by their index numbers
3. Weld segments together
4. Mark pipes as used
5. Continue to Pile 262

**Option B: Batch by Similar Patterns**
1. Filter the spreadsheet to find repeated patterns
2. Example: Many piles use 50.3' + 50.2' pipes
3. Gather all pipes matching those patterns
4. Assemble in batches for welding efficiency

**Option C: Prioritize by Weld Count**
1. Sort by "Welds" column
2. Build all 1-weld piles first (36 piles - simpler)
3. Then build 2-weld piles (226 piles)

### Step 4: Track Progress

Add a "Status" column to track completion:
1. Insert new column after "Welds"
2. Mark each pile: "Pending" / "In Progress" / "Complete"
3. Use conditional formatting for visual status

---

## Working with the Excel File

### Filtering by Pipe Index

To find which pile uses a specific pipe:
1. Click filter dropdown on "Pipe Index" column
2. Enter the pipe number (e.g., 682)
3. Shows which pile that pipe belongs to

### Sorting Options

| Sort By | Purpose |
|---------|---------|
| Pile Number | Sequential assembly order |
| Pipe Index | Verify no duplicates |
| Actual Length | Group similar-length pipes |
| Waste | See most/least efficient piles |
| Welds | Prioritize simpler assemblies |

### Creating a Pick List

To generate a list of pipes needed for specific piles:
1. Filter Pile Details by desired pile numbers
2. Copy the Pipe Index column
3. Sort and use as a picking list for inventory

### Printing for Shop Floor

For physical use:
1. Set print area to Pile Details sheet
2. Page Layout > Print Titles > Repeat header row
3. Consider printing in landscape
4. Include Pipe Index column prominently

---

## Understanding Unused Pipes

The **Unused Pipes** sheet lists 8 pipes not included in any pile:

| Pipe Index | Length (ft) | Why Unused |
|------------|-------------|------------|
| 44 | 20.6' | No valid combination reaches 100-120' |
| 78 | 22.0' | No valid combination reaches 100-120' |
| 192 | 26.8' | No valid combination reaches 100-120' |
| 236 | 28.1' | No valid combination reaches 100-120' |
| 409 | 35.0' | No valid combination reaches 100-120' |
| 433 | 36.1' | No valid combination reaches 100-120' |
| 471 | 38.1' | No valid combination reaches 100-120' |
| 537 | 41.4' | No valid combination reaches 100-120' |

These pipes cannot be combined with remaining available inventory to form valid 100-120' piles. They can be:
- Saved for future projects
- Used for other purposes
- Combined with new inventory later

---

## Troubleshooting

### "I can't find pipe #X in my inventory"

The Pipe Index corresponds to the row number (0-indexed) in your original `pipe_lengths_clean.csv` file:
- Pipe Index 0 = Row 1 in CSV
- Pipe Index 100 = Row 101 in CSV

### "The actual length doesn't match my measurement"

Small variations are expected. The optimizer uses lengths rounded to 0.1'. A pipe listed as 50.3' could physically measure 50.25' to 50.34'.

### "A pipe is damaged/missing"

If a specific pipe is unavailable:
1. Find a substitute pipe of similar length (within 0.1')
2. The pile total will change slightly but should remain valid (100-120')
3. Document the substitution

### "I want to verify the math myself"

For any pile:
1. Look up each Pipe Index in the original CSV to get exact lengths
2. Sum the Actual Length values
3. Confirm total is between 100.0' and 120.0'
4. Waste = Total - 100.0'

---

## Quick Reference

| Statistic | Value |
|-----------|-------|
| Total Piles | 262 |
| 2-pipe piles (1 weld) | 36 |
| 3-pipe piles (2 welds) | 226 |
| **Total welds needed** | **488** |
| Pipes consumed | 750 |
| Pipes leftover | 8 |
| Average waste | 0.08' per pile |
| Total waste | 20.2' |

### Weld Calculation
- 36 piles × 1 weld = 36 welds
- 226 piles × 2 welds = 452 welds
- **Total: 488 welds**

---

## Files Reference

| File | Purpose |
|------|---------|
| `pipe_optimization_V2_VERIFIED.xlsx` | Main output with pipe indices |
| `pipe_lengths_clean.csv` | Original inventory (index reference) |
| `pipe_optimizer_v2.py` | Optimization script |
| `METHODS.md` | Technical methodology documentation |
| `INSTRUCTION_GUIDE.md` | This file |

---

## Contact & Support

For questions about:
- **Optimization methodology**: See `METHODS.md`
- **Re-running with different parameters**: Modify `pipe_optimizer_v2.py`
- **Verifying results**: Use the Pipe Index column to audit
