# Goodness-of-Fit Test Base (goftest.py)

## Overview

This file contains the base `NystroemKSD` class, which is the foundation for our Rényi-Nyström KSD implementation.

## What It Does

### Main Class

**`NystroemKSD`**:
- Base class for Nyström-accelerated Kernel Stein Discrepancy
- Inherits from both `GofTest` (from `kgof`) and `Divergence` (from `rfsd`)
- Implements the core Nyström approximation algorithm
- **Source**: This is from the original Nyström-KSD repository (Kalinke et al., AISTATS 2025)

## How It Connects to the Project

### Used By

1. **`src/renyiksd.py`**:
   - Our `RenyiNystroemKSD` class extends `NystroemKSD`
   - Adds Rényi-based landmark selection on top of the base implementation
   - This is where we add our original contribution

2. **Notebooks**:
   - The notebooks implement their own PyTorch version
   - This file provides the NumPy-based reference implementation
   - Used for understanding and validation

## Key Features

1. **Nyström Approximation**:
   - Reduces computational complexity from O(n²) to O(mn + m³)
   - Uses m << n landmarks for efficiency
   - Projects onto subspace spanned by landmarks

2. **Goodness-of-Fit Testing**:
   - Can perform hypothesis tests (H0: sample follows p)
   - Uses bootstrap for null distribution
   - Provides p-values and test statistics

## Note

This is **not our original work** - it's from the Nyström-KSD repository. We extend it in `src/renyiksd.py` with Rényi landmark selection, which is our contribution.

