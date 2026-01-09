# Rényi KSD Implementation (renyiksd.py)

## Overview

This file contains **our original implementation** of Rényi-based Nyström-KSD, adapted from the base Nyström-KSD implementation for use in GAN training.

## What It Does

### Main Components

1. **`select_renyi_landmarks(X, k, m)`**:
   - Greedy algorithm for selecting landmarks using Rényi entropy criterion
   - Minimizes the quadratic energy 1^T K 1 over the chosen subset
   - Returns indices of m landmarks selected from n samples
   - **This is our original contribution** - Rényi-based landmark selection

2. **`RenyiNystroemKSD` class**:
   - Extends the base `NystroemKSD` class from `src/goftest.py`
   - Uses Rényi landmark selection instead of uniform sampling
   - Implements the `compute_stat()` method with Rényi landmarks
   - **This is our adaptation** of the base Nyström-KSD

## How It Connects to the Project

### Relationship to Notebooks

- **`src/notebooks/ksd_gan_cifar10_training.ipynb`**:
  - The notebooks contain **standalone PyTorch implementations** of Rényi-Nyström KSD
  - This file (`renyiksd.py`) is a **NumPy-based reference implementation**
  - The notebooks don't import from this file but implement similar functionality
  - This file serves as a reference and can be used for testing/validation

- **`src/test_renyiksd.py`**:
  - Unit tests verify our implementation against reference implementations
  - Tests landmark selection and KSD computation
  - Validates correctness of the Rényi landmark selection algorithm

### Relationship to Base Implementation

- **Base**: `NystroemKSD` in `src/goftest.py` (from Kalinke et al., AISTATS 2025)
- **Our Extension**: `RenyiNystroemKSD` adds Rényi-based landmark selection
- **Key Difference**: Uses informative landmark selection instead of uniform sampling

## Key Features

1. **Rényi Entropy Criterion**:
   - Selects landmarks that minimize quadratic Rényi energy
   - Ensures diverse and informative landmark selection
   - More effective than uniform sampling for heterogeneous data

2. **Compatible Interface**:
   - Matches the base `NystroemKSD` interface
   - Can be used as a drop-in replacement
   - Maintains O(mn + m³) computational complexity

## Testing

Run tests with:
```bash
python src/test_renyiksd.py
```

Tests verify:
- Landmark selection correctness
- KSD statistic computation
- Goodness-of-fit test performance

## Note

This is **our original work**, adapted from the base Nyström-KSD implementation. The notebooks contain a PyTorch version optimized for GAN training, while this file provides a NumPy-based reference implementation.

