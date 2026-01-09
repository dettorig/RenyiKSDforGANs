# Random Feature Stein Discrepancies (rfsd)

## Source and Attribution

This library is adapted from the **random-feature-stein-discrepancies** repository by Huggins and Mackey (2018):
- **Original Source**: https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/
- **Paper**: "Random Feature Stein Discrepancies" (NIPS 2018)
- **Authors**: Jonathan H. Huggins, Lester Mackey

This library was also included as a dependency in the Nyström-KSD repository (`nystroem-ksd/lib/random-feature-stein-discrepancies/`).

## What It Provides

The `rfsd` library provides:
- **Random feature methods** for Stein discrepancy computation
- **Efficient approximations** of kernel Stein discrepancies
- **Divergence classes** for measuring distribution differences
- **Kernel implementations** optimized for random features
- **Inference utilities** for MCMC and sampling

## How It Connects to Our Project

### Used By

1. **`src/goftest.py`**:
   - Imports `Divergence` class from `rfsd.rfsd`
   - The `NystroemKSD` class inherits from both `GofTest` and `Divergence`

2. **Our Implementation**:
   - We use concepts from this library for computational efficiency
   - The random feature approach inspired our feature-space implementations
   - Used as reference for efficient KSD computation patterns

### Our Modifications

- **No direct modifications** to the `rfsd/` library itself
- We use it as a **reference** for efficient computation methods
- Our adaptations focus on **feature-space Stein kernels** in the notebooks, which draw inspiration from random feature methods

## Key Components

- **`rfsd.py`**: Main divergence and KSD classes
- **`kernel.py`**: Random feature kernel implementations
- **`distributions.py`**: Distribution classes for testing
- **`inference.py`**: MCMC and sampling utilities
- **`experiments/`**: Experimental code (not directly used in our project)

## Connection to Notebooks

While we don't directly import from `rfsd` in our notebooks, the concepts from this library influenced our approach:
- **Feature-space kernels**: Similar to random feature methods
- **Computational efficiency**: Both aim to reduce O(n²) complexity
- **Stein discrepancy computation**: Shared mathematical foundation

## Note

This is an **adapted library**, not our original work. We use it as a reference for efficient computation methods, but our main implementation is in the notebooks using PyTorch and our own feature-space approach.

