# Kernel Goodness-of-Fit Library (kgof)

## Source and Attribution

This library is adapted from the **kernel-gof** repository by Jitkrittum et al. (2017):
- **Original Source**: https://github.com/wittawatj/kernel-gof
- **Paper**: "A Linear-Time Kernel Goodness-of-Fit Test" (NIPS 2017)
- **Authors**: Wittawat Jitkrittum, Wenkai Xu, Zoltan Szabo, Kenji Fukumizu, Arthur Gretton

This library was also included as a dependency in the Nyström-KSD repository (`nystroem-ksd/lib/kernel-gof/`).

## What It Provides

The `kgof` library provides:
- **Base classes** for kernel-based goodness-of-fit testing
- **Kernel implementations** (Gaussian, IMQ, etc.)
- **Density classes** for representing probability distributions
- **Data structures** for handling test samples
- **Utilities** for kernel computations and testing

## How It Connects to Our Project

### Used By

1. **`src/goftest.py`**:
   - Imports `GofTest` base class and `bootstrapper_rademacher` from `kgof.goftest`
   - Used to create the `NystroemKSD` base class

2. **`src/renyiksd.py`**:
   - Our `RenyiNystroemKSD` class extends `NystroemKSD`, which inherits from `kgof.goftest.GofTest`
   - Uses kernel and density classes from `kgof` for testing

3. **`src/test_renyiksd.py`**:
   - Uses `kgof.kernel.KGauss` for testing
   - Uses `kgof.density.IsotropicNormal` for test distributions
   - Uses `kgof.goftest.KernelSteinTest` as reference implementation

### Our Modifications

- **No direct modifications** to the `kgof/` library itself
- We use it as a **reference and base** for our implementations
- Our adaptations are in `src/renyiksd.py` and the notebooks, which extend the base functionality

## Key Components

- **`goftest.py`**: Base classes for goodness-of-fit tests
- **`kernel.py`**: Kernel function implementations (Gaussian, IMQ, etc.)
- **`density.py`**: Probability density representations
- **`data.py`**: Data structures for test samples
- **`util.py`**: Utility functions for kernel computations

## Testing

The library includes its own test suite in `kgof/test/`. We also have tests in `src/test_renyiksd.py` that use this library as a reference.

## Note

This is an **adapted library**, not our original work. We use it as a foundation for our Rényi-Nyström KSD implementation, which extends the base functionality for GAN training.

