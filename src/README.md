# Source Code Directory

This directory contains all the source code for the Rényi-Nyström KSD-GAN project.

## Directory Structure

```
src/
├── notebooks/              # Jupyter notebooks with training and evaluation code
├── kgof/                   # Kernel goodness-of-fit library (adapted)
├── rfsd/                   # Random feature Stein discrepancies (adapted)
├── renyiksd.py            # Our Rényi KSD implementation (original work)
├── score.py               # Our score function utilities (original work)
├── goftest.py             # Goodness-of-fit test base classes
└── test_renyiksd.py       # Unit tests for Rényi KSD
```

## Key Files

### Our Original Implementations

- **`renyiksd.py`**: Our adaptation of Nyström-KSD with Rényi-based landmark selection
  - Used by the notebooks for landmark selection during GAN training
  - See `src/renyiksd.py` README for details

- **`score.py`**: Our implementation for computing DDPM score functions
  - Provides utilities for working with pretrained diffusion models
  - Used in notebooks to compute score functions for KSD computation
  - See `src/score.py` README for details

### Supporting Libraries (Adapted)

- **`kgof/`**: Kernel goodness-of-fit library
  - Adapted from Jitkrittum et al. (2017)
  - Provides base classes and utilities for kernel-based testing
  - See `src/kgof/README.md` for details

- **`rfsd/`**: Random feature Stein discrepancies
  - Adapted from Huggins and Mackey (2018)
  - Used for computational efficiency improvements
  - See `src/rfsd/README.md` for details

- **`goftest.py`**: Base classes for goodness-of-fit testing
  - Contains `NystroemKSD` base class from the original repository
  - Extended by our `RenyiNystroemKSD` in `renyiksd.py`

## Connection to Notebooks

The main training and evaluation code is in `src/notebooks/ksd_gan_cifar10_training.ipynb`, which:

1. **Implements everything inline**: The notebooks contain complete PyTorch implementations of:
   - `RenyiNystroemKSD` class (PyTorch version)
   - `select_renyi_landmarks_stein()` function
   - `score_fn_xt()` for DDPM score functions
   - Feature maps (RandomProjection, FrozenResNetFeatureMap)
   - Full training loops

2. **Standalone reference files**: 
   - `renyiksd.py` and `score.py` are NumPy-based reference implementations
   - Can be used independently for testing or as utilities
   - The notebooks don't import from these files but implement similar functionality in PyTorch

3. **Base libraries**:
   - `kgof/` and `rfsd/` provide base classes and concepts
   - The notebooks implement PyTorch versions inspired by these libraries
   - `goftest.py` contains the base `NystroemKSD` class from the original repository

## Testing

Run unit tests with:
```bash
python src/test_renyiksd.py
```

This tests the Rényi KSD implementation against reference implementations.

