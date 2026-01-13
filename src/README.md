# Source Code Directory

This directory contains all the source code for the Rényi-Nyström KSD-GAN project.

## Development Process: Two-Phase Approach

### Phase 1: Preliminary Testing (This Directory)

**Before implementing the GAN**, we first validated that our Rényi-based landmark selection method could effectively distinguish distributions compared to the standard Nyström approach (which uses random landmark selection).

**What we tested:**
- **Base Nyström-KSD** (`goftest.py`): Uses random landmark selection (`rng.choice()`)
- **Our Rényi-Nyström KSD** (`renyiksd.py`): Uses Rényi-based landmark selection that minimizes quadratic Rényi energy
- **Goodness-of-fit testing** (`test_renyiksd.py`): Verified that Rényi selection can distinguish distributions

**Test Results:**
- ✅ All tests passed
- ✅ Rényi landmark selection successfully distinguishes distributions (H0 vs H1 hypothesis testing)
- ✅ Validated that our method works before moving to GAN implementation

**Key Insight:** This preliminary testing phase proved that Rényi landmark selection is effective for distribution discrimination, giving us confidence to proceed with GAN training.

### Phase 2: GAN Implementation (Notebooks)

After validating the approach, we implemented the full GAN training pipeline in the notebooks (`src/notebooks/`). The notebooks contain **standalone PyTorch implementations** that don't import from this directory, but build on the concepts validated here.

## Directory Structure

```
src/
├── notebooks/              # Phase 2: GAN training and evaluation (PyTorch)
├── kgof/                   # Kernel goodness-of-fit library (adapted)
├── rfsd/                   # Random feature Stein discrepancies (adapted)
├── renyiksd.py            # Phase 1: Our Rényi KSD implementation (NumPy)
├── goftest.py             # Base Nyström-KSD with random selection
└── test_renyiksd.py       # Phase 1: Tests comparing Rényi vs random selection
```

## Key Files

### Phase 1: Distribution Discrimination Testing

- **`goftest.py`**: Contains `NystroemKSD` base class
  - Uses **random landmark selection** (`rng.choice()` at line 100)
  - This is the baseline we compared against

- **`renyiksd.py`**: Our Rényi-Nyström KSD implementation
  - Extends `NystroemKSD` with Rényi-based landmark selection
  - Uses `select_renyi_landmarks()` to choose informative landmarks
  - **This is our original contribution** - proving Rényi selection works

- **`test_renyiksd.py`**: Validation tests
  - `test_renyi_h_p_matches_reference_gauss()`: Verifies Stein kernel computation
  - `test_renyi_stat_close_to_quadratic()`: Validates KSD statistic computation
  - `test_renyi_h0_and_h1()`: **Key test** - Verifies Rényi selection can distinguish distributions
    - H0: Sample follows distribution p → should NOT reject (p-value > 0.2)
    - H1: Sample from shifted distribution → should reject (p-value < 0.05)
  - **All tests passed** ✅

### Supporting Libraries (Adapted)

- **`kgof/`**: Kernel goodness-of-fit library
  - Adapted from Jitkrittum et al. (2017)
  - Provides base classes and utilities for kernel-based testing
  - See `src/kgof/README.md` for details

- **`rfsd/`**: Random feature Stein discrepancies
  - Adapted from Huggins and Mackey (2018)
  - Used for computational efficiency improvements
  - See `src/rfsd/README.md` for details

## Connection to Notebooks (Phase 2)

The notebooks in `src/notebooks/` contain the **GAN implementation phase**:

1. **Do NOT import from this directory**: The notebooks implement everything from scratch in PyTorch
2. **Build on validated concepts**: The Rényi landmark selection concept validated here is implemented in PyTorch for image data
3. **Different use case**: 
   - Phase 1 (this code): Goodness-of-fit testing on simple distributions
   - Phase 2 (notebooks): GAN training on CIFAR-10 images with feature-space Stein kernels

## Running the Tests

To verify the Phase 1 validation:

```bash
python src/test_renyiksd.py
```

This runs all three tests:
1. Stein kernel computation correctness
2. KSD statistic accuracy
3. **Distribution discrimination** (H0/H1 hypothesis testing)

All tests should pass, confirming that Rényi landmark selection effectively distinguishes distributions compared to random selection.

## Summary

This directory represents **Phase 1** of our project: validating that Rényi-based landmark selection works for distribution discrimination. Once validated, we moved to **Phase 2** (notebooks) to implement the full GAN training pipeline. The notebooks don't use this code directly, but the concepts and validation from this phase informed the GAN implementation.
