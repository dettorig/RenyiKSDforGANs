# Notebooks Directory

This directory contains all Jupyter notebooks for training, evaluation, and experimentation.

## Notebooks

### Main Training Notebook

**`ksd_gan_cifar10_training.ipynb`** - Primary implementation and training code

This is the main notebook containing:
- **Complete KSD-GAN training implementation** - Our original contribution
- **Standard GAN baseline** - For comparison
- **Feature map implementations**:
  - Random projection feature map
  - ResNet18 feature map (alternative)
- **Training loops** for both KSD-GAN and standard GAN
- **Evaluation code**:
  - FID and KID metric computation
  - KSD value plotting across timesteps
  - Sample visualization
- **All plots and figures** used in the project

**How it connects to the project:**
- This is where all our original GAN training code lives
- Contains **complete standalone PyTorch implementations** of:
  - Rényi-Nyström KSD with feature-space Stein kernels
  - Rényi landmark selection (`select_renyi_landmarks_stein()`)
  - DDPM score functions (`score_fn_xt()`)
  - Feature maps (RandomProjection, FrozenResNetFeatureMap)
- Implements feature-space Stein kernels for image data
- Contains the complete training pipeline from data loading to evaluation
- **Note**: The notebooks are self-contained. `src/renyiksd.py` and `src/score.py` are reference implementations that can be used independently.

### Backup Notebook

**`ksd_gan_cifar10_backup.ipynb`** - Backup/alternative implementation

- Similar structure to the main training notebook
- Contains alternative implementations and experiments
- Useful for reference and comparison

### Additional Experiments

**`RenyiKSDtorch.ipynb`** - Additional experiments and prototyping

- Early experiments and development work
- Prototyping different approaches
- Additional analysis and visualizations

## Running the Notebooks

1. **Prerequisites**: Install all dependencies (see main README.md)
2. **Data**: CIFAR-10 will be automatically downloaded on first run
3. **GPU**: Recommended for training (can take 2-4 hours)
4. **Execution**: Run cells sequentially or use "Run All"

## Key Sections in Main Notebook

1. **Setup and Imports** - Environment configuration
2. **DDPM Score Function Loading** - Pretrained model setup
3. **Data Loading** - CIFAR-10 dataset preparation
4. **Feature Maps** - Random projection and ResNet implementations
5. **KSD Implementation** - Nyström-KSD with Rényi landmarks
6. **Generator Architecture** - Custom upsampling generator
7. **Training Loops** - Both KSD-GAN and standard GAN
8. **Evaluation** - Metrics and visualizations

## Output

The notebooks generate:
- Trained generator models (in memory, can be saved)
- Sample visualizations
- FID/KID metric values
- KSD plots across timesteps
- Training loss curves

All results are displayed inline in the notebook cells.

