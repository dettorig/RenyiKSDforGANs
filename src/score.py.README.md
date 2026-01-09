# Score Function Utilities (score.py)

## Overview

This file contains **our original implementation** of utilities for computing DDPM (Denoising Diffusion Probabilistic Model) score functions. These utilities are essential for KSD computation in our GAN training.

## What It Does

### Main Functions

1. **`score_fn(x, sigma=0.1)`**:
   - Computes the score function ∇ log p_t(x_t) using a pretrained DDPM
   - Maps noise level σ to diffusion timestep t
   - Returns score and timestep
   - **Core function** used in KSD computation

2. **`map_sigma_to_t(sigma)`**:
   - Maps noise level σ to the corresponding diffusion timestep t
   - Uses the DDPM scheduler's alpha_cumprod values
   - Essential for correct score function computation

3. **`score_from_noise_pred(eps_pred, sigma)`**:
   - Converts noise prediction to score function
   - Formula: score = -(1/σ) * eps_pred
   - Standard DDPM score function relationship

4. **`load_cifar_batch(batch_size=8)`**:
   - Utility for loading CIFAR-10 batches
   - Handles data normalization to [-1, 1] range
   - Used for testing and validation

## How It Connects to the Project

### Relationship to Notebooks

- **`src/notebooks/ksd_gan_cifar10_training.ipynb`**:
  - The notebooks contain **standalone PyTorch implementations** of score functions
  - The notebooks implement `score_fn_xt()` inline (PyTorch version)
  - This file (`score.py`) is a **standalone reference implementation**
  - The notebooks don't import from this file but implement similar functionality
  - This file serves as a reference and can be used independently for testing

### Role in KSD-GAN Training

1. **Score Function Computation**:
   - KSD requires score functions ∇ log p_t(x_t) for the Stein kernel
   - We use pretrained DDPM to provide these score functions
   - This file provides the interface to the DDPM model

2. **Integration with Training**:
   - During training, we compute scores at different timesteps t
   - These scores are used in the Stein kernel h_p(x, y)
   - Essential for the KSD loss computation

## Key Features

1. **Pretrained Model Integration**:
   - Uses Hugging Face's `google/ddpm-cifar10-32` model
   - No training required - uses pretrained weights
   - Efficient inference-only usage

2. **Timestep Mapping**:
   - Correctly maps between noise levels and diffusion timesteps
   - Handles the DDPM scheduler's alpha_cumprod schedule
   - Ensures accurate score function computation

3. **Error Handling**:
   - Includes fallback for data loading issues
   - Provides informative error messages
   - Test code included in `__main__` block

## Testing

The file includes test code that:
- Loads a CIFAR-10 batch
- Computes score functions at different noise levels
- Validates score function norms
- Tests noise prediction accuracy

Run tests with:
```bash
python src/score.py
```

## Note

This is **our original work**, created specifically for this project. It provides a clean interface to pretrained DDPM models for score function computation, which is essential for our KSD-GAN training approach.

