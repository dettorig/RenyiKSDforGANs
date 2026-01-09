# Accelerated Kernel Stein Discrepancy with Rényi Landmark Selection for Stable and Efficient GAN Training

**Authors:** Michael Carlo, Giovanni Dettori, Ryan Gumsheimer  
**Professor:** Austin J. Stromme

**Date:** November 2025

## Repository Link

**GitHub Repository:** [https://github.com/dettorig/RenyiKSDforGANs](https://github.com/dettorig/RenyiKSDforGANs)

All code, data processing scripts, and notebooks are available in this repository. The repository includes everything needed to reproduce the results presented in this project.

## Attribution and Acknowledgments

**Code Development:** This code was developed with assistance from ChatGPT for implementation guidance. The core ideas and adaptations are our original work, with ChatGPT helping with implementation details and debugging.

**Base Implementation:** This project builds upon the Nyström-KSD implementation from Kalinke, Szabó, and Sriperumbudur (AISTATS 2025). The original repository is available at [https://github.com/FlopsKa/nystroem-ksd](https://github.com/FlopsKa/nystroem-ksd) and is also included in the `nystroem-ksd/` directory. Our adaptations for GAN training are detailed in the "Code Description and Attribution" section below.



## Project Overview

This project investigates replacing the classical adversarial discriminator in GAN training with a kernel-based distance metric, specifically Kernel Stein Discrepancy (KSD) with Rényi-based landmark selection. The goal is to assess whether a kernelized objective can provide an alternative that improves training stability and efficiency without compromising sample quality.

## Repository Structure

```
RenyiKSDforGANs/
├── README.md                          # This file
├── PROPOSAL.md                        # Project proposal document
├── src/
│   ├── notebooks/
│   │   ├── ksd_gan_cifar10_training.ipynb    # Main training notebook (our implementation)
│   │   ├── ksd_gan_cifar10_backup.ipynb      # Backup/alternative implementation (our implementation)
│   │   └── RenyiKSDtorch.ipynb               # Additional experiments (our implementation)
│   ├── renyiksd.py                    # Our Rényi KSD adaptation (original implementation)
│   ├── score.py                       # Our score function utilities (original implementation)
│   ├── goftest.py                     # Goodness-of-fit testing
│   ├── test_renyiksd.py               # Unit tests
│   ├── kgof/                          # Kernel goodness-of-fit library
│   └── rfsd/                          # Random feature Stein discrepancies
└── nystroem-ksd/                      # Original Nyström-KSD repo (Kalinke et al., AISTATS 2025)
```

## Data and Code Implementation

As part of our research, we proposed replicating the GAN architecture as done in Goodfellow et al. (2014) on benchmark datasets and comparing it with our proposed KSD approach. Our implementation focused on two datasets:

### Datasets


**CIFAR-10 Dataset** (Primary Benchmark):
   - 60,000 32×32 color images in 10 classes
   - 6,000 images per class
   - Split into 50,000 training images and 10,000 test images
   - Available on Kaggle: https://www.kaggle.com/datasets/ayush1220/cifar10
   - Used as the main benchmark for real-world application scenarios

## Main Results

### Phase 1: Distribution Discrimination Testing

**Motivation:** Standard Kernel Stein Discrepancy (KSD) computation has O(n²) complexity, which becomes prohibitively expensive for large datasets. We use the **Nyström acceleration method** to reduce this to O(m²) where m << n, by selecting only m landmark points instead of using all n points.

**Our Contribution:** Instead of randomly selecting landmarks (as in the base Nyström-KSD implementation), we choose landmarks that **maximize entropy** to select the most informative points. This ensures we get the best approximation quality with fewer landmarks.

**What we tested:**
- **Base Nyström-KSD** (`src/goftest.py`): Uses random landmark selection (`rng.choice()`) - reduces complexity but landmarks may not be optimal
- **Our Rényi-Nyström KSD** (`src/renyiksd.py`): Uses entropy-maximizing landmark selection (minimizes quadratic Rényi energy `1^T K 1`) - same O(m²) complexity but with more informative landmarks
- **Goodness-of-fit testing** (`src/test_renyiksd.py`): Verified that our entropy-based selection can effectively distinguish distributions

**Test Results:**
- ✅ All tests passed
- ✅ Rényi landmark selection successfully distinguishes distributions (H0 vs H1 hypothesis testing)
- ✅ Validated that our method works before moving to GAN implementation
- ✅ Achieved O(m²) complexity instead of O(n²) while maintaining discrimination power

**Key Insight:** By selecting the most informative landmarks (maximizing entropy), we can achieve the same distribution discrimination capability with far fewer points, making KSD computation feasible for large-scale applications like GAN training. See `src/README.md` for more details on Phase 1 testing.

### Phase 2: Quantitative Evaluation (CIFAR-10)

After validating our Rényi landmark selection approach in Phase 1, we implemented the full GAN training pipeline in the notebooks. We evaluated both KSD-GAN and standard GAN baselines using Fréchet Inception Distance (FID) and Kernel Inception Distance (KID):

| Method | FID | KID |
|--------|-----|-----|
| **Standard GAN** | **104.02** | **0.0669** |
| **KSD-GAN (Random Projection)** | 304.81 | 0.3159 |
| **KSD-GAN (ResNet Features)** | Similar range | Similar range |

**Key Findings:**
- The standard GAN baseline achieved significantly better quantitative metrics (lower FID and KID indicate better quality)
- KSD-GAN training showed higher loss values and more instability, particularly at low noise levels (small t values)
- The Rényi-Nyström KSD implementation successfully reduced computational complexity from O(n²) to O(m²) where m << n
- Training stability was improved through careful gradient flow management (gradients only through fake samples)

### Qualitative Observations & Visual Results
- **Training Stability:** KSD-GAN training exhibited occasional loss spikes, especially at low diffusion timesteps (t < 100)
- **Sample Quality:** Visual inspection showed that KSD-GAN samples were generated but with lower fidelity compared to standard GAN
- **Computational Efficiency:** The Nyström approximation with Rényi landmark selection successfully reduced computational cost

To evaluate the generative performance, we compared visual outputs across three configurations. While KSD-based methods successfully generated recognizable image patterns, the adversarial baseline remained superior in terms of high-frequency detail.

| Training Method | Sample Generation | Qualitative Analysis |
| :--- | :---: | :--- |
| **KSD-GAN (Random Projection)** | <img src="https://github.com/user-attachments/assets/6da34f49-5f55-46cc-bd25-67622b083d92" width="300" /> | **Primary Result:** Successfully captures the global color distribution and basic shapes of CIFAR-10 classes. Our most stable KSD variant. |
| **KSD-GAN (ResNet Features)** | <img src="https://github.com/user-attachments/assets/fc9b2c15-0491-416c-a458-0bf94e1580f8" width="300" /> | **Secondary Result:** Features from pre-trained ResNet-18 (after 3h training) resulted in lower fidelity and less structural coherence compared to random projection. |
| **Standard GAN (Baseline)** | <img src="https://github.com/user-attachments/assets/37beb60a-24d1-4dcd-9544-8331eb9ecec6" width="300" /> | **Reference:** The adversarial discriminator produces the sharpest results, highlighting the performance gap our KSD approach aims to bridge. |

**Summary of Findings:**
* **Stability:** KSD-GAN training exhibited occasional loss spikes, especially at low diffusion timesteps ($t < 100$).
* **Fidelity:** Visual inspection confirms KSD-GAN samples are recognizable but have lower fidelity compared to standard GANs.
* **Efficiency:** The Nyström approximation with Rényi landmark selection successfully reduced computational cost, making kernelized GAN training more feasible.

### Implementation Approach

**Code Sharing and Version Control:**
- All code is version-controlled using GitHub to facilitate collaboration, branching, and as part of project deliverables
- The repository structure allows for easy navigation and reproducibility

**Computational Resources:**
- Primary development and training conducted using Google Colab for GPU access
- GPU acceleration (CUDA) was essential for efficient training of both KSD-GAN and standard GAN models

### Training Configuration

**KSD-GAN Training Parameters:**
- **Training Steps:** 1,000 iterations (for both random projection and ResNet feature implementations)
- **Batch Size:** 64 (random projection) / 32 (ResNet features)
- **Learning Rate:** 2e-4 with Adam optimizer (β₁=0.5, β₂=0.999)
- **Feature Map Dimensions:** 512-dimensional random projection
- **Landmark Selection:** Rényi-based selection with m ≈ 4√n landmarks
- **MMD Stabilizer Weight:** λ = 0.50
- **Timestep Sampling:** Single timestep per iteration (K_T = 1)
- **Gradient Clipping:** 50.0 (random projection) / 5.0 (ResNet features)

**Standard GAN Baseline Parameters:**
- **Training Steps:** 1,000 iterations
- **Batch Size:** 64
- **Learning Rate:** 2e-4 for both generator and discriminator
- **Discriminator Steps:** 1 step per generator step
- **Label Smoothing:** 0.9 for real labels

**Training Time Estimates:**
- **KSD-GAN Training:** ~2-4 hours on a single GPU (depending on hardware and feature map choice)
- **Standard GAN Training:** ~ minutes on a single GPU
- **Evaluation (FID/KID):** ~ minutes for 5,000 samples

### What We Tried

During the project development, we experimented with several approaches and configurations:

1. **Feature Map Variants:**
   - **Random Projection:** Fixed random linear projection with orthogonal scaling (primary implementation)
   - **ResNet Features:** Frozen ResNet18 layer2 features (ImageNet-pretrained) as an alternative
   - Both approaches were tested to compare feature representation quality

2. **Training Strategies:**
   - Multi-timestep mixture sampling (initially planned but simplified to single timestep)
   - Different MMD stabilizer weights (settled on λ = 0.50)
   - Various gradient clipping thresholds to manage training stability

3. **Architectural Choices:**
   - Upsampling-based generator to reduce checkerboard artifacts
   - Generator output in [-1, 1] range with tanh activation
   - DCGAN-style discriminator for baseline comparison

4. **Stability Improvements:**
   - Feature normalization computed from real data only (per timestep)
   - Landmark selection performed without gradients
   - Careful gradient flow control (only through fake samples)
   - Ridge regularization in Nyström-KSD computation

5. **Evaluation Metrics:**
   - Fréchet Inception Distance (FID) using Inception v3 features
   - Kernel Inception Distance (KID) for additional quality assessment
   - One-sample KSD values across different diffusion timesteps

### Challenges Encountered

- **Training Instability:** Initial implementations showed high loss variance, especially at low noise levels (small t values)
- **Computational Cost:** Even with Nyström approximation, KSD computation is more expensive than standard discriminator
- **Hyperparameter Sensitivity:** Required careful tuning of bandwidth (median heuristic), ridge parameter, and landmark count
- **Sample Quality:** Quantitative metrics indicated standard GAN performed better, suggesting need for further optimization

This iterative experimentation process helped us understand the trade-offs between computational efficiency, training stability, and sample quality in KSD-based GAN training.

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training, but not strictly required)
- PyTorch 1.12+
- Jupyter Notebook or Google Colab

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dettorig/RenyiKSDforGANs
   cd RenyiKSDforGANs
   ```

2. **Install dependencies:**

   **Option 1: Using requirements.txt (Recommended)**
   ```bash
   pip install -r requirements.txt
   ```

   **Option 2: Manual installation**
   ```bash
   pip install torch torchvision
   pip install diffusers
   pip install numpy scipy matplotlib
   pip install scikit-learn
   pip install autograd future past  # Required for Phase 1 testing code
   pip install jupyter  # For running notebooks
   ```

   The `requirements.txt` file includes all dependencies needed for both Phase 1 (testing) and Phase 2 (GAN training) of the project.

### Data Setup

The CIFAR-10 dataset will be automatically downloaded when running the notebooks. The dataset is available from:
- Official: https://www.cs.toronto.edu/~kriz/cifar.html
- Kaggle: https://www.kaggle.com/datasets/ayush1220/cifar10

**Note:** For GPU acceleration, ensure CUDA is available. If using Google Colab, the notebook can be opened directly via the Colab badge at the top of the notebook.

## Result Replication

### Step 1: Environment Setup

1. Follow the Setup instructions above to clone the repository and install dependencies

2. Ensure you have GPU access (CUDA) for efficient training

### Step 2: Run Training Notebooks

#### Option A: Random Projection Feature Map

Open and run `src/notebooks/ksd_gan_cifar10_training.ipynb`:

1. The notebook will:
   - Load a pretrained DDPM model for CIFAR-10 (from Hugging Face)
   - Initialize a random projection feature map (512-dimensional)
   - Train a KSD-GAN generator for 1000 steps
   - Train a standard GAN baseline for comparison
   - Evaluate both models using FID/KID metrics

2. Key parameters (can be modified in the notebook):
   - `steps = 1000` - Number of training iterations
   - `batch_size = 64` - Batch size for training
   - `d_proj = 512` - Random projection dimension
   - `K_T = 1` - Number of timesteps sampled per iteration
   - `LAMBDA_MMD = 0.50` - MMD stabilizer weight

#### Option B: ResNet Feature Map

The same notebook also includes an implementation using frozen ResNet18 features (ImageNet-pretrained). This can be run by executing the corresponding cells in the notebook.

### Step 3: Evaluation

The notebook includes evaluation code that:
- Computes FID and KID metrics using Inception v3 features
- Generates sample visualizations
- Plots KSD values across different diffusion timesteps

### Expected Runtime

- **KSD-GAN Training:** ~2-4 hours on a single GPU (depending on hardware)
- **Standard GAN Training:** ~1-2 hours on a single GPU
- **Evaluation:** ~30 minutes for FID/KID computation on 5000 samples

## Code Description and Attribution

**Note on Code Development:** This code was developed with assistance from ChatGPT for implementation guidance. The core ideas and adaptations are our original work, with ChatGPT helping with implementation details and debugging.

**All Notebooks:** All notebooks in `src/notebooks/` were implemented by us. They contain standalone PyTorch implementations that don't import from other source files in this repository.

### Original Implementation (Our Work)

The following components were implemented from scratch for this project:

1. **Rényi-Nyström KSD Implementation** (`src/renyiksd.py`):
   - **Our adaptation of Nyström-KSD** - Rényi-based landmark selection algorithm
   - `select_renyi_landmarks()` function: Greedy selection that minimizes quadratic Rényi energy
   - `RenyiNystroemKSD` class: Extends base Nyström-KSD with Rényi landmark selection
   - Adapted from the base Nyström-KSD implementation for our GAN training use case
   - **Note**: This is a NumPy-based reference implementation. The notebooks contain standalone PyTorch implementations.

2. **Score Function Utilities** (`src/score.py`):
   - **Our implementation** for computing DDPM score functions
   - `score_fn()`: Computes score function ∇ log p_t(x_t) using pretrained DDPM
   - `map_sigma_to_t()`: Maps noise level σ to diffusion timestep t
   - `score_from_noise_pred()`: Converts noise prediction to score
   - `load_cifar_batch()`: Utility for loading CIFAR-10 batches
   - **Note**: This is a standalone reference implementation. The notebooks contain inline PyTorch implementations.

3. **All Notebooks in `src/notebooks/`** (All implemented by us):
   - **`ksd_gan_cifar10_training.ipynb`**: Main training notebook - **This is our main original contribution** - adapting Nyström-KSD for GAN training
     - Complete PyTorch implementation of Rényi-Nyström KSD with feature-space Stein kernels
     - `RenyiNystroemKSD` class: PyTorch implementation optimized for GAN training
     - `select_renyi_landmarks_stein()`: Rényi landmark selection for image data
     - `score_fn_xt()`: Score function computation using pretrained DDPM
     - Feature-space Stein kernel computation (random projection and ResNet features)
     - Two-sample KSD estimator with gradient flow control
     - Integration of KSD loss into GAN training loop
     - Multi-timestep sampling strategy
     - Feature normalization and bandwidth selection (median heuristic)
     - MMD stabilizer implementation
     - All training code, evaluation metrics, and visualizations
   - **`ksd_gan_cifar10_backup.ipynb`**: Backup/alternative implementation with similar functionality
   - **`RenyiKSDtorch.ipynb`**: Additional experiments and testing
   - **Note**: All notebooks contain standalone PyTorch implementations. `src/renyiksd.py` and `src/score.py` are reference/utility implementations that can be used independently.

4. **Generator Architecture**:
   - Custom upsampling-based generator (UpBlock + GenX0)
   - Designed to reduce checkerboard artifacts
   - Implemented in the notebooks

5. **Standard GAN Baseline**:
   - DCGAN-style discriminator implementation
   - Training loop for comparison purposes

### Adapted/Modified Code (With Proper Attribution)

1. **Nyström-KSD Base Implementation** (`nystroem-ksd/` directory):
   - **Source:** This is the official repository for "Nyström Kernel Stein Discrepancy" by Kalinke, Szabó, and Sriperumbudur (AISTATS 2025)
   - **Repository:** [https://github.com/FlopsKa/nystroem-ksd](https://github.com/FlopsKa/nystroem-ksd)
   - **What we used:** The base Nyström-KSD implementation from their repository
   - **What we modified:** We adapted it for GAN training in our notebooks, specifically:
     - Modified for feature-space Stein kernels (random projection and ResNet features)
     - Adapted for two-sample testing between real and fake distributions
     - Integrated into PyTorch training loop with gradient flow control
     - Added Rényi landmark selection (inspired by their work but implemented for our use case)
   - **Location of our adaptations:** All adaptations are in `src/notebooks/ksd_gan_cifar10_training.ipynb`
   - **Original repository:** The `nystroem-ksd/` directory contains the original implementation from the paper authors

2. **Kernel Goodness-of-Fit Library** (`src/kgof/` and `nystroem-ksd/lib/kernel-gof/`):
   - **Source:** Jitkrittum et al. (2017) - https://github.com/wittawatj/kernel-gof
   - Included as a dependency in the Nyström-KSD repository
   - Used as reference for KSD computation patterns
   - Modified for our specific use case in the notebooks

3. **Random Feature Stein Discrepancies** (`src/rfsd/` and `nystroem-ksd/lib/random-feature-stein-discrepancies/`):
   - **Source:** Huggins and Mackey (2018) - https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/
   - Included as a dependency in the Nyström-KSD repository
   - Modified to work with our feature maps in the notebooks

4. **DDPM Score Function**:
   - Uses pretrained DDPM model from Hugging Face (`google/ddpm-cifar10-32`)
   - Score function computation adapted from standard DDPM formulation
   - No modifications to the pretrained model weights

### External Dependencies

- **PyTorch**: Deep learning framework
- **Diffusers**: For pretrained DDPM models
- **Torchvision**: For dataset loading and pretrained models (ResNet, Inception)
- **NumPy/SciPy**: Numerical computations

### Key Modifications and Rationale

1. **Feature Normalization from Real Data Only:**
   - **Why:** Ensures stable training by preventing gradient flow through normalization statistics
   - **Implementation:** `mu` and `std` computed from real samples only, then applied to both real and fake

2. **Landmark Selection Without Gradients:**
   - **Why:** Prevents landmark selection from affecting generator gradients
   - **Implementation:** All landmark selection performed under `torch.no_grad()` context

3. **Gradient Flow Control:**
   - **Why:** Only fake samples should contribute to generator gradients
   - **Implementation:** Real samples and landmarks are detached before KSD computation

4. **MMD Stabilizer:**
   - **Why:** Provides additional regularization and stability
   - **Implementation:** Small MMD term (λ=0.5) added to KSD loss in the same feature space

5. **Multi-timestep Sampling:**
   - **Why:** Training at multiple noise levels improves coverage
   - **Implementation:** Mixture sampler that focuses increasingly on low noise levels (small t) as training progresses

## Technical Details

### Rényi Landmark Selection

The landmark selection minimizes the Rényi entropy approximation:
\[
E(\tilde{X}) = \frac{1}{m^2} 1_m^\top \Omega(\tilde{X}) 1_m
\]
where \(\Omega\) is the Stein kernel Gram matrix. This ensures diverse and informative landmark selection.

### Nyström-KSD Estimator

The full KSD has O(n²) complexity. The Nyström approximation projects onto a subspace spanned by m landmarks:
\[
\text{KSD}_N^2(Q, P) = \delta^T (H_{mm} + \lambda I)^{-1} \delta
\]
where \(\delta = \beta_{\text{fake}} - \beta_{\text{real}}\) and \(H_{mm}\) is the landmark-landmark kernel matrix.

### Feature Maps

Two feature map implementations:
1. **Random Projection:** Fixed random linear projection \(z = xP^T\) with orthogonal scaling
2. **ResNet Features:** Frozen ResNet18 layer2 features (ImageNet-pretrained)

## Challenges and Limitations

1. **Training Instability:** KSD-GAN showed higher loss variance, especially at low noise levels
2. **Sample Quality:** Quantitative metrics (FID/KID) indicate standard GAN performs better
3. **Computational Cost:** Even with Nyström approximation, KSD computation is more expensive than standard discriminator
4. **Hyperparameter Sensitivity:** Requires careful tuning of bandwidth, ridge parameter, and landmark count

## Future Work

1. **Improved Feature Maps:** Experiment with learned feature maps or better pretrained features
2. **Better Timestep Sampling:** Develop more sophisticated strategies for selecting diffusion timesteps
3. **Stability Improvements:** Investigate gradient clipping, loss normalization, or alternative kernel choices
4. **Extended Evaluation:** Test on additional datasets (MNIST, CelebA) and longer training runs

## References

1. **Kalinke, F., Szabó, Z., Sriperumbudur, B. K.** (2025). *Nyström Kernel Stein Discrepancy*. AISTATS 2025.  
   - Repository: [https://github.com/FlopsKa/nystroem-ksd](https://github.com/FlopsKa/nystroem-ksd)
   - Original Nyström-KSD implementation repository included in `nystroem-ksd/` directory
   - Our adaptation for GAN training is in `src/notebooks/ksd_gan_cifar10_training.ipynb`

2. **Goodfellow, I., et al.** (2014). *Generative Adversarial Nets*. NIPS 2014.

3. **Liu, Q., et al.** (2016). *A Kernelized Stein Discrepancy for Goodness-of-fit Tests*. ICML 2016.

4. **Jitkrittum, W., et al.** (2017). *A Linear-Time Kernel Goodness-of-Fit Test*. NIPS 2017.  
   - Kernel goodness-of-fit library: https://github.com/wittawatj/kernel-gof

5. **Huggins, J. H., Mackey, L.** (2018). *Random Feature Stein Discrepancies*. NIPS 2018.  
   - Repository: https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/

6. **Krizhevsky, A., Hinton, G.** (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.

## License

This project is for academic research purposes. Please refer to individual library licenses for dependencies.

## Contact

For questions or issues, please open an issue on the GitHub repository.
