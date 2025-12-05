# Accelerated Kernel Stein Discrepancy with R´enyi Landmark Selection for Stable and Efficient GAN Training  
**Michael Carlo, Giovanni Dettori, Ryan Gumsheimer**  
**November 2025**

---

## 1. Project Plan

Our project investigates the replacement of the classical adversarial discriminator in GAN training with a kernel-based distance metric, namely Kernel Stein Discrepancy (KSD). The goal is to assess whether a kernelized objective can provide an alternative that improves training stability and efficiency without compromising sample quality.

We will begin by reproducing the standard GAN setup using benchmark image datasets such as MNIST and CIFAR-10 to establish a reliable baseline and validate our implementation. Building on this, we will integrate KSD loss with R´enyi landmark selection into the training pipeline, using it as a direct substitute for the adversarial loss. Finally, we will compare the two training paradigms—adversarial versus kernel-based—using both quantitative metrics (e.g., Fréchet Inception Distance, FID) and qualitative sample evaluation. All experiments will be implemented in Python, version-controlled via GitHub, and executed with GPU acceleration where appropriate.

---

## 2. Background and Motivation

Goodness-of-fit testing and model evaluation are central problems in modern machine learning and statistics. Given samples from an unknown distribution \(Q\) and a target model \(P\) with density \(p(x)\), we often wish to determine how well \(Q\) approximates \(P\). Classical divergence measures such as the Kullback–Leibler or Jensen–Shannon divergences are theoretically appealing but typically require access to the density \(q(x)\) of the model distribution, which is unavailable in implicit generative models like GANs.

Kernel methods provide a powerful alternative by embedding probability measures into a reproducing kernel Hilbert space (RKHS), thereby allowing discrepancies between distributions to be expressed as distances between their embeddings. Formally, every positive-definite kernel \(k(x, x')\) defines a unique Hilbert space of functions \(\mathcal{H}_k\), where evaluation is continuous and satisfies  
\[
f(x) = \langle f, k(\cdot, x)\rangle_{\mathcal{H}_k}, \quad  
\langle k(\cdot, x), k(\cdot, x')\rangle_{\mathcal{H}_k} = k(x, x').
\]

This property allows nonlinear relationships in the data space to appear as linear relations in the induced Hilbert space.

Each probability distribution \(P\) on \(X\) can be represented by its mean embedding  
\[
\mu_P := \mathbb{E}_{X \sim P}[k(\cdot, X)] \in \mathcal{H}_k,
\]
which satisfies
\[
\mathbb{E}_{X \sim P}[f(X)] = \langle f, \mu_P\rangle_{\mathcal{H}_k}.
\]

Comparing two distributions thus reduces to comparing \(\mu_P\) and \(\mu_Q\).

A broad class of divergences can be expressed as integral probability metrics (IPMs):  
\[
D_F(P, Q) = \sup_{f \in F} \left| \mathbb{E}_{X \sim P}[f(X)] - \mathbb{E}_{Y \sim Q}[f(Y)] \right|.
\]

Choosing \(F\) as the unit ball of the RKHS yields the Maximum Mean Discrepancy (MMD):  
\[
\text{MMD}(P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}_k}.
\]

When \(k\) is characteristic, the MMD uniquely identifies probability distributions.  
However, it still requires sampling from both \(P\) and \(Q\), or explicit knowledge of normalized densities.

### Stein’s Method and Kernel Stein Discrepancy (KSD)

Stein’s identity states that for a smooth log-density \( \log p(x) \) with score function  
\[
\nabla_x \log p(x)
\]
and a suitable test function \(f\),  
\[
\mathbb{E}_{X \sim P}\big[\nabla_x \log p(X)^\top f(X) + \nabla_x \cdot f(X)\big] = 0.
\]

The Kernel Stein Discrepancy (KSD) compares \(Q\) to \(P\) without requiring access to \(q(x)\):  
\[
\text{KSD}^2(Q, P) = \| \mathbb{E}_{X \sim Q}[T_p k(\cdot, X)] \|_{\mathcal{H}_k}^2.
\]

The empirical estimators of KSD are:

- V-statistic:  
\[
\text{KSD}^2_V(Q, P) = \frac{1}{n^2} \sum_{i,j} h_p(x_i, x_j)
\]

- U-statistic:  
\[
\text{KSD}^2_U(Q, P) = \frac{1}{n(n-1)} \sum_{i \neq j} h_p(x_i, x_j)
\]

Both have \(O(n^2)\) computational cost.

---

## 3. Proposed Research

To address computational cost, the Nyström-KSD estimator projects the empirical embedding onto a subspace spanned by a small set of \(m \ll n\) Nyström points.

However, uniform landmark selection may be suboptimal in heterogeneous or high-dimensional datasets. We propose **R´enyi-based landmark selection**, inspired by quadratic R´enyi entropy:
\[
H_R = - \log \int p(x)^2 dx.
\]

Given empirical density \(\hat{p}\), we approximate:
\[
\int \hat{p}(x)^2 dx = \frac{1}{N^2} 1_v^\top \Omega 1_v,
\]
where \(\Omega\) is the Gram matrix.

Thus we select landmarks \(\tilde{X}\) by minimizing:
\[
E(\tilde{X}) = \frac{1}{m^2} 1_m^\top \Omega(\tilde{X}) 1_m.
\]

This ensures diverse and informative landmark selection.

---

### KSD for GAN Training

GANs solve:
\[
\min_\theta \max_\phi L_{\text{GAN}}(\theta, \phi)
\]

This is unstable and indirectly measures discrepancy.  
Instead, we train the generator using:
\[
L_{\text{KSD}}(\theta) = \text{KSD}_N^2(Q_\theta, P_{\text{data}})
\]

This removes the discriminator entirely and provides a well-defined statistical discrepancy.

---

## 4. Data and Code Implementation

We will:

1. Reproduce GAN training on MNIST (70,000 grayscale images).
2. Compare adversarial vs KSD training.
3. Extend to CIFAR-10 (60,000 color images).
4. Use GitHub for version control.
5. Use Google Colab for GPU execution.

---

## References

1. CIFAR-10 dataset — https://www.kaggle.com/datasets/ayush1220/cifar10  
2. MNIST dataset — https://www.kaggle.com/datasets/hojjatk/mnist-dataset  
3. Cribeiro-Ramallo et al., *Minimax lower bound of KSD estimation*, 2025  
4. Girolami, *Orthogonal series density estimation*, 2002  
5. Goodfellow et al., *GANs*, 2014  
6. Kalinke, Szabó, Sriperumbudur, *Nyström KSD*, AISTATS 2025  
7. Krizhevsky & Hinton, *Tiny Images*, 2009  
8. LeCun et al., *Gradient-based learning*, 1998  

