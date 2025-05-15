# NPTBench: A Benchmark Suite for Neural Posterior Testing

__Update:__ this repository accompanies the paper "From NPE to NPT: a benchmark suite for neural posterior testing" (Tianyu Chen, Vansh Bansal, and James Scott, 2024)

Neural posterior estimation (NPE) methods learn approximate Bayesian posteriors q(theta | y) using neural networks trained on simulated data. But checking whether q(theta | y) is a good approximation to the true posterior p(theta | y) is a distinct problem, which we call neural posterior testing (NPT).

This benchmark suite provides a controlled and extensible framework for evaluating NPT methods. It introduces test problems where the true posterior is known, and applies parameterized perturbations that simulate common NPE failure modes such as bias, overdispersion, geometric misalignment, and multimodality.

This suite began as a series of experiments at:
https://github.com/TianyuCodings/cDiff  
We have subsequently moved development to this repository.

## Overview: What this suite does

- Provides synthetic test problems with known posteriors
- Introduces controlled perturbations parameterized by a scalar gamma 
- Supports quantitative comparison of NPT methods via power curves parametrized by gamma.  
- Includes implementations of SBC, TARP, and C2ST

## Core benchmark: Perturbed multivariate normal

Our main benchmark uses multivariate normal posteriors:

    p(theta | y) = N(mu_y, Sigma_y)

where the mean and covariance mu_y and Sigma_y depend on data, simulated as y ~ N(1_m, I_m). Perturbations define approximate posteriors q(theta | y; gamma), where gamma controls the deviation from the true posterior.


## Supported perturbation types

For for each case, we define a true posterior of the form:
$$
p(\theta \mid y) = \mathcal{N}(\mu_y, \Sigma_y)
$$
Perturbed posteriors are defined as $q(\theta \mid y; \gamma)$, where $\gamma \geq 0$ controls the deviation from the true posterior. This allows us to generate power curves by plotting power/rejection rate as a function of gamma.

### Mean shift

Introduces bias in the posterior location.

$$
q(\theta \mid y) = \mathcal{N}\left((1 + \gamma) \mu_y, \Sigma_y\right)
$$


### Covariance inflation

Uniformly inflates the posterior covariance, modeling overdispersed uncertainty.

$$
q(\theta \mid y) = \mathcal{N}\left(\mu_y, (1 + \gamma) \Sigma_y\right)
$$


### Anisotropic distortion

Adds low-rank noise along a structured direction, specifically the minimum-variance eigenvector of $\Sigma_y$.

$$
q(\theta \mid y) = \mathcal{N}\left(\mu_y, \Sigma_y + \gamma \cdot \Delta\right)
$$

where

$$
\Delta = \mathbf{v}_{\min} \mathbf{v}_{\min}^\top
$$

and $\mathbf{v}_{\min}$ is the eigenvector of $\Sigma_y$ with the smallest eigenvalue.


### Heavy tails

Replaces the Gaussian with a multivariate $t$-distribution to simulate tail mismatch.

$$
q(\theta \mid y) = t_\nu(\mu_y, \Sigma_y), \quad \nu = \frac{1}{\gamma + \varepsilon}
$$

for small $\varepsilon > 0$.


### Extra modes

Creates a symmetric bimodal mixture by adding a reflected mode.

$$
q(\theta \mid y) = (1 - \gamma) \, \mathcal{N}(\mu_y, \Sigma_y) + \gamma \, \mathcal{N}(-\mu_y, \Sigma_y)
$$


### Mode collapse

The true posterior is bimodal, but the approximate posterior collapses to a single mode.

$$
\text{True: } p(\theta \mid y) = (1 - \gamma) \, \mathcal{N}(\mu_y, \Sigma_y) + \gamma \, \mathcal{N}(-\mu_y, \Sigma_y)
$$

$$
\text{Approx: } q(\theta \mid y) = \mathcal{N}(\mu_y, \Sigma_y)
$$



## Other perturbations

The paper also includes more complex scenarios:

- Manifold-based posteriors: Gaussian samples are mapped through a nonlinear function f, concentrating the posterior on a curved manifold in R^d. Perturbations simulate drift along or off the manifold.
- Diffusion-based degradation: A diffusion model is trained on structured posterior samples. Earlier-stage samples represent degraded approximations to the final step.
- Information thinning: The conditioning variable y is corrupted by noise to simulate encoder underfitting. As gamma increases, less information about y is retained in q.

Implementation for these more complex benchmarks is ongoing.  

## Evaluated methods

We currently support benchmarking of:

- SBC (Simulation-Based Calibration)
- TARP
- C2ST (Classifier Two-Sample Test)

Each method is evaluated by its ability to detect mismatch as the perturbation strength gamma increases.

## Citation

If you use this benchmark suite, please cite:

Chen, T., Bansal, V., and Scott, J. (2024).  "From NPE to NPT: a benchmark suite for neural posterior testing."
