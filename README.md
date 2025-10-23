# Integrators for Multiplicative Noise

*Brownian dynamics integrators for multiplicative noise, implemented in Julia.*

This codebase was developed for the paper "Efficient Langevin sampling with position-dependent diffusion" - Read the paper [here](https://arxiv.org/abs/2501.02943). 
This codebase is also an extension of [this repo](https://github.com/dominicp6/Transforms-For-Brownian-Dynamics), which was developed for the paper "Numerical Methods with Coordinate Transforms for Efficient Brownian Dynamics Simulations" - Read the paper [here](https://arxiv.org/abs/2307.02913).

If you use this work, please use the following citations:

```
@article{bronasco2025efficient,
  title={Efficient Langevin sampling with position-dependent diffusion},
  author={Bronasco, Eugen and Leimkuhler, Benedict and Phillips, Dominic and Vilmart, Gilles},
  journal={arXiv preprint arXiv:2501.02943},
  year={2025}
}
```
```
@article{phillips2025numerics,
  title={Numerics with coordinate transforms for efficient Brownian dynamics simulations},
  author={Phillips, Dominic and Leimkuhler, Benedict and Matthews, Charles},
  journal={Molecular Physics},
  volume={123},
  number={7-8},
  pages={e2347546},
  year={2025},
  publisher={Taylor \& Francis}
}
```

## Motivation
Numerical integrators of Stochastic Differential Equations (SDEs) work well for constant (additive) noise but often lose performance, or fail to converge, for variable (multiplicative) noise. 
**Brownian dynamics** is one of the most important classes of SDE processes, with applications across the physical, biological, and data-driven sciences. 
We develop a new, second-order integrator for the invariant measure for Brownian dynamics with multiplicative noise. The codebase includes implementations of other common integrators for comparison.

## Installation

To install this project, follow these steps:

1. Clone the repository.

`git clone https://github.com/dominicp6/Transforms-For-Brownian-Dynamics`

2. Navigate to the project directory.

3. Install the required Julia packages:

The code has only been tested with the version numbers provided. If you are experiencing package syntax errors, consider downgrading the packages to the specified versions:

 - "QuadGK"     => v"2.8.2"
 - "Statistics" => v"1.9.0"
 - "JSON"       => v"0.21.4"
 - "HDF5"       => v"0.16.15"
 - "StatsBase"  => v"0.34.0"
 - "Plots"      => v"1.38.17"
 - "HCubature"  => v"1.5.1"
 - "FHist"      => v"0.10.2"

Done!

## Contributing

We welcome contributions! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

### Brownian Dynamics
Brownian dynamics is defined through an Ito stochastic differential equation (SDE), which in one dimension reads

$$
    dx_t = - D(x_t) \frac{dV(x_t)}{dx}dt + kT \frac{d D(x_t)}{dx}dt + \sqrt{2 kT D(x_t)} dW_t,
$$

where $`t \in \mathbb{R}_{>0}`$ is time, $`x_t \in \mathbb{R}`$ is the state variable, $`W_t`$ is a one-dimensional Wiener process, $`V : \mathbb{R} \xrightarrow{} \mathbb{R}`$ is a potential energy function, $`D : \mathbb{R} \xrightarrow{} \mathbb{R}_{>0}`$ is the diffusion coefficient, $`k`$ is the Boltzmann constant and $`T`$ is the temperature in degrees Kelvin. Note that the diffusion coefficient $`D(x)`$ is a function of $x$ which means that we have configuration-dependent noise, also known as multiplicative noise. 

In higher dimensions, 

$$
   d\mathbf{X}_t = -(\Sigma(\mathbf{X}_t)\Sigma(\mathbf{X}_t)^T) \nabla V(\mathbf{X}_t) dt + kT \text{div}(\Sigma\Sigma^T)(\mathbf{X}_t) dt + \sqrt{2 kT} \Sigma(\mathbf{X}_t)d\mathbf{W}_t,
$$

where $`\mathbf{X}_t \in \mathbb{R}^n`$ is the state variable, $`\mathbf{W}_t`$ is an n-dimensional Wiener process, $`V: \mathbb{R}^n \xrightarrow{} \mathbb{R}`$ is a potential function, and $`\Sigma\Sigma^T: \mathbb{R}^n \xrightarrow{} \mathbb{R}^n \times \mathbb{R}^n`$ is a configuration-dependent diffusion tensor that is everywhere positive definite.

We assume that $`V`$ is confining in a way that ensures ergodicity of the dynamics and, therefore, there exists a unique invariant distribution $`\rho(\mathbf{X})`$ - a probability distribution that does not change under the process dynamics. For Brownian dynamics, the invariant distribution is the canonical ensemble; $`\rho(\mathbf{X}) \propto \exp{\left(- V(\mathbf{X})/kT\right)}`$. Another consequence of ergodicity is that the long-time time averages converge to phase-space averages, i.e.

$$
\int_{\mathbb{R}^n} f(\mathbf{X})\rho(\mathbf{X})d\mathbf{X} = \lim_{T \rightarrow \infty} \frac{1}{T} \int_{t=0}^T f(\mathbf{X}_t)dt,
$$

for reasonably well-behaved functions $`f: \mathbb{R}^n \rightarrow \mathbb{R}`$.

### Second-Order Post-Processor Method For Variable Diffusion

**Definition (PVD-2):** The Second-Order Post-processed method for Variable Diffusion (PVD-2) is an integrator for Brownian dynamics. It is defined as:

$$
X_{n+1} = X_n + hF(Y_n) + \hat\Phi^\Sigma_h \left(X_n + \frac{1}{4} h F(Y_{n-1}); \xi_n\right),
$$

$$
Y_n = X_n + \frac{1}{2} \sqrt{h} \sigma \Sigma(X_n) \xi_n, \quad \text{with } Y_{-1} = X_0,
$$

where $\xi_n \sim \mathcal{N}(0, I_d)$ are independent standard Gaussian vectors, $F = -(\Sigma^\top \Sigma)\nabla V + \frac{2}{\sigma^2} \nabla \cdot (\Sigma^\top \Sigma)$ is the drift term, and $\Phi_h^\Sigma(X_n; \xi_n) = X_n + \hat{\Phi}_h^\Sigma(X_n; \xi_n)$ is any one-step integrator of weak order 2, applied to the pure-noise system:

$$
dX = \sigma \Sigma(X) \, dW.
$$

**Theorem:** Under smoothness and ergodicity assumptions, the PVD-2 scheme achieves **second-order accuracy** for the invariant measure. For any smooth test function $\phi$:

$$
\left| \lim_{N\rightarrow +\infty} \mathrm{a.s.}
\frac{1}{N+1} \sum^N_{n=0} \phi(Y_n) - \int_{\mathbb{R}^d} \phi(x) \, d\pi(x) \right| \leq Ch^2
$$

**Proof Sketch:** Proving this theorem directly is exceptionally difficult due to the complexity of the integrator's error expansion. The proof relies on the modern algebraic framework of **exotic aromatic B-series**. This technique translates complex differential operators (which describe the error of the integrator) into simpler combinatorial objects (decorated trees). By showing that the error terms, represented as trees, cancel out when averaged over the invariant distribution, we prove the method's second-order accuracy without a brute-force calculation. The full details can be found in our accompanying paper.


