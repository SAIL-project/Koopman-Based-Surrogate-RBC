# Koopman-Based Surrogate Modelling of Turbulent Rayleigh-Bénard Convection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Conference](http://img.shields.io/badge/Paper-IJCNN_2024-blue)](https://ieeexplore.ieee.org/document/10651496)
[![arXiv](https://img.shields.io/badge/arXiv-2405.06425-red.svg)](https://arxiv.org/abs/2405.06425)


## Abstract
Several related works have introduced Koopman-based Machine Learning architectures as a surrogate model for dynamical systems. These architectures aim to learn non-linear measurements (also known as observables) of the system’s state that evolve by a linear operator and are, therefore, amenable to model-based linear control techniques. So far, mainly simple systems have been targeted, and Koopman architectures as reduced-order models for more complex dynamics have not been fully explored. Hence, we use a Koopman-inspired architecture called the Linear Recurrent Autoencoder Network (LRAN) for learning reduced-order dynamics in convection flows of a Rayleigh Bénard Convection (RBC) system at different amounts of turbulence. The data is obtained from direct numerical simulations of the RBC system. A traditional fluid dynamics method, the Kernel Dynamic Mode Decomposition (KDMD), is used to compare the LRAN. For both methods, we performed hyperparameter sweeps to identify optimal settings. We used a Normalized Sum of Square Error measure for the quantitative evaluation of the models, and we also studied the model predictions qualitatively. We obtained more accurate predictions with the LRAN than with KDMD in the most turbulent setting. We conjecture that this is due to the LRAN’s flexibility in learning complicated observables from data, thereby serving as a viable surrogate model for the main structure of fluid dynamics in turbulent convection settings. In contrast, KDMD was more effective in lower turbulence settings due to the repetitiveness of the convection flow. The feasibility of Koopman-based surrogate models for turbulent fluid flows opens possibilities for efficient model-based control techniques useful in a variety of industrial settings.

## Rayleigh-Bénard Convection
This work uses Fourier Neural Operators to model Rayleigh-Bénard Convection (RBC). RBC describes convection processes in a layer of fluid cooled from the top and heated from the bottom via the partial differential equations:

**Rayleigh-Bénard Convection**

$$\begin{aligned}
& \frac{\partial u}{\partial t} + (u \cdot \nabla) u = -\nabla p + \sqrt{\frac{Pr}{Ra}} \nabla^2 u + T j \\
& \frac{\partial T}{\partial t} + u \cdot \nabla T = \frac{1}{\sqrt{Ra Pr}} \nabla^2 T\ \\
& \nabla \cdot u = 0 \\
\end{aligned}$$

The surrogate models are trained on data generated by a Direct Numerical Simulation based on [Shenfun](https://github.com/spectralDNS/shenfun) with the following parameters:
| Parameter       | Value                | | Parameter      | Value    |
|-----------------|----------------------|-|----------------|----------|
| Domain          | ((-1, 1),(0, $2\pi$))| | ($T_t$, $T_b$) | (1,2)    |
| Grid            | 64 x 96              | | $\Delta t$     | 0.025    |
| Rayleigh Number | {1e5, 1e6, 2e6, 5e6} | | Episode Length | 300      |
| Prandtl Number  | 0.7                  | | Cook Time      | 200      |

## Results
TODO

## Citation
If you find our work useful, please cite us via:

```bibtex
@INPROCEEDINGS{10651496,
  author={Markmann, Thorben and Straat, Michiel and Hammer, Barbara},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Koopman-Based Surrogate Modelling of Turbulent Rayleigh-Bénard Convection}, 
  year={2024},
  pages={1-8},
  doi={10.1109/IJCNN60899.2024.10651496}}

```

## Setup
```bash
pip install ...
```

## How to run
```bash
python ...
```