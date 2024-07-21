# ASTM-D86 Numerical Simulation

This repository contains a numerical simulation of the ASTM-D86 distillation process for a hypothetical petroleum cut. The batch distillation process is approximated by several flash vaporization steps, with each step involving a small temperature rise. Each step is modeled as a multi-component flash calculation using the Peng-Robinson Equation of State (EOS). The Wilson correlation is utilized to generate initial guesses for the rigorous flash calculations.

## Project Overview

The simulation aims to replicate the ASTM-D86 distillation method, which is commonly used to characterize the boiling range of petroleum products. The distillation process is broken down into incremental temperature steps, and each step is treated as a flash vaporization. This approach provides a detailed and accurate representation of the distillation curve.

### Key Features
- **Multi-component Flash Calculations**: Each flash calculation is conducted using the Peng-Robinson EOS to ensure accuracy in phase equilibrium predictions.
- **Initial Guesses via Wilson Correlation**: The Wilson correlation provides reliable initial guesses, enhancing the convergence of the flash calculations.
- **Incremental Temperature Steps**: The distillation process is divided into small temperature increments, ensuring detailed resolution of the distillation curve.

## Numerical Methods Implemented

The following numerical methods are employed in the project to solve the nonlinear equations and perform the necessary calculations:

### Nonlinear Equation Solvers
- **Newton Method**: Utilized for solving nonlinear systems of equations with quadratic convergence.
- **Newton-Broyden Method**: An extension of the Newton method, incorporating quasi-Newton updates to approximate the Jacobian.
- **Newton-Householder Method**: Another variant of the Newton method, used for improving convergence properties in certain cases.

### Interpolation and Matrix Methods
- **Cubic Spline Interpolation**: Used for interpolating data points to create a smooth distillation curve.
- **Jacobian Matrix Approximation**: Essential for the Newton-based methods to solve nonlinear systems.
- **LU Decomposition**: Applied to solve linear systems efficiently as part of the iterative methods.
- **Inverse Matrix Calculation via Gaussian Elimination**: Used to compute matrix inverses, necessary for solving linear equations.

Check the `utils.py` file, the implementation of some useful numerical methods are in there.

## Usage Instructions

### Prerequisites
Ensure you have the following software and libraries installed:
- Python 3.x
- NumPy
- SciPy
- Matplotlib (for plotting results)
