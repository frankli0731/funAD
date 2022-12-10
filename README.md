# team21
[![Continuous Integration Test Coverage for Milestone 2](https://code.harvard.edu/CS107/team21/actions/workflows/coverage.yml/badge.svg?branch=milestone2_dev)](https://code.harvard.edu/CS107/team21/actions/workflows/coverage.yml)
[![test AD for Milestone 2](https://code.harvard.edu/CS107/team21/actions/workflows/test.yml/badge.svg?branch=milestone2_dev)](https://code.harvard.edu/CS107/team21/actions/workflows/test.yml)

## Brief Introduction

funAD is a PyPi-distributed package that executes forward-mode of automatic differentiation, enabling users to solve functional derivatives with high computational efficiency and machine precision.

This project/package is the fruit of Harvard CS107/AC207 class final project in 2022 Fall. Our package utilize forward mode and dual number. Additionally, we also allow users to define their own seeds vector for the Jacobian Matrix and the option to calculate local maxima and minima through gradient descent.

To install, run the following command in your terminal

`pip install funAD`

For details instruction on how to use this package, please follow the steps in the [usage page](https://code.harvard.edu/CS107/team21/tree/main/docs/documentation.ipynb).

**Group Number:**

Group 21

**Group Members:**

_Zhecheng Yao zhechengyao@g.harvard.edu_

_Hanlin Zhu hzhu@g.harvard.edu_

_Xu Tang xutang@g.harvard.edu_

_Tiantong Li tiantongli@g.harvard.edu_

Harvard University, Fall 2022

## Broader Impact

`funAD` is a useful tool to reduce the computational burden by reducing the amount of manual computation and providing accurate results. Our package supports the forward mode of automatic differentiation, as well as several optimizers, including gradient descent and Adam. `funAD` enables users to compute derivatives of complex functions with machine precision and high computational efficiency. Additional optimization function is implemented to generate local minima and maxima for the user. These functions mentioned above are applicable to many research areas. For example, computational biologists often need to perform the above calculations in order to estimate population growth and peaks for the target species.  Optimizers such as gradient descent help minimize loss functions, which play a significant role in the development of neural networks, computer vision, and many statistical models. With the assistance of optimization functions and its computationally efficient nature, `funAD' can be very useful in training machine learning models and neural networks.This may also benefit economists and financial researchers who can minimize cost functions and maximize profit functions in the context of industrial, business and investment analysis to generate the best returns for their companies.

Unlike some models trained based on datasets, our package is essentially a specified calculator that follows mathematical expressions and rules. The package itself is not biased in any way. However, users should be careful not to use it in a discriminative way such as training machine learning models on biased datasets. Users need to discipline themselves not to use our package to bypass the calculation step when they need to find derivative results manually as a math assignment.  Although the code is restricted to code.harvard.edu for grading purposes, our package is uploaded to PyPI, which is free to download and relatively easy to implement. Our email addresses are listed on PyPI's `funAD` package download page. Users can easily contact us with any feedback and suggestions.
