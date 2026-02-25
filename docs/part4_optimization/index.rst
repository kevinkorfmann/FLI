.. _part4:

========================================
Part IV: Optimization for Likelihood
========================================

Maximum likelihood estimation reduces, in almost every case, to an optimization
problem: find the parameter values that make the observed data most probable.
Parts I--III of this book developed the *what* --- what the likelihood function is,
what properties its maximizer enjoys, and what we can infer from it. Part IV
addresses the *how*: the numerical algorithms that actually locate that maximizer.

We begin with **gradient methods** (:ref:`ch12_gradient`), the workhorses of
modern optimization. Starting from nothing more than a Taylor expansion, we
derive gradient descent, explore step-size strategies, and then follow the
historical arc through stochastic gradient descent, momentum, and the adaptive
methods (Adam and friends) that dominate machine-learning practice.

**Newton and scoring methods** (:ref:`ch13_newton`) exploit second-order
information --- the curvature of the log-likelihood --- to achieve dramatically
faster local convergence. We derive Newton--Raphson, Fisher scoring, and the
iteratively reweighted least-squares (IRLS) algorithm used in every generalized
linear model package.

Computing and inverting a full Hessian is expensive. **Quasi-Newton methods**
(:ref:`ch14_quasi_newton`) approximate the Hessian on the fly using only
gradient information. We derive BFGS and its limited-memory cousin L-BFGS,
discuss trust-region strategies, and compare the practical trade-offs.

The **EM algorithm** (:ref:`ch15_em`) is the method of choice when the
likelihood involves latent variables or missing data. We give a self-contained
derivation via Jensen's inequality, prove monotone ascent, and work through
Gaussian mixture models in full detail. Extensions --- ECM, MCEM, and
variational EM --- round out the chapter.

Finally, **constrained optimization** (:ref:`ch16_constrained`) handles the
reality that parameters often live on restricted sets (probabilities must sum
to one, variances must be positive). Lagrange multipliers, KKT conditions,
barrier methods, and the practical trick of reparameterization are all
developed here.

Together, these five chapters give a self-contained toolkit for turning any
likelihood function into a concrete parameter estimate.


.. toctree::
   :maxdepth: 2
   :caption: Optimization

   gradient_methods
   newton_methods
   quasi_newton
   em_algorithm
   constrained_optimization
