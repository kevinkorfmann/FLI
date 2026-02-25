.. _ch12_gradient:

========================================
Chapter 12: Gradient Methods
========================================

.. raw:: html

   <!-- This chapter uses a running example: logistic regression for customer churn. -->

You have a dataset with 10,000 customers and 200 features. You want to fit a
logistic regression to predict who will churn. Newton's method would need a
200 x 200 Hessian at every iteration---expensive but feasible. Now imagine a
neural network with 10,000,000 parameters. That Hessian has
:math:`10^{14}` entries. It will never fit in memory.

Gradient methods solve this problem. They use only the gradient---a single
vector of the same dimension as the parameter---so the cost per iteration is
:math:`O(p)` instead of :math:`O(p^2)` or :math:`O(p^3)`. This chapter
derives every major variant from first principles, beginning with the Taylor
expansion that motivates the basic update rule, and ending with the adaptive
methods---AdaGrad, RMSProp, Adam---that power modern deep learning.

Because we are maximizing a log-likelihood, most of our discussion is phrased
as *gradient ascent*. The mathematics is identical to gradient *descent* on the
negative log-likelihood; only the sign differs.

.. admonition:: Why Gradient Methods Matter

   If you take away one family of algorithms from this entire book, let it be
   gradient methods.  Every optimizer you will meet in practice --- from the
   simplest line-search routine to the fanciest neural-network trainer --- is
   either a gradient method or builds directly on gradient information.
   Understanding how and why they work gives you the vocabulary to diagnose
   convergence problems, choose hyper-parameters, and decide when to reach for
   something more powerful like Newton's method.


12.1 Gradient Descent from the Taylor Expansion
================================================

Setting Up the Problem
----------------------

Suppose we want to minimize a differentiable function
:math:`f : \mathbb{R}^p \to \mathbb{R}`. Given a current iterate
:math:`\boldsymbol{\theta}_k`, we seek a new point
:math:`\boldsymbol{\theta}_{k+1}` with a smaller function value. The simplest
systematic way to choose a direction is to approximate :math:`f` locally by its
first-order Taylor expansion.

First-Order Taylor Approximation
---------------------------------

Think of it this way: if you are standing on a hilly surface and can only see
the slope directly beneath your feet, which way should you step to go downhill
fastest?  The Taylor expansion formalizes this local view.

Around :math:`\boldsymbol{\theta}_k`, the Taylor expansion reads

.. math::

   f(\boldsymbol{\theta}_k + \mathbf{d})
   \;\approx\;
   f(\boldsymbol{\theta}_k)
   \;+\;
   \nabla f(\boldsymbol{\theta}_k)^{\!\top} \mathbf{d},

where :math:`\mathbf{d} \in \mathbb{R}^p` is a displacement vector. We want to
choose :math:`\mathbf{d}` so that the right-hand side is as small as possible.
However, without a constraint on the size of :math:`\mathbf{d}`, we could make
the linear term arbitrarily negative by taking :math:`\|\mathbf{d}\| \to \infty`
--- but the approximation would then be useless. So we restrict
:math:`\|\mathbf{d}\| = \eta` for some small step size :math:`\eta > 0`.

This constraint is what keeps us honest: the Taylor approximation is only
accurate near the current point, so we should not wander too far from it.

Deriving the Steepest-Descent Direction
-----------------------------------------

We solve

.. math::

   \min_{\mathbf{d}} \;\nabla f(\boldsymbol{\theta}_k)^{\!\top} \mathbf{d}
   \quad\text{subject to}\quad
   \|\mathbf{d}\|_2 = \eta.

By the Cauchy--Schwarz inequality,

.. math::

   \nabla f(\boldsymbol{\theta}_k)^{\!\top} \mathbf{d}
   \;\geq\;
   -\|\nabla f(\boldsymbol{\theta}_k)\|_2 \,\|\mathbf{d}\|_2
   \;=\;
   -\eta\,\|\nabla f(\boldsymbol{\theta}_k)\|_2,

with equality when :math:`\mathbf{d}` points in the direction opposite to the
gradient. Therefore the optimal displacement is

.. math::

   \mathbf{d}^*
   = -\eta \,
   \frac{\nabla f(\boldsymbol{\theta}_k)}{\|\nabla f(\boldsymbol{\theta}_k)\|_2}.

Absorbing the normalization into the step size, we obtain the canonical
**gradient-descent update rule**:

.. math::
   :label: gd_update

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
     - \alpha_k \,\nabla f(\boldsymbol{\theta}_k),

where :math:`\alpha_k > 0` is the *learning rate* (or step size) at iteration
:math:`k`.

This single equation is the engine behind a vast number of algorithms.  Every
variant we will meet in this chapter --- momentum, AdaGrad, Adam --- is a
creative modification of this update. Let us see it run on a concrete problem.

.. code-block:: python

   # Gradient descent on a 2D quadratic: the formula becomes code
   # Update rule: theta = theta - alpha * grad
   import numpy as np

   np.random.seed(42)
   A = np.array([[3.0, 0.5], [0.5, 1.0]])   # positive-definite
   b = np.array([1.0, 2.0])
   theta = np.array([5.0, 5.0])              # starting point
   alpha = 0.25                               # learning rate

   x_star = np.linalg.solve(A, b)            # analytic solution

   print(f"{'Iter':>4s}  {'f(theta)':>10s}  {'||grad||':>10s}  {'||theta - theta*||':>18s}")
   print("-" * 50)
   for k in range(15):
       grad = A @ theta - b                   # gradient of f(theta) = 0.5*theta'A*theta - b'theta
       f_val = 0.5 * theta @ A @ theta - b @ theta
       dist = np.linalg.norm(theta - x_star)
       print(f"{k:4d}  {f_val:10.4f}  {np.linalg.norm(grad):10.4f}  {dist:18.6f}")
       theta = theta - alpha * grad           # THE update rule

   print(f"\nGD solution:    {theta}")
   print(f"Exact solution: {x_star}")

Notice how the code ``theta = theta - alpha * grad`` is a literal transcription
of :eq:`gd_update`. The table shows the function value, gradient norm, and
distance to the optimum shrinking at each step.

For **gradient ascent** --- the form we use when maximizing a log-likelihood
:math:`\ell(\boldsymbol{\theta})` --- the sign flips:

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
     + \alpha_k \,\nabla \ell(\boldsymbol{\theta}_k).


12.2 Step-Size Selection
=========================

The step size :math:`\alpha_k` is the single most important hyper-parameter in
gradient descent. Too large and the iterates diverge; too small and convergence
is glacially slow. Let us see both failure modes on our churn problem before
learning how to fix them.

.. admonition:: Intuition

   Imagine walking downhill in thick fog.  Your step size is how far you commit
   to walking before stopping to re-check the slope.  If you take huge strides,
   you might overshoot the valley floor and end up climbing the opposite wall.
   If you take tiny baby steps, you will eventually reach the bottom, but it
   could take all day.  Step-size selection is about finding the sweet spot.

.. code-block:: python

   # The effect of step size: too large, too small, just right
   import numpy as np
   from scipy.special import expit  # sigmoid function

   np.random.seed(42)

   # Generate a small logistic regression dataset (churn prediction)
   n, p = 500, 5
   X = np.random.randn(n, p)
   beta_true = np.array([0.8, -1.2, 0.5, 0.0, 1.5])
   prob = expit(X @ beta_true)
   y = (np.random.rand(n) < prob).astype(float)

   def log_likelihood(beta, X, y):
       z = X @ beta
       return np.sum(y * z - np.log(1 + np.exp(z)))

   def gradient(beta, X, y):
       p_hat = expit(X @ beta)
       return X.T @ (y - p_hat)

   print(f"{'Step size':>10s}  {'Iter':>5s}  {'Log-lik':>10s}  {'||grad||':>10s}  {'Status'}")
   print("-" * 55)
   for alpha, label in [(0.1, "too large"), (0.0001, "too small"), (0.005, "good")]:
       beta = np.zeros(p)
       for k in range(200):
           grad = gradient(beta, X, y)
           beta = beta + alpha * grad    # gradient ASCENT
           ll = log_likelihood(beta, X, y)
           if np.isnan(ll) or np.isinf(ll):
               print(f"{alpha:10.4f}  {k:5d}  {'DIVERGED':>10s}  {'---':>10s}  {label}")
               break
       else:
           print(f"{alpha:10.4f}  {k:5d}  {ll:10.2f}  {np.linalg.norm(grad):10.4f}  {label}")

The too-large step size causes the log-likelihood to explode. The too-small
step size barely moves after 200 iterations. The "good" step size converges
efficiently.

Fixed Step Size
---------------

The simplest strategy is to use a constant :math:`\alpha_k = \alpha` for all
:math:`k`. For a function with :math:`L`-Lipschitz-continuous gradient (meaning
:math:`\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} -
\mathbf{y}\|`), gradient descent converges provided :math:`\alpha < 2/L`, and
the optimal fixed step size is :math:`\alpha = 1/L`. In practice :math:`L` is
rarely known, so one resorts to the adaptive strategies below.

Exact Line Search
-----------------

Choose :math:`\alpha_k` to minimize :math:`f` along the ray:

.. math::

   \alpha_k = \arg\min_{\alpha > 0}\;
   f\!\bigl(\boldsymbol{\theta}_k - \alpha\,\nabla f(\boldsymbol{\theta}_k)\bigr).

This is a one-dimensional optimization problem. For quadratic objectives
:math:`f(\boldsymbol{\theta}) = \tfrac{1}{2}\boldsymbol{\theta}^{\!\top}
\mathbf{A}\boldsymbol{\theta} - \mathbf{b}^{\!\top}\boldsymbol{\theta}`, the
exact solution is

.. math::

   \alpha_k
   = \frac{\|\nabla f_k\|^2}{\nabla f_k^{\!\top} \mathbf{A}\,\nabla f_k},

where :math:`\nabla f_k = \mathbf{A}\boldsymbol{\theta}_k - \mathbf{b}`.
Exact line search is useful for analysis but rarely practical for
high-dimensional problems.

Backtracking Line Search (Armijo Condition)
---------------------------------------------

The **Armijo condition** provides a practical "sufficient decrease" check.
Given parameters :math:`0 < c_1 < 1` (typically :math:`c_1 = 10^{-4}`) and a
shrinkage factor :math:`0 < \rho < 1` (typically :math:`\rho = 0.5`):

1. Start with a trial step size :math:`\alpha`.
2. **While** the following condition is *violated*:

   .. math::

      f(\boldsymbol{\theta}_k - \alpha\,\nabla f_k)
      \;\leq\;
      f(\boldsymbol{\theta}_k)
      - c_1 \,\alpha\,\|\nabla f_k\|^2,

   set :math:`\alpha \leftarrow \rho\,\alpha`.

3. Accept :math:`\alpha_k = \alpha`.

The left-hand side is the actual decrease; the right-hand side demands that it
be at least a fraction :math:`c_1` of the decrease predicted by the linear
model. This is cheap (one function evaluation per trial) and robust.

.. code-block:: python

   # Backtracking line search on the churn logistic regression
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 500, 5
   X = np.random.randn(n, p)
   beta_true = np.array([0.8, -1.2, 0.5, 0.0, 1.5])
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   def neg_ll(beta):
       z = X @ beta
       return -np.sum(y * z - np.log(1 + np.exp(z)))

   def neg_ll_grad(beta):
       return -(X.T @ (y - expit(X @ beta)))

   beta = np.zeros(p)
   c1, rho = 1e-4, 0.5

   print(f"{'Iter':>4s}  {'neg-LL':>10s}  {'||grad||':>10s}  {'step size':>10s}")
   print("-" * 42)
   for k in range(50):
       g = neg_ll_grad(beta)
       f_old = neg_ll(beta)
       alpha = 1.0
       # Backtracking: shrink alpha until Armijo condition holds
       while neg_ll(beta - alpha * g) > f_old - c1 * alpha * np.dot(g, g):
           alpha *= rho
       beta = beta - alpha * g
       if k % 5 == 0 or k < 5:
           print(f"{k:4d}  {neg_ll(beta):10.4f}  {np.linalg.norm(g):10.4f}  {alpha:10.6f}")

   print(f"\nRecovered beta: {np.round(beta, 3)}")
   print(f"True beta:      {beta_true}")

Notice how the step size adapts: early iterations use larger steps when the
gradient is steep, and later iterations take smaller steps as the optimum is
approached.

Wolfe Conditions
----------------

The Armijo condition alone can accept steps that are too short. The **Wolfe
conditions** add a *curvature condition*:

.. math::

   \nabla f(\boldsymbol{\theta}_k - \alpha\,\nabla f_k)^{\!\top}
   (-\nabla f_k)
   \;\geq\;
   c_2 \,(-\nabla f_k)^{\!\top}(-\nabla f_k)
   = c_2\,\|\nabla f_k\|^2,

with :math:`c_1 < c_2 < 1` (typically :math:`c_2 = 0.9`). Together with the
Armijo condition, this ensures the step is neither too long nor too short. The
Wolfe conditions are essential for quasi-Newton methods (:ref:`ch14_quasi_newton`)
because they guarantee positive definiteness of the BFGS update.


12.3 Convergence Analysis
==========================

We sketch the convergence theory for gradient descent with a fixed step size
:math:`\alpha = 1/L`.

.. admonition:: Why Care About Convergence Rates?

   Convergence rates tell you *how many iterations you need* to reach a given
   accuracy.  A rate of :math:`O(1/k)` means that halving the error requires
   roughly doubling the number of iterations.  A linear rate like
   :math:`(1-\mu/L)^k` means each iteration shaves off a fixed fraction of the
   remaining error --- much faster for well-conditioned problems.  Knowing
   these rates helps you decide whether gradient descent is good enough or
   whether you need a heavier tool.

Convex Functions
----------------

If :math:`f` is convex with :math:`L`-Lipschitz gradient, then after :math:`k`
iterations,

.. math::

   f(\boldsymbol{\theta}_k) - f(\boldsymbol{\theta}^*)
   \;\leq\;
   \frac{L\,\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2k}.

This is a :math:`O(1/k)` rate --- sublinear convergence.

**Proof sketch.** The Lipschitz gradient condition implies the *descent lemma*:

.. math::

   f(\boldsymbol{\theta}_{k+1})
   \leq f(\boldsymbol{\theta}_k)
   - \frac{1}{2L}\|\nabla f(\boldsymbol{\theta}_k)\|^2.

Summing this telescope and using convexity
(:math:`f(\boldsymbol{\theta}_k) - f^* \leq \nabla f_k^{\!\top}(\boldsymbol{\theta}_k - \boldsymbol{\theta}^*)`)
yields the :math:`O(1/k)` bound.

Strongly Convex Functions
--------------------------

If :math:`f` is additionally :math:`\mu`-strongly convex
(:math:`f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^{\!\top}(\mathbf{y}-\mathbf{x}) + \frac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2`),
then gradient descent achieves **linear convergence** (exponential decrease):

.. math::

   f(\boldsymbol{\theta}_k) - f(\boldsymbol{\theta}^*)
   \;\leq\;
   \left(1 - \frac{\mu}{L}\right)^k
   \bigl[f(\boldsymbol{\theta}_0) - f(\boldsymbol{\theta}^*)\bigr].

The ratio :math:`\kappa = L/\mu` is the *condition number*. When :math:`\kappa`
is large (the function is "ill-conditioned"), convergence is slow. This
motivates preconditioning and second-order methods
(:ref:`ch13_newton`, :ref:`ch14_quasi_newton`).

Let us verify this theory numerically. We run gradient descent on quadratics
with different condition numbers and watch the convergence rate match the
prediction.

.. code-block:: python

   # Convergence rate vs. condition number: theory meets practice
   import numpy as np

   np.random.seed(42)

   def gd_quadratic(L, mu, n_iters=80):
       """Run GD on f(x) = 0.5*(L*x1^2 + mu*x2^2) and track errors."""
       theta = np.array([1.0, 1.0])
       alpha = 2.0 / (L + mu)          # optimal fixed step size
       errors = []
       for _ in range(n_iters):
           grad = np.array([L * theta[0], mu * theta[1]])
           theta = theta - alpha * grad
           errors.append(0.5 * (L * theta[0]**2 + mu * theta[1]**2))
       return errors

   print(f"{'kappa':>6s}  {'Predicted rate':>14s}  {'Observed rate':>14s}  "
         f"{'f-f* after 80':>14s}")
   print("-" * 56)
   for kappa in [2, 10, 50, 200]:
       L, mu = float(kappa), 1.0
       errs = gd_quadratic(L, mu)
       # Observed rate: (err[k] / err[k-1]) for large k
       observed_rate = errs[-1] / errs[-2] if errs[-2] > 0 else 0
       predicted_rate = ((kappa - 1) / (kappa + 1))**2  # per-iteration on f
       print(f"{kappa:6d}  {predicted_rate:14.6f}  {observed_rate:14.6f}  {errs[-1]:14.2e}")

The table confirms that ill-conditioned problems (:math:`\kappa = 200`) converge
much more slowly, exactly as the theory predicts.


12.4 Stochastic Gradient Descent
==================================

Motivation
----------

In statistics and machine learning the objective is often an average over
:math:`n` observations:

.. math::

   f(\boldsymbol{\theta})
   = \frac{1}{n}\sum_{i=1}^{n} f_i(\boldsymbol{\theta}),

where :math:`f_i` is the contribution of the :math:`i`-th data point (for
example, :math:`f_i = -\log p(x_i \mid \boldsymbol{\theta})`). Computing the
full gradient :math:`\nabla f = \frac{1}{n}\sum_i \nabla f_i` costs :math:`O(n)`
per iteration. When :math:`n` is in the millions, this is prohibitive.

Here is the core insight: you do not need the *exact* gradient to make
progress.  An unbiased *estimate* of the gradient, even a noisy one, still
moves you in the right direction on average.

The SGD Update
--------------

**Stochastic gradient descent (SGD)** replaces the full gradient with a
single-sample (or mini-batch) estimate. At iteration :math:`k`, draw an index
:math:`i_k` uniformly at random from :math:`\{1,\dots,n\}` and update

.. math::
   :label: sgd_update

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   - \alpha_k \,\nabla f_{i_k}(\boldsymbol{\theta}_k).

Because :math:`\mathbb{E}[\nabla f_{i_k}] = \nabla f`, the stochastic gradient
is an **unbiased estimator** of the true gradient. The cost per iteration drops
from :math:`O(n)` to :math:`O(1)`.

Let us see both full-batch GD and SGD fitting our churn logistic regression,
side by side. SGD uses 10x more "iterations" but each one touches only a single
data point, making it 500x cheaper in total work per pass.

.. code-block:: python

   # Full-batch gradient ascent vs. SGD on logistic regression (churn)
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 2000, 10
   X = np.random.randn(n, p)
   beta_true = np.random.randn(p) * 0.5
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   def log_lik(beta):
       z = X @ beta
       return np.sum(y * z - np.log(1 + np.exp(z)))

   # Full-batch gradient ascent
   beta_gd = np.zeros(p)
   print("--- Full-batch Gradient Ascent ---")
   print(f"{'Iter':>4s}  {'Log-lik':>12s}  {'||grad||':>10s}")
   for k in range(100):
       grad = X.T @ (y - expit(X @ beta_gd))
       beta_gd = beta_gd + 0.001 * grad
       if k % 20 == 0:
           print(f"{k:4d}  {log_lik(beta_gd):12.2f}  {np.linalg.norm(grad):10.4f}")

   # SGD with decaying step size
   beta_sgd = np.zeros(p)
   print("\n--- Stochastic Gradient Ascent ---")
   print(f"{'Epoch':>5s}  {'Log-lik':>12s}  {'||beta - beta_gd||':>20s}")
   for epoch in range(10):
       perm = np.random.permutation(n)
       for j in range(n):
           i = perm[j]
           grad_i = X[i] * (y[i] - expit(X[i] @ beta_sgd))
           lr = 0.05 / (1 + 0.001 * (epoch * n + j))
           beta_sgd = beta_sgd + lr * grad_i
       print(f"{epoch:5d}  {log_lik(beta_sgd):12.2f}  "
             f"{np.linalg.norm(beta_sgd - beta_gd):20.6f}")

   print(f"\nMax |beta_sgd - beta_gd|: {np.max(np.abs(beta_sgd - beta_gd)):.4f}")

Notice the SGD log-likelihood is noisier epoch to epoch, but it reaches the
same neighborhood as full-batch GD while doing much less work per step.

Mini-Batch SGD
--------------

In practice one draws a mini-batch :math:`\mathcal{B}_k \subset \{1,\dots,n\}`
of size :math:`|\mathcal{B}_k| = B` and uses

.. math::

   \mathbf{g}_k = \frac{1}{B}\sum_{i \in \mathcal{B}_k} \nabla f_i(\boldsymbol{\theta}_k).

The variance of this estimator is

.. math::

   \operatorname{Var}(\mathbf{g}_k)
   = \frac{1}{B}\,\operatorname{Var}\bigl(\nabla f_i(\boldsymbol{\theta}_k)\bigr),

so increasing :math:`B` reduces variance (and noise) by a factor of
:math:`1/B`, but each iteration becomes :math:`B` times more expensive.
Typical values of :math:`B` range from 32 to 512 in deep learning.

.. code-block:: python

   # Effect of mini-batch size on gradient variance
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 2000, 10
   X = np.random.randn(n, p)
   beta = np.random.randn(p) * 0.3
   y = (np.random.rand(n) < expit(X @ beta)).astype(float)

   # True full gradient
   full_grad = X.T @ (y - expit(X @ beta)) / n

   print(f"{'Batch size':>10s}  {'Mean ||g - g_full||':>20s}  {'Variance reduction':>18s}")
   print("-" * 55)
   for B in [1, 8, 32, 128, 512]:
       errors = []
       for _ in range(200):
           idx = np.random.choice(n, B, replace=False)
           g_batch = X[idx].T @ (y[idx] - expit(X[idx] @ beta)) / B
           errors.append(np.linalg.norm(g_batch - full_grad))
       mean_err = np.mean(errors)
       print(f"{B:10d}  {mean_err:20.6f}  {1.0/B:18.4f}")

The error drops as :math:`1/\sqrt{B}`, confirming the variance formula.

Learning-Rate Schedules
-----------------------

With a fixed step size, SGD cannot converge exactly because the noise in the
gradient estimate causes the iterates to fluctuate around the optimum. The
classical remedy is a **decaying step size** satisfying

.. math::

   \sum_{k=0}^{\infty} \alpha_k = \infty,
   \qquad
   \sum_{k=0}^{\infty} \alpha_k^2 < \infty.

The first condition ensures the iterates can reach any point; the second
ensures the noise is eventually damped. A standard choice is
:math:`\alpha_k = c / (k + k_0)` for constants :math:`c, k_0 > 0`.

.. admonition:: Common Pitfall

   A frequent mistake is to use a learning rate that is too large for SGD.
   Unlike full-batch gradient descent, SGD has noisy updates, so the effective
   learning rate needs to be smaller.  If you see the loss oscillating wildly
   or diverging, the first thing to try is a smaller learning rate or a more
   aggressive decay schedule.


12.5 Momentum
==============

Plain gradient descent (or SGD) can oscillate in narrow valleys, making slow
progress along the bottom while bouncing between the walls. Momentum methods
address this by accumulating a "velocity" that smooths out oscillations.

.. admonition:: Intuition

   Picture a marble rolling down a curved surface.  Unlike a mathematical
   point that teleports to whatever the gradient says, the marble has *inertia*
   --- it keeps some of its previous velocity.  When the surface zigzags, the
   marble's inertia carries it through the oscillations, and it races along the
   valley floor.  That physical intuition is exactly what momentum does for
   optimization.

Polyak (Heavy-Ball) Momentum
-----------------------------

Polyak (1964) proposed augmenting the gradient step with a fraction of the
previous displacement:

.. math::
   :label: polyak_momentum

   \mathbf{v}_{k+1} &= \beta\,\mathbf{v}_k - \alpha\,\nabla f(\boldsymbol{\theta}_k), \\
   \boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k + \mathbf{v}_{k+1},

where :math:`\mathbf{v}_0 = \mathbf{0}` and :math:`0 \leq \beta < 1` is the
momentum coefficient (typically :math:`\beta = 0.9`).

**Why it helps.** Expanding the recursion shows that the effective update is a
weighted average of all past gradients with exponentially decaying weights:

.. math::

   \mathbf{v}_{k+1}
   = -\alpha \sum_{j=0}^{k} \beta^{k-j}\,\nabla f(\boldsymbol{\theta}_j).

Components of the gradient that are consistent across iterations accumulate;
components that alternate in sign cancel out. This damps oscillations and
accelerates progress along consistent directions.

.. code-block:: python

   # Momentum vs. plain GD on an ill-conditioned quadratic
   import numpy as np

   np.random.seed(42)
   # Condition number 50: GD oscillates, momentum is smooth
   A = np.diag([50.0, 1.0])
   b = np.array([1.0, 1.0])
   x_star = np.linalg.solve(A, b)

   def f(theta):
       return 0.5 * theta @ A @ theta - b @ theta

   # Plain GD
   theta_gd = np.array([5.0, 5.0])
   alpha_gd = 2.0 / (50 + 1)  # optimal for GD

   # GD with momentum
   theta_mom = np.array([5.0, 5.0])
   vel = np.zeros(2)
   alpha_mom = 0.03
   beta_mom = 0.9

   print(f"{'Iter':>4s}  {'f(GD)':>12s}  {'f(Momentum)':>12s}  {'Speedup':>8s}")
   print("-" * 42)
   for k in range(60):
       # Plain GD step
       grad_gd = A @ theta_gd - b
       theta_gd = theta_gd - alpha_gd * grad_gd

       # Momentum step
       grad_mom = A @ theta_mom - b
       vel = beta_mom * vel - alpha_mom * grad_mom
       theta_mom = theta_mom + vel

       if k % 10 == 0:
           f_gd = f(theta_gd) - f(x_star)
           f_mom = f(theta_mom) - f(x_star)
           ratio = f_gd / max(f_mom, 1e-15)
           print(f"{k:4d}  {f_gd:12.6f}  {f_mom:12.6f}  {ratio:8.1f}x")

Momentum converges dramatically faster on this ill-conditioned problem.

Nesterov Accelerated Gradient (NAG)
-------------------------------------

Nesterov (1983) refined momentum with a "lookahead" evaluation:

.. math::
   :label: nag

   \mathbf{v}_{k+1} &= \beta\,\mathbf{v}_k
     - \alpha\,\nabla f(\boldsymbol{\theta}_k + \beta\,\mathbf{v}_k), \\
   \boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k + \mathbf{v}_{k+1}.

The key difference is that the gradient is evaluated not at the current point
:math:`\boldsymbol{\theta}_k`, but at the "lookahead" point
:math:`\boldsymbol{\theta}_k + \beta\,\mathbf{v}_k` --- roughly where
momentum would carry us. This anticipatory correction leads to provably
better convergence:

- For convex :math:`L`-smooth functions, NAG achieves an
  :math:`O(1/k^2)` rate, compared with :math:`O(1/k)` for plain gradient
  descent. This is *optimal* among first-order methods (Nesterov, 1983).
- For strongly convex functions with condition number :math:`\kappa`, NAG
  converges at rate :math:`(1 - 1/\sqrt{\kappa})^k`, compared with
  :math:`(1 - 1/\kappa)^k` for gradient descent --- a substantial
  improvement when :math:`\kappa` is large.


12.6 Adaptive Methods
======================

A single global learning rate is rarely ideal when different parameters have
vastly different curvatures. Adaptive methods maintain per-parameter learning
rates that are adjusted automatically based on the history of gradients.

.. admonition:: What's the Intuition?

   Think about training a model where some features appear frequently and
   others rarely.  A single learning rate will either be too big for the
   frequent features (causing instability) or too small for the rare ones
   (causing slow learning).  Adaptive methods solve this by giving each
   parameter its own learning rate, automatically tuned from the gradient
   history.

AdaGrad
-------

Duchi, Hazan, and Singer (2011) introduced **AdaGrad**. The idea is to give
parameters with historically large gradients a *smaller* learning rate, and
parameters with small gradients a *larger* rate. Define

.. math::

   G_{k,j} = \sum_{t=0}^{k} \bigl(\nabla_j f(\boldsymbol{\theta}_t)\bigr)^2,

where :math:`\nabla_j f` denotes the :math:`j`-th component of the gradient.
The update for component :math:`j` is

.. math::
   :label: adagrad

   \theta_{k+1,j}
   = \theta_{k,j}
   - \frac{\alpha}{\sqrt{G_{k,j}} + \epsilon}\;\nabla_j f(\boldsymbol{\theta}_k),

where :math:`\epsilon > 0` (e.g., :math:`10^{-8}`) prevents division by zero.

In vector form, with :math:`\odot` denoting element-wise multiplication and
:math:`\mathbf{g}_k = \nabla f(\boldsymbol{\theta}_k)`:

.. math::

   \mathbf{G}_k &= \mathbf{G}_{k-1} + \mathbf{g}_k \odot \mathbf{g}_k, \\
   \boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k
     - \frac{\alpha}{\sqrt{\mathbf{G}_k} + \epsilon} \odot \mathbf{g}_k.

**Strengths.** AdaGrad works well for sparse features (common in NLP): rare
features get large effective step sizes.

**Weakness.** The accumulated sum :math:`G_{k,j}` grows monotonically, so the
effective step size shrinks to zero, potentially causing premature convergence.

RMSProp
-------

Hinton (unpublished lecture notes, 2012) proposed fixing AdaGrad's decaying
learning rate by using an *exponentially weighted moving average* instead of a
plain sum:

.. math::

   \mathbf{v}_k &= \gamma\,\mathbf{v}_{k-1}
     + (1-\gamma)\,\mathbf{g}_k \odot \mathbf{g}_k, \\
   \boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k
     - \frac{\alpha}{\sqrt{\mathbf{v}_k} + \epsilon} \odot \mathbf{g}_k,

with :math:`\gamma \approx 0.9`. This forgets old gradients, preventing the
denominator from growing without bound.

Adam
----

Kingma and Ba (2015) combined RMSProp's adaptive denominator with momentum in
the numerator, plus **bias correction** to account for initialization at zero.
The full update is:

.. math::
   :label: adam

   \mathbf{m}_k &= \beta_1\,\mathbf{m}_{k-1} + (1-\beta_1)\,\mathbf{g}_k,
   \qquad &\text{(first moment estimate)} \\
   \mathbf{v}_k &= \beta_2\,\mathbf{v}_{k-1} + (1-\beta_2)\,\mathbf{g}_k \odot \mathbf{g}_k,
   \qquad &\text{(second moment estimate)} \\
   \hat{\mathbf{m}}_k &= \frac{\mathbf{m}_k}{1 - \beta_1^k},
   \qquad &\text{(bias-corrected first moment)} \\
   \hat{\mathbf{v}}_k &= \frac{\mathbf{v}_k}{1 - \beta_2^k},
   \qquad &\text{(bias-corrected second moment)} \\
   \boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k
     - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_k} + \epsilon}\,\hat{\mathbf{m}}_k.

Default hyperparameters: :math:`\beta_1 = 0.9`, :math:`\beta_2 = 0.999`,
:math:`\alpha = 10^{-3}`, :math:`\epsilon = 10^{-8}`.

**Why bias correction?** At :math:`k=1` with :math:`\mathbf{m}_0 = \mathbf{0}`:

.. math::

   \mathbf{m}_1 = (1-\beta_1)\,\mathbf{g}_1.

The expected value is :math:`(1-\beta_1)\,\mathbb{E}[\mathbf{g}_1]`, which is
biased toward zero by a factor :math:`(1-\beta_1)`. Dividing by
:math:`1 - \beta_1^k` corrects this; as :math:`k \to \infty` the correction
vanishes since :math:`\beta_1^k \to 0`.

Now let us implement all of these from scratch and run them on the **same**
logistic regression churn problem, printing iteration-by-iteration progress so
we can compare convergence directly.

.. code-block:: python

   # Head-to-head: SGD, SGD+Momentum, AdaGrad, RMSProp, Adam on churn data
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 2000, 20
   X = np.random.randn(n, p)
   beta_true = np.random.randn(p) * 0.5
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   def neg_log_lik(beta):
       z = X @ beta
       return -np.sum(y * z - np.log(1 + np.exp(z))) / n

   def grad_nll(beta):
       return -(X.T @ (y - expit(X @ beta))) / n

   results = {}
   for name in ["SGD", "SGD+Mom", "AdaGrad", "RMSProp", "Adam"]:
       theta = np.zeros(p)
       m = np.zeros(p)       # first moment (Adam/momentum)
       v = np.zeros(p)       # second moment (Adam/RMSProp) or accum (AdaGrad)
       G = np.zeros(p)       # AdaGrad accumulator
       losses = []

       for k in range(1, 501):
           g = grad_nll(theta)

           if name == "SGD":
               theta = theta - 0.5 * g
           elif name == "SGD+Mom":
               m = 0.9 * m + g
               theta = theta - 0.05 * m
           elif name == "AdaGrad":
               G = G + g**2
               theta = theta - 0.5 * g / (np.sqrt(G) + 1e-8)
           elif name == "RMSProp":
               v = 0.9 * v + 0.1 * g**2
               theta = theta - 0.05 * g / (np.sqrt(v) + 1e-8)
           elif name == "Adam":
               m = 0.9 * m + 0.1 * g
               v = 0.999 * v + 0.001 * g**2
               m_hat = m / (1 - 0.9**k)
               v_hat = v / (1 - 0.999**k)
               theta = theta - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

           losses.append(neg_log_lik(theta))

       results[name] = losses

   # Print convergence table
   print(f"{'Iter':>5s}", end="")
   for name in results:
       print(f"  {name:>10s}", end="")
   print()
   print("-" * (5 + 12 * len(results)))
   for k in [0, 9, 24, 49, 99, 199, 499]:
       print(f"{k+1:5d}", end="")
       for name in results:
           print(f"  {results[name][k]:10.4f}", end="")
       print()

   # Final comparison
   print(f"\n{'Method':>10s}  {'Final neg-LL':>12s}")
   print("-" * 26)
   for name, losses in results.items():
       print(f"{name:>10s}  {losses[-1]:12.6f}")

Comparison of Adaptive Methods
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Per-param rate
     - Momentum
     - Bias correction
     - Typical use
   * - SGD + momentum
     - No
     - Yes
     - N/A
     - Convex, well-tuned
   * - AdaGrad
     - Yes
     - No
     - No
     - Sparse features
   * - RMSProp
     - Yes
     - No
     - No
     - Non-stationary
   * - Adam
     - Yes
     - Yes
     - Yes
     - General default


12.7 Application to Log-Likelihood Maximization
=================================================

In maximum likelihood estimation we maximize

.. math::

   \ell(\boldsymbol{\theta})
   = \sum_{i=1}^n \log p(x_i \mid \boldsymbol{\theta}).

This is equivalent to minimizing :math:`f(\boldsymbol{\theta}) = -\ell(\boldsymbol{\theta})`.
All the methods above apply directly.

Gradient Ascent for MLE
------------------------

The gradient-ascent update is

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   + \alpha_k \sum_{i=1}^n
     \frac{\partial}{\partial \boldsymbol{\theta}}
     \log p(x_i \mid \boldsymbol{\theta}_k).

Each summand :math:`\nabla_{\boldsymbol{\theta}} \log p(x_i \mid \boldsymbol{\theta})` is
the **score contribution** of observation :math:`i` (see
:ref:`ch13_newton` for its role in Fisher scoring).

The Full Churn Example
-----------------------

We now put everything together. We fit a logistic regression to predict
customer churn using gradient ascent with Adam, printing a detailed convergence
table at each epoch: log-likelihood, gradient norm, and effective step size.

.. code-block:: python

   # Full logistic regression MLE via gradient ascent with Adam
   # Scenario: predicting customer churn from usage features
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 5000, 30

   # Simulate churn data with realistic feature structure
   X = np.random.randn(n, p)
   # Only first 10 features matter; rest are noise
   beta_true = np.zeros(p)
   beta_true[:10] = np.array([0.8, -1.2, 0.5, 0.3, -0.7, 1.1, -0.4, 0.6, -0.9, 0.2])
   prob = expit(X @ beta_true)
   y = (np.random.rand(n) < prob).astype(float)
   print(f"Churn rate: {y.mean():.1%}")

   # Gradient ascent with Adam (mini-batch)
   beta = np.zeros(p)
   m, v = np.zeros(p), np.zeros(p)
   lr, B = 0.005, 128

   print(f"\n{'Epoch':>5s}  {'Log-lik':>12s}  {'||grad||':>10s}  "
         f"{'eff. step':>10s}  {'Accuracy':>10s}")
   print("-" * 55)

   global_step = 0
   for epoch in range(20):
       perm = np.random.permutation(n)
       for start in range(0, n, B):
           idx = perm[start:start + B]
           p_hat = expit(X[idx] @ beta)
           grad = X[idx].T @ (y[idx] - p_hat) / len(idx)

           global_step += 1
           m = 0.9 * m + 0.1 * grad
           v = 0.999 * v + 0.001 * grad**2
           m_hat = m / (1 - 0.9**global_step)
           v_hat = v / (1 - 0.999**global_step)
           step = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
           beta = beta + step

       # End-of-epoch diagnostics
       full_grad = X.T @ (y - expit(X @ beta)) / n
       ll = np.sum(y * (X @ beta) - np.log(1 + np.exp(X @ beta)))
       acc = np.mean((expit(X @ beta) > 0.5) == y)
       eff_step = np.mean(np.abs(step))
       print(f"{epoch:5d}  {ll:12.2f}  {np.linalg.norm(full_grad):10.6f}  "
             f"{eff_step:10.6f}  {acc:10.1%}")

   # Check recovery of true coefficients
   print(f"\nTrue nonzero betas:  {beta_true[:10]}")
   print(f"Estimated betas:     {np.round(beta[:10], 3)}")
   print(f"Max |noise coeff|:   {np.max(np.abs(beta[10:])):.4f}")

This output shows the full story: the log-likelihood climbing, the gradient
norm shrinking, the step size adapting, and the accuracy improving. The
recovered coefficients match the true values, and the noise features stay near
zero.

Stochastic MLE
--------------

For large :math:`n`, stochastic gradient ascent draws a mini-batch and updates

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   + \alpha_k \,\frac{n}{B}
     \sum_{i \in \mathcal{B}_k}
     \nabla_{\boldsymbol{\theta}} \log p(x_i \mid \boldsymbol{\theta}_k).

The factor :math:`n/B` rescales the mini-batch sum to approximate the full-data
gradient. This is the basis of *stochastic maximum likelihood* and is used
extensively in training deep generative models.

When to Use Gradient Methods for MLE
--------------------------------------

Gradient methods are the tool of choice when:

- The number of parameters :math:`p` is very large (thousands or more), making
  second-order methods impractical.
- The sample size :math:`n` is so large that full-batch gradient evaluation is
  too expensive.
- The model is a neural network or other complex, non-convex model where
  second-order information is hard to exploit.

For low-dimensional, smooth, concave log-likelihoods (e.g., GLMs with
:math:`p \ll n`), Newton-type methods (:ref:`ch13_newton`) or quasi-Newton
methods (:ref:`ch14_quasi_newton`) are usually far more efficient.


12.8 Summary
=============

This chapter derived gradient descent from a first-order Taylor expansion,
explored step-size strategies (fixed, Armijo backtracking, Wolfe conditions),
established convergence rates for convex and strongly convex functions, and
developed the stochastic, momentum, and adaptive variants that are standard in
modern practice. Gradient methods provide a versatile foundation; the next
chapters build on them by incorporating curvature information
(:ref:`ch13_newton`, :ref:`ch14_quasi_newton`) and by handling latent-variable
structure (:ref:`ch15_em`) and constraints (:ref:`ch16_constrained`).
