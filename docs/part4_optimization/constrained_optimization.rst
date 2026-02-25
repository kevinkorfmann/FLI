.. _ch16_constrained:

========================================
Chapter 16: Constrained Optimization
========================================

You manage a portfolio of 5 assets. You want to maximize expected return, but
you have a risk budget: the portfolio variance must not exceed 2%. The weights
must be non-negative (no short selling) and sum to one (fully invested). This
is a constrained optimization problem---and if you ignore the constraints, your
optimizer will happily tell you to put 500% of your money in the highest-return
asset and short-sell everything else.

.. code-block:: python

   # What happens when you ignore constraints
   import numpy as np

   np.random.seed(42)
   p = 5
   mu = np.array([0.12, 0.10, 0.07, 0.03, 0.14])   # expected returns
   A = np.random.randn(p, p) * 0.1
   Sigma = A.T @ A + 0.05 * np.eye(p)                # covariance of returns

   # Unconstrained "optimal" portfolio: just maximize mu^T w
   # Without constraints, w -> infinity in the direction of max return
   w_unconstrained = mu / np.linalg.norm(mu)  # just point at highest return
   w_unconstrained *= 10  # scale up for "more return"
   print("Unconstrained 'solution':")
   print(f"  Weights: {np.round(w_unconstrained, 2)}")
   print(f"  Sum of weights: {w_unconstrained.sum():.2f} (should be 1.00)")
   print(f"  Min weight: {w_unconstrained.min():.2f} (should be >= 0)")
   print(f"  Portfolio variance: {w_unconstrained @ Sigma @ w_unconstrained:.4f} (budget: 0.02)")
   print("\nThis is nonsense. We NEED constraints.")

In many likelihood problems the parameters cannot take arbitrary values.
Probabilities must be non-negative and sum to one. Variances must be positive.
Correlation matrices must be positive semi-definite. This chapter develops the
theory and algorithms for optimization under constraints, starting from
Lagrange multipliers for equality constraints, extending to the
Karush--Kuhn--Tucker (KKT) conditions for inequality constraints, and then
covering the main algorithmic approaches: augmented Lagrangian methods,
barrier (interior-point) methods, and penalty methods. We close with the
practical alternative of reparameterization, which transforms a constrained
problem into an unconstrained one.

.. admonition:: Why Constrained Optimization Matters

   Every time you fit a model with probabilities, variances, or proportions,
   you are doing constrained optimization --- whether you realize it or not.
   The unconstrained methods from earlier chapters (:ref:`ch12_gradient`,
   :ref:`ch13_newton`, :ref:`ch14_quasi_newton`) are powerful, but they can
   happily produce a negative variance or probabilities that sum to 1.3.
   Constrained optimization gives you the mathematical framework to keep
   parameters in their proper domains while still finding the optimum.


16.1 Equality Constraints and Lagrange Multipliers
====================================================

Problem Formulation
--------------------

Consider the problem

.. math::
   :label: eq_constrained

   \min_{\boldsymbol{\theta} \in \mathbb{R}^p}\;
   f(\boldsymbol{\theta})
   \qquad\text{subject to}\qquad
   h_j(\boldsymbol{\theta}) = 0,
   \quad j = 1, \dots, m,

where :math:`f` and :math:`h_1, \dots, h_m` are continuously differentiable.

Geometric Intuition
--------------------

At a constrained minimum :math:`\boldsymbol{\theta}^*`, the gradient of
:math:`f` cannot have a component in the feasible set --- otherwise we could
decrease :math:`f` by moving along the constraint surface. More precisely, the
gradient :math:`\nabla f(\boldsymbol{\theta}^*)` must be *normal* to the
constraint surface.

Think of it this way: you are hiking on a mountain and are confined to a trail
(the constraint surface).  At the lowest point of the trail, the slope of the
mountain must be perpendicular to the trail --- if there were any downhill
component along the trail, you could walk further and go lower.

The constraint surface is (locally) a manifold defined by
:math:`\mathbf{h}(\boldsymbol{\theta}) = \mathbf{0}`. Its tangent space at
:math:`\boldsymbol{\theta}^*` is the null space of the Jacobian matrix

.. math::

   \mathbf{J}_h(\boldsymbol{\theta}^*)
   = \begin{pmatrix}
     \nabla h_1(\boldsymbol{\theta}^*)^{\!\top} \\
     \vdots \\
     \nabla h_m(\boldsymbol{\theta}^*)^{\!\top}
     \end{pmatrix}
   \in \mathbb{R}^{m \times p}.

The normal space to the constraint surface is spanned by the rows of
:math:`\mathbf{J}_h`, i.e., by the gradients :math:`\nabla h_j`. The condition
that :math:`\nabla f` lies in this normal space is

.. math::

   \nabla f(\boldsymbol{\theta}^*)
   = \sum_{j=1}^m \lambda_j^*\,\nabla h_j(\boldsymbol{\theta}^*),

or equivalently

.. math::
   :label: tangent_cond

   \nabla f(\boldsymbol{\theta}^*)
   - \sum_{j=1}^m \lambda_j^*\,\nabla h_j(\boldsymbol{\theta}^*)
   = \mathbf{0}.

The scalars :math:`\lambda_1^*, \dots, \lambda_m^*` are the **Lagrange
multipliers**.

The Lagrangian
--------------

Define the **Lagrangian function**

.. math::
   :label: lagrangian

   \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\lambda})
   = f(\boldsymbol{\theta})
   - \sum_{j=1}^m \lambda_j\,h_j(\boldsymbol{\theta}).

Reading this formula: the Lagrangian takes the objective :math:`f` and subtracts
a penalty for each constraint, weighted by the multiplier :math:`\lambda_j`.
If :math:`\lambda_j` is large, the corresponding constraint exerts a strong
pull on the solution. The Lagrangian converts the constrained problem into an
unconstrained one: finding a stationary point of
:math:`\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\lambda})` with respect to
*both* :math:`\boldsymbol{\theta}` and :math:`\boldsymbol{\lambda}` gives you
the constrained optimum and the multipliers simultaneously.

The necessary conditions for a constrained minimum are found by setting the
partial derivatives to zero:

.. math::
   :label: lag_conditions

   \nabla_{\boldsymbol{\theta}} \mathcal{L}
   &= \nabla f(\boldsymbol{\theta})
   - \sum_{j=1}^m \lambda_j\,\nabla h_j(\boldsymbol{\theta})
   = \mathbf{0}, \\
   \nabla_{\boldsymbol{\lambda}} \mathcal{L}
   &= -\mathbf{h}(\boldsymbol{\theta}) = \mathbf{0}.

The first equation is :eq:`tangent_cond` --- the gradient of :math:`f` must lie
in the span of the constraint gradients. The second equation is simply the
requirement that all constraints are satisfied (:math:`h_j = 0`). Together
they form a system of :math:`p + m` equations in :math:`p + m` unknowns
:math:`(\boldsymbol{\theta}, \boldsymbol{\lambda})`, which can be solved by
Newton's method or other root-finding techniques.

Let us solve this both analytically and numerically and compare.

.. code-block:: python

   # Lagrange multipliers: analytical vs. numerical
   # Problem: minimize x^2 + 2y^2 subject to x + y = 1
   import numpy as np
   from scipy.optimize import fsolve

   # --- Analytical solution ---
   # Lagrangian: L = x^2 + 2y^2 - lambda*(x + y - 1)
   # dL/dx = 2x - lambda = 0  =>  x = lambda/2
   # dL/dy = 4y - lambda = 0  =>  y = lambda/4
   # Constraint: x + y = 1  =>  lambda/2 + lambda/4 = 1  =>  lambda = 4/3
   lam_analytic = 4.0 / 3.0
   x_analytic = lam_analytic / 2.0
   y_analytic = lam_analytic / 4.0

   # --- Numerical solution (fsolve) ---
   def lagrange_system(vars):
       x, y, lam = vars
       return [2*x - lam,        # dL/dx = 0
               4*y - lam,        # dL/dy = 0
               x + y - 1]        # constraint

   sol = fsolve(lagrange_system, [0.5, 0.5, 1.0])
   x_num, y_num, lam_num = sol

   print("Minimize x^2 + 2y^2 subject to x + y = 1")
   print(f"\n{'':>12s}  {'x':>8s}  {'y':>8s}  {'lambda':>8s}  {'f(x,y)':>8s}")
   print("-" * 50)
   print(f"{'Analytical':>12s}  {x_analytic:8.4f}  {y_analytic:8.4f}  "
         f"{lam_analytic:8.4f}  {x_analytic**2 + 2*y_analytic**2:8.4f}")
   print(f"{'Numerical':>12s}  {x_num:8.4f}  {y_num:8.4f}  "
         f"{lam_num:8.4f}  {x_num**2 + 2*y_num**2:8.4f}")
   print(f"\nConstraint satisfied: x + y = {x_num + y_num:.8f}")

.. admonition:: Intuition

   The Lagrange multiplier :math:`\lambda^*` has a beautiful economic
   interpretation: it tells you the *rate* at which the optimal value of
   :math:`f` changes if you relax the constraint by an infinitesimal amount.
   In operations research, this is called the **shadow price** of the
   constraint.  If :math:`\lambda^*` is large, the constraint is "expensive"
   --- the objective would improve significantly if the constraint were
   loosened.

.. code-block:: python

   # Shadow price interpretation: relax the constraint and observe
   import numpy as np
   from scipy.optimize import fsolve

   def solve_relaxed(c):
       """Solve min x^2 + 2y^2 s.t. x + y = c, return (x*, y*, f*)."""
       # Analytical: x = 2c/3, y = c/3, f = 2c^2/3
       return 2*c/3, c/3, 2*c**2/3

   print("Shadow price of the constraint x + y = c:")
   print(f"{'c':>6s}  {'x*':>8s}  {'y*':>8s}  {'f*':>8s}  {'df*/dc':>8s}")
   print("-" * 44)
   prev_f = None
   for c in [0.9, 0.95, 1.0, 1.05, 1.1]:
       x, y, f = solve_relaxed(c)
       df_dc = f - prev_f if prev_f is not None else 0.0
       # The analytical derivative df*/dc = 4c/3, at c=1 this is 4/3 = lambda*
       print(f"{c:6.2f}  {x:8.4f}  {y:8.4f}  {f:8.4f}  {df_dc / 0.05:8.4f}")
       prev_f = f

   print(f"\ndf*/dc at c=1 = {4.0/3.0:.4f} = lambda* (the Lagrange multiplier)")

Second-Order Sufficient Conditions
------------------------------------

The first-order conditions :eq:`lag_conditions` are necessary but not
sufficient. A second-order sufficient condition for a local minimum is that the
**bordered Hessian** (or equivalently the Hessian of the Lagrangian restricted
to the tangent space of the constraints) is positive definite:

.. math::

   \mathbf{v}^{\!\top}
   \nabla^2_{\boldsymbol{\theta}\boldsymbol{\theta}} \mathcal{L}
   (\boldsymbol{\theta}^*, \boldsymbol{\lambda}^*)\,\mathbf{v} > 0
   \quad\text{for all } \mathbf{v} \neq \mathbf{0}
   \text{ with } \mathbf{J}_h(\boldsymbol{\theta}^*)\mathbf{v} = \mathbf{0},

where

.. math::

   \nabla^2_{\boldsymbol{\theta}\boldsymbol{\theta}} \mathcal{L}
   = \nabla^2 f - \sum_{j=1}^m \lambda_j\,\nabla^2 h_j.


16.2 Inequality Constraints and KKT Conditions
=================================================

Problem Formulation
--------------------

The general constrained optimization problem is

.. math::
   :label: gen_constrained

   \min_{\boldsymbol{\theta}}\; f(\boldsymbol{\theta})
   \qquad\text{subject to}\qquad
   &h_j(\boldsymbol{\theta}) = 0, \quad j = 1, \dots, m, \\
   &g_i(\boldsymbol{\theta}) \leq 0, \quad i = 1, \dots, q.

Inequality constraints bring a new subtlety: at the solution, some constraints
will be *active* (holding with equality) and others will be *inactive* (strict
inequality).  The KKT conditions elegantly handle both cases.

Deriving the KKT Conditions
-----------------------------

Define the Lagrangian with both equality and inequality constraints:

.. math::
   :label: kkt_lagrangian

   \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\lambda}, \boldsymbol{\mu})
   = f(\boldsymbol{\theta})
   - \sum_{j=1}^m \lambda_j\,h_j(\boldsymbol{\theta})
   + \sum_{i=1}^q \mu_i\,g_i(\boldsymbol{\theta}).

At a constrained minimum :math:`(\boldsymbol{\theta}^*, \boldsymbol{\lambda}^*,
\boldsymbol{\mu}^*)`, the following **Karush--Kuhn--Tucker (KKT) conditions**
must hold (under a constraint qualification such as LICQ):

1. **Stationarity:**

   .. math::

      \nabla_{\boldsymbol{\theta}} \mathcal{L}
      = \nabla f(\boldsymbol{\theta}^*)
      - \sum_{j=1}^m \lambda_j^*\,\nabla h_j(\boldsymbol{\theta}^*)
      + \sum_{i=1}^q \mu_i^*\,\nabla g_i(\boldsymbol{\theta}^*)
      = \mathbf{0}.

2. **Primal feasibility:**

   .. math::

      h_j(\boldsymbol{\theta}^*) = 0 \quad\forall\, j,
      \qquad
      g_i(\boldsymbol{\theta}^*) \leq 0 \quad\forall\, i.

3. **Dual feasibility:**

   .. math::

      \mu_i^* \geq 0 \quad\forall\, i.

4. **Complementary slackness:**

   .. math::
      :label: comp_slack

      \mu_i^*\,g_i(\boldsymbol{\theta}^*) = 0 \quad\forall\, i.

Complementary Slackness Explained
-----------------------------------

Condition :eq:`comp_slack` says that for each inequality constraint, either the
constraint is *active* (:math:`g_i(\boldsymbol{\theta}^*) = 0`) and the
multiplier can be positive, or the constraint is *inactive*
(:math:`g_i(\boldsymbol{\theta}^*) < 0`) and the multiplier must be zero.

**Intuition:** An inactive constraint does not affect the solution --- it is as
if the constraint were not there. Only active constraints "push" on the
solution, and the multiplier :math:`\mu_i^*` measures the strength of that push.

The KKT conditions are *sufficient* for a global minimum when :math:`f` is
convex and the constraints define a convex feasible set (i.e., :math:`g_i`
convex, :math:`h_j` affine).

Let us verify the KKT conditions in code on the portfolio problem.

.. code-block:: python

   # Checking KKT conditions on the portfolio problem
   import numpy as np
   from scipy.optimize import minimize

   np.random.seed(42)
   p = 5
   mu = np.array([0.12, 0.10, 0.07, 0.03, 0.14])
   A = np.random.randn(p, p) * 0.1
   Sigma = A.T @ A + 0.05 * np.eye(p)
   sigma2_max = 0.02

   # Solve with scipy
   constraints = [
       {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
       {'type': 'ineq', 'fun': lambda w: sigma2_max - w @ Sigma @ w},
   ]
   bounds = [(0, 1)] * p
   w0 = np.ones(p) / p
   res = minimize(lambda w: -mu @ w, w0, method='SLSQP',
                  bounds=bounds, constraints=constraints)
   w_star = res.x

   # Check all four KKT conditions
   print("=== KKT Condition Check ===\n")

   # 1. Primal feasibility
   eq_violation = abs(np.sum(w_star) - 1)
   risk = w_star @ Sigma @ w_star
   print("1. PRIMAL FEASIBILITY")
   print(f"   Sum of weights = {np.sum(w_star):.8f} (violation: {eq_violation:.2e})")
   print(f"   Portfolio variance = {risk:.6f} <= {sigma2_max} : "
         f"{'OK' if risk <= sigma2_max + 1e-8 else 'VIOLATED'}")
   for i in range(p):
       status = "ACTIVE" if w_star[i] < 1e-6 else "inactive"
       print(f"   w[{i}] = {w_star[i]:.6f} >= 0 : {status}")

   # 2. Complementary slackness (for bound constraints)
   print("\n2. COMPLEMENTARY SLACKNESS")
   print("   For each w_i >= 0 constraint:")
   for i in range(p):
       # mu_i * w_i should = 0 (either mu_i = 0 or w_i = 0)
       active = w_star[i] < 1e-6
       print(f"   Asset {i}: w={w_star[i]:.6f}, "
             f"{'constraint active (multiplier can be > 0)' if active else 'constraint inactive (multiplier = 0)'}")

   # 3. Risk constraint
   risk_active = abs(risk - sigma2_max) < 1e-4
   print(f"\n   Risk constraint: {'ACTIVE' if risk_active else 'inactive'} "
         f"(variance = {risk:.6f}, budget = {sigma2_max})")

   # Solution quality
   print(f"\n=== Optimal Portfolio ===")
   print(f"{'Asset':>5s}  {'Weight':>8s}  {'Return':>8s}  {'Contribution':>12s}")
   print("-" * 38)
   for i in range(p):
       print(f"{i:5d}  {w_star[i]:8.4f}  {mu[i]:8.2%}  {w_star[i]*mu[i]:12.4%}")
   print(f"{'Total':>5s}  {np.sum(w_star):8.4f}  {mu @ w_star:8.2%}")


16.3 Examples in Maximum Likelihood
=====================================

Multinomial MLE with Simplex Constraint
-----------------------------------------

Suppose :math:`X = (X_1, \dots, X_K) \sim \text{Multinomial}(n, \boldsymbol{\pi})`
with :math:`\boldsymbol{\pi} = (\pi_1, \dots, \pi_K)`. The log-likelihood is

.. math::

   \ell(\boldsymbol{\pi})
   = \sum_{k=1}^K X_k \log \pi_k + \text{const}.

We maximize subject to :math:`\sum_k \pi_k = 1` and :math:`\pi_k \geq 0`.

**Using Lagrange multipliers** for the equality constraint (the positivity
constraints turn out to be inactive at the interior solution):

.. math::

   \mathcal{L} = \sum_{k=1}^K X_k \log \pi_k + \lambda\Bigl(1 - \sum_{k=1}^K \pi_k\Bigr).

Setting :math:`\partial\mathcal{L}/\partial\pi_k = 0`:

.. math::

   \frac{X_k}{\pi_k} - \lambda = 0
   \quad\Longrightarrow\quad
   \pi_k = \frac{X_k}{\lambda}.

Summing over :math:`k`: :math:`1 = n/\lambda`, so :math:`\lambda = n` and

.. math::

   \hat{\pi}_k = \frac{X_k}{n},

as expected. Let us verify this Lagrange multiplier derivation numerically.

.. code-block:: python

   # Multinomial MLE: Lagrange multiplier solution vs. direct formula
   import numpy as np
   from scipy.optimize import minimize

   np.random.seed(42)
   K = 6
   counts = np.array([45, 30, 15, 5, 3, 2])
   n = counts.sum()

   # Analytical solution from Lagrange multipliers
   pi_analytical = counts / n

   # Numerical solution using scipy with constraints
   constraints = [{'type': 'eq', 'fun': lambda pi: np.sum(pi) - 1}]
   bounds = [(1e-10, 1)] * K
   pi0 = np.ones(K) / K
   res = minimize(lambda pi: -np.sum(counts * np.log(pi)), pi0,
                  method='SLSQP', bounds=bounds, constraints=constraints)
   pi_numerical = res.x

   print(f"{'Category':>8s}  {'Count':>5s}  {'Analytical':>10s}  {'Numerical':>10s}  {'Match':>6s}")
   print("-" * 48)
   for k in range(K):
       match = "yes" if abs(pi_analytical[k] - pi_numerical[k]) < 1e-6 else "no"
       print(f"{k+1:8d}  {counts[k]:5d}  {pi_analytical[k]:10.4f}  "
             f"{pi_numerical[k]:10.4f}  {match:>6s}")
   print(f"\nLagrange multiplier lambda = n = {n}")

The Full Portfolio Optimization
----------------------------------

Now let us solve the portfolio problem properly with constraints, comparing
the analytical Lagrange approach with numerical optimization.

.. code-block:: python

   # Portfolio optimization: solve analytically (equality only) and numerically
   import numpy as np
   from scipy.optimize import minimize

   np.random.seed(42)
   p = 5
   mu = np.array([0.12, 0.10, 0.07, 0.03, 0.14])
   A = np.random.randn(p, p) * 0.1
   Sigma = A.T @ A + 0.05 * np.eye(p)

   # Minimum-variance portfolio (equality constraint only: w^T 1 = 1)
   # Lagrangian: L = 0.5 w^T Sigma w - lambda*(1^T w - 1)
   # dL/dw = Sigma w - lambda 1 = 0  =>  w = lambda * Sigma^{-1} 1
   # Constraint: 1^T w = 1  =>  lambda = 1 / (1^T Sigma^{-1} 1)
   Sigma_inv = np.linalg.inv(Sigma)
   ones = np.ones(p)
   lam_mv = 1.0 / (ones @ Sigma_inv @ ones)
   w_mv_analytic = lam_mv * Sigma_inv @ ones

   # Numerical
   res_mv = minimize(lambda w: 0.5 * w @ Sigma @ w, np.ones(p)/p,
                     method='SLSQP',
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
   w_mv_numerical = res_mv.x

   print("Minimum-variance portfolio (equality constraint only):")
   print(f"{'Asset':>5s}  {'Analytic':>10s}  {'Numerical':>10s}")
   print("-" * 30)
   for i in range(p):
       print(f"{i:5d}  {w_mv_analytic[i]:10.4f}  {w_mv_numerical[i]:10.4f}")
   print(f"{'Var':>5s}  {w_mv_analytic @ Sigma @ w_mv_analytic:10.6f}  "
         f"{w_mv_numerical @ Sigma @ w_mv_numerical:10.6f}")

   # Full problem: maximize return with risk budget + no short selling
   sigma2_max = 0.02
   constraints_full = [
       {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
       {'type': 'ineq', 'fun': lambda w: sigma2_max - w @ Sigma @ w},
   ]
   bounds = [(0, 1)] * p
   res_full = minimize(lambda w: -mu @ w, np.ones(p)/p,
                       method='SLSQP', bounds=bounds, constraints=constraints_full)

   print(f"\nFull constrained portfolio (risk budget {sigma2_max}):")
   print(f"  Weights: {np.round(res_full.x, 4)}")
   print(f"  Return:  {mu @ res_full.x:.4f}")
   print(f"  Risk:    {res_full.x @ Sigma @ res_full.x:.6f}")

Positive-Definiteness Constraint on Covariance
-------------------------------------------------

In multivariate normal MLE, :math:`\boldsymbol{\Sigma}` must be positive
definite. This is a matrix inequality constraint. In practice, this is handled
by reparameterization (Section 16.7) rather than by explicit KKT conditions ---
for instance, working with the Cholesky factor :math:`\mathbf{L}` such that
:math:`\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^{\!\top}`.

Variance Positivity
--------------------

For a normal model :math:`\mathcal{N}(\mu, \sigma^2)`, the constraint
:math:`\sigma^2 > 0` is automatically satisfied by the MLE
:math:`\hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2 > 0` (for
non-degenerate data). But in mixed models or penalized settings, the boundary
:math:`\sigma^2 = 0` can be relevant, and the KKT conditions (specifically
complementary slackness) govern the behavior.


16.4 Augmented Lagrangian Method
==================================

Motivation
----------

The basic Lagrangian approach requires solving the KKT system exactly, which
can be difficult. The **augmented Lagrangian method** adds a quadratic penalty
term to improve numerical conditioning.

Formulation
-----------

For equality constraints :math:`h_j(\boldsymbol{\theta}) = 0`, the augmented
Lagrangian is

.. math::
   :label: aug_lag

   \mathcal{L}_\rho(\boldsymbol{\theta}, \boldsymbol{\lambda})
   = f(\boldsymbol{\theta})
   - \sum_{j=1}^m \lambda_j\,h_j(\boldsymbol{\theta})
   + \frac{\rho}{2}\sum_{j=1}^m h_j(\boldsymbol{\theta})^2,

where :math:`\rho > 0` is a penalty parameter. The last term penalizes
constraint violations, encouraging feasibility even when the Lagrange
multiplier estimates are imperfect.

The augmented Lagrangian combines the best of both worlds: the Lagrange
multipliers handle the constraint "exactly" in the limit, while the quadratic
penalty provides numerical stability along the way.

Algorithm
---------

.. code-block:: text

   Initialize: theta_0, lambda_0, rho_0
   For k = 0, 1, 2, ...
       1. Minimize L_rho(theta, lambda_k) with respect to theta -> theta_{k+1}
       2. Update multipliers: lambda_{k+1,j} = lambda_{k,j} - rho_k h_j(theta_{k+1})
       3. Optionally increase rho_k -> rho_{k+1}
       4. Check convergence

The multiplier update in step 2 is a **dual ascent** step: it moves the
multipliers in the direction that increases the Lagrangian, encouraging the
constraints to be satisfied more tightly.

Let us implement the augmented Lagrangian from scratch and watch it converge.

.. code-block:: python

   # Augmented Lagrangian method: min x^2 + 2y^2 s.t. x + y = 1
   import numpy as np

   np.random.seed(42)

   # Objective and constraint
   def f(x, y):
       return x**2 + 2*y**2

   def h(x, y):
       return x + y - 1  # constraint: h = 0

   def grad_f(x, y):
       return np.array([2*x, 4*y])

   def grad_h(x, y):
       return np.array([1.0, 1.0])

   # Augmented Lagrangian method
   theta = np.array([0.0, 0.0])  # initial guess
   lam = 0.0                      # initial multiplier
   rho = 1.0                      # penalty parameter

   print(f"{'Iter':>4s}  {'x':>8s}  {'y':>8s}  {'lambda':>8s}  {'rho':>6s}  "
         f"{'f(x,y)':>8s}  {'|h(x,y)|':>10s}")
   print("-" * 62)

   for k in range(15):
       # Inner loop: minimize augmented Lagrangian w.r.t. theta using GD
       for _ in range(100):
           x, y = theta
           hval = h(x, y)
           grad = grad_f(x, y) - lam * grad_h(x, y) + rho * hval * grad_h(x, y)
           theta = theta - 0.05 * grad

       x, y = theta
       hval = h(x, y)
       print(f"{k:4d}  {x:8.5f}  {y:8.5f}  {lam:8.4f}  {rho:6.1f}  "
             f"{f(x, y):8.5f}  {abs(hval):10.2e}")

       # Update multiplier (dual ascent)
       lam = lam - rho * hval

       # Optionally increase rho
       if abs(hval) > 0.01:
           rho *= 1.5

       if abs(hval) < 1e-8:
           print(f"Converged at iteration {k}.")
           break

   print(f"\nSolution: x={theta[0]:.6f}, y={theta[1]:.6f}")
   print(f"Exact:    x={2/3:.6f}, y={1/3:.6f}")

**Advantages over a pure penalty method:** The augmented Lagrangian converges
to the correct solution for a *finite* value of :math:`\rho`; a pure penalty
method requires :math:`\rho \to \infty`, causing ill-conditioning.

Connection to ADMM
--------------------

The **Alternating Direction Method of Multipliers (ADMM)** applies the
augmented Lagrangian method to a problem that has been split into two blocks:

.. math::

   \min\; f_1(\boldsymbol{\theta}_1) + f_2(\boldsymbol{\theta}_2)
   \quad\text{s.t.}\quad
   \mathbf{A}_1\boldsymbol{\theta}_1 + \mathbf{A}_2\boldsymbol{\theta}_2
   = \mathbf{b}.

The augmented Lagrangian is minimized alternately over
:math:`\boldsymbol{\theta}_1` and :math:`\boldsymbol{\theta}_2`, followed by a
multiplier update. ADMM is widely used in machine learning for distributed
optimization and for problems with structured regularizers (e.g., LASSO,
group LASSO).


16.5 Barrier (Interior-Point) Methods
========================================

Motivation
----------

Interior-point methods approach the constrained optimum from strictly inside
the feasible region, using a barrier function that goes to infinity at the
boundary.

.. admonition:: Intuition

   Imagine you are optimizing in a room, and the walls represent constraints.
   A barrier function creates a "force field" near the walls that repels you
   from getting too close.  As the optimization proceeds, the force field is
   gradually weakened (by increasing :math:`t`), allowing you to creep closer
   and closer to the walls --- but you never actually hit them.  In the limit,
   you end up right at the constrained optimum, possibly touching one or more
   walls.

Log-Barrier Function
---------------------

For inequality constraints :math:`g_i(\boldsymbol{\theta}) \leq 0`, the
**log-barrier** function is

.. math::
   :label: log_barrier

   B(\boldsymbol{\theta}) = -\sum_{i=1}^q \log\bigl(-g_i(\boldsymbol{\theta})\bigr).

Note that :math:`B(\boldsymbol{\theta}) \to +\infty` as any constraint becomes
active (:math:`g_i \to 0^-`), so minimizing :math:`f + (1/t)\,B` keeps the
iterates strictly inside the feasible region.

The Barrier Problem
--------------------

For a parameter :math:`t > 0`, solve the unconstrained problem

.. math::
   :label: barrier_prob

   \min_{\boldsymbol{\theta}}\;
   f(\boldsymbol{\theta}) + \frac{1}{t}\,B(\boldsymbol{\theta})
   = f(\boldsymbol{\theta})
   - \frac{1}{t}\sum_{i=1}^q \log\bigl(-g_i(\boldsymbol{\theta})\bigr).

As :math:`t \to \infty`, the barrier term vanishes and the solution approaches
the constrained optimum.

Let us implement the barrier method on our portfolio problem and watch the
barrier parameter grow as the solution converges to the constraint boundary.

.. code-block:: python

   # Barrier method for portfolio optimization
   # Maximize mu^T w s.t. w^T Sigma w <= sigma2_max, sum(w)=1, w >= 0
   # Reformulated as: min -mu^T w with barrier for inequality constraints
   import numpy as np

   np.random.seed(42)
   p = 3  # small for clarity
   mu = np.array([0.12, 0.07, 0.14])
   A = np.random.randn(p, p) * 0.1
   Sigma = A.T @ A + 0.05 * np.eye(p)
   sigma2_max = 0.03

   # Constraints: w >= 0 (p constraints), w^T Sigma w <= sigma2_max (1 constraint)
   # We handle sum(w) = 1 by projecting after each step

   def barrier_obj(w, t):
       """Objective + (1/t) * barrier."""
       obj = -mu @ w
       # Barrier for w_i > 0
       if np.any(w <= 0):
           return 1e10
       obj -= (1.0 / t) * np.sum(np.log(w))
       # Barrier for sigma2_max - w^T Sigma w > 0
       slack = sigma2_max - w @ Sigma @ w
       if slack <= 0:
           return 1e10
       obj -= (1.0 / t) * np.log(slack)
       return obj

   def barrier_grad(w, t):
       """Gradient of objective + (1/t) * barrier."""
       grad = -mu
       grad -= (1.0 / t) / w                           # d/dw [-log(w_i)] = -1/w_i
       slack = sigma2_max - w @ Sigma @ w
       grad -= (1.0 / t) * (-2 * Sigma @ w) / slack    # d/dw [-log(slack)]
       return grad

   def project_simplex(w):
       """Project w onto the simplex {w >= 0, sum(w) = 1}."""
       # Simple projection: normalize positive part
       w = np.maximum(w, 1e-10)
       return w / np.sum(w)

   w = np.ones(p) / p  # start at equal weights (feasible)

   print(f"{'t':>8s}  ", end="")
   for i in range(p):
       print(f"{'w'+str(i):>8s}  ", end="")
   print(f"{'Return':>8s}  {'Risk':>8s}  {'Barrier obj':>12s}")
   print("-" * 70)

   for t in [1, 5, 20, 100, 500, 2000, 10000]:
       # Inner loop: projected gradient descent
       for step in range(500):
           g = barrier_grad(w, t)
           lr = 0.0005 / (1 + step * 0.001)
           w = w - lr * g
           w = project_simplex(w)
           # Ensure feasibility
           while w @ Sigma @ w >= sigma2_max:
               w = 0.95 * w + 0.05 * np.ones(p) / p

       ret = mu @ w
       risk = w @ Sigma @ w
       bobj = barrier_obj(w, t)
       print(f"{t:8d}  ", end="")
       for i in range(p):
           print(f"{w[i]:8.4f}  ", end="")
       print(f"{ret:8.4f}  {risk:8.6f}  {bobj:12.6f}")

   print(f"\nAs t -> inf, the barrier weakens and the solution approaches")
   print(f"the constrained optimum. Risk budget: {sigma2_max}")

The Central Path
-----------------

The set of solutions :math:`\{\boldsymbol{\theta}^*(t) : t > 0\}` traces a
smooth curve in parameter space called the **central path**. As :math:`t`
increases, the points on the central path approach the constrained optimum.

The first-order optimality condition for :eq:`barrier_prob` is

.. math::

   \nabla f(\boldsymbol{\theta})
   + \frac{1}{t}\sum_{i=1}^q \frac{\nabla g_i(\boldsymbol{\theta})}{-g_i(\boldsymbol{\theta})}
   = \mathbf{0}.

Defining :math:`\mu_i = 1/(t \cdot (-g_i(\boldsymbol{\theta})))`, this becomes

.. math::

   \nabla f(\boldsymbol{\theta})
   + \sum_{i=1}^q \mu_i\,\nabla g_i(\boldsymbol{\theta}) = \mathbf{0},
   \qquad
   \mu_i\,g_i(\boldsymbol{\theta}) = -1/t,

which converges to the KKT conditions (with complementary slackness
:math:`\mu_i g_i = 0`) as :math:`t \to \infty`.

Algorithm
---------

.. code-block:: text

   Initialize: theta_0 strictly feasible, t_0 > 0, growth factor kappa > 1
   For k = 0, 1, 2, ...
       1. Solve the barrier problem (16.7) using Newton's method -> theta_k*
       2. Increase t: t_{k+1} = kappa * t_k
       3. Use theta_k* as the starting point for the next barrier problem
       4. Stop when q/t < epsilon (duality gap bound)

Interior-point methods have excellent theoretical properties: they can solve
convex problems with :math:`q` inequality constraints in
:math:`O(\sqrt{q}\,\log(1/\epsilon))` Newton steps.


16.6 Penalty Methods
======================

Quadratic Penalty Method
--------------------------

The simplest approach to constrained optimization is to add a penalty for
constraint violations. For equality constraints:

.. math::
   :label: quad_penalty

   \min_{\boldsymbol{\theta}}\;
   f(\boldsymbol{\theta})
   + \frac{\rho}{2}\sum_{j=1}^m h_j(\boldsymbol{\theta})^2.

For inequality constraints :math:`g_i(\boldsymbol{\theta}) \leq 0`:

.. math::

   \min_{\boldsymbol{\theta}}\;
   f(\boldsymbol{\theta})
   + \frac{\rho}{2}\sum_{i=1}^q \bigl[\max(0, g_i(\boldsymbol{\theta}))\bigr]^2.

Let us implement the penalty method and see how increasing :math:`\rho` drives
the solution toward feasibility---and also how it causes ill-conditioning.

.. code-block:: python

   # Penalty method: min x^2 + 2y^2 s.t. x + y = 1
   # Compare with exact solution (2/3, 1/3) and augmented Lagrangian
   import numpy as np

   np.random.seed(42)
   exact_x, exact_y = 2.0/3, 1.0/3
   exact_f = exact_x**2 + 2*exact_y**2

   print(f"{'rho':>10s}  {'x':>8s}  {'y':>8s}  {'f(x,y)':>8s}  "
         f"{'|h(x,y)|':>10s}  {'||error||':>10s}  {'cond(H)':>10s}")
   print("-" * 78)

   for rho in [1, 10, 100, 1000, 10000, 100000]:
       # Penalized objective: f(x,y) + (rho/2)*(x+y-1)^2
       # Gradient: [2x + rho*(x+y-1), 4y + rho*(x+y-1)]
       # Hessian:  [[2+rho, rho], [rho, 4+rho]]
       # Solve: H * [x,y] = [rho, rho]  (from setting gradient = 0)
       H = np.array([[2 + rho, rho], [rho, 4 + rho]])
       b = np.array([rho, rho])
       sol = np.linalg.solve(H, b)
       x, y = sol
       fval = x**2 + 2*y**2
       hval = x + y - 1
       err = np.sqrt((x - exact_x)**2 + (y - exact_y)**2)
       cond = np.linalg.cond(H)
       print(f"{rho:10d}  {x:8.6f}  {y:8.6f}  {fval:8.6f}  "
             f"{abs(hval):10.2e}  {err:10.2e}  {cond:10.1f}")

   print(f"\nExact: x={exact_x:.6f}, y={exact_y:.6f}, f={exact_f:.6f}")
   print("Note: as rho increases, h(x,y)->0 but cond(H) grows -> ill-conditioning!")

Convergence
-----------

**Theorem.** Let :math:`\boldsymbol{\theta}(\rho)` be the minimizer of the
quadratic penalty problem for a given :math:`\rho`. Under mild conditions, as
:math:`\rho \to \infty`, any limit point of :math:`\boldsymbol{\theta}(\rho)`
is a solution of the constrained problem.

*Proof sketch.* For any feasible point :math:`\bar{\boldsymbol{\theta}}`:

.. math::

   f(\boldsymbol{\theta}(\rho))
   + \frac{\rho}{2}\|\mathbf{h}(\boldsymbol{\theta}(\rho))\|^2
   \leq f(\bar{\boldsymbol{\theta}})
   + \frac{\rho}{2}\|\mathbf{h}(\bar{\boldsymbol{\theta}})\|^2
   = f(\bar{\boldsymbol{\theta}}),

since :math:`\mathbf{h}(\bar{\boldsymbol{\theta}}) = \mathbf{0}`. Therefore
:math:`\|\mathbf{h}(\boldsymbol{\theta}(\rho))\|^2 \leq
2[f(\bar{\boldsymbol{\theta}}) - f(\boldsymbol{\theta}(\rho))]/\rho`. As
:math:`\rho \to \infty`, the constraint violation shrinks to zero (assuming
:math:`f` is bounded below on the feasible set).

**Drawback:** For large :math:`\rho`, the penalized objective becomes
ill-conditioned (its Hessian has eigenvalues of order :math:`\rho`), making
the subproblem hard to solve. This motivates the augmented Lagrangian method
(Section 16.4), which achieves feasibility without driving :math:`\rho` to
infinity.

.. admonition:: Common Pitfall

   The quadratic penalty method is tempting because of its simplicity, but
   beware of the ill-conditioning trap.  As :math:`\rho` grows, the Hessian
   becomes dominated by the penalty term, making Newton's method and even
   gradient descent struggle with numerical precision.  If you find yourself
   needing :math:`\rho > 10^6`, switch to the augmented Lagrangian method or
   reparameterize the problem.


16.7 Reparameterization as an Alternative to Constraints
==========================================================

In practice, the most common way to handle parameter constraints in
statistical models is to transform the parameters so that the transformed
parameters are unconstrained. This is especially convenient with gradient-based
optimization.

.. admonition:: Why Reparameterize?

   Reparameterization is often the simplest and most robust approach.  Instead
   of fighting the constraints with multipliers and penalties, you change
   variables so the constraints simply *vanish*.  The transformed problem is
   unconstrained, so you can use any method from Chapters 12--14 directly.
   The catch is that the transformation must be smooth and invertible, and the
   transformed parameter space may have different curvature properties.

Positivity: The Log Transform
-------------------------------

If :math:`\sigma^2 > 0`, define :math:`\phi = \log \sigma^2`. Then
:math:`\phi \in \mathbb{R}` is unconstrained, and :math:`\sigma^2 = e^{\phi}`
is automatically positive.

The gradient transforms as

.. math::

   \frac{\partial \ell}{\partial \phi}
   = \frac{\partial \ell}{\partial \sigma^2}\,
     \frac{\partial \sigma^2}{\partial \phi}
   = \frac{\partial \ell}{\partial \sigma^2}\,e^{\phi}
   = \sigma^2\,\frac{\partial \ell}{\partial \sigma^2}.

.. code-block:: python

   # Log reparameterization: MLE of normal variance
   import numpy as np

   np.random.seed(42)
   n = 100
   sigma2_true = 4.0
   x = np.random.normal(0, np.sqrt(sigma2_true), n)

   # Direct: gradient ascent on sigma^2 (must stay positive!)
   sigma2 = 1.0
   print("--- Without reparameterization (constrained) ---")
   print(f"{'Iter':>4s}  {'sigma^2':>10s}  {'grad':>10s}")
   for k in range(8):
       grad = -n / (2 * sigma2) + np.sum(x**2) / (2 * sigma2**2)
       sigma2 = sigma2 + 0.01 * grad
       sigma2 = max(sigma2, 1e-6)  # MANUAL clipping to enforce positivity
       print(f"{k:4d}  {sigma2:10.4f}  {grad:10.4f}")

   # Reparameterized: gradient ascent on phi = log(sigma^2) (unconstrained!)
   phi = 0.0  # log(1.0) = 0
   print("\n--- With log reparameterization (unconstrained) ---")
   print(f"{'Iter':>4s}  {'phi':>10s}  {'sigma^2':>10s}  {'grad_phi':>10s}")
   for k in range(8):
       sigma2_reparam = np.exp(phi)
       # Chain rule: d(ell)/d(phi) = sigma^2 * d(ell)/d(sigma^2)
       grad_sigma2 = -n / (2 * sigma2_reparam) + np.sum(x**2) / (2 * sigma2_reparam**2)
       grad_phi = sigma2_reparam * grad_sigma2  # no clipping needed!
       phi = phi + 0.01 * grad_phi
       print(f"{k:4d}  {phi:10.4f}  {np.exp(phi):10.4f}  {grad_phi:10.4f}")

   print(f"\nTrue sigma^2 = {sigma2_true:.4f}")
   print(f"Sample variance = {np.var(x):.4f}")

Bounded Parameters: The Logit Transform
------------------------------------------

If :math:`p \in (0, 1)`, define :math:`\eta = \log(p / (1-p))`. Then
:math:`\eta \in \mathbb{R}`, and :math:`p = 1/(1 + e^{-\eta})` is
automatically in :math:`(0, 1)`.

More generally, for :math:`\theta \in (a, b)`:

.. math::

   \eta = \log\frac{\theta - a}{b - \theta},
   \qquad
   \theta = a + (b-a)\,\frac{1}{1 + e^{-\eta}}.

.. code-block:: python

   # Logit reparameterization for a binomial proportion
   import numpy as np
   from scipy.special import expit, logit

   np.random.seed(42)
   p_true = 0.73
   n = 50
   x = np.random.binomial(n, p_true)  # x successes out of n trials
   print(f"Observed: {x}/{n} = {x/n:.2f}, True p = {p_true}")

   # Gradient ascent on eta = logit(p), unconstrained
   eta = 0.0  # start at p = 0.5
   print(f"\n{'Iter':>4s}  {'eta':>8s}  {'p':>8s}  {'grad_eta':>10s}  {'log-lik':>10s}")
   for k in range(15):
       p = expit(eta)
       # Log-likelihood: x*log(p) + (n-x)*log(1-p)
       ll = x * np.log(p) + (n - x) * np.log(1 - p)
       # d(ell)/d(eta) = x - n*p  (a beautiful simplification!)
       grad_eta = x - n * p
       eta = eta + 0.05 * grad_eta
       print(f"{k:4d}  {eta:8.4f}  {p:8.4f}  {grad_eta:10.4f}  {ll:10.4f}")

   print(f"\nMLE: p = {x/n:.4f} (exact), p = {expit(eta):.4f} (reparameterized GD)")

Simplex Constraint: The Softmax Transform
--------------------------------------------

If :math:`\boldsymbol{\pi} = (\pi_1, \dots, \pi_K)` with
:math:`\pi_k \geq 0` and :math:`\sum_k \pi_k = 1`, define unconstrained
:math:`\boldsymbol{\eta} = (\eta_1, \dots, \eta_{K-1}) \in \mathbb{R}^{K-1}` and set

.. math::
   :label: softmax

   \pi_k = \frac{e^{\eta_k}}{\sum_{j=1}^{K} e^{\eta_j}},
   \qquad k = 1, \dots, K,

where :math:`\eta_K = 0` for identifiability (one degree of freedom is removed
by the sum-to-one constraint).

The softmax transform is smooth and invertible (on the interior of the
simplex), so gradient methods can be applied to :math:`\boldsymbol{\eta}`
without any constraint-handling machinery.

.. code-block:: python

   # Softmax reparameterization: multinomial MLE without constraints
   import numpy as np
   from scipy.special import softmax

   np.random.seed(42)
   K = 5
   counts = np.array([45, 30, 15, 5, 5])
   n = counts.sum()
   pi_true = counts / n

   # Unconstrained parameters (K-1 free, last fixed at 0)
   eta = np.zeros(K - 1)

   print(f"{'Iter':>4s}  ", end="")
   for k in range(K):
       print(f"{'pi_'+str(k+1):>8s}  ", end="")
   print(f"{'||pi - true||':>14s}")
   print("-" * 65)

   for step in range(200):
       # Forward: softmax to get probabilities
       eta_full = np.append(eta, 0.0)
       pi = softmax(eta_full)

       # Gradient: d(ell)/d(eta_k) = counts_k - n*pi_k for k=1,...,K-1
       grad = counts[:K-1] - n * pi[:K-1]
       eta = eta + 0.01 * grad

       if step % 40 == 0 or step == 199:
           err = np.linalg.norm(pi - pi_true)
           print(f"{step:4d}  ", end="")
           for k in range(K):
               print(f"{pi[k]:8.4f}  ", end="")
           print(f"{err:14.6f}")

   print(f"\nTrue:  {pi_true}")
   print(f"Sum:   {pi.sum():.8f} (always exactly 1 by construction)")

The key insight: the softmax guarantees :math:`\pi_k > 0` and
:math:`\sum_k \pi_k = 1` by construction, so we never need to worry about
constraint violations.

Positive-Definite Matrices: The Cholesky Factor
--------------------------------------------------

For a covariance matrix :math:`\boldsymbol{\Sigma}` that must be positive
definite, write :math:`\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^{\!\top}`
where :math:`\mathbf{L}` is lower triangular with positive diagonal entries.
Optimize over the entries of :math:`\mathbf{L}`, parameterizing the diagonal as
:math:`L_{jj} = e^{\phi_j}` to ensure positivity.

This gives :math:`p(p+1)/2` unconstrained parameters, which is the correct
number of degrees of freedom for a :math:`p \times p` symmetric positive
definite matrix.

.. code-block:: python

   # Cholesky reparameterization for covariance MLE
   import numpy as np

   np.random.seed(42)
   d = 3
   # True covariance
   Sigma_true = np.array([[4.0, 1.0, 0.5],
                           [1.0, 2.0, 0.3],
                           [0.5, 0.3, 1.0]])
   L_true = np.linalg.cholesky(Sigma_true)

   # Generate data
   n = 500
   X = np.random.randn(n, d) @ L_true.T

   # Parameterize: L has d*(d+1)/2 free parameters
   # Diagonal: L_jj = exp(phi_j) (always positive)
   # Off-diagonal: unconstrained
   n_params = d * (d + 1) // 2
   params = np.zeros(n_params)  # initial: L = I

   def params_to_L(params, d):
       """Convert unconstrained params to lower triangular L."""
       L = np.zeros((d, d))
       idx = 0
       for i in range(d):
           for j in range(i + 1):
               if i == j:
                   L[i, j] = np.exp(params[idx])  # positive diagonal
               else:
                   L[i, j] = params[idx]           # free off-diagonal
               idx += 1
       return L

   def neg_log_lik(params, X, d):
       """Negative log-likelihood for multivariate normal (mu=0)."""
       L = params_to_L(params, d)
       Sigma = L @ L.T
       n = len(X)
       sign, logdet = np.linalg.slogdet(Sigma)
       Sigma_inv = np.linalg.inv(Sigma)
       return 0.5 * n * logdet + 0.5 * np.sum(X @ Sigma_inv * X)

   # Optimize using numerical gradient descent
   print(f"{'Iter':>4s}  {'neg-LL':>12s}  {'||Sigma - Sigma_hat||':>22s}")
   print("-" * 44)
   lr = 0.0001
   for step in range(500):
       # Numerical gradient
       grad = np.zeros(n_params)
       eps = 1e-5
       f0 = neg_log_lik(params, X, d)
       for j in range(n_params):
           params_p = params.copy()
           params_p[j] += eps
           grad[j] = (neg_log_lik(params_p, X, d) - f0) / eps
       params = params - lr * grad

       if step % 100 == 0 or step == 499:
           L = params_to_L(params, d)
           Sigma_hat = L @ L.T
           err = np.linalg.norm(Sigma_hat - Sigma_true, 'fro')
           print(f"{step:4d}  {f0:12.2f}  {err:22.4f}")

   L_final = params_to_L(params, d)
   Sigma_hat = L_final @ L_final.T
   print(f"\nTrue Sigma:\n{Sigma_true}")
   print(f"\nEstimated Sigma:\n{np.round(Sigma_hat, 3)}")
   print(f"Positive definite: {np.all(np.linalg.eigvalsh(Sigma_hat) > 0)}")

Correlation Matrices: Fisher's z-Transform
---------------------------------------------

For a correlation parameter :math:`r \in (-1, 1)`, use

.. math::

   z = \operatorname{arctanh}(r) = \frac{1}{2}\log\frac{1+r}{1-r},
   \qquad
   r = \tanh(z).

Then :math:`z \in \mathbb{R}` is unconstrained.

.. code-block:: python

   # Fisher's z-transform for correlation estimation
   import numpy as np

   np.random.seed(42)
   r_true = 0.85
   n = 100

   # Generate correlated data
   Sigma = np.array([[1.0, r_true], [r_true, 1.0]])
   L = np.linalg.cholesky(Sigma)
   data = np.random.randn(n, 2) @ L.T

   # Gradient ascent on z = arctanh(r), unconstrained
   z = 0.0  # start at r = 0
   print(f"{'Iter':>4s}  {'z':>8s}  {'r=tanh(z)':>10s}  {'log-lik':>10s}")
   for k in range(20):
       r = np.tanh(z)
       # Bivariate normal log-likelihood (mu=0, sigma=1, correlation r)
       det = 1 - r**2
       ll = -0.5 * n * np.log(det) - 0.5 / det * np.sum(
           data[:, 0]**2 - 2*r*data[:, 0]*data[:, 1] + data[:, 1]**2)
       # d(ell)/d(r)
       dll_dr = n * r / det + (1 / det**2) * np.sum(
           -r * (data[:, 0]**2 + data[:, 1]**2) + (1 + r**2) * data[:, 0] * data[:, 1])
       # Chain rule: d(ell)/d(z) = d(ell)/d(r) * d(r)/d(z) = dll_dr * (1 - r^2)
       grad_z = dll_dr * (1 - r**2)
       z = z + 0.001 * grad_z
       if k % 4 == 0:
           print(f"{k:4d}  {z:8.4f}  {np.tanh(z):10.4f}  {ll:10.2f}")

   print(f"\nTrue r = {r_true}, Sample r = {np.corrcoef(data.T)[0,1]:.4f}, "
         f"Estimated r = {np.tanh(z):.4f}")

All Four Reparameterizations at a Glance
-------------------------------------------

.. code-block:: python

   # Summary: all reparameterization strategies
   import numpy as np

   print(f"{'Constraint':>25s}  {'Transform':>25s}  {'Inverse':>25s}")
   print("-" * 80)
   examples = [
       ("sigma^2 > 0",       "phi = log(sigma^2)",    "sigma^2 = exp(phi)"),
       ("p in (0, 1)",        "eta = logit(p)",         "p = sigmoid(eta)"),
       ("r in (-1, 1)",       "z = arctanh(r)",         "r = tanh(z)"),
       ("pi on simplex",      "eta_k unconstrained",    "pi = softmax(eta)"),
       ("Sigma pos. def.",    "L = chol(Sigma)",        "Sigma = L @ L.T"),
   ]
   for constraint, transform, inverse in examples:
       print(f"{constraint:>25s}  {transform:>25s}  {inverse:>25s}")

   # Numerical demo: all transforms preserve constraints
   print("\nNumerical verification:")
   phi = -2.5
   print(f"  exp({phi}) = {np.exp(phi):.4f} > 0: True")
   eta = 3.7
   p = 1 / (1 + np.exp(-eta))
   print(f"  sigmoid({eta}) = {p:.4f} in (0,1): {0 < p < 1}")
   z = 1.2
   r = np.tanh(z)
   print(f"  tanh({z}) = {r:.4f} in (-1,1): {-1 < r < 1}")
   eta_vec = np.array([1.0, -0.5, 0.3])
   from scipy.special import softmax
   pi = softmax(np.append(eta_vec, 0.0))
   print(f"  softmax({list(np.round(eta_vec, 1))} + [0]) = {np.round(pi, 3)}, "
         f"sum={pi.sum():.6f}")

When to Reparameterize vs. Use Constrained Methods
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Scenario
     - Reparameterize
     - Constrained method
   * - Simple positivity/boundedness
     - Preferred
     - Overkill
   * - Simplex (few categories)
     - Preferred (softmax)
     - Fine too
   * - Complex coupled constraints
     - May be difficult
     - Preferred
   * - Linear constraints (:math:`\mathbf{A}\boldsymbol{\theta} = \mathbf{b}`)
     - Null-space projection
     - Lagrangian
   * - Interpretability of Hessian
     - Harder (need Jacobian)
     - Original parameterization


16.8 Putting It All Together: Constrained MLE
================================================

A typical workflow for constrained maximum likelihood is:

1. **Identify the constraints.** Are they simple (positivity, simplex) or
   complex (coupled inequalities, linear equalities)?

2. **Try reparameterization first.** For simple constraints, the log, logit,
   softmax, or Cholesky transforms eliminate the constraints. Apply any
   unconstrained method from :ref:`ch12_gradient`, :ref:`ch13_newton`, or
   :ref:`ch14_quasi_newton`.

3. **If reparameterization is infeasible,** use:

   - **Lagrange multipliers** (for small problems with equality constraints).
   - **Interior-point methods** (for convex problems with many inequality
     constraints).
   - **Augmented Lagrangian / ADMM** (for large-scale or separable problems).
   - **Penalty methods** (for quick-and-dirty solutions).

4. **Verify the KKT conditions** at the solution. Check that all constraints
   are satisfied, multipliers have the correct sign, and complementary
   slackness holds.

5. **Compute standard errors** using the constrained observed information
   (which may involve the bordered Hessian or the Hessian of the Lagrangian
   restricted to the constraint manifold).

Let us run a final comprehensive example comparing all three approaches---
penalty, augmented Lagrangian, and reparameterization---on the same problem.

.. code-block:: python

   # Head-to-head: three approaches to the same constrained problem
   # Problem: MLE of Bernoulli p with constraint p in (0.2, 0.8)
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n = 30
   p_true = 0.65
   x = np.random.binomial(1, p_true, n)
   p_mle = x.mean()
   print(f"Observed: {x.sum()}/{n} successes, unconstrained MLE = {p_mle:.4f}")

   # Method 1: Quadratic penalty for constraint p >= 0.2 and p <= 0.8
   p = 0.5
   for rho in [10, 100, 1000]:
       for _ in range(200):
           grad = x.sum() / p - (n - x.sum()) / (1 - p)
           # Penalty gradient for g1: 0.2 - p <= 0 and g2: p - 0.8 <= 0
           if p < 0.2:
               grad -= rho * (0.2 - p)
           if p > 0.8:
               grad -= rho * (p - 0.8)
           p = p + 0.001 * grad
           p = np.clip(p, 0.01, 0.99)

   p_penalty = p
   print(f"\nPenalty method:          p = {p_penalty:.6f}")

   # Method 2: Reparameterization via logit with bounds
   # Map p in (0.2, 0.8) to eta in R via eta = log((p-0.2)/(0.8-p))
   a, b = 0.2, 0.8
   eta = 0.0  # start at midpoint p = 0.5
   for _ in range(500):
       p = a + (b - a) * expit(eta)
       grad_p = x.sum() / p - (n - x.sum()) / (1 - p)
       dp_deta = (b - a) * expit(eta) * (1 - expit(eta))
       grad_eta = grad_p * dp_deta
       eta = eta + 0.01 * grad_eta

   p_reparam = a + (b - a) * expit(eta)
   print(f"Reparameterization:     p = {p_reparam:.6f}")

   # Method 3: Projected gradient (project onto [0.2, 0.8] after each step)
   p = 0.5
   for _ in range(500):
       grad = x.sum() / p - (n - x.sum()) / (1 - p)
       p = p + 0.001 * grad
       p = np.clip(p, 0.2, 0.8)  # project onto feasible set

   p_projected = p
   print(f"Projected gradient:     p = {p_projected:.6f}")

   # Analytical: MLE is x.mean(), clipped to [0.2, 0.8]
   p_exact = np.clip(p_mle, 0.2, 0.8)
   print(f"Exact (clipped MLE):    p = {p_exact:.6f}")


16.9 Summary
==============

Constrained optimization is an essential part of likelihood-based inference.
Lagrange multipliers handle equality constraints by introducing dual variables
whose values measure the sensitivity of the optimum to the constraints. The
KKT conditions extend this framework to inequality constraints, with
complementary slackness determining which constraints are active. Algorithmic
approaches --- augmented Lagrangian, barrier methods, and penalty methods ---
provide practical ways to solve constrained problems numerically.

In statistical practice, the most common approach is **reparameterization**:
the log transform for positivity, the logit for bounded parameters, the
softmax for simplex constraints, and the Cholesky decomposition for positive
definiteness. These transforms convert the constrained problem into an
unconstrained one, allowing direct application of the gradient methods
(:ref:`ch12_gradient`), Newton methods (:ref:`ch13_newton`), and quasi-Newton
methods (:ref:`ch14_quasi_newton`) developed in earlier chapters. When
reparameterization is not feasible or when the constraints have complex
structure, the formal constrained-optimization methods of this chapter provide
the necessary tools.
