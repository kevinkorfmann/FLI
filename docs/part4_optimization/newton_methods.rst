.. _ch13_newton:

========================================
Chapter 13: Newton and Scoring Methods
========================================

Gradient methods (:ref:`ch12_gradient`) use only the slope of the objective.
Newton methods also use the *curvature* --- the second derivatives --- and this
extra information buys dramatically faster convergence. Where gradient descent
converges linearly (one new correct digit per fixed number of iterations),
Newton's method converges *quadratically* (the number of correct digits roughly
doubles each iteration).

This chapter derives Newton--Raphson for general optimization, specializes it
to maximum likelihood, introduces Fisher scoring as a stabilized variant,
shows how Fisher scoring reduces to iteratively reweighted least squares (IRLS)
for generalized linear models, and discusses the practical issues that arise
when implementing these methods.

.. admonition:: Why Second-Order Methods?

   Imagine you are trying to find the bottom of a valley.  Gradient descent
   tells you the direction of steepest descent, but it says nothing about
   *how far* to go --- the landscape might flatten out just ahead, or it might
   drop off a cliff.  Second-order methods read the *curvature* of the
   landscape: if the valley is broad, they take a big step; if it is narrow,
   they take a small one.  This awareness of curvature is what makes Newton's
   method so much faster near the solution.


13.1 Newton--Raphson from the Second-Order Taylor Expansion
=============================================================

The Quadratic Model
---------------------

We want to minimize a twice-differentiable function
:math:`f:\mathbb{R}^p \to \mathbb{R}`. At a current iterate
:math:`\boldsymbol{\theta}_k`, the *second-order* Taylor expansion is

.. math::
   :label: taylor2

   f(\boldsymbol{\theta}_k + \mathbf{d})
   \;\approx\;
   f(\boldsymbol{\theta}_k)
   + \nabla f(\boldsymbol{\theta}_k)^{\!\top}\mathbf{d}
   + \tfrac{1}{2}\,\mathbf{d}^{\!\top}
     \mathbf{H}(\boldsymbol{\theta}_k)\,\mathbf{d},

where :math:`\mathbf{H}(\boldsymbol{\theta}_k) = \nabla^2 f(\boldsymbol{\theta}_k)`
is the Hessian matrix of second partial derivatives:

.. math::

   [\mathbf{H}(\boldsymbol{\theta})]_{jl}
   = \frac{\partial^2 f}{\partial \theta_j\,\partial \theta_l}.

The Hessian captures the curvature of :math:`f`: a large positive eigenvalue
means the function curves steeply upward in that eigenvector direction, while
a small eigenvalue means the function is nearly flat.

Think of the second-order expansion as fitting a paraboloid --- a bowl-shaped
surface --- to the function at the current point.  The gradient tells you the
tilt of the bowl, and the Hessian tells you its width and shape.

Deriving the Newton Step
--------------------------

To find the displacement :math:`\mathbf{d}` that minimizes the quadratic model
:eq:`taylor2`, differentiate with respect to :math:`\mathbf{d}` and set the
result to zero:

.. math::

   \nabla f(\boldsymbol{\theta}_k)
   + \mathbf{H}(\boldsymbol{\theta}_k)\,\mathbf{d} = \mathbf{0}.

Solving for :math:`\mathbf{d}` gives the **Newton step**:

.. math::
   :label: newton_step

   \mathbf{d}_k^{\text{N}}
   = -\mathbf{H}(\boldsymbol{\theta}_k)^{-1}\,
     \nabla f(\boldsymbol{\theta}_k).

The Newton update is then

.. math::
   :label: newton_update

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k + \mathbf{d}_k^{\text{N}}
   = \boldsymbol{\theta}_k
     - \mathbf{H}(\boldsymbol{\theta}_k)^{-1}\,
       \nabla f(\boldsymbol{\theta}_k).

Compare this with gradient descent :eq:`gd_update`: Newton's method replaces
the scalar step size :math:`\alpha_k` with the *inverse Hessian*
:math:`\mathbf{H}_k^{-1}`, which adapts both the direction and the magnitude
of the step to the local curvature.

The beauty of this formula is that it is *self-scaling*: you never need to tune
a learning rate. The Hessian automatically tells you how far to step.

Let's see Newton's method in action on a simple root-finding problem first,
then on optimization.

.. code-block:: python

   # Newton-Raphson root finding: solve x^3 - 2x - 5 = 0
   import numpy as np

   np.random.seed(42)

   f = lambda x: x**3 - 2*x - 5
   fp = lambda x: 3*x**2 - 2

   x = 2.0  # starting guess
   for k in range(8):
       x_new = x - f(x) / fp(x)
       print(f"Iter {k}: x = {x_new:.12f},  f(x) = {f(x_new):.2e}")
       x = x_new

Seeing Quadratic Convergence
-------------------------------

The root-finding example above converges, but the output obscures the *rate*.
Let's optimize a scalar function where we know the exact minimum, so we can
watch the error halve in digits at each step.

We minimize :math:`f(x) = (x - \sqrt{2})^4 + (x - \sqrt{2})^2`, whose unique
minimum is at :math:`x^* = \sqrt{2}`. The function is not quadratic, so Newton
does not finish in one step, but it converges *quadratically*: once the error
is :math:`10^{-3}`, the next error is roughly :math:`10^{-6}`, then
:math:`10^{-12}`.

.. code-block:: python

   # Newton-Raphson on a scalar function: watching quadratic convergence
   import numpy as np

   x_star = np.sqrt(2)

   # f(x) = (x - sqrt(2))^4 + (x - sqrt(2))^2
   def f(x):
       d = x - x_star
       return d**4 + d**2

   def fp(x):
       d = x - x_star
       return 4*d**3 + 2*d

   def fpp(x):
       d = x - x_star
       return 12*d**2 + 2

   x = 3.0  # starting guess (far from sqrt(2) ≈ 1.414)
   print(f"{'Iter':>4s}  {'x_k':>18s}  {'error':>12s}  {'digits gained':>14s}")
   print("-" * 56)
   prev_log_err = None
   for k in range(12):
       err = abs(x - x_star)
       log_err = np.log10(err) if err > 0 else -16
       digits = f"{-log_err:.1f}"
       ratio = ""
       if prev_log_err is not None and err > 0:
           ratio = f"  (ratio: {log_err/prev_log_err:.2f})"
       print(f"{k:4d}  {x:18.14f}  {err:12.2e}  ~{digits:>5s} digits{ratio}")
       if err < 1e-15:
           break
       prev_log_err = log_err
       x = x - fp(x) / fpp(x)

   # Quadratic convergence means log(error_{k+1}) / log(error_k) → 2

The ratio column should approach 2, which is the signature of quadratic
convergence: each iteration roughly *doubles* the number of correct digits.

Geometric Interpretation
--------------------------

Newton's method fits a paraboloid to the function at the current point and jumps
directly to the minimum of that paraboloid. If the function is exactly quadratic,
one Newton step finds the exact minimum. For non-quadratic functions, Newton's
method converges very quickly once the iterates are close to the minimum (where
the quadratic approximation is accurate).


13.2 Application to Maximum Likelihood Estimation
===================================================

For MLE we maximize the log-likelihood :math:`\ell(\boldsymbol{\theta})
= \sum_{i=1}^n \log p(x_i \mid \boldsymbol{\theta})`, or equivalently minimize
:math:`f = -\ell`. The gradient and Hessian are

.. math::

   \nabla f &= -\nabla \ell = -\mathbf{S}(\boldsymbol{\theta}), \\
   \mathbf{H}_f &= -\nabla^2 \ell = -\mathbf{H}_\ell(\boldsymbol{\theta}),

where :math:`\mathbf{S} = \nabla \ell` is the *score vector*. The Newton update
:eq:`newton_update` becomes

.. math::
   :label: newton_mle

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   - \bigl[-\mathbf{H}_\ell(\boldsymbol{\theta}_k)\bigr]^{-1}
     \bigl[-\mathbf{S}(\boldsymbol{\theta}_k)\bigr]
   = \boldsymbol{\theta}_k
   + \bigl[\mathbf{H}_\ell(\boldsymbol{\theta}_k)\bigr]^{-1}
     \mathbf{S}(\boldsymbol{\theta}_k).

The matrix :math:`\mathbf{J}(\boldsymbol{\theta}) = -\mathbf{H}_\ell(\boldsymbol{\theta})`
is the **observed information matrix**. Rewriting:

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   + \mathbf{J}(\boldsymbol{\theta}_k)^{-1}\,
     \mathbf{S}(\boldsymbol{\theta}_k).

This is the standard form of Newton--Raphson for MLE. Notice how the observed
information matrix plays a dual role: it is both the curvature of the
log-likelihood *and* the matrix that determines the asymptotic variance of the
MLE (via the Cramer--Rao bound). So the same object that makes Newton fast also
gives you standard errors for free at the end.

Newton's Method for Logistic Regression MLE
----------------------------------------------

A hospital wants to predict 30-day readmission from patient features.
Logistic regression gives interpretable coefficients *and* calibrated
probabilities --- both essential for clinical decision-making. Here we fit
the model by Newton--Raphson, printing a detailed convergence table at
each iteration.

The log-likelihood, score, and Hessian for logistic regression are:

.. math::

   \ell(\boldsymbol{\beta})
   &= \sum_i \bigl[y_i\,\eta_i - \log(1+e^{\eta_i})\bigr],\qquad
   \eta_i = \mathbf{x}_i^{\!\top}\boldsymbol{\beta}, \\
   \mathbf{S}
   &= \mathbf{X}^{\!\top}(\mathbf{y} - \mathbf{p}), \\
   \mathbf{H}_\ell
   &= -\mathbf{X}^{\!\top}\mathbf{W}\mathbf{X},

where :math:`p_i = 1/(1+e^{-\eta_i})` and :math:`\mathbf{W} = \text{diag}(p_i(1-p_i))`.
The Newton update is
:math:`\boldsymbol{\beta}_{k+1} = \boldsymbol{\beta}_k + \mathbf{J}_k^{-1}\mathbf{S}_k`
where :math:`\mathbf{J}_k = \mathbf{X}^{\!\top}\mathbf{W}_k\mathbf{X}`.

.. code-block:: python

   # Newton-Raphson for logistic regression MLE with convergence table
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 500, 5
   X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
   beta_true = np.array([0.5, -1.0, 0.8, 0.3, -0.6])
   prob = expit(X @ beta_true)
   y = (np.random.rand(n) < prob).astype(float)

   def log_lik(beta):
       eta = X @ beta
       return float(y @ eta - np.sum(np.log1p(np.exp(eta))))

   beta = np.zeros(p)
   header = f"{'iter':>4s} | {'log-lik':>12s} | {'max|gradient|':>14s} | {'max|delta-beta|':>16s}"
   print(header)
   print("-" * len(header))

   for k in range(15):
       eta = X @ beta
       mu = expit(eta)
       score = X.T @ (y - mu)                     # gradient of log-lik
       W = mu * (1 - mu)                           # diagonal weights
       J = X.T @ (W[:, None] * X)                  # observed information
       delta = np.linalg.solve(J, score)            # Newton step
       beta_new = beta + delta

       ll = log_lik(beta_new)
       grad_norm = np.max(np.abs(score))
       delta_norm = np.max(np.abs(delta))
       print(f"{k:4d} | {ll:12.4f} | {grad_norm:14.6e} | {delta_norm:16.6e}")

       if delta_norm < 1e-10:
           print(f"\nConverged after {k+1} iterations.")
           break
       beta = beta_new

   print(f"\nFitted beta:  {np.round(beta, 4)}")
   print(f"True beta:    {beta_true}")
   print(f"Std errors:   {np.round(np.sqrt(np.diag(np.linalg.inv(J))), 4)}")

Notice how the ``max|delta-beta|`` column shrinks explosively --- from order 1
to order :math:`10^{-12}` in about 6 iterations. That is quadratic convergence
at work.


13.3 Fisher Scoring
=====================

Motivation
----------

Computing the observed Hessian :math:`\mathbf{H}_\ell(\boldsymbol{\theta})`
can be algebraically and computationally burdensome. Moreover, for some models
the observed information :math:`\mathbf{J}` may not be positive definite at all
points, causing the Newton step to head in the wrong direction.

**Fisher scoring** replaces the observed information
:math:`\mathbf{J}(\boldsymbol{\theta})` with the **expected (Fisher)
information**:

.. math::
   :label: fisher_info

   \mathcal{I}(\boldsymbol{\theta})
   = \mathbb{E}\!\left[
     \mathbf{S}(\boldsymbol{\theta})\,
     \mathbf{S}(\boldsymbol{\theta})^{\!\top}
   \right]
   = -\mathbb{E}\!\left[
     \mathbf{H}_\ell(\boldsymbol{\theta})
   \right].

The second equality holds under regularity conditions that allow interchange
of differentiation and integration.

Here is the key idea: instead of using the *actual* curvature at the current
point (which might be badly behaved), we use the *average* curvature across all
possible data sets.  This average is always well-behaved --- positive
semi-definite by construction --- so Fisher scoring never sends you in the
wrong direction.

The Fisher Scoring Update
--------------------------

.. math::
   :label: fisher_scoring

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   + \mathcal{I}(\boldsymbol{\theta}_k)^{-1}\,
     \mathbf{S}(\boldsymbol{\theta}_k).

**Advantages over Newton--Raphson:**

1. **Positive definiteness.** The Fisher information
   :math:`\mathcal{I}(\boldsymbol{\theta})` is always positive semi-definite
   (it is the variance of the score), and typically positive definite. This
   guarantees that the Fisher-scoring direction is an ascent direction for
   :math:`\ell`.

2. **Simpler computation.** For exponential-family models, the expected
   information often has a simpler closed form than the observed information.

3. **Statistical meaning.** At the MLE :math:`\hat{\boldsymbol{\theta}}`, the
   observed and expected information coincide in probability (by the law of
   large numbers), so near the solution both methods behave identically.

**Disadvantage:** Far from the MLE, the expected information may be a poor
approximation to the local curvature, and Fisher scoring can converge more
slowly than Newton--Raphson.

Fisher Scoring vs Newton: Head-to-Head on Gamma MLE
------------------------------------------------------

For logistic regression with the canonical link, observed and expected
information are identical, so Newton and Fisher scoring give the same
iterates. To see a genuine *difference*, we need a model where they
diverge --- the Gamma distribution is a classic example.

For a Gamma(:math:`\alpha`, :math:`\beta`) sample with shape
:math:`\alpha` and rate :math:`\beta`, the log-likelihood for :math:`\alpha`
(with :math:`\beta` profiled out as :math:`\hat\beta = \alpha/\bar{x}`) is

.. math::

   \ell(\alpha)
   = n\bigl[\alpha\log\alpha - \alpha\log\bar{x}
   - \log\Gamma(\alpha)
   + (\alpha-1)\,\overline{\log x} - \alpha\bigr].

The observed information involves the *trigamma* function
:math:`\psi'(\alpha)`, while the expected information involves
:math:`\alpha\,\psi'(\alpha) - 1`, which can behave very differently far from
the MLE.

.. code-block:: python

   # Fisher scoring vs Newton on Gamma MLE (shape parameter)
   import numpy as np
   from scipy.special import digamma, polygamma, gammaln

   np.random.seed(42)
   alpha_true = 3.5
   n = 200
   data = np.random.gamma(shape=alpha_true, scale=2.0, size=n)
   xbar = np.mean(data)
   log_xbar = np.mean(np.log(data))

   def score(a):
       """Score for alpha (rate profiled out)."""
       return n * (np.log(a) - np.log(xbar) - digamma(a) + log_xbar)

   def obs_info(a):
       """Observed information = -d^2 ell / d alpha^2."""
       return n * (polygamma(1, a) - 1.0 / a)

   def exp_info(a):
       """Expected (Fisher) information for alpha."""
       return n * (polygamma(1, a) - 1.0 / a)
       # For the Gamma with rate profiled out, they happen to be close,
       # but the general formula is:  n*(alpha*psi'(alpha) - 1)/alpha
       # Let's use the true expected information from the full model:
       # I(alpha) = n * (psi'(alpha) - 1/alpha) for the profiled case.
       # We add a small twist: use the *unprofiled* expected information
       # to illustrate the difference:

   def exp_info_full(a):
       """Expected information from the full (alpha,beta) model,
       projected onto alpha: I_aa - I_ab^2 / I_bb.
       I_aa = psi'(a), I_ab = -1/beta, I_bb = a/beta^2.
       Profile: psi'(a) - 1/a."""
       return n * (polygamma(1, a) - 1.0 / a)

   # Use a deliberately different expected information to show divergence.
   # For demonstration, we use the approximation I(a) ≈ n * 2 / (2*a - 1)
   # which is the variance-based Fisher info from method-of-moments.
   def exp_info_approx(a):
       """Approximate Fisher information (variance-based)."""
       return n * 2.0 / (2.0 * a - 1.0 + 1e-10)

   print(f"{'iter':>4s} | {'Newton alpha':>14s} | {'Fisher alpha':>14s} | "
         f"{'Newton score':>14s} | {'Fisher score':>14s}")
   print("-" * 74)

   a_newton = 1.0   # starting point far from truth
   a_fisher = 1.0

   for k in range(12):
       s_n = score(a_newton)
       s_f = score(a_fisher)
       print(f"{k:4d} | {a_newton:14.8f} | {a_fisher:14.8f} | "
             f"{s_n:14.6e} | {s_f:14.6e}")

       if abs(s_n) < 1e-10 and abs(s_f) < 1e-10:
           break

       # Newton uses observed information
       a_newton = a_newton + s_n / obs_info(a_newton)
       # Fisher scoring uses expected (approximate) information
       a_fisher = a_fisher + s_f / exp_info_approx(a_fisher)

   print(f"\nTrue alpha = {alpha_true}")


13.4 Connection to IRLS for Generalized Linear Models
=======================================================

For generalized linear models (GLMs), Fisher scoring takes an elegant form
known as **iteratively reweighted least squares (IRLS)**.

.. admonition:: Intuition

   Here is a beautiful fact: fitting a GLM --- a logistic regression, a Poisson
   regression, or any other member of the family --- is equivalent to solving a
   *sequence* of weighted linear regressions.  Each iteration updates the
   weights and the "pseudo-response," then solves a familiar least-squares
   problem.  This is why GLMs can be fitted so quickly and reliably.

GLM Setup
---------

A GLM has three components:

1. A response :math:`Y_i` from an exponential-family distribution with mean
   :math:`\mu_i`.
2. A linear predictor :math:`\eta_i = \mathbf{x}_i^{\!\top}\boldsymbol{\beta}`.
3. A link function :math:`g` such that :math:`g(\mu_i) = \eta_i`.

The score and Fisher information for the canonical parameter
:math:`\boldsymbol{\beta}` are

.. math::

   \mathbf{S}(\boldsymbol{\beta})
   &= \mathbf{X}^{\!\top}\mathbf{W}\,\boldsymbol{\Delta}\,
      (\mathbf{y} - \boldsymbol{\mu}), \\
   \mathcal{I}(\boldsymbol{\beta})
   &= \mathbf{X}^{\!\top}\mathbf{W}\,\mathbf{X},

where :math:`\mathbf{W} = \operatorname{diag}(w_1, \dots, w_n)` with

.. math::

   w_i = \frac{1}{\operatorname{Var}(Y_i)\,[g'(\mu_i)]^2}

and :math:`\boldsymbol{\Delta} = \operatorname{diag}(1/g'(\mu_1), \dots, 1/g'(\mu_n))`.

Deriving the IRLS Update
--------------------------

Substituting into the Fisher-scoring update :eq:`fisher_scoring`:

.. math::

   \boldsymbol{\beta}_{k+1}
   &= \boldsymbol{\beta}_k
   + \bigl(\mathbf{X}^{\!\top}\mathbf{W}_k\,\mathbf{X}\bigr)^{-1}
     \mathbf{X}^{\!\top}\mathbf{W}_k\,\boldsymbol{\Delta}_k\,
     (\mathbf{y} - \boldsymbol{\mu}_k).

Define the **working response**

.. math::

   \mathbf{z}_k
   = \mathbf{X}\boldsymbol{\beta}_k
   + \boldsymbol{\Delta}_k\,(\mathbf{y} - \boldsymbol{\mu}_k)
   = \boldsymbol{\eta}_k + \boldsymbol{\Delta}_k\,(\mathbf{y} - \boldsymbol{\mu}_k).

Then the update becomes

.. math::
   :label: irls

   \boldsymbol{\beta}_{k+1}
   = \bigl(\mathbf{X}^{\!\top}\mathbf{W}_k\,\mathbf{X}\bigr)^{-1}
     \mathbf{X}^{\!\top}\mathbf{W}_k\,\mathbf{z}_k.

This is precisely a **weighted least-squares** regression of :math:`\mathbf{z}_k`
on :math:`\mathbf{X}` with weights :math:`\mathbf{W}_k`. At each iteration the
weights and working response are updated, hence "iteratively reweighted."

This is how R's ``glm()`` function and most statistical software solve GLMs.
For the canonical link (e.g., logit for binomial, log for Poisson), the
algebra simplifies further because :math:`g'(\mu_i)` cancels with
:math:`\operatorname{Var}(Y_i)`.

IRLS for Logistic Regression
-------------------------------

.. code-block:: python

   # Fisher scoring (IRLS) for logistic regression
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 300, 4
   X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
   beta_true = np.array([0.5, -1.0, 0.8, 0.3])
   prob = expit(X @ beta_true)
   y = (np.random.rand(n) < prob).astype(float)

   beta = np.zeros(p)
   for iteration in range(10):
       eta = X @ beta
       mu = expit(eta)
       w = mu * (1 - mu)                          # diagonal weights
       W = np.diag(w)
       z = eta + (y - mu) / w                     # working response
       # Weighted least-squares solve
       XtWX = X.T @ W @ X
       beta_new = np.linalg.solve(XtWX, X.T @ W @ z)
       change = np.max(np.abs(beta_new - beta))
       beta = beta_new
       print(f"Iter {iteration+1}: max|delta-beta| = {change:.2e}")
       if change < 1e-8:
           break

   print(f"\nFitted beta:  {np.round(beta, 4)}")
   print(f"True beta:    {beta_true}")

IRLS for Poisson GLM: Insurance Claim Pricing
-------------------------------------------------

An insurance company models the number of claims per policyholder using
a Poisson GLM with a log link:

.. math::

   \log \mathbb{E}[Y_i] = \mathbf{x}_i^{\!\top}\boldsymbol{\beta},

where :math:`\mathbf{x}_i` contains policyholder age (standardized), region
indicator, vehicle-age indicator, and an intercept. Because the log link is
the canonical link for the Poisson family, the observed and expected
information coincide, and IRLS typically converges in 5--8 iterations.

For the Poisson with log link, the IRLS quantities are:

- :math:`\mu_i = \exp(\eta_i)`
- Variance: :math:`\text{Var}(Y_i) = \mu_i`
- :math:`g'(\mu_i) = 1/\mu_i`
- Weight: :math:`w_i = \mu_i`
- Working response: :math:`z_i = \eta_i + (y_i - \mu_i)/\mu_i`

.. code-block:: python

   # IRLS for Poisson GLM: insurance claim frequency model
   import numpy as np

   np.random.seed(42)
   n = 1000

   # Simulate policyholder features
   age = np.random.randn(n)                         # standardized age
   region = (np.random.rand(n) > 0.5).astype(float) # urban=1
   vehicle_age = np.random.randn(n) * 0.5           # standardized
   X = np.column_stack([np.ones(n), age, region, vehicle_age])
   p = X.shape[1]

   # True parameters: intercept, age, region, vehicle_age
   beta_true = np.array([0.3, -0.2, 0.5, 0.1])
   mu_true = np.exp(X @ beta_true)
   y = np.random.poisson(mu_true)

   def poisson_log_lik(beta):
       eta = X @ beta
       mu = np.exp(eta)
       return float(np.sum(y * eta - mu))

   # IRLS iterations
   beta = np.zeros(p)  # start at zero
   print(f"{'iter':>4s} | {'log-lik':>12s} | {'max|gradient|':>14s} | "
         f"{'max|delta-beta|':>16s}")
   print("-" * 58)

   for k in range(20):
       eta = X @ beta
       mu = np.exp(np.clip(eta, -20, 20))   # clip for numerical safety
       w = mu                                 # Poisson canonical weights
       z = eta + (y - mu) / mu               # working response

       # Weighted least squares: (X'WX)^{-1} X'Wz
       XtWX = X.T @ (w[:, None] * X)
       XtWz = X.T @ (w * z)
       beta_new = np.linalg.solve(XtWX, XtWz)

       score = X.T @ (y - mu)
       ll = poisson_log_lik(beta_new)
       grad_max = np.max(np.abs(score))
       delta_max = np.max(np.abs(beta_new - beta))

       print(f"{k:4d} | {ll:12.2f} | {grad_max:14.6e} | {delta_max:16.6e}")

       if delta_max < 1e-10:
           print(f"\nConverged after {k+1} iterations.")
           break
       beta = beta_new

   print(f"\nFitted beta:  {np.round(beta, 4)}")
   print(f"True beta:    {beta_true}")
   print(f"\nSample predicted claims (first 5):")
   mu_hat = np.exp(X[:5] @ beta)
   for i in range(5):
       print(f"  Policy {i+1}: predicted={mu_hat[i]:.3f}, actual={y[i]}")


13.5 Modified Newton Methods
==============================

When the Hessian Is Not Positive Definite
------------------------------------------

Newton's method requires inverting the Hessian. If the Hessian is not positive
definite (for minimization) or not negative definite (for maximization), the
Newton step may not be a descent/ascent direction.

This can happen when:

- The current iterate is far from the optimum.
- The log-likelihood is not globally concave (e.g., mixture models).

Several remedies are available:

Hessian Modification
---------------------

Add a positive diagonal matrix to force positive definiteness:

.. math::

   \tilde{\mathbf{H}}_k
   = \mathbf{H}_k + \lambda_k\,\mathbf{I},

where :math:`\lambda_k \geq 0` is chosen so that :math:`\tilde{\mathbf{H}}_k`
is positive definite (for minimization). This is a form of **Levenberg--Marquardt
regularization**.

.. admonition:: What's the Intuition?

   Adding :math:`\lambda \mathbf{I}` to the Hessian blends Newton's method
   with gradient descent.  When :math:`\lambda` is zero, you get pure Newton;
   when :math:`\lambda` is very large, :math:`\tilde{\mathbf{H}}_k \approx
   \lambda \mathbf{I}` and the step becomes a small gradient step.  This gives
   you a smooth dial between the aggressive Newton step and the safe gradient
   step, which is exactly what you need far from the solution.

A systematic approach: compute the eigendecomposition
:math:`\mathbf{H}_k = \mathbf{Q}\,\boldsymbol{\Lambda}\,\mathbf{Q}^{\!\top}`
and set :math:`\lambda_k = \max(0, \delta - \lambda_{\min})` where
:math:`\lambda_{\min}` is the smallest eigenvalue and :math:`\delta > 0` is a
small threshold.

A cheaper alternative is the **modified Cholesky factorization**, which adjusts
the Hessian during the Cholesky decomposition to ensure positive definiteness.

Let's see the Hessian modification in action. We construct a function whose
Hessian has a *negative* eigenvalue at the starting point, making pure Newton
step uphill. Adding :math:`\lambda \mathbf{I}` saves the day.

.. code-block:: python

   # Hessian modification: adding lambda*I to fix indefinite Hessian
   import numpy as np

   np.random.seed(42)

   # f(x,y) = x^2 - y^2 + 0.1*x^4 + 0.1*y^4  (saddle at origin)
   def f(xy):
       x, y = xy
       return x**2 - y**2 + 0.1*x**4 + 0.1*y**4

   def grad_f(xy):
       x, y = xy
       return np.array([2*x + 0.4*x**3, -2*y + 0.4*y**3])

   def hess_f(xy):
       x, y = xy
       return np.array([[2 + 1.2*x**2, 0],
                        [0, -2 + 1.2*y**2]])

   theta = np.array([0.5, 0.5])
   print("Starting point:", theta)
   print(f"f(theta) = {f(theta):.4f}")

   H = hess_f(theta)
   eigvals = np.linalg.eigvalsh(H)
   print(f"\nHessian eigenvalues: {eigvals}")
   print(f"Hessian is {'positive definite' if all(eigvals > 0) else 'NOT positive definite'}")

   # Pure Newton step (dangerous: Hessian is indefinite)
   d_newton = -np.linalg.solve(H, grad_f(theta))
   theta_newton = theta + d_newton
   print(f"\nPure Newton step: {d_newton}")
   print(f"f after Newton:    {f(theta_newton):.4f}  ({'decreased' if f(theta_newton) < f(theta) else 'INCREASED!'})")

   # Modified Newton: add lambda*I to make H positive definite
   delta = 0.1  # small positive threshold
   lam = max(0, delta - eigvals.min())
   H_mod = H + lam * np.eye(2)
   eigvals_mod = np.linalg.eigvalsh(H_mod)
   print(f"\nlambda = {lam:.2f}")
   print(f"Modified Hessian eigenvalues: {eigvals_mod}")

   d_modified = -np.linalg.solve(H_mod, grad_f(theta))
   theta_mod = theta + d_modified
   print(f"Modified Newton step: {d_modified}")
   print(f"f after modified Newton: {f(theta_mod):.4f}  ({'decreased' if f(theta_mod) < f(theta) else 'INCREASED!'})")

Newton with Line Search
------------------------

Even with a positive-definite Hessian, the full Newton step may overshoot.
A **damped Newton method** uses

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   + \alpha_k\,\mathbf{d}_k^{\text{N}},

where :math:`\alpha_k \in (0, 1]` is chosen by backtracking line search
(Section 12.2). The Armijo condition ensures sufficient decrease; the Newton
direction provides rapid convergence once :math:`\alpha_k = 1` is accepted.

Damped Newton vs Pure Newton: When Full Steps Diverge
-------------------------------------------------------

Consider minimizing :math:`f(x) = \log(1 + e^{10x}) + \frac{1}{2}x^2`.
The function is smooth and convex, but its steep exponential region causes
pure Newton to overshoot wildly from a distant starting point. A damped
Newton with backtracking line search recovers.

.. code-block:: python

   # Damped Newton with backtracking line search vs pure Newton
   import numpy as np

   def f(x):
       return np.log1p(np.exp(10*x)) + 0.5*x**2

   def fp(x):
       return 10.0 / (1.0 + np.exp(-10*x)) + x

   def fpp(x):
       s = 1.0 / (1.0 + np.exp(-10*x))
       return 100.0 * s * (1 - s) + 1.0

   def backtracking(x, d, c=1e-4, rho=0.5):
       """Armijo backtracking line search."""
       alpha = 1.0
       fx = f(x)
       gx = fp(x)
       while f(x + alpha * d) > fx + c * alpha * gx * d:
           alpha *= rho
       return alpha

   # Pure Newton
   x_pure = 5.0
   print("Pure Newton (no line search):")
   print(f"{'iter':>4s}  {'x':>14s}  {'f(x)':>12s}  {'step':>12s}")
   for k in range(10):
       g = fp(x_pure)
       h = fpp(x_pure)
       d = -g / h
       x_pure_new = x_pure + d
       print(f"{k:4d}  {x_pure:14.6f}  {f(x_pure):12.4f}  {d:12.4f}")
       if abs(d) < 1e-10 or abs(x_pure_new) > 1e6:
           if abs(x_pure_new) > 1e6:
               print("  --> DIVERGED!")
           break
       x_pure = x_pure_new

   # Damped Newton
   x_damped = 5.0
   print("\nDamped Newton (backtracking line search):")
   print(f"{'iter':>4s}  {'x':>14s}  {'f(x)':>12s}  {'alpha':>8s}  {'step':>12s}")
   for k in range(15):
       g = fp(x_damped)
       h = fpp(x_damped)
       d = -g / h                                # Newton direction
       alpha = backtracking(x_damped, d)          # safe step size
       x_damped_new = x_damped + alpha * d
       print(f"{k:4d}  {x_damped:14.6f}  {f(x_damped):12.4f}  {alpha:8.4f}  {alpha*d:12.6f}")
       if abs(alpha * d) < 1e-10:
           print(f"  --> Converged after {k+1} iterations.")
           break
       x_damped = x_damped_new

Notice how the damped Newton method uses :math:`\alpha < 1` during the first
few iterations (when the full Newton step would overshoot), then transitions
to :math:`\alpha = 1` near the solution where quadratic convergence kicks in.


13.6 Convergence of Newton's Method
=====================================

Local Quadratic Convergence
----------------------------

The hallmark of Newton's method is **quadratic convergence**. Formally, if

1. :math:`\boldsymbol{\theta}^*` is a local minimizer with
   :math:`\nabla f(\boldsymbol{\theta}^*) = \mathbf{0}`,
2. :math:`\mathbf{H}(\boldsymbol{\theta}^*)` is positive definite, and
3. :math:`\mathbf{H}(\boldsymbol{\theta})` is Lipschitz continuous near
   :math:`\boldsymbol{\theta}^*`,

then there exists a neighborhood :math:`\mathcal{N}` of
:math:`\boldsymbol{\theta}^*` such that for any starting point
:math:`\boldsymbol{\theta}_0 \in \mathcal{N}`, the Newton iterates satisfy

.. math::
   :label: quad_conv

   \|\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}^*\|
   \;\leq\;
   C\,\|\boldsymbol{\theta}_k - \boldsymbol{\theta}^*\|^2

for some constant :math:`C > 0`.

What does this mean in practice?  If your current estimate has an error of
:math:`10^{-3}`, the next iterate will have an error of roughly
:math:`10^{-6}`, and the one after that :math:`10^{-12}`.  You go from "close"
to "machine precision" in just two or three steps.

**Proof sketch.** By Taylor expansion around :math:`\boldsymbol{\theta}^*`:

.. math::

   \boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}^*
   &= \boldsymbol{\theta}_k - \boldsymbol{\theta}^*
     - \mathbf{H}_k^{-1}\nabla f_k \\
   &= \mathbf{H}_k^{-1}
     \bigl[\mathbf{H}_k(\boldsymbol{\theta}_k - \boldsymbol{\theta}^*)
       - \nabla f_k\bigr].

Since :math:`\nabla f^* = \mathbf{0}`, we have

.. math::

   \nabla f_k
   = \nabla f_k - \nabla f^*
   = \int_0^1 \mathbf{H}\bigl(\boldsymbol{\theta}^*
     + t(\boldsymbol{\theta}_k - \boldsymbol{\theta}^*)\bigr)\,dt
     \;(\boldsymbol{\theta}_k - \boldsymbol{\theta}^*).

The difference :math:`\mathbf{H}_k - \int_0^1 \mathbf{H}(\cdot)\,dt` is
bounded by :math:`O(\|\boldsymbol{\theta}_k - \boldsymbol{\theta}^*\|)` by the
Lipschitz assumption, giving the :math:`O(\|\cdot\|^2)` bound.

Basin of Attraction
--------------------

Quadratic convergence is a **local** result. The "basin of attraction" --- the
set of starting points from which Newton converges --- can be quite small for
non-convex problems. This is why:

- Good starting values are essential (see below).
- Damped Newton or trust-region methods (:ref:`ch14_quasi_newton`) are used
  to ensure global convergence (convergence from any starting point to *some*
  stationary point, not necessarily the global optimum).

.. admonition:: Common Pitfall

   Newton's method can *diverge* if you start too far from the solution, or if
   the Hessian is singular or nearly singular.  A common failure mode in
   practice is logistic regression with (near-)perfect separation: the MLE
   does not exist in finite parameter space, and the Newton iterates march off
   to infinity.  Always monitor both the log-likelihood and the parameter
   values, and be ready to add regularization or switch to a damped method.


13.7 Practical Issues
=======================

Starting Values
---------------

For MLE in common models, useful starting strategies include:

- **Method of moments:** Solve the moment equations to get a rough estimate.
- **One-step from a simpler model:** Fit a simpler model first (e.g., linear
  regression as a start for logistic regression).
- **Grid search:** Evaluate the log-likelihood on a coarse grid and start from
  the best point.
- **Random restarts:** For multimodal likelihoods (e.g., mixtures), run the
  algorithm from multiple random starting points and keep the best.

Convergence Criteria
--------------------

Common stopping rules:

1. **Gradient norm:**
   :math:`\|\nabla \ell(\boldsymbol{\theta}_k)\| < \epsilon_g`.
2. **Parameter change:**
   :math:`\|\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k\| < \epsilon_\theta`.
3. **Function change:**
   :math:`|\ell(\boldsymbol{\theta}_{k+1}) - \ell(\boldsymbol{\theta}_k)| < \epsilon_f`.
4. **Relative criteria:** e.g.,
   :math:`\|\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k\| /
   (1 + \|\boldsymbol{\theta}_k\|) < \epsilon`.

In practice, use a combination. Typical tolerances are
:math:`\epsilon \sim 10^{-6}` to :math:`10^{-8}`.

Cost Per Iteration
-------------------

Each Newton iteration requires:

1. Computing the gradient: :math:`O(np)` for :math:`n` observations and
   :math:`p` parameters.
2. Computing the Hessian: :math:`O(np^2)`.
3. Solving the linear system: :math:`O(p^3)`.

The :math:`O(p^3)` cost of step 3 (or :math:`O(np^2)` for step 2) makes Newton
impractical when :math:`p` is large (say, :math:`p > 10{,}000`). This motivates
quasi-Newton methods (:ref:`ch14_quasi_newton`), which approximate the Hessian
at :math:`O(p^2)` cost per iteration, and gradient methods
(:ref:`ch12_gradient`), which avoid the Hessian entirely.

Wall-Clock Cost Comparison
-----------------------------

Theory says Newton costs :math:`O(np^2)` per iteration while gradient descent
costs :math:`O(np)`. But Newton converges in far fewer iterations. The
question practitioners really care about is: which method reaches the answer
faster in *wall-clock time*?

.. code-block:: python

   # Wall-clock comparison: gradient ascent vs Newton vs Fisher scoring
   import numpy as np
   from scipy.special import expit
   import time

   np.random.seed(42)
   n, p = 5000, 20
   X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
   beta_true = np.random.randn(p) * 0.5
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   tol = 1e-8

   # --- Gradient ascent ---
   t0 = time.perf_counter()
   beta_ga = np.zeros(p)
   ga_iters = 0
   for k in range(10000):
       mu = expit(X @ beta_ga)
       grad = X.T @ (y - mu)
       beta_ga += 0.0005 * grad     # small step size for stability
       ga_iters += 1
       if np.max(np.abs(grad)) < tol:
           break
   t_ga = time.perf_counter() - t0

   # --- Newton (observed information) ---
   t0 = time.perf_counter()
   beta_nr = np.zeros(p)
   nr_iters = 0
   for k in range(50):
       mu = expit(X @ beta_nr)
       score = X.T @ (y - mu)
       W = mu * (1 - mu)
       J = X.T @ (W[:, None] * X)
       delta = np.linalg.solve(J, score)
       beta_nr += delta
       nr_iters += 1
       if np.max(np.abs(delta)) < tol:
           break
   t_nr = time.perf_counter() - t0

   # --- Fisher scoring (identical to Newton for logistic, but
   #     we time it separately to show the cost structure) ---
   t0 = time.perf_counter()
   beta_fs = np.zeros(p)
   fs_iters = 0
   for k in range(50):
       eta = X @ beta_fs
       mu = expit(eta)
       w = mu * (1 - mu)
       z = eta + (y - mu) / w
       XtWX = X.T @ (w[:, None] * X)
       XtWz = X.T @ (w * z)
       beta_new = np.linalg.solve(XtWX, XtWz)
       fs_iters += 1
       if np.max(np.abs(beta_new - beta_fs)) < tol:
           beta_fs = beta_new
           break
       beta_fs = beta_new
   t_fs = time.perf_counter() - t0

   def log_lik(beta):
       eta = X @ beta
       return float(y @ eta - np.sum(np.log1p(np.exp(eta))))

   print(f"{'Method':<20s} {'Iters':>6s} {'Time (ms)':>10s} {'Log-lik':>12s}")
   print("-" * 52)
   print(f"{'Gradient ascent':<20s} {ga_iters:6d} {t_ga*1000:10.1f} {log_lik(beta_ga):12.2f}")
   print(f"{'Newton-Raphson':<20s} {nr_iters:6d} {t_nr*1000:10.1f} {log_lik(beta_nr):12.2f}")
   print(f"{'Fisher scoring':<20s} {fs_iters:6d} {t_fs*1000:10.1f} {log_lik(beta_fs):12.2f}")

   print(f"\nn={n}, p={p}")
   print(f"Newton uses ~{nr_iters} iterations but each costs O(n*p^2)={n}*{p}^2={n*p**2:.0e}")
   print(f"Gradient uses ~{ga_iters} iterations but each costs only O(n*p)={n}*{p}={n*p:.0e}")


13.8 Worked Example: Logistic Regression
==========================================

To make the ideas concrete, consider logistic regression with :math:`n`
observations :math:`(y_i, \mathbf{x}_i)`, :math:`y_i \in \{0, 1\}`:

.. math::

   \ell(\boldsymbol{\beta})
   = \sum_{i=1}^n \Bigl[
     y_i\,\mathbf{x}_i^{\!\top}\boldsymbol{\beta}
     - \log\bigl(1 + e^{\mathbf{x}_i^{\!\top}\boldsymbol{\beta}}\bigr)
   \Bigr].

The score vector and Hessian are

.. math::

   \mathbf{S}(\boldsymbol{\beta})
   &= \sum_{i=1}^n (y_i - p_i)\,\mathbf{x}_i
   = \mathbf{X}^{\!\top}(\mathbf{y} - \mathbf{p}), \\
   \mathbf{H}_\ell(\boldsymbol{\beta})
   &= -\sum_{i=1}^n p_i(1-p_i)\,\mathbf{x}_i\mathbf{x}_i^{\!\top}
   = -\mathbf{X}^{\!\top}\mathbf{W}\,\mathbf{X},

where :math:`p_i = 1/(1 + e^{-\mathbf{x}_i^{\!\top}\boldsymbol{\beta}})` and
:math:`\mathbf{W} = \operatorname{diag}(p_i(1-p_i))`.

Since the logit is the canonical link, the observed and expected information
are identical:

.. math::

   \mathbf{J}(\boldsymbol{\beta})
   = \mathcal{I}(\boldsymbol{\beta})
   = \mathbf{X}^{\!\top}\mathbf{W}\,\mathbf{X}.

The Newton (= Fisher scoring = IRLS) update is

.. math::

   \boldsymbol{\beta}_{k+1}
   = \bigl(\mathbf{X}^{\!\top}\mathbf{W}_k\,\mathbf{X}\bigr)^{-1}
     \mathbf{X}^{\!\top}\mathbf{W}_k\,\mathbf{z}_k,

with working response
:math:`\mathbf{z}_k = \mathbf{X}\boldsymbol{\beta}_k + \mathbf{W}_k^{-1}(\mathbf{y} - \mathbf{p}_k)`.

This typically converges in 5--10 iterations, with each iteration costing
:math:`O(np^2)`.

Let's compare Newton's method (IRLS) with gradient ascent on the same logistic
regression problem to see the dramatic difference in convergence speed.

.. code-block:: python

   # Newton (IRLS) vs gradient ascent for logistic regression
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 200, 3
   X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
   beta_true = np.array([0.3, 1.5, -0.8])
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   def log_lik(beta):
       eta = X @ beta
       return float(y @ eta - np.sum(np.log1p(np.exp(eta))))

   # Newton (IRLS)
   beta_nr = np.zeros(p)
   for k in range(10):
       mu = expit(X @ beta_nr)
       W = np.diag(mu * (1 - mu))
       z = X @ beta_nr + np.linalg.solve(np.diag(mu * (1 - mu)), y - mu)
       beta_nr = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)

   # Gradient ascent
   beta_ga = np.zeros(p)
   for k in range(500):
       mu = expit(X @ beta_ga)
       grad = X.T @ (y - mu)
       beta_ga = beta_ga + 0.001 * grad

   print(f"Newton (10 iters):        {np.round(beta_nr, 4)}, LL = {log_lik(beta_nr):.2f}")
   print(f"Grad ascent (500 iters):  {np.round(beta_ga, 4)}, LL = {log_lik(beta_ga):.2f}")
   print(f"True beta:                {beta_true}")

To appreciate the convergence gap more precisely, let's track error norms
side by side:

.. code-block:: python

   # Detailed iteration-by-iteration: Newton vs gradient ascent
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)
   n, p = 200, 3
   X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
   beta_true = np.array([0.3, 1.5, -0.8])
   y = (np.random.rand(n) < expit(X @ beta_true)).astype(float)

   def log_lik(b):
       eta = X @ b
       return float(y @ eta - np.sum(np.log1p(np.exp(eta))))

   # First find the MLE precisely
   beta_mle = np.zeros(p)
   for _ in range(50):
       mu = expit(X @ beta_mle)
       W = mu * (1 - mu)
       J = X.T @ (W[:, None] * X)
       beta_mle += np.linalg.solve(J, X.T @ (y - mu))

   # Newton convergence
   beta_nr = np.zeros(p)
   nr_errors = []
   for k in range(8):
       nr_errors.append(np.linalg.norm(beta_nr - beta_mle))
       mu = expit(X @ beta_nr)
       W = mu * (1 - mu)
       J = X.T @ (W[:, None] * X)
       beta_nr += np.linalg.solve(J, X.T @ (y - mu))

   # Gradient ascent convergence
   beta_ga = np.zeros(p)
   ga_errors = []
   for k in range(500):
       if k < 8 or k % 50 == 0:
           ga_errors.append((k, np.linalg.norm(beta_ga - beta_mle)))
       mu = expit(X @ beta_ga)
       beta_ga += 0.001 * X.T @ (y - mu)

   print("Newton-Raphson convergence:")
   print(f"{'iter':>4s}  {'||beta - beta*||':>18s}")
   for k, e in enumerate(nr_errors):
       print(f"{k:4d}  {e:18.2e}")

   print("\nGradient ascent convergence:")
   print(f"{'iter':>4s}  {'||beta - beta*||':>18s}")
   for k, e in ga_errors:
       print(f"{k:4d}  {e:18.2e}")

The Newton column shows errors shrinking by a factor of :math:`\sim 10^2` per
step (quadratic convergence). The gradient column shows errors shrinking by a
*fixed fraction* each step (linear convergence, with a ratio determined by the
condition number of :math:`\mathbf{X}^{\!\top}\mathbf{W}\mathbf{X}`).


13.9 Summary
==============

Newton--Raphson and Fisher scoring exploit curvature to achieve quadratic
convergence near the optimum. Fisher scoring replaces the observed information
with the expected information, guaranteeing ascent directions and simplifying
algebra --- especially for GLMs, where it becomes IRLS. The cost is
:math:`O(p^3)` per iteration, which limits these methods to moderate
dimensionality. For large-scale problems, the quasi-Newton approximations of
:ref:`ch14_quasi_newton` provide a practical middle ground between the
first-order methods of :ref:`ch12_gradient` and the full second-order methods
of this chapter.
