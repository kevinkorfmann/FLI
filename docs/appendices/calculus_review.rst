.. _appendix_calculus:

================================
Appendix B: Calculus Review
================================

This appendix reviews the calculus tools that underpin likelihood-based
inference. Every rule and formula is tied to a concrete statistical
application and verified in code. The goal is not to re-teach calculus but to
show *why* each tool matters the moment you open a likelihood function.

If you have taken a standard calculus sequence, the rules below will be
familiar. What is different here is the lens: every derivative is a score
function, every integral is a normalizing constant, and every Taylor
expansion is an approximation to a log-likelihood.


Differentiation Rules
======================

Throughout, we let :math:`f` and :math:`g` denote differentiable real-valued
functions.

Sum Rule
--------

.. math::

   \frac{d}{dx}\bigl[f(x) + g(x)\bigr] = f'(x) + g'(x).

**Why it matters for statistics:** the sum rule is the reason log-likelihoods
are so much easier than likelihoods. Taking the log of a product turns it
into a sum, and then we differentiate each observation's contribution
independently.

.. code-block:: python

   # Sum rule in action: log-likelihood of n independent observations
   # is a SUM of individual log-likelihoods
   import numpy as np
   from scipy.stats import poisson

   data = np.array([3, 5, 2, 7, 4])
   lam = 4.0

   # Log-likelihood: sum of individual terms
   individual_terms = [poisson.logpmf(x, mu=lam) for x in data]
   total_ll = sum(individual_terms)

   print("Individual log-likelihood contributions:")
   for i, (x, ll) in enumerate(zip(data, individual_terms)):
       print(f"  x_{i} = {x}: log f(x|lam) = {ll:.4f}")
   print(f"Total log-likelihood = {total_ll:.4f}")
   print(f"Verify: {np.sum(poisson.logpmf(data, mu=lam)):.4f}")

Product Rule
------------

.. math::

   \frac{d}{dx}\bigl[f(x)\,g(x)\bigr] = f'(x)\,g(x) + f(x)\,g'(x).

You encounter the product rule when differentiating likelihood contributions
where both the location and scale depend on the same parameter --- for
example, in heteroscedastic models.

.. code-block:: python

   # Product rule: derivative of x * exp(-x^2)
   # This shape appears in the score of a Rayleigh distribution
   import numpy as np

   x = 2.0
   # f(x) = x, g(x) = exp(-x^2)
   # d/dx [x * exp(-x^2)] = 1 * exp(-x^2) + x * (-2x) * exp(-x^2)
   #                       = exp(-x^2) * (1 - 2x^2)

   analytical = np.exp(-x**2) * (1 - 2 * x**2)

   eps = 1e-7
   h = lambda t: t * np.exp(-t**2)
   numerical = (h(x + eps) - h(x)) / eps

   print(f"Product rule (analytical): {analytical:.8f}")
   print(f"Finite difference:         {numerical:.8f}")

Chain Rule
----------

If :math:`h(x) = f(g(x))`, then:

.. math::

   h'(x) = f'\bigl(g(x)\bigr)\,g'(x).

The chain rule is the single most important differentiation tool in
likelihood theory. **The score function is the chain rule applied to**
:math:`\log(\cdot)` **composed with** :math:`L(\cdot)`:

.. math::

   s(\theta) = \frac{d}{d\theta}\log L(\theta)
   = \frac{1}{L(\theta)}\cdot\frac{dL}{d\theta}.

.. code-block:: python

   # Chain rule = score function: d/d(theta) log L(theta)
   import numpy as np
   from scipy.stats import poisson

   data = np.array([3, 5, 2, 7, 4])
   theta = 4.0  # Poisson rate

   # The chain rule gives us the score:
   # d/d(lam) log L = d/d(lam) [sum x_i * log(lam) - n*lam - sum log(x_i!)]
   #                = sum(x_i) / lam - n
   score_chain_rule = data.sum() / theta - len(data)

   # Verify numerically
   def loglik(lam):
       return np.sum(poisson.logpmf(data, mu=lam))

   eps = 1e-7
   score_numerical = (loglik(theta + eps) - loglik(theta)) / eps

   print(f"Score via chain rule: {score_chain_rule:.8f}")
   print(f"Score via finite diff: {score_numerical:.8f}")
   print(f"MLE (where score = 0): lam_hat = {data.mean()}")

**Logarithmic differentiation.** For :math:`L(\theta) > 0`:

.. math::

   \frac{d}{d\theta}\log L(\theta)
   = \frac{1}{L(\theta)}\,\frac{dL}{d\theta}.

This converts differentiating a product (likelihood) into differentiating a
sum (log-likelihood) --- one of the most powerful tricks in statistics.

.. code-block:: python

   # Logarithmic differentiation for a Binomial likelihood
   # L(p) = C * p^s * (1-p)^(n-s)
   # log L = const + s*log(p) + (n-s)*log(1-p)
   # d/dp log L = s/p - (n-s)/(1-p)
   import numpy as np

   n, s = 100, 73
   p = 0.7

   # Score via log differentiation
   score = s / p - (n - s) / (1 - p)

   # Score via direct differentiation of L, then dividing by L
   # dL/dp = C * [s * p^(s-1) * (1-p)^(n-s) - (n-s) * p^s * (1-p)^(n-s-1)]
   # dL/dp / L = s/p - (n-s)/(1-p)  (same thing!)
   print(f"Score at p={p}: {score:.4f}")
   print(f"Score at MLE p={s/n}: {s/(s/n) - (n-s)/(1-s/n):.10f}")


Partial Derivatives and Gradients
==================================

For a function :math:`f : \mathbb{R}^n \to \mathbb{R}`, the **gradient** is

.. math::

   \nabla f(\mathbf{x})
   = \begin{pmatrix}
   \partial f / \partial x_1 \\
   \vdots \\
   \partial f / \partial x_n
   \end{pmatrix}.

Setting :math:`\nabla \ell = \mathbf{0}` gives the MLE in multiparameter
models.

.. code-block:: python

   # Gradient of a Normal log-likelihood with two parameters (mu, sigma)
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   data = np.random.normal(loc=3.0, scale=2.0, size=100)

   def loglik(mu, log_sigma):
       sigma = np.exp(log_sigma)
       return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

   # Analytical gradient for Normal:
   #   d/d(mu) = sum(x_i - mu) / sigma^2
   #   d/d(sigma) = -n/sigma + sum(x_i - mu)^2 / sigma^3
   mu_hat = data.mean()
   sigma_hat = data.std()

   grad_mu = np.sum(data - mu_hat) / sigma_hat**2
   grad_sigma = -len(data) / sigma_hat + np.sum((data - mu_hat)**2) / sigma_hat**3

   print(f"MLE: mu = {mu_hat:.4f}, sigma = {sigma_hat:.4f}")
   print(f"Gradient at MLE: d/d(mu) = {grad_mu:.6f}, d/d(sigma) = {grad_sigma:.6f}")
   print(f"(Both should be ~0 at the MLE)")

   # Numerical verification
   eps = 1e-7
   grad_mu_num = (loglik(mu_hat + eps, np.log(sigma_hat))
                  - loglik(mu_hat, np.log(sigma_hat))) / eps
   print(f"Numerical d/d(mu) at MLE: {grad_mu_num:.6f}")

**Multivariate chain rule and the Jacobian.** If :math:`f` depends on
:math:`\mathbf{x}` and :math:`\mathbf{x}` depends on :math:`\mathbf{t}`:

.. math::

   \nabla_{\mathbf{t}} f
   = \mathbf{J}^\top \nabla_{\mathbf{x}} f,

where :math:`\mathbf{J}` is the Jacobian. This is the foundation of both
backpropagation and the delta method in statistics.

.. code-block:: python

   # Delta method: SE of a transformation g(theta_hat)
   # If theta_hat ~ N(theta, se^2), then g(theta_hat) ~ N(g(theta), [g'(theta)]^2 * se^2)
   import numpy as np

   # Example: Poisson rate lambda, want SE of exp(-lambda) = P(X=0)
   np.random.seed(42)
   data = np.random.poisson(3.0, size=100)
   lam_hat = data.mean()
   se_lam = np.sqrt(lam_hat / len(data))

   # g(lambda) = exp(-lambda), g'(lambda) = -exp(-lambda)
   g_hat = np.exp(-lam_hat)
   g_prime = -np.exp(-lam_hat)
   se_g = abs(g_prime) * se_lam  # delta method

   print(f"lam_hat = {lam_hat:.4f}, SE(lam) = {se_lam:.4f}")
   print(f"P(X=0) = exp(-lam_hat) = {g_hat:.6f}")
   print(f"SE of P(X=0) via delta method = {se_g:.6f}")
   print(f"95% CI for P(X=0): [{g_hat - 1.96*se_g:.6f}, {g_hat + 1.96*se_g:.6f}]")


Taylor Series
==============

Univariate Taylor Expansion
----------------------------

.. math::

   f(x) = f(a) + f'(a)\,(x - a) + \frac{f''(a)}{2!}\,(x - a)^2 + \cdots

The second-order expansion is the basis for the **Laplace approximation** and
for understanding the curvature of the log-likelihood near its maximum.

.. code-block:: python

   # Taylor approximations to log(x) around a=1
   # This is relevant because log appears in every log-likelihood
   import numpy as np

   a = 1.0
   x_test = 1.5

   f_true = np.log(x_test)
   taylor_1 = (x_test - a)                                   # order 1
   taylor_2 = taylor_1 - (1/2) * (x_test - a)**2             # order 2
   taylor_3 = taylor_2 + (1/3) * (x_test - a)**3             # order 3

   print(f"log({x_test}) = {f_true:.6f}")
   print(f"{'Order':>6} {'Approx':>10} {'Error':>10}")
   print("-" * 30)
   for order, approx in [(1, taylor_1), (2, taylor_2), (3, taylor_3)]:
       print(f"{order:6d} {approx:10.6f} {abs(approx - f_true):10.6f}")

Application: quadratic approximation to the log-likelihood
-------------------------------------------------------------

Expanding :math:`\ell(\theta)` about the MLE
:math:`\hat\theta` (where :math:`\ell'(\hat\theta) = 0`):

.. math::

   \ell(\theta)
   \approx \ell(\hat\theta)
   + \tfrac{1}{2}\,\ell''(\hat\theta)\,(\theta - \hat\theta)^2.

The linear term vanishes because the score is zero at the MLE. What remains
is a quadratic --- and a quadratic in the exponent is a Gaussian. **This is
precisely why the MLE is approximately normally distributed.**

.. code-block:: python

   # Quadratic approximation to a Poisson log-likelihood
   import numpy as np
   from scipy.stats import poisson

   np.random.seed(42)
   data = np.array([3, 7, 2, 5, 4, 6, 3, 5])
   n = len(data)
   theta_mle = data.mean()

   def loglik(theta):
       return np.sum(poisson.logpmf(data, mu=theta))

   # Analytical Hessian: d^2 ell / d(lambda)^2 = -sum(x_i) / lambda^2
   hessian = -data.sum() / theta_mle**2
   obs_info = -hessian  # observed Fisher information

   # Quadratic approximation
   theta_grid = np.linspace(2.0, 7.0, 200)
   ell_true = np.array([loglik(t) for t in theta_grid])
   ell_quad = loglik(theta_mle) + 0.5 * hessian * (theta_grid - theta_mle)**2

   # Compare at specific points
   print(f"MLE = {theta_mle:.2f}")
   print(f"Observed info = {obs_info:.4f}")
   print(f"SE = 1/sqrt(info) = {1/np.sqrt(obs_info):.4f}")
   print(f"Exact SE = sqrt(lam/n) = {np.sqrt(theta_mle/n):.4f}")
   print()
   print(f"{'theta':>8} {'True ell':>12} {'Quad approx':>12} {'Error':>10}")
   print("-" * 46)
   for t in [3.0, 3.5, 4.0, 4.375, 5.0, 5.5, 6.0]:
       true_val = loglik(t)
       quad_val = loglik(theta_mle) + 0.5 * hessian * (t - theta_mle)**2
       print(f"{t:8.2f} {true_val:12.4f} {quad_val:12.4f} {abs(true_val-quad_val):10.4f}")

.. admonition:: Common Pitfall

   The quadratic approximation is only good *near* the MLE. If the true
   log-likelihood is strongly skewed (small samples, parameters near
   boundaries), Wald intervals based on this approximation can be misleading.
   Profile likelihood intervals are more robust.

Multivariate Taylor Expansion
------------------------------

For :math:`f : \mathbb{R}^n \to \mathbb{R}`, expanded about :math:`\mathbf{a}`:

.. math::

   f(\mathbf{x}) \approx f(\mathbf{a})
   + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a})
   + \tfrac{1}{2}\,(\mathbf{x} - \mathbf{a})^\top
     \mathbf{H}(\mathbf{a})\,(\mathbf{x} - \mathbf{a}),

where :math:`\mathbf{H}` is the Hessian matrix.

.. code-block:: python

   # Multivariate Taylor: Normal log-likelihood with (mu, log_sigma)
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   data = np.random.normal(loc=5.0, scale=2.0, size=50)
   n = len(data)

   def loglik(params):
       mu, log_sig = params
       sig = np.exp(log_sig)
       return np.sum(norm.logpdf(data, mu, sig))

   # MLE
   mu_hat = data.mean()
   sig_hat = data.std()
   mle = np.array([mu_hat, np.log(sig_hat)])

   # Numerical Hessian at MLE
   eps = 1e-5
   H = np.zeros((2, 2))
   for i in range(2):
       for j in range(2):
           pp = mle.copy(); pp[i] += eps; pp[j] += eps
           pm = mle.copy(); pm[i] += eps; pm[j] -= eps
           mp = mle.copy(); mp[i] -= eps; mp[j] += eps
           mm = mle.copy(); mm[i] -= eps; mm[j] -= eps
           H[i, j] = (loglik(pp) - loglik(pm) - loglik(mp) + loglik(mm)) / (4 * eps**2)

   obs_info = -H
   cov = np.linalg.inv(obs_info)
   se = np.sqrt(np.diag(cov))

   print(f"MLE: mu = {mu_hat:.4f}, sigma = {sig_hat:.4f}")
   print(f"SEs: SE(mu) = {se[0]:.4f}, SE(log sigma) = {se[1]:.4f}")
   print(f"\nHessian at MLE:")
   print(np.array2string(H, precision=2))
   print(f"\nObserved information (= -H):")
   print(np.array2string(obs_info, precision=2))


Optimization
=============

Necessary Conditions for Extrema
----------------------------------

**First-order condition (score equation):**

.. math::

   f'(x^*) = 0 \quad \text{(univariate)}, \qquad
   \nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{(multivariate)}.

**Second derivative test:** at a critical point :math:`x^*`:

- :math:`f''(x^*) < 0 \implies` local maximum (what we want for log-likelihoods).
- The negative Hessian :math:`-\mathbf{H}` is the **observed information matrix**.

.. code-block:: python

   # Finding the MLE numerically: Gamma distribution
   import numpy as np
   from scipy.optimize import minimize_scalar
   from scipy.stats import gamma

   np.random.seed(42)
   alpha_true, beta_true = 3.0, 2.0
   data = gamma.rvs(a=alpha_true, scale=1/beta_true, size=50)

   def neg_loglik(alpha):
       return -np.sum(gamma.logpdf(data, a=alpha, scale=1/beta_true))

   result = minimize_scalar(neg_loglik, bounds=(0.5, 10.0), method='bounded')
   alpha_hat = result.x

   # Second derivative test: should be positive for neg_loglik (= minimum)
   eps = 1e-5
   d2 = (neg_loglik(alpha_hat+eps) - 2*neg_loglik(alpha_hat)
         + neg_loglik(alpha_hat-eps)) / eps**2

   se = 1.0 / np.sqrt(d2)  # d2 = observed info for neg_loglik

   print(f"MLE alpha = {alpha_hat:.4f} (true: {alpha_true})")
   print(f"d^2(-ell)/d(alpha)^2 at MLE = {d2:.4f} (>0 confirms minimum of -ell)")
   print(f"SE(alpha) = {se:.4f}")
   print(f"95% CI: [{alpha_hat - 1.96*se:.4f}, {alpha_hat + 1.96*se:.4f}]")

Convexity
---------

A function :math:`f` is **convex** if for all :math:`\mathbf{x}, \mathbf{y}`
and :math:`t \in [0,1]`:

.. math::

   f\bigl(t\,\mathbf{x} + (1-t)\,\mathbf{y}\bigr)
   \leq t\,f(\mathbf{x}) + (1-t)\,f(\mathbf{y}).

Many log-likelihoods for exponential family distributions are **concave**
(negative Hessian is positive definite), which guarantees the MLE is unique.

.. code-block:: python

   # Concavity check: logistic regression log-likelihood
   import numpy as np

   np.random.seed(42)
   n = 50
   X = np.column_stack([np.ones(n), np.random.randn(n)])
   beta_true = np.array([0.5, 1.5])
   prob = 1 / (1 + np.exp(-X @ beta_true))
   y = (np.random.rand(n) < prob).astype(float)

   def loglik(beta):
       eta = X @ beta
       return np.sum(y * eta - np.log(1 + np.exp(eta)))

   # Check Hessian at several points --- should always be negative semi-definite
   for beta_test in [np.array([0, 0]), np.array([1, 2]), np.array([-1, 0.5])]:
       eps = 1e-5
       H = np.zeros((2, 2))
       for i in range(2):
           for j in range(2):
               bp, bm = beta_test.copy(), beta_test.copy()
               bpp = beta_test.copy(); bpp[i] += eps; bpp[j] += eps
               bpm = beta_test.copy(); bpm[i] += eps; bpm[j] -= eps
               bmp = beta_test.copy(); bmp[i] -= eps; bmp[j] += eps
               bmm = beta_test.copy(); bmm[i] -= eps; bmm[j] -= eps
               H[i, j] = (loglik(bpp) - loglik(bpm)
                           - loglik(bmp) + loglik(bmm)) / (4 * eps**2)

       eigenvalues = np.linalg.eigvalsh(H)
       print(f"beta = {beta_test}: eigenvalues of H = "
             f"[{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}] "
             f"{'<= 0 (concave)' if all(eigenvalues <= 1e-10) else 'NOT concave'}")

.. admonition:: Why concavity matters

   When the log-likelihood is concave, any critical point is the global
   maximum. This is the case for logistic regression, Poisson regression,
   and all canonical-link GLMs. When the log-likelihood is *not* concave
   (mixture models, neural networks), algorithms like EM may converge to
   local maxima.


Integration Techniques
=======================

Integration by Substitution
-----------------------------

.. math::

   \int f(g(x))\,g'(x)\,dx = \int f(u)\,du.

This is how we compute normalizing constants of transformed random variables.

.. code-block:: python

   # Substitution: verify that Y = log(X) has density f_Y(y) = f_X(e^y) * e^y
   # when X ~ Gamma(alpha, beta)
   import numpy as np
   from scipy.stats import gamma
   from scipy import integrate

   alpha, beta = 3.0, 2.0

   # The density of Y = log(X) where X ~ Gamma
   def f_Y(y):
       x = np.exp(y)
       return gamma.pdf(x, a=alpha, scale=1/beta) * x  # Jacobian |dx/dy| = e^y

   # Verify it integrates to 1
   integral, _ = integrate.quad(f_Y, -10, 20)
   print(f"int f_Y(y) dy = {integral:.8f} (should be 1)")

   # Verify moments: E[Y] = E[log X] = psi(alpha) - log(beta) (digamma)
   from scipy.special import digamma
   EY_theory = digamma(alpha) - np.log(beta)
   EY_numerical, _ = integrate.quad(lambda y: y * f_Y(y), -10, 20)
   print(f"E[log X] numerical: {EY_numerical:.6f}")
   print(f"E[log X] theory:    {EY_theory:.6f}")

Integration by Parts
---------------------

.. math::

   \int u\,dv = uv - \int v\,du.

Integration by parts is essential for deriving moments and for the Gamma
function recursion :math:`\Gamma(\alpha+1) = \alpha\,\Gamma(\alpha)`.

.. code-block:: python

   # Integration by parts proves Gamma(alpha+1) = alpha * Gamma(alpha)
   # Verify computationally:
   import numpy as np
   from scipy.special import gamma as gamma_fn

   for alpha in [1.0, 2.5, 5.0, 10.0]:
       lhs = gamma_fn(alpha + 1)
       rhs = alpha * gamma_fn(alpha)
       print(f"Gamma({alpha+1:.1f}) = {lhs:.6f},  "
             f"{alpha:.1f} * Gamma({alpha:.1f}) = {rhs:.6f},  "
             f"match: {np.isclose(lhs, rhs)}")

Leibniz Integral Rule
----------------------

When the limits are constant:

.. math::

   \frac{d}{d\theta} \int_a^b f(x, \theta)\,dx
   = \int_a^b \frac{\partial f}{\partial \theta}\,dx.

This rule justifies **differentiating under the integral sign** --- the
technical step behind deriving score functions and Fisher information from
the likelihood.

.. code-block:: python

   # Leibniz rule: d/d(beta) of the Gamma normalizing constant
   import numpy as np
   from scipy import integrate
   from scipy.special import gamma as gamma_fn

   alpha = 3.0
   beta = 2.0

   # I(beta) = int_0^inf x^(a-1) exp(-beta*x) dx = Gamma(a) / beta^a
   def I(b):
       val, _ = integrate.quad(lambda x: x**(alpha-1) * np.exp(-b*x), 0, np.inf)
       return val

   # Numerical dI/dbeta
   eps = 1e-7
   dI_numerical = (I(beta + eps) - I(beta)) / eps

   # Analytical: dI/dbeta = -alpha * Gamma(alpha) / beta^(alpha+1)
   dI_analytical = -alpha * gamma_fn(alpha) / beta**(alpha + 1)

   print(f"dI/d(beta) numerical:  {dI_numerical:.8f}")
   print(f"dI/d(beta) analytical: {dI_analytical:.8f}")

.. admonition:: When Leibniz fails

   The Leibniz rule requires the support to be independent of :math:`\theta`.
   When the support *does* depend on the parameter --- for example, the
   Uniform(0, :math:`\theta`) distribution --- the rule fails, and the MLE
   behaves in non-standard ways (the MLE is :math:`\max(x_i)` and converges
   at rate :math:`1/n` instead of :math:`1/\sqrt{n}`).

.. code-block:: python

   # When Leibniz fails: Uniform(0, theta) MLE
   import numpy as np

   np.random.seed(42)
   theta_true = 5.0

   for n in [10, 100, 1000, 10000]:
       data = np.random.uniform(0, theta_true, size=n)
       theta_hat = data.max()  # MLE
       # Bias and rate of convergence
       error = theta_true - theta_hat
       print(f"n = {n:5d}: MLE = {theta_hat:.6f}, "
             f"error = {error:.6f}, "
             f"n * error = {n * error:.4f}")

   print("\n(n * error stays roughly constant => O(1/n) rate, not O(1/sqrt(n)))")


The Gamma Function
===================

.. math::

   \Gamma(\alpha) = \int_0^\infty t^{\alpha - 1}\,e^{-t}\,dt,
   \qquad \alpha > 0.

The Gamma function is the normalizing constant that makes the Gamma, Beta,
Chi-squared, Student's t, and F distributions integrate to one.

.. code-block:: python

   # Gamma function: recursion, factorials, and special values
   import numpy as np
   from scipy.special import gamma as gamma_fn
   from math import factorial

   # Recursion: Gamma(a+1) = a * Gamma(a)
   alpha = 3.7
   print(f"Gamma({alpha+1}) = {gamma_fn(alpha+1):.6f}")
   print(f"{alpha} * Gamma({alpha}) = {alpha * gamma_fn(alpha):.6f}")

   # Factorial connection
   print(f"\n{'n':>3} {'Gamma(n)':>12} {'(n-1)!':>12}")
   for n in range(1, 7):
       print(f"{n:3d} {gamma_fn(n):12.1f} {factorial(n-1):12d}")

   # Special value
   print(f"\nGamma(1/2) = {gamma_fn(0.5):.6f}")
   print(f"sqrt(pi)   = {np.sqrt(np.pi):.6f}")

The digamma function
---------------------

.. math::

   \psi(\alpha) = \frac{d}{d\alpha}\log\Gamma(\alpha).

The digamma function appears in the score function of every distribution
whose normalizing constant involves :math:`\Gamma(\alpha)`.

.. code-block:: python

   # Digamma: the score function of the Gamma normalizing constant
   import numpy as np
   from scipy.special import digamma, gammaln

   alpha = 2.5

   # Analytical
   psi = digamma(alpha)

   # Numerical: d/d(alpha) log Gamma(alpha)
   eps = 1e-7
   psi_num = (gammaln(alpha + eps) - gammaln(alpha)) / eps

   print(f"digamma({alpha}) = {psi:.8f}")
   print(f"numerical:        {psi_num:.8f}")

   # Application: score of Gamma(alpha, beta) w.r.t. alpha
   beta = 2.0
   np.random.seed(42)
   data = np.random.gamma(alpha, 1/beta, size=100)

   # Score: log(beta) - psi(alpha) + mean(log x)
   score_alpha = np.log(beta) - digamma(alpha) + np.mean(np.log(data))
   print(f"\nGamma score w.r.t. alpha at true value: {score_alpha:.4f} (should be ~0)")

The Beta function
------------------

.. math::

   B(\alpha, \beta)
   = \int_0^1 t^{\alpha - 1}(1 - t)^{\beta - 1}\,dt
   = \frac{\Gamma(\alpha)\,\Gamma(\beta)}{\Gamma(\alpha + \beta)}.

.. code-block:: python

   # Beta function: integration vs Gamma formula
   import numpy as np
   from scipy.special import gamma as gamma_fn, beta as beta_fn
   from scipy import integrate

   alpha, beta = 2.5, 3.0

   numerical, _ = integrate.quad(lambda t: t**(alpha-1) * (1-t)**(beta-1), 0, 1)
   via_gamma = gamma_fn(alpha) * gamma_fn(beta) / gamma_fn(alpha + beta)
   via_scipy = beta_fn(alpha, beta)

   print(f"B({alpha},{beta}) by integration: {numerical:.8f}")
   print(f"B({alpha},{beta}) via Gamma:      {via_gamma:.8f}")
   print(f"B({alpha},{beta}) via scipy:       {via_scipy:.8f}")

   # Application: verify Beta(alpha, beta) density integrates to 1
   def beta_pdf(x, a, b):
       return x**(a-1) * (1-x)**(b-1) / beta_fn(a, b)

   integral, _ = integrate.quad(beta_pdf, 0, 1, args=(alpha, beta))
   print(f"\nint Beta({alpha},{beta}) pdf = {integral:.8f} (should be 1)")


Stirling's Approximation
==========================

.. math::

   \log(n!) \approx n\log n - n + \tfrac{1}{2}\log(2\pi n).

Stirling's approximation simplifies log-likelihoods involving factorials
(multinomial, Poisson limit of Binomial) and appears in asymptotic
arguments for MLE normality.

.. code-block:: python

   # Stirling's approximation: accuracy at various n
   import numpy as np
   from scipy.special import gammaln

   print(f"{'n':>6}  {'log(n!)':>12}  {'Stirling':>12}  {'Rel Error':>12}")
   print("-" * 48)
   for n in [5, 10, 20, 50, 100, 1000]:
       exact = gammaln(n + 1)
       stirling = n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)
       rel_err = abs(stirling - exact) / abs(exact)
       print(f"{n:>6}  {exact:>12.4f}  {stirling:>12.4f}  {rel_err:>12.2e}")

   print("\n(Relative error drops below 1% for n >= 10)")

.. code-block:: python

   # Application: Stirling in the Poisson-Binomial connection
   # P(X = k) for Binomial(n, lam/n) -> Poisson(lam) as n -> inf
   import numpy as np
   from scipy.special import comb
   from scipy.stats import poisson

   lam = 3.0
   k = 4

   print(f"P(X={k}) for Binomial(n, {lam}/n) vs Poisson({lam}):")
   print(f"{'n':>8}  {'Binomial':>12}  {'Poisson':>12}  {'Abs Diff':>12}")
   print("-" * 48)
   for n in [10, 50, 100, 1000, 10000]:
       p = lam / n
       binom_prob = comb(n, k, exact=True) * p**k * (1-p)**(n-k)
       pois_prob = poisson.pmf(k, lam)
       print(f"{n:8d}  {binom_prob:12.8f}  {pois_prob:12.8f}  {abs(binom_prob-pois_prob):12.2e}")


Useful Integrals
==================

Gaussian Integral
------------------

.. math::

   \int_{-\infty}^{\infty} e^{-ax^2 + bx}\,dx
   = \sqrt{\frac{\pi}{a}}\,\exp\!\left(\frac{b^2}{4a}\right),
   \qquad a > 0.

This identity is used every time we derive the normalizing constant of a
normal distribution or "complete the square" in an exponent.

.. code-block:: python

   # Gaussian integrals: numerical verification
   import numpy as np
   from scipy import integrate

   # Basic: int exp(-x^2) dx = sqrt(pi)
   val, _ = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
   print(f"int exp(-x^2) dx = {val:.6f},  sqrt(pi) = {np.sqrt(np.pi):.6f}")

   # With parameters: int exp(-ax^2 + bx) dx
   a, b = 2.0, 3.0
   val_ab, _ = integrate.quad(lambda x: np.exp(-a*x**2 + b*x), -np.inf, np.inf)
   expected = np.sqrt(np.pi / a) * np.exp(b**2 / (4*a))
   print(f"int exp(-{a}x^2+{b}x) dx = {val_ab:.6f},  formula = {expected:.6f}")

   # Application: verify Normal density integrates to 1
   mu, sigma = 3.0, 2.0
   normal_integral, _ = integrate.quad(
       lambda x: (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2)),
       -np.inf, np.inf)
   print(f"\nN({mu},{sigma}^2) integrates to {normal_integral:.8f}")

.. code-block:: python

   # Completing the square: the trick behind conjugate Bayesian updates
   # Posterior of Normal mean with Normal prior
   import numpy as np
   from scipy import integrate

   # Prior: mu ~ N(mu_0, tau^2)
   mu_0, tau = 0.0, 5.0
   # Data: x_1, ..., x_n ~ N(mu, sigma^2), sigma known
   np.random.seed(42)
   sigma = 2.0
   data = np.random.normal(3.0, sigma, size=10)
   n = len(data)
   x_bar = data.mean()

   # Posterior by completing the square:
   # precision: 1/tau^2 + n/sigma^2
   post_prec = 1/tau**2 + n/sigma**2
   post_var = 1 / post_prec
   post_mean = post_var * (mu_0/tau**2 + n*x_bar/sigma**2)

   print(f"Prior:     N({mu_0}, {tau**2})")
   print(f"Data mean: {x_bar:.4f} (n={n})")
   print(f"Posterior: N({post_mean:.4f}, {post_var:.4f})")
   print(f"Post SD:   {np.sqrt(post_var):.4f}")

   # The posterior precision = prior precision + data precision
   print(f"\nPrior precision:    {1/tau**2:.4f}")
   print(f"Data precision:     {n/sigma**2:.4f}")
   print(f"Posterior precision: {post_prec:.4f}")

Gamma Function Integral
-------------------------

.. math::

   \int_0^\infty x^{\alpha - 1}\,e^{-\beta x}\,dx
   = \frac{\Gamma(\alpha)}{\beta^\alpha}.

This is the normalizing constant of the Gamma distribution and appears
whenever we integrate out a rate or precision parameter.

.. code-block:: python

   # Gamma integral: normalizing constant verification
   import numpy as np
   from scipy import integrate
   from scipy.special import gamma as gamma_fn

   alpha, beta = 3.5, 2.0

   numerical, _ = integrate.quad(
       lambda x: x**(alpha-1) * np.exp(-beta*x), 0, np.inf)
   analytical = gamma_fn(alpha) / beta**alpha

   print(f"Numerical:       {numerical:.8f}")
   print(f"Gamma(a)/beta^a: {analytical:.8f}")

   # Application: verify Gamma(alpha, beta) density integrates to 1
   def gamma_pdf(x, a, b):
       return b**a / gamma_fn(a) * x**(a-1) * np.exp(-b*x)

   integral, _ = integrate.quad(gamma_pdf, 0, np.inf, args=(alpha, beta))
   print(f"\nGamma({alpha},{beta}) pdf integrates to {integral:.8f}")

.. admonition:: Conjugacy connection

   In Bayesian inference, if the prior on a Poisson rate is
   Gamma(:math:`\alpha_0, \beta_0`) and we observe :math:`n` data points
   summing to :math:`s`, the posterior is
   Gamma(:math:`\alpha_0 + s, \beta_0 + n`). The Gamma integral is what
   makes this conjugate update work.

.. code-block:: python

   # Bayesian conjugacy: Poisson-Gamma in action
   import numpy as np

   np.random.seed(42)
   # Prior: lambda ~ Gamma(alpha_0, beta_0)
   alpha_0, beta_0 = 2.0, 1.0  # weak prior

   # Data: n Poisson observations
   lam_true = 4.0
   data = np.random.poisson(lam_true, size=20)
   n = len(data)
   s = data.sum()

   # Posterior: Gamma(alpha_0 + s, beta_0 + n)
   alpha_post = alpha_0 + s
   beta_post = beta_0 + n
   post_mean = alpha_post / beta_post
   post_var = alpha_post / beta_post**2

   # Frequentist MLE for comparison
   mle = data.mean()
   se_mle = np.sqrt(mle / n)

   print(f"Data: n={n}, sum={s}, mean={data.mean():.2f}")
   print(f"Prior mean:     {alpha_0/beta_0:.4f}")
   print(f"Posterior mean: {post_mean:.4f}")
   print(f"Posterior SD:   {np.sqrt(post_var):.4f}")
   print(f"MLE:            {mle:.4f}")
   print(f"MLE SE:         {se_mle:.4f}")
   print(f"True lambda:    {lam_true}")

Moments of the Gaussian
-------------------------

For :math:`X \sim \mathcal{N}(\mu, \sigma^2)`:

.. math::

   E[X^2] &= \mu^2 + \sigma^2, \\
   E[X^4] &= \mu^4 + 6\mu^2\sigma^2 + 3\sigma^4.

These moments appear when computing the Fisher information for models
involving squared observations.

.. code-block:: python

   # Gaussian moments: simulation vs formula
   import numpy as np

   np.random.seed(42)
   mu, sigma = 2.0, 1.5
   X = np.random.normal(mu, sigma, size=1_000_000)

   print(f"{'Moment':>8} {'Simulated':>12} {'Formula':>12}")
   print("-" * 36)
   print(f"{'E[X^2]':>8} {np.mean(X**2):12.4f} {mu**2 + sigma**2:12.4f}")
   print(f"{'E[X^4]':>8} {np.mean(X**4):12.2f} "
         f"{mu**4 + 6*mu**2*sigma**2 + 3*sigma**4:12.2f}")

The Chi-Squared Integral
--------------------------

If :math:`X_1, \ldots, X_n \sim \mathcal{N}(0,1)` i.i.d., then
:math:`Q = \sum X_i^2 \sim \chi^2_n`.

.. math::

   f_Q(q) = \frac{1}{2^{n/2}\,\Gamma(n/2)}\,q^{n/2 - 1}\,e^{-q/2}.

This connects to likelihood ratio tests via Wilks' theorem:
:math:`-2\log\Lambda \xrightarrow{d} \chi^2_r`.

.. code-block:: python

   # Chi-squared: simulation and connection to likelihood ratio tests
   import numpy as np
   from scipy.stats import chi2

   np.random.seed(42)
   n = 5
   n_sim = 100_000

   # Simulate: sum of squared standard normals
   Z = np.random.randn(n_sim, n)
   Q = np.sum(Z**2, axis=1)

   print(f"Chi-squared({n}) distribution:")
   print(f"  Mean  -- simulated: {Q.mean():.3f},  theory: {n}")
   print(f"  Var   -- simulated: {Q.var():.3f},  theory: {2*n}")

   # Connection to Wilks' theorem:
   # Under H0, -2 log(L0/L1) ~ chi2(r) where r = # restrictions
   # This is why the chi2 critical values matter for hypothesis testing
   print(f"\nChi2 critical values (for likelihood ratio tests):")
   for df in [1, 2, 5, 10]:
       for alpha in [0.05, 0.01]:
           cv = chi2.ppf(1 - alpha, df=df)
           print(f"  df={df}, alpha={alpha}: critical value = {cv:.4f}")
