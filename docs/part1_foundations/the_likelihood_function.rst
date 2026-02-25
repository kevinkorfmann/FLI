.. _ch4_likelihood:

====================================
Chapter 4: The Likelihood Function
====================================

Everything in Part I has been building to this chapter.  We have learned how
to describe uncertainty with probability (:ref:`ch1_probability`), how to work
with random variables and their properties (:ref:`ch2_random_variables`), and
how to recognize the most common probability distributions
(:ref:`ch3_distributions`).  Now we turn the entire framework around: instead
of reasoning *from* a known distribution *to* data, we reason *from* observed
data *back to* the distribution that most plausibly generated it.

The **likelihood function** is the central object that makes this reversal
possible.  It is the foundation of maximum likelihood estimation, likelihood
ratio tests, the EM algorithm, and much of modern statistical inference.

Throughout this chapter we will follow a single running scenario: **you are a
quality engineer testing light bulbs**.  Your company manufactures LED bulbs
and claims an average lifetime of 1000 hours.  You pull bulbs off the line,
run them until failure, and record the lifetimes.  The question is: *which
Exponential rate parameter* :math:`\lambda` *best explains the data you see?*

.. contents:: Chapter Contents
   :local:
   :depth: 2


4.1 Likelihood vs. Probability: The Key Distinction
=====================================================

Before writing any formulas, we must be completely clear about a conceptual
distinction that trips up many newcomers.

**Probability** answers the question: *Given a fully specified model (known
parameters), how likely is a particular dataset?*  We fix the parameters and
vary the data.

**Likelihood** answers the *reverse* question: *Given observed data, how
plausible is each possible parameter value?*  We fix the data and vary the
parameters.

The same mathematical expression appears in both cases --- only the
*interpretation* changes.

.. admonition:: Intuition

   You test one light bulb and it lasts 1050 hours.  Probability asks: "If
   :math:`\lambda = 0.001`, what is the probability density at 1050?"
   Likelihood asks: "Given that we *observed* 1050 hours, which value of
   :math:`\lambda` makes this observation most plausible?"  Same formula,
   different question.

Let us make this concrete with our light bulb.  The Exponential PDF is
:math:`f(x \mid \lambda) = \lambda e^{-\lambda x}`.  A single bulb lasted
:math:`x = 1050` hours.

.. code-block:: python

   # Likelihood vs. probability: same formula, different perspective
   import numpy as np

   x_obs = 1050  # observed lifetime (hours)

   # --- Probability view: fix lambda, ask about different x values ---
   lam_fixed = 0.001
   print("PROBABILITY VIEW: fix lambda=0.001, vary x")
   for x in [500, 750, 1000, 1050, 1500, 2000]:
       f_x = lam_fixed * np.exp(-lam_fixed * x)
       print(f"  f(x={x:4d} | lambda={lam_fixed}) = {f_x:.6f}")

   # --- Likelihood view: fix x=1050, ask about different lambda values ---
   print(f"\nLIKELIHOOD VIEW: fix x={x_obs}, vary lambda")
   for lam in [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002]:
       L = lam * np.exp(-lam * x_obs)
       print(f"  L(lambda={lam:.4f} | x={x_obs}) = {L:.6f}")

**A crucial point:**  The likelihood function is *not* a probability
distribution over :math:`\lambda`.  It does not integrate to 1 over the
parameter space.  It is a *relative* measure of plausibility: we compare
likelihood values at different :math:`\lambda` to decide which parameter values
are better supported by the data.

.. code-block:: python

   # The likelihood does NOT integrate to 1 over the parameter space
   import numpy as np

   x_obs = 1050
   lam_grid = np.linspace(0.0001, 0.005, 10_000)
   L_values = lam_grid * np.exp(-lam_grid * x_obs)

   # Numerical integral over lambda (trapezoidal rule)
   integral = np.trapz(L_values, lam_grid)
   print(f"Integral of L(lambda | x={x_obs}) over lambda in [0.0001, 0.005]:")
   print(f"  = {integral:.6f}")
   print(f"  This is NOT 1.  The likelihood is not a probability distribution.")


4.2 Formal Definition of the Likelihood Function
==================================================

Let :math:`x = (x_1, x_2, \dots, x_n)` be observed data, and let
:math:`f(x \mid \theta)` denote either:

- the joint PMF (if the data are discrete), or
- the joint PDF (if the data are continuous),

where :math:`\theta` is the parameter (or vector of parameters) of the model.

**Definition.**  The **likelihood function** is

.. math::

   L(\theta \mid x) = f(x \mid \theta),

viewed as a function of :math:`\theta` for fixed (observed) :math:`x`.

Think of it this way: the same formula :math:`f(x \mid \theta)` wears two
hats.  When you vary :math:`x` with :math:`\theta` fixed, it is a probability
distribution.  When you vary :math:`\theta` with :math:`x` fixed, it is a
likelihood function.  Notation like :math:`L(\theta \mid x)` reminds you which
hat it is wearing.

When the observations :math:`x_1, \dots, x_n` are **independent and
identically distributed (i.i.d.)**, the joint density factors:

.. math::

   L(\theta \mid x) = \prod_{i=1}^{n} f(x_i \mid \theta).

This product form is what makes the independence assumption so powerful ---
without it, writing down the likelihood would require knowledge of all
dependence structures.

Now let us build the likelihood for our light bulb data.  Suppose we test
:math:`n = 5` bulbs and observe lifetimes :math:`x = (1020, 980, 1100, 950, 1050)`.
Each bulb's lifetime is modeled as :math:`X_i \sim \text{Exp}(\lambda)`.

.. code-block:: python

   # Building the likelihood step by step for Exponential data
   import numpy as np

   # Our observed light bulb lifetimes (hours)
   data = np.array([1020, 980, 1100, 950, 1050])
   n = len(data)

   # For a candidate lambda, the likelihood is:
   # L(lambda | x) = prod_{i=1}^{n} lambda * exp(-lambda * x_i)
   #               = lambda^n * exp(-lambda * sum(x_i))

   lam_candidate = 0.001  # candidate value

   # Step-by-step computation
   print(f"Data: {data}")
   print(f"n = {n}, sum(x) = {data.sum()}, x_bar = {data.mean():.1f}")
   print(f"\nStep-by-step for lambda = {lam_candidate}:")
   L_product = 1.0
   for i, x_i in enumerate(data):
       f_i = lam_candidate * np.exp(-lam_candidate * x_i)
       L_product *= f_i
       print(f"  f(x_{i+1}={x_i}) = {lam_candidate} * exp(-{lam_candidate}*{x_i})"
             f" = {f_i:.8f}")
   print(f"  L(lambda) = product = {L_product:.4e}")

   # Compact formula: lambda^n * exp(-lambda * sum(x))
   L_compact = lam_candidate**n * np.exp(-lam_candidate * data.sum())
   print(f"  Compact:  lambda^n * exp(-lambda*sum(x)) = {L_compact:.4e}")
   print(f"  Match: {np.isclose(L_product, L_compact)}")


4.2.1 How the Likelihood Changes as Data Accumulate
-----------------------------------------------------

One of the most illuminating things you can do is watch the likelihood function
*sharpen* as you collect more data.  With one bulb, many values of
:math:`\lambda` are plausible.  With 20 bulbs, the likelihood becomes a narrow
spike around the truth.

.. code-block:: python

   # Likelihood sharpening: n=1, n=5, n=20 light bulbs
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001  # true rate (mean lifetime = 1000 hrs)
   all_data = np.random.exponential(scale=1/lambda_true, size=20)

   lam_grid = np.linspace(0.0003, 0.003, 300)

   print("How the log-likelihood sharpens with more data:")
   print("(Each row: lambda value -> log-likelihood for n=1, 5, 20)\n")

   for n_obs in [1, 5, 20]:
       x = all_data[:n_obs]
       x_sum = x.sum()

       # Log-likelihood: n*log(lambda) - lambda*sum(x)
       logL = n_obs * np.log(lam_grid) - lam_grid * x_sum

       # Normalize so max = 0 for comparison
       logL_norm = logL - logL.max()

       # Find MLE
       lam_mle = n_obs / x_sum  # MLE for exponential rate

       print(f"  n = {n_obs:2d}: MLE = {lam_mle:.6f}, data mean = {x.mean():.1f} hrs")

       # Print a table showing the shape
       print(f"    {'lambda':>10s}  {'logL - max':>12s}  {'shape':>30s}")
       for lam_val in [0.0005, 0.0007, 0.0009, 0.001, 0.0012, 0.0015, 0.002]:
           ll = n_obs * np.log(lam_val) - lam_val * x_sum
           ll_n = ll - logL.max()
           bar_len = max(0, int(30 + ll_n))  # rough visual
           bar = '#' * min(bar_len, 30)
           print(f"    {lam_val:10.4f}  {ll_n:12.2f}  {bar}")
       print()


4.3 The Log-Likelihood
========================

Working with products of many small numbers is numerically and algebraically
unpleasant.  The **log-likelihood** fixes both problems.

**Definition.**

.. math::

   \ell(\theta \mid x) = \ln L(\theta \mid x).

**Why logarithms?**

1. **Monotonicity.**  The natural logarithm is a strictly increasing function.
   Therefore maximizing :math:`\ell(\theta)` is equivalent to maximizing
   :math:`L(\theta)` --- the location of the maximum does not change.

2. **Products become sums.**  For i.i.d. data,

   .. math::

      \ell(\theta \mid x)
      = \ln \prod_{i=1}^{n} f(x_i \mid \theta)
      = \sum_{i=1}^{n} \ln f(x_i \mid \theta).

   Sums are far easier to differentiate, manipulate, and compute.

3. **Numerical stability.**  Products of many probabilities can underflow to
   zero in floating-point arithmetic.  Summing log-probabilities avoids this.

4. **Exponential families.**  For exponential-family distributions (which
   include most distributions in :ref:`ch3_distributions`), the log-likelihood
   has an especially clean form.

Let us see the numerical stability issue in action and derive the
log-likelihood for our Exponential bulb model.

For :math:`X_i \sim \text{Exp}(\lambda)`:

.. math::

   \ell(\lambda \mid x) = \sum_{i=1}^n \ln(\lambda e^{-\lambda x_i})
   = n \ln \lambda - \lambda \sum_{i=1}^n x_i.

.. code-block:: python

   # Log-likelihood for Exponential: algebraically clean and numerically stable
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001
   data = np.random.exponential(scale=1/lambda_true, size=100)
   n = len(data)
   x_sum = data.sum()

   # Direct likelihood: product of 100 small densities
   lam_test = 0.001
   log_densities = np.log(lam_test) - lam_test * data
   densities = np.exp(log_densities)
   L_direct = np.prod(densities)

   # Log-likelihood: sum of log-densities
   logL = np.sum(log_densities)

   # Compact formula: n*ln(lambda) - lambda*sum(x)
   logL_compact = n * np.log(lam_test) - lam_test * x_sum

   print(f"n = {n} light bulbs")
   print(f"Direct L (product of {n} densities): {L_direct}")
   print(f"Log-likelihood (sum):                {logL:.4f}")
   print(f"Compact formula:                     {logL_compact:.4f}")
   print(f"\nThe direct product underflowed to {L_direct}, but the")
   print(f"log-likelihood {logL:.4f} is a perfectly usable number.")

.. admonition:: Common Pitfall

   When implementing likelihood computations, *always* work with the
   log-likelihood.  Even for moderate sample sizes (say, :math:`n = 100`),
   the product :math:`\prod f(x_i \mid \theta)` can be astronomically small
   (e.g., :math:`10^{-200}`), causing floating-point underflow.  The
   log-likelihood stays in a numerically reasonable range.


4.3.1 Plotting the Log-Likelihood for Light Bulbs
---------------------------------------------------

Let us compute the log-likelihood across a grid of :math:`\lambda` values and
find the maximum.  This is the most direct way to "see" what the data say about
the parameter.

.. code-block:: python

   # Log-likelihood curve for the light bulb data
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001
   data = np.random.exponential(scale=1/lambda_true, size=20)
   n = len(data)
   x_sum = data.sum()
   x_bar = data.mean()

   # Grid of candidate lambda values
   lam_grid = np.linspace(0.0003, 0.003, 300)

   # Log-likelihood: ell(lambda) = n * ln(lambda) - lambda * sum(x)
   logL = n * np.log(lam_grid) - lam_grid * x_sum

   # MLE: set derivative to zero -> lambda_hat = n / sum(x) = 1 / x_bar
   lambda_hat = n / x_sum  # equivalently, 1 / x_bar

   # Log-likelihood at MLE
   logL_at_mle = n * np.log(lambda_hat) - lambda_hat * x_sum

   print(f"Data: n={n}, sum(x)={x_sum:.1f}, x_bar={x_bar:.1f}")
   print(f"MLE: lambda_hat = n/sum(x) = 1/x_bar = {lambda_hat:.6f}")
   print(f"     (true lambda = {lambda_true})")
   print(f"\nLog-likelihood table:")
   print(f"  {'lambda':>10s}  {'ell(lambda)':>14s}  {'ell - ell_max':>14s}")
   for lam in np.linspace(0.0005, 0.002, 16):
       ll = n * np.log(lam) - lam * x_sum
       print(f"  {lam:10.6f}  {ll:14.2f}  {ll - logL_at_mle:14.2f}")

   print(f"\n  The maximum is at lambda_hat = {lambda_hat:.6f}")
   print(f"  ell(lambda_hat) = {logL_at_mle:.2f}")


4.4 Finding the MLE Analytically
===================================

The **maximum likelihood estimate (MLE)** is the parameter value that maximizes
the likelihood (or equivalently, the log-likelihood).  For our Exponential
model, we can find it by calculus.

The log-likelihood is

.. math::

   \ell(\lambda) = n \ln \lambda - \lambda \sum_{i=1}^n x_i.

Setting the derivative to zero:

.. math::

   \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
   \quad \Longrightarrow \quad
   \hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}.

Let us verify this is a maximum (not a minimum) by checking the second
derivative:

.. math::

   \frac{d^2\ell}{d\lambda^2} = -\frac{n}{\lambda^2} < 0 \quad \text{for all } \lambda > 0.

The log-likelihood is concave, so the critical point is indeed the global
maximum.

.. code-block:: python

   # MLE for Exponential: analytical solution and numerical verification
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   lambda_true = 0.001
   data = np.random.exponential(scale=1/lambda_true, size=20)
   n = len(data)
   x_sum = data.sum()
   x_bar = data.mean()

   # Analytical MLE
   lambda_hat = 1 / x_bar  # = n / x_sum

   # Numerical verification: minimize negative log-likelihood
   def neg_loglik(lam):
       if lam <= 0:
           return np.inf
       return -(n * np.log(lam) - lam * x_sum)

   result = minimize_scalar(neg_loglik, bounds=(1e-6, 0.01), method='bounded')
   lambda_hat_numerical = result.x

   print(f"Data: n={n}, x_bar={x_bar:.2f}")
   print(f"  Analytical MLE:  lambda_hat = 1/x_bar = {lambda_hat:.6f}")
   print(f"  Numerical MLE:   lambda_hat = {lambda_hat_numerical:.6f}")
   print(f"  True lambda:     {lambda_true}")
   print(f"  Match: {np.isclose(lambda_hat, lambda_hat_numerical, rtol=1e-4)}")

   # Verify: score = 0 at MLE
   score_at_mle = n / lambda_hat - x_sum
   print(f"\n  Score at MLE: d(ell)/d(lambda) = n/lambda_hat - sum(x)")
   print(f"    = {n}/{lambda_hat:.6f} - {x_sum:.1f}")
   print(f"    = {score_at_mle:.10f}  (should be 0)")

   # Verify: second derivative is negative (concave)
   d2ell = -n / lambda_hat**2
   print(f"  Second derivative: -n/lambda^2 = {d2ell:.2f}  (< 0, so it is a max)")


4.5 The Score Function
========================

The **score function** (or simply the "score") is the gradient of the
log-likelihood with respect to the parameter.  It tells you the direction
and steepness of the log-likelihood surface at any parameter value.

**Definition.**  For a scalar parameter :math:`\theta`,

.. math::

   S(\theta) = \frac{\partial}{\partial\theta}\,\ell(\theta \mid x)
             = \frac{\partial}{\partial\theta}\,\ln f(x \mid \theta).

Equivalently, using the chain rule,

.. math::

   S(\theta) = \frac{1}{L(\theta)}\,\frac{\partial L(\theta)}{\partial\theta}
             = \frac{\partial \ln L(\theta)}{\partial\theta}.

For a parameter vector :math:`\boldsymbol{\theta} = (\theta_1, \dots, \theta_k)^T`,
the score is a vector:

.. math::

   \mathbf{S}(\boldsymbol{\theta})
   = \nabla_{\boldsymbol{\theta}}\,\ell(\boldsymbol{\theta} \mid x)
   = \left(\frac{\partial \ell}{\partial \theta_1}, \dots,
           \frac{\partial \ell}{\partial \theta_k}\right)^T.

At the maximum likelihood estimate :math:`\hat{\theta}`, the score equals
zero: :math:`S(\hat{\theta}) = 0`.  This is the **score equation** (or
likelihood equation), the starting point for finding MLEs.

For our Exponential light bulb model, the score for a single observation is:

.. math::

   S(\lambda) = \frac{\partial}{\partial\lambda}\left[\ln\lambda - \lambda x\right]
              = \frac{1}{\lambda} - x.

For :math:`n` observations:

.. math::

   S_n(\lambda) = \frac{n}{\lambda} - \sum_{i=1}^n x_i.

.. code-block:: python

   # Score function for Exponential: verify S(lambda_hat) = 0
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001
   data = np.random.exponential(scale=1/lambda_true, size=20)
   n = len(data)
   x_sum = data.sum()
   lambda_hat = n / x_sum

   # Score: S(lambda) = n/lambda - sum(x)
   print("Score function: S(lambda) = n/lambda - sum(x)")
   print(f"  Data: n={n}, sum(x)={x_sum:.2f}\n")
   for lam in [0.0005, 0.0008, lambda_hat, 0.0012, 0.0015]:
       score = n / lam - x_sum
       label = " <-- MLE" if np.isclose(lam, lambda_hat, rtol=1e-6) else ""
       print(f"  S({lam:.6f}) = {score:10.2f}{label}")

   print(f"\n  Score is positive below the MLE (go up) and negative above (go down).")
   print(f"  At the MLE, the score is exactly zero: the log-likelihood is flat.")


4.5.1 The Score Has Zero Expectation
--------------------------------------

A fundamental property:

**Theorem.**  Under regularity conditions,

.. math::

   E_\theta[S(\theta)] = 0.

In words: when :math:`\theta` is the *true* parameter, the expected score is
zero.  The intuition is that if you are at the true parameter, the
log-likelihood is (on average) at its peak.  The average slope at a peak is
zero.

**Proof (continuous case).**  By definition,

.. math::

   E_\theta[S(\theta)]
   = \int S(\theta)\,f(x \mid \theta)\,dx
   = \int \frac{\partial}{\partial\theta}\ln f(x \mid \theta)\;f(x \mid \theta)\,dx.

Using :math:`\frac{\partial}{\partial\theta}\ln f = \frac{1}{f}\frac{\partial f}{\partial\theta}`:

.. math::

   = \int \frac{1}{f(x \mid \theta)}\,\frac{\partial f(x \mid \theta)}{\partial\theta}\;
     f(x \mid \theta)\,dx
   = \int \frac{\partial f(x \mid \theta)}{\partial\theta}\,dx.

Now, assuming we can interchange differentiation and integration (the key
regularity condition):

.. math::

   = \frac{\partial}{\partial\theta}\int f(x \mid \theta)\,dx
   = \frac{\partial}{\partial\theta}\,1 = 0. \quad \square

The regularity conditions are: (i) the support of :math:`f` does not depend on
:math:`\theta`, and (ii) the derivative and integral can be interchanged
(satisfied under mild smoothness conditions).

.. code-block:: python

   # Verify E[S(lambda)] = 0 at the true lambda, by simulation
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001

   # For Exp(lambda), score for one observation: S = 1/lambda - x
   # At the true lambda, E[S] = 1/lambda - E[X] = 1/lambda - 1/lambda = 0
   n_sim = 200_000
   x_sim = np.random.exponential(scale=1/lambda_true, size=n_sim)
   scores = 1/lambda_true - x_sim

   print(f"E[S(lambda)] at lambda_true = {lambda_true}")
   print(f"  Analytical: 1/lambda - E[X] = 1/{lambda_true} - {1/lambda_true} = 0")
   print(f"  Simulated:  mean of S = {scores.mean():.4f}  (should be ~0)")
   print(f"  Var(S):     {scores.var():.2f}")
   print(f"  Theory Var: 1/lambda^2 = {1/lambda_true**2:.2f}")


4.6 Fisher Information
========================

The score function measures the *slope* of the log-likelihood.  **Fisher
information** measures how much *curvature* the log-likelihood has --- that is,
how sharply peaked it is around the true parameter.  More curvature means the
data are more informative about :math:`\theta`.

Imagine you are the quality engineer.  You have tested 10 light bulbs.  The
log-likelihood has some width around its peak.  Now imagine you test 100 bulbs.
The log-likelihood becomes much narrower --- you know :math:`\lambda` much more
precisely.  Fisher information quantifies exactly this sharpening.

**Definition.**

.. math::

   I(\theta) = E_\theta\!\left[S(\theta)^2\right]
             = E_\theta\!\left[\left(\frac{\partial \ell}{\partial\theta}\right)^2\right].

Since :math:`E[S(\theta)] = 0`, this is also the variance of the score:

.. math::

   I(\theta) = \operatorname{Var}_\theta\!\left[S(\theta)\right].


4.6.1 Alternative Formula via Second Derivative
-------------------------------------------------

Under the same regularity conditions that gave us :math:`E[S] = 0`, there is
an equivalent and often more convenient formula:

**Theorem.**

.. math::

   I(\theta) = -E_\theta\!\left[\frac{\partial^2 \ell}{\partial\theta^2}\right].

This tells us that Fisher information equals the expected negative curvature
of the log-likelihood.  In practice, this formula is often easier to compute
because taking two derivatives is straightforward algebra.

**Proof.**  Differentiate :math:`E[S(\theta)] = 0` with respect to
:math:`\theta`.  We have

.. math::

   0 = \frac{\partial}{\partial\theta}\int \frac{\partial \ln f}{\partial\theta}\,
   f(x \mid \theta)\,dx.

Applying the product rule under the integral (again using the interchange
condition):

.. math::

   0 = \int \left[\frac{\partial^2 \ln f}{\partial\theta^2}\,f
       + \frac{\partial \ln f}{\partial\theta}\,
         \frac{\partial f}{\partial\theta}\right]dx.

Note :math:`\frac{\partial f}{\partial\theta} = f\,\frac{\partial \ln f}{\partial\theta}`,
so the second term is

.. math::

   \int \left(\frac{\partial \ln f}{\partial\theta}\right)^2 f\,dx
   = E\!\left[\left(\frac{\partial \ell}{\partial\theta}\right)^2\right]
   = I(\theta).

Therefore

.. math::

   0 = E\!\left[\frac{\partial^2 \ell}{\partial\theta^2}\right] + I(\theta),

which gives

.. math::

   I(\theta) = -E\!\left[\frac{\partial^2 \ell}{\partial\theta^2}\right]. \quad \square


4.6.2 Fisher Information for the Exponential (Light Bulb Model)
------------------------------------------------------------------

Let us compute the Fisher information for our :math:`\text{Exp}(\lambda)` model
step by step.

For a single observation :math:`X \sim \text{Exp}(\lambda)`:

.. math::

   \ell_1(\lambda) = \ln \lambda - \lambda x.

Score:

.. math::

   S(\lambda) = \frac{1}{\lambda} - x.

Second derivative:

.. math::

   \frac{\partial^2 \ell_1}{\partial \lambda^2} = -\frac{1}{\lambda^2}.

This does not depend on :math:`x`, so the expectation is trivial:

.. math::

   I_1(\lambda) = -E\!\left[-\frac{1}{\lambda^2}\right] = \frac{1}{\lambda^2}.

For :math:`n` i.i.d. observations:

.. math::

   I_n(\lambda) = \frac{n}{\lambda^2}.

.. code-block:: python

   # Fisher information for Exp(lambda): I_n = n / lambda^2
   import numpy as np

   lambda_true = 0.001

   print("Fisher information for Exp(lambda): I_n(lambda) = n / lambda^2")
   print(f"  lambda = {lambda_true}\n")
   for n in [10, 20, 50, 100, 500]:
       I_n = n / lambda_true**2
       se  = lambda_true / np.sqrt(n)  # SE = lambda / sqrt(n)
       ci_half = 1.96 * se
       print(f"  n = {n:3d}:  I_n = {I_n:.0e},  "
             f"SE(lambda_hat) = {se:.6f},  "
             f"95% CI width = {2*ci_half:.6f}")

   print(f"\n  10x more data -> sqrt(10)x more precise.")
   print(f"  SE(n=100) / SE(n=10) = {(lambda_true/np.sqrt(100)) / (lambda_true/np.sqrt(10)):.4f}"
         f"  (= 1/sqrt(10) = {1/np.sqrt(10):.4f})")


4.6.3 Fisher Information as Measurement Precision
---------------------------------------------------

How precisely can you estimate :math:`\lambda`?  If you test 10 bulbs vs 100
bulbs, how much better off are you?  Fisher information answers this directly.

The **Cramer-Rao lower bound** states that for any unbiased estimator
:math:`\hat{\theta}`,

.. math::

   \operatorname{Var}(\hat{\theta}) \geq \frac{1}{I_n(\theta)}.

For the MLE of the Exponential rate, the asymptotic variance is exactly
:math:`1/I_n(\lambda) = \lambda^2/n`, giving a standard error of
:math:`\lambda/\sqrt{n}`.

.. code-block:: python

   # Fisher information as precision: 10 bulbs vs 100 bulbs
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001

   print("Testing 10 vs 100 light bulbs: how much precision do you gain?\n")
   for n in [10, 100]:
       # Simulate many experiments of size n
       n_experiments = 50_000
       mle_estimates = np.array([
           1.0 / np.random.exponential(1/lambda_true, size=n).mean()
           for _ in range(n_experiments)
       ])

       I_n = n / lambda_true**2
       se_theory = lambda_true / np.sqrt(n)

       print(f"  n = {n}:")
       print(f"    Fisher info I_n = {I_n:.2e}")
       print(f"    SE (theory):  {se_theory:.6f}")
       print(f"    SE (simulated): {mle_estimates.std():.6f}")
       print(f"    95% CI width: +/- {1.96 * se_theory:.6f}")
       print(f"    Simulated MLE mean: {mle_estimates.mean():.6f} (true: {lambda_true})")
       print()

   print(f"  Going from 10 to 100 bulbs makes your estimate sqrt(10) = "
         f"{np.sqrt(10):.2f}x more precise.")


4.6.4 Fisher Information for i.i.d. Data
------------------------------------------

If :math:`x_1, \dots, x_n` are i.i.d. with common density :math:`f(x \mid \theta)`,
the total log-likelihood is :math:`\ell(\theta) = \sum_{i=1}^n \ln f(x_i \mid \theta)`.
The Fisher information for the full sample is

.. math::

   I_n(\theta) = n\,I_1(\theta),

where :math:`I_1(\theta)` is the Fisher information from a single observation.
Information *adds* across independent observations --- this is a key property.

.. admonition:: Intuition

   Every independent observation contributes the same amount of information
   about :math:`\theta`.  Ten observations are ten times as informative as
   one.  This is the formal justification for the intuitive idea that
   "more data = more precision."


4.6.5 Multiparameter Fisher Information
------------------------------------------

When :math:`\boldsymbol{\theta} = (\theta_1, \dots, \theta_k)^T`, the Fisher
information is a :math:`k \times k` matrix:

.. math::

   [\mathcal{I}(\boldsymbol{\theta})]_{jl}
   = E\!\left[\frac{\partial \ell}{\partial\theta_j}\,
               \frac{\partial \ell}{\partial\theta_l}\right]
   = -E\!\left[\frac{\partial^2 \ell}{\partial\theta_j\,\partial\theta_l}\right].

This **Fisher information matrix** generalizes the scalar case and plays a
central role in the asymptotic theory of maximum likelihood estimation (the
Cramer--Rao bound, asymptotic normality of the MLE, etc.).


4.6.6 Example: Fisher Information for the Normal
---------------------------------------------------

Let :math:`X_1, \dots, X_n` be i.i.d. :math:`N(\mu, \sigma^2)` with
:math:`\boldsymbol{\theta} = (\mu, \sigma^2)^T`.

The log-likelihood for a single observation is

.. math::

   \ell_1(\mu, \sigma^2) = -\frac{1}{2}\ln(2\pi) - \frac{1}{2}\ln\sigma^2
   - \frac{(x-\mu)^2}{2\sigma^2}.

Computing the second partial derivatives and taking expectations yields the
Fisher information matrix for one observation:

.. math::

   \mathcal{I}_1(\mu, \sigma^2) = \begin{pmatrix}
   \dfrac{1}{\sigma^2} & 0 \\[6pt]
   0 & \dfrac{1}{2\sigma^4}
   \end{pmatrix}.

The off-diagonal zeros mean :math:`\mu` and :math:`\sigma^2` are
*information-orthogonal*: data provide independent information about the mean
and the variance.

For :math:`n` observations:

.. math::

   \mathcal{I}_n(\mu, \sigma^2) = n\,\mathcal{I}_1 = \begin{pmatrix}
   \dfrac{n}{\sigma^2} & 0 \\[6pt]
   0 & \dfrac{n}{2\sigma^4}
   \end{pmatrix}.

.. code-block:: python

   # Fisher information matrix for Normal(mu, sigma^2)
   import numpy as np

   sigma = 12.0   # heart rate std dev (from Ch 3 hospital scenario)
   sigma2 = sigma**2
   n = 25

   # Fisher info matrix for one observation
   I1 = np.array([[1/sigma2, 0],
                   [0, 1/(2*sigma2**2)]])
   In = n * I1

   print(f"Normal(mu, sigma^2={sigma2}), n={n}")
   print(f"\nFisher info matrix (one observation):")
   print(f"  I_1 = [[{I1[0,0]:.6f}, {I1[0,1]:.6f}],")
   print(f"         [{I1[1,0]:.6f}, {I1[1,1]:.8f}]]")
   print(f"\nFisher info matrix ({n} observations):")
   print(f"  I_n = [[{In[0,0]:.4f}, {In[0,1]:.4f}],")
   print(f"         [{In[1,0]:.4f}, {In[1,1]:.6f}]]")
   print(f"\nStandard errors:")
   print(f"  SE(mu_hat)     = sigma/sqrt(n) = {sigma/np.sqrt(n):.4f}")
   print(f"  SE(sigma2_hat) = sigma^2*sqrt(2/n) = {sigma2*np.sqrt(2/n):.4f}")


4.6.7 Example: Fisher Information for the Bernoulli
-----------------------------------------------------

Let :math:`X \sim \text{Bernoulli}(p)`.  The log-likelihood for a single
observation is

.. math::

   \ell(p) = x \ln p + (1-x)\ln(1-p).

Score:

.. math::

   S(p) = \frac{x}{p} - \frac{1-x}{1-p}.

Check: :math:`E[S(p)] = \frac{p}{p} - \frac{1-p}{1-p} = 1 - 1 = 0`. Good.

Second derivative:

.. math::

   \frac{\partial^2 \ell}{\partial p^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}.

Expected value:

.. math::

   -E\!\left[\frac{\partial^2 \ell}{\partial p^2}\right]
   = \frac{E[X]}{p^2} + \frac{1-E[X]}{(1-p)^2}
   = \frac{p}{p^2} + \frac{1-p}{(1-p)^2}
   = \frac{1}{p} + \frac{1}{1-p}
   = \frac{1}{p(1-p)}.

Therefore

.. math::

   I(p) = \frac{1}{p(1-p)}.

Notice that the information is largest when :math:`p` is near 0 or 1 (where
the distribution is most "decisive") and smallest at :math:`p = 0.5` (maximum
uncertainty).

For :math:`n` i.i.d. Bernoulli observations:

.. math::

   I_n(p) = \frac{n}{p(1-p)}.

.. code-block:: python

   # Fisher information for Bernoulli: I(p) = 1/[p(1-p)]
   import numpy as np

   print("Fisher information I(p) = 1 / [p(1-p)] for Bernoulli:")
   print(f"{'p':>6s}  {'I(p)':>8s}  {'SE(p_hat,n=100)':>16s}  {'visual':>20s}")
   for p in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
       I_p = 1 / (p * (1 - p))
       se  = np.sqrt(p * (1-p) / 100)  # SE for n=100
       bar = '#' * int(I_p / 2)
       print(f"{p:6.2f}  {I_p:8.2f}  {se:16.4f}  {bar}")

   # Verify by simulation
   np.random.seed(42)
   p_true = 0.3
   n = 100
   scores = []
   for _ in range(100_000):
       x = np.random.binomial(1, p_true)
       s = x / p_true - (1 - x) / (1 - p_true)
       scores.append(s)
   scores = np.array(scores)

   print(f"\nSimulation check at p={p_true}, n=1:")
   print(f"  E[S(p)]  = {scores.mean():.4f}  (should be 0)")
   print(f"  Var[S(p)] = {scores.var():.4f}  (should be I(p) = {1/(p_true*(1-p_true)):.4f})")


4.7 Sufficient Statistics
===========================

In many problems, the entire dataset :math:`x = (x_1, \dots, x_n)` can be
compressed into a smaller summary without any loss of information about
:math:`\theta`.  These summaries are called **sufficient statistics**.

Why does this matter for our quality engineer?  If you have tested 1000 light
bulbs, you do not need to store all 1000 individual lifetimes.  For the
Exponential model, all the information about :math:`\lambda` is contained in
just one number: the sum of lifetimes :math:`T = \sum x_i`.  You can throw
away the individual data and do inference just as well.

**Definition.**  A statistic :math:`T = T(x)` is **sufficient** for
:math:`\theta` if the conditional distribution of the data given :math:`T`
does not depend on :math:`\theta`:

.. math::

   f(x \mid T(x) = t,\;\theta) = f(x \mid T(x) = t) \quad \text{for all } \theta.

*Plain English:*  Once you know the value of :math:`T`, the remaining "shape"
of the data carries no additional information about :math:`\theta`.

.. code-block:: python

   # Sufficient statistic for Exponential: T = sum(x)
   import numpy as np

   np.random.seed(42)
   lambda_true = 0.001
   data = np.random.exponential(scale=1/lambda_true, size=20)
   n = len(data)
   T = data.sum()  # sufficient statistic

   print(f"20 light bulb lifetimes:")
   print(f"  data = [{', '.join(f'{x:.0f}' for x in data[:5])}, ..., "
         f"{', '.join(f'{x:.0f}' for x in data[-3:])}]")
   print(f"  T = sum(x_i) = {T:.1f}")
   print(f"  x_bar = T/n = {T/n:.1f}")
   print(f"\nThe likelihood depends on data ONLY through T:")
   print(f"  L(lambda | x) = lambda^n * exp(-lambda * T)")
   print(f"               = lambda^{n} * exp(-lambda * {T:.1f})")
   print(f"\n  lambda   L(lambda | T)")
   for lam in [0.0005, 0.0008, 0.001, 0.0012, 0.0015]:
       L = lam**n * np.exp(-lam * T)
       print(f"  {lam:.4f}   {L:.6e}")

   print(f"\n  Any dataset with the same T = {T:.1f} gives the SAME likelihood")
   print(f"  for every lambda. That is what sufficiency means.")


4.7.1 The Neyman--Fisher Factorization Theorem
------------------------------------------------

Checking the definition directly is often inconvenient.  The **factorization
theorem** provides a much simpler test.

**Theorem (Neyman--Fisher).**  A statistic :math:`T(x)` is sufficient for
:math:`\theta` if and only if the joint density (or PMF) can be written as

.. math::

   f(x \mid \theta) = g(T(x),\;\theta)\;h(x),

where

- :math:`g` depends on the data *only through* :math:`T(x)` and may depend on
  :math:`\theta`,
- :math:`h` depends on the data but **not** on :math:`\theta`.

*Plain English:*  Factor the density into two pieces.  One piece involves
:math:`\theta` but sees the data only through :math:`T`.  The other piece
sees the full data but does not involve :math:`\theta`.  If such a
factorization exists, :math:`T` is sufficient.

**Proof sketch (discrete case).**

(:math:`\Rightarrow`)  If :math:`T` is sufficient, write

.. math::

   f(x \mid \theta)
   = f(x \mid T(x), \theta)\,P(T(x) = T(x) \mid \theta)
   = \underbrace{f(x \mid T(x))}_{h(x)} \cdot \underbrace{P(T = T(x) \mid \theta)}_{g(T(x), \theta)}.

(:math:`\Leftarrow`)  If :math:`f(x \mid \theta) = g(T(x),\theta)\,h(x)`,
then

.. math::

   P(T = t \mid \theta) = \sum_{x: T(x)=t} g(t, \theta)\,h(x)
   = g(t, \theta)\sum_{x: T(x)=t} h(x),

and

.. math::

   f(x \mid T(x)=t, \theta)
   = \frac{f(x \mid \theta)}{P(T=t \mid \theta)}
   = \frac{g(t,\theta)\,h(x)}{g(t,\theta)\sum_{x':T(x')=t}h(x')}
   = \frac{h(x)}{\sum_{x':T(x')=t}h(x')},

which does not depend on :math:`\theta`.  Hence :math:`T` is sufficient.
:math:`\square`


4.7.2 Examples of Sufficient Statistics
-----------------------------------------

Let us apply the factorization theorem to three models and verify with code.

**Exponential (light bulbs).**  For :math:`X_i \sim \text{Exp}(\lambda)`:

.. math::

   L(\lambda) = \lambda^n \exp\!\left(-\lambda \sum x_i\right).

Take :math:`T = \sum x_i`, :math:`g(T, \lambda) = \lambda^n e^{-\lambda T}`,
:math:`h(x) = 1`.  So :math:`T = \sum X_i` is sufficient for :math:`\lambda`.

**Bernoulli / Binomial.**  For :math:`X_1, \dots, X_n` i.i.d.
:math:`\text{Bernoulli}(p)`:

.. math::

   L(p) = p^{\sum x_i}(1-p)^{n - \sum x_i}.

Take :math:`T = \sum x_i`, :math:`g(T, p) = p^T(1-p)^{n-T}`, :math:`h(x) = 1`.
So :math:`T = \sum X_i` is sufficient for :math:`p`.

**Normal (both parameters unknown).**  For :math:`X_i \sim N(\mu, \sigma^2)`:

.. math::

   L(\mu, \sigma^2) \propto (\sigma^2)^{-n/2}
   \exp\!\left(-\frac{1}{2\sigma^2}\sum (x_i - \mu)^2\right).

Expand :math:`\sum(x_i-\mu)^2 = \sum x_i^2 - 2\mu\sum x_i + n\mu^2`.  The
likelihood depends on the data only through :math:`\sum x_i` and
:math:`\sum x_i^2`.  So :math:`T = (\sum X_i,\;\sum X_i^2)` is sufficient for
:math:`(\mu, \sigma^2)`.

**Poisson.**  For :math:`X_i \sim \text{Pois}(\lambda)`:

.. math::

   L(\lambda) = \prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_i!}
   = \frac{e^{-n\lambda}\,\lambda^{\sum x_i}}{\prod x_i!}.

So :math:`T = \sum X_i` is sufficient, with :math:`g(T, \lambda) = e^{-n\lambda}\lambda^T`
and :math:`h(x) = 1/\prod x_i!`.

.. code-block:: python

   # Sufficiency: two datasets with the same T give the same likelihood
   import numpy as np

   # Two DIFFERENT datasets with the same sum
   data_A = np.array([800, 900, 1000, 1100, 1200])
   data_B = np.array([950, 950, 1000, 1050, 1050])
   n = 5

   T_A = data_A.sum()
   T_B = data_B.sum()
   print(f"Dataset A: {data_A}  ->  T = {T_A}")
   print(f"Dataset B: {data_B}  ->  T = {T_B}")
   print(f"Same T? {T_A == T_B}")

   print(f"\nLikelihood values (should be identical):")
   print(f"  {'lambda':>10s}  {'L(A)':>14s}  {'L(B)':>14s}  {'match':>6s}")
   for lam in [0.0008, 0.001, 0.0012]:
       L_A = lam**n * np.exp(-lam * T_A)
       L_B = lam**n * np.exp(-lam * T_B)
       print(f"  {lam:10.4f}  {L_A:14.6e}  {L_B:14.6e}  {np.isclose(L_A, L_B)}")

   print(f"\n  The raw data differ, but the likelihood is the same because")
   print(f"  T = sum(x_i) is sufficient for lambda.")


4.7.3 Minimal Sufficiency
---------------------------

A sufficient statistic :math:`T` is **minimal sufficient** if it is a function
of every other sufficient statistic.  Minimal sufficient statistics achieve the
maximum possible data compression without losing information.

For exponential families, the natural sufficient statistics are minimal
sufficient.


4.8 The Likelihood Principle
==============================

The **likelihood principle** is a foundational claim about how statistical
evidence should be interpreted.

**Statement.**  All the evidence that the data provide about the parameter
:math:`\theta` is contained in the likelihood function
:math:`L(\theta \mid x)`.  Specifically, if two experiments produce likelihood
functions that are proportional as functions of :math:`\theta`,

.. math::

   L_1(\theta \mid x_1) = c \cdot L_2(\theta \mid x_2)
   \quad \text{for all } \theta,

where :math:`c > 0` does not depend on :math:`\theta`, then the two
experiments provide identical evidence about :math:`\theta`.


4.8.1 Discussion
-----------------

The likelihood principle has profound consequences:

1. **Stopping rules are irrelevant.**  Consider a researcher who flips a coin
   and decides to stop either after 12 flips or after 3 heads, whichever comes
   first.  Under the likelihood principle, the inference about :math:`p`
   depends only on the data actually observed (say, 3 heads in 10 flips), not
   on the stopping intention.  The likelihood function is the same regardless
   of the stopping rule.

2. **Bayesian inference automatically satisfies the likelihood principle.**
   The posterior :math:`\pi(\theta \mid x) \propto L(\theta \mid x)\,\pi(\theta)`
   depends on the data only through the likelihood.

3. **Some frequentist procedures violate it.**  P-values and confidence
   intervals can depend on the sample space of outcomes that *could have*
   occurred but *did not*, which is information outside the likelihood.

4. **Connection to sufficiency.**  The likelihood principle can be derived from
   two more basic principles: the **sufficiency principle** (inference should
   depend on the data only through a sufficient statistic) and the **conditionality
   principle** (if an experiment is selected at random, inference should be
   conditional on which experiment was performed).  This remarkable result is
   due to Birnbaum (1962).

The likelihood principle does not dictate *how* to use the likelihood ---
whether to maximize it, integrate over it with a prior, or something else ---
only that the likelihood contains all the evidential content of the data.

.. code-block:: python

   # Likelihood principle: two stopping rules, same likelihood
   import numpy as np
   from scipy.special import comb

   # Experiment 1: flip 12 times, observe 3 heads (fixed n)
   # L_1(p) = C(12,3) * p^3 * (1-p)^9

   # Experiment 2: flip until 3 heads, took 12 flips (fixed k)
   # L_2(p) = C(11,2) * p^3 * (1-p)^9
   # (NegBin: last flip is a head, so C(11,2) ways to arrange 2 heads in first 11)

   print("Two experiments, same data (3 heads in 12 flips):\n")
   print(f"  {'p':>6s}  {'L1 (fixed n)':>14s}  {'L2 (fixed k)':>14s}  {'ratio':>8s}")
   for p in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
       L1 = comb(12, 3) * p**3 * (1-p)**9
       L2 = comb(11, 2) * p**3 * (1-p)**9
       print(f"  {p:6.2f}  {L1:14.6f}  {L2:14.6f}  {L1/L2:8.4f}")

   print(f"\n  The ratio L1/L2 = C(12,3)/C(11,2) = {comb(12,3)/comb(11,2):.4f} is constant.")
   print(f"  The likelihoods are proportional, so by the likelihood principle,")
   print(f"  both experiments provide IDENTICAL evidence about p.")


4.9 Bringing It All Together: Complete Light Bulb Analysis
===========================================================

Let us trace the logical arc of this chapter with one complete computation on
our light bulb data.

1. We start with a probability model :math:`f(x \mid \theta)` and observed
   data :math:`x`.
2. Viewing :math:`f(x \mid \theta)` as a function of :math:`\theta` gives the
   **likelihood** :math:`L(\theta \mid x)`.
3. Taking the logarithm gives the **log-likelihood**
   :math:`\ell(\theta \mid x)`, which is easier to work with.
4. The **score** :math:`S(\theta) = \ell'(\theta)` has zero expectation under
   the true parameter and is zero at the MLE.
5. **Fisher information** :math:`I(\theta) = \operatorname{Var}(S)` measures
   the curvature of :math:`\ell`, quantifying how much data tell us about
   :math:`\theta`.
6. **Sufficient statistics** compress the data without losing information about
   :math:`\theta`.
7. The **likelihood principle** asserts that the likelihood function captures
   all the evidence in the data.

.. code-block:: python

   # Complete likelihood analysis: light bulb data
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)

   # ---- Step 1: Data ----
   lambda_true = 0.001  # true failure rate (mean lifetime 1000 hrs)
   data = np.random.exponential(scale=1/lambda_true, size=50)
   n = len(data)
   T = data.sum()           # sufficient statistic
   x_bar = data.mean()

   # ---- Step 2: MLE ----
   lambda_hat = n / T       # = 1 / x_bar

   # ---- Step 3: Log-likelihood at MLE ----
   logL_at_mle = n * np.log(lambda_hat) - lambda_hat * T

   # ---- Step 4: Score at MLE ----
   score_at_mle = n / lambda_hat - T

   # ---- Step 5: Fisher information and SE ----
   I_n = n / lambda_hat**2
   se_mle = 1.0 / np.sqrt(I_n)  # = lambda_hat / sqrt(n)
   ci_lo = lambda_hat - 1.96 * se_mle
   ci_hi = lambda_hat + 1.96 * se_mle

   # ---- Step 6: Numerical verification of MLE ----
   result = minimize_scalar(
       lambda lam: -(n * np.log(lam) - lam * T),
       bounds=(1e-6, 0.01), method='bounded'
   )

   print("=" * 64)
   print("  COMPLETE LIKELIHOOD ANALYSIS: LIGHT BULB LIFETIMES")
   print("=" * 64)
   print(f"\n  Model:            X_i ~ Exp(lambda)")
   print(f"  True lambda:      {lambda_true}")
   print(f"  Data:             n = {n} bulbs")
   print(f"  Sample mean:      x_bar = {x_bar:.2f} hours")
   print(f"  Sufficient stat:  T = sum(x_i) = {T:.1f}")
   print(f"\n  --- Maximum Likelihood Estimation ---")
   print(f"  MLE (analytic):   lambda_hat = 1/x_bar = {lambda_hat:.6f}")
   print(f"  MLE (numerical):  lambda_hat = {result.x:.6f}")
   print(f"  ell(lambda_hat):  {logL_at_mle:.2f}")
   print(f"\n  --- Score Function ---")
   print(f"  S(lambda_hat) = n/lambda - T = {score_at_mle:.10f}  (= 0 at MLE)")
   print(f"\n  --- Fisher Information ---")
   print(f"  I_n(lambda_hat) = n/lambda^2 = {I_n:.2f}")
   print(f"  SE(lambda_hat)  = lambda/sqrt(n) = {se_mle:.6f}")
   print(f"  95% CI:  ({ci_lo:.6f}, {ci_hi:.6f})")
   print(f"  True lambda {lambda_true} in CI? "
         f"{'Yes' if ci_lo <= lambda_true <= ci_hi else 'No'}")

   # ---- Log-likelihood table ----
   print(f"\n  --- Log-Likelihood Curve ---")
   print(f"  {'lambda':>10s}  {'ell(lambda)':>12s}  {'ell - max':>10s}")
   for lam in np.linspace(0.0006, 0.0016, 11):
       ll = n * np.log(lam) - lam * T
       marker = " <-- MLE" if abs(lam - lambda_hat) < 0.00005 else ""
       print(f"  {lam:10.6f}  {ll:12.2f}  {ll - logL_at_mle:10.2f}{marker}")

These ideas form the theoretical core of everything that follows.  In the
subsequent parts of this book, we will:

- Build a catalogue of likelihoods for specific models (Part II).
- Develop maximum likelihood estimation and its properties (Part III).
- Study the optimization algorithms that find the MLE (Part IV).
- Explore advanced topics including Bayesian inference, model selection, and
  computational methods (Part V).


4.10 Summary
==============

- The **likelihood function** :math:`L(\theta \mid x) = f(x \mid \theta)` is
  the probability (density) of the observed data, viewed as a function of the
  parameter.
- The **log-likelihood** :math:`\ell = \ln L` turns products into sums and
  preserves optima thanks to the monotonicity of the logarithm.
- The **score function** :math:`S(\theta) = \partial\ell / \partial\theta` has
  the key property :math:`E[S] = 0`.
- **Fisher information** :math:`I(\theta) = E[S^2] = -E[\ell'']` quantifies
  the information the data carry about :math:`\theta`.  It is additive for
  independent observations.
- A **sufficient statistic** :math:`T(x)` captures all information about
  :math:`\theta`.  The **Neyman--Fisher factorization theorem** provides a
  practical test for sufficiency.
- The **likelihood principle** states that the likelihood function is the sole
  carrier of evidential content in the data.
