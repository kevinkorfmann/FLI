.. _ch5_discrete:

==========================================
Chapter 5 --- Discrete Likelihoods
==========================================

This chapter catalogues the most important discrete probability distributions.
For every distribution we follow the same programme:

1. **Motivating scenario** --- a concrete problem that demands this distribution.
2. State the probability mass function (PMF) and compute it in code.
3. Write the log-likelihood and sweep over parameter values to find the peak.
4. Derive the score function and verify it equals zero at the MLE.
5. Compute the Fisher information, standard error, and a 95% confidence interval.
6. Print a summary line: MLE, SE, 95% CI, true parameter.

.. admonition:: Why this programme matters

   By following the same steps for every distribution, you build a mental
   template that transfers to *any* new model you encounter. Once you have
   internalised the pattern---scenario, PMF, log-likelihood, score, information,
   MLE---the mechanics become automatic and you can focus on the modelling
   decisions that really matter.

.. contents:: Distributions in this chapter
   :local:
   :depth: 1


.. _sec_bernoulli:

5.1 Bernoulli Distribution
===========================

Motivating scenario
--------------------

A quality-control engineer inspects circuit boards coming off a production
line.  Each board is either *defective* (1) or *non-defective* (0).  She wants
to estimate the defect rate :math:`p` from a sample of 100 boards, quantify
her uncertainty, and decide whether the line meets a 5% defect-rate target.
Every more complex discrete distribution in this chapter can be built from
Bernoulli building blocks, so mastering this case first is essential.

PMF
---

Let :math:`X \in \{0, 1\}` with success probability :math:`p \in (0,1)`.
The probability mass function is

.. math::

   f(x \mid p) = p^{x}(1-p)^{1-x}, \qquad x \in \{0,1\}.

When :math:`x=1` this gives :math:`p`; when :math:`x=0` it gives :math:`1-p`.

.. code-block:: python

   # Bernoulli PMF: compute P(X=k) for each outcome
   import numpy as np

   p = 0.7
   for k in [0, 1]:
       pmf_k = p**k * (1 - p)**(1 - k)
       print(f"P(X={k} | p={p}) = {pmf_k:.4f}")

   # Verify: probabilities sum to 1
   total = sum(p**k * (1 - p)**(1 - k) for k in [0, 1])
   print(f"Sum of PMF = {total:.4f}")

Moments: :math:`E[X] = p` and :math:`\text{Var}(X) = p(1-p)`.

.. code-block:: python

   # Bernoulli moments: verify E[X] and Var(X) by simulation
   np.random.seed(42)
   p_true = 0.7
   n_sim = 100_000
   samples = np.random.binomial(1, p_true, size=n_sim)

   print(f"E[X]   = {samples.mean():.4f}  (theory: {p_true})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {p_true*(1-p_true):.4f})")

Log-likelihood
--------------

Given :math:`n` i.i.d. observations :math:`x_1, \dots, x_n`, define
:math:`s = \sum_{i=1}^{n} x_i` (the number of successes). The log-likelihood
is

.. math::

   \ell(p) = s \ln p + (n - s) \ln(1 - p).

The entire dataset collapses into a single number :math:`s` --- the sufficient
statistic.

.. code-block:: python

   # Bernoulli log-likelihood: sweep over p, find the peak
   import numpy as np

   np.random.seed(42)
   p_true = 0.7
   n = 100
   data = np.random.binomial(1, p_true, size=n)
   s = data.sum()

   p_grid = np.linspace(0.01, 0.99, 200)
   loglik = s * np.log(p_grid) + (n - s) * np.log(1 - p_grid)

   p_peak = p_grid[np.argmax(loglik)]
   print(f"Grid search MLE: p_hat = {p_peak:.4f}")
   print(f"Exact MLE:       p_hat = {s/n:.4f}")
   print(f"True p:                  {p_true}")

Score function
--------------

Differentiating with respect to :math:`p`:

.. math::

   S(p) = \frac{d\ell}{dp} = \frac{s}{p} - \frac{n-s}{1-p}.

The score balances two competing pulls: the successes push :math:`p` upward, and
the failures push it downward.

.. code-block:: python

   # Bernoulli score: verify score = 0 at the MLE
   p_hat = s / n
   score_at_mle = s / p_hat - (n - s) / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}  (should be 0)")

   # Score at a wrong value --- should be nonzero
   score_at_0_5 = s / 0.5 - (n - s) / (1 - 0.5)
   print(f"Score at p=0.5 = {score_at_0_5:.4f}  (nonzero: pushes toward MLE)")

Fisher information
------------------

The Fisher information for a single observation is

.. math::

   \mathcal{I}(p) = \frac{1}{p(1-p)}.

For :math:`n` observations the total information is
:math:`n\,\mathcal{I}(p) = \dfrac{n}{p(1-p)}`.

.. admonition:: Intuition

   The Fisher information is largest when :math:`p` is near 0 or 1 (extreme
   probabilities) and smallest when :math:`p = 0.5`. When almost every trial
   gives the same result, each observation is highly informative about
   :math:`p`. When outcomes are maximally uncertain, each observation tells
   you relatively less.

MLE and inference
------------------

Setting the score to zero gives

.. math::

   \hat{p}_{\text{MLE}} = \frac{s}{n} = \bar{x}.

The standard error is :math:`\text{SE} = 1/\sqrt{n\,\mathcal{I}(\hat{p})}
= \sqrt{\hat{p}(1-\hat{p})/n}`.

.. code-block:: python

   # Bernoulli MLE: full summary with SE and 95% CI
   import numpy as np

   np.random.seed(42)
   p_true = 0.7
   n = 100
   data = np.random.binomial(1, p_true, size=n)

   s = data.sum()
   p_hat = s / n

   # Fisher information and SE
   fisher_info_total = n / (p_hat * (1 - p_hat))
   se = 1.0 / np.sqrt(fisher_info_total)

   # 95% Wald confidence interval
   ci_lo = p_hat - 1.96 * se
   ci_hi = p_hat + 1.96 * se

   # Verify score = 0 at MLE
   score = s / p_hat - (n - s) / (1 - p_hat)

   print(f"MLE = {p_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true p = {p_true}")
   print(f"Score at MLE = {score:.2e}")


.. _sec_binomial:

5.2 Binomial Distribution
==========================

Motivating scenario
--------------------

A pharmaceutical company runs a clinical trial where each of :math:`n = 50`
patients undergoes :math:`m = 10` treatment sessions.  Each session is recorded
as a response or non-response.  The goal is to estimate the per-session
response probability :math:`p`, together with a confidence interval that will
be reported to the regulator.  The Binomial is "Bernoulli at scale" --- we
count successes within each patient, then combine across patients.

PMF
---

Let :math:`X \sim \text{Binomial}(m, p)` where :math:`m` is known and
:math:`p \in (0,1)`:

.. math::

   f(x \mid p) = \binom{m}{x} p^x (1-p)^{m-x}, \qquad x = 0, 1, \dots, m.

.. code-block:: python

   # Binomial PMF: compute P(X=k) for several k
   import numpy as np
   from scipy.special import comb

   m, p = 10, 0.65
   print(f"{'k':>3}  {'P(X=k)':>10}")
   print("-" * 16)
   for k in range(m + 1):
       pmf_k = comb(m, k, exact=True) * p**k * (1 - p)**(m - k)
       print(f"{k:3d}  {pmf_k:10.6f}")

Moments: :math:`E[X] = mp` and :math:`\text{Var}(X) = mp(1-p)`.

.. code-block:: python

   # Binomial moments: simulation check
   np.random.seed(42)
   m, p_true = 10, 0.65
   samples = np.random.binomial(m, p_true, size=100_000)

   print(f"E[X]   = {samples.mean():.4f}  (theory: {m*p_true})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {m*p_true*(1-p_true):.4f})")

Log-likelihood
--------------

For :math:`n` independent observations, let :math:`S = \sum_{i=1}^{n} x_i`.
The parameter-dependent part of the log-likelihood is

.. math::

   \ell(p) \propto S \ln p + (nm - S)\ln(1-p).

.. code-block:: python

   # Binomial log-likelihood: sweep over p, find the peak
   import numpy as np

   np.random.seed(42)
   m, p_true, n = 10, 0.65, 50
   data = np.random.binomial(m, p_true, size=n)
   S = data.sum()

   p_grid = np.linspace(0.01, 0.99, 300)
   loglik = S * np.log(p_grid) + (n * m - S) * np.log(1 - p_grid)

   p_peak = p_grid[np.argmax(loglik)]
   print(f"Grid search MLE: p_hat = {p_peak:.4f}")
   print(f"Exact MLE:       p_hat = {S/(n*m):.4f}")
   print(f"True p:                  {p_true}")

Score function
--------------

.. math::

   S(p) = \frac{S}{p} - \frac{nm - S}{1-p}.

This has exactly the same form as the Bernoulli score with
:math:`s \to S` and :math:`n \to nm`.

.. code-block:: python

   # Binomial score: verify score = 0 at MLE
   p_hat = S / (n * m)
   score_at_mle = S / p_hat - (n * m - S) / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

For a single :math:`\text{Binomial}(m,p)` observation:

.. math::

   \mathcal{I}(p) = \frac{m}{p(1-p)}.

For :math:`n` observations: :math:`n\mathcal{I}(p) = \dfrac{nm}{p(1-p)}`.

MLE and inference
------------------

.. math::

   \hat{p} = \frac{S}{nm} = \frac{\bar{x}}{m}.

.. code-block:: python

   # Binomial MLE: full summary
   import numpy as np

   np.random.seed(42)
   m, p_true, n = 10, 0.65, 50
   data = np.random.binomial(m, p_true, size=n)

   S = data.sum()
   p_hat = S / (n * m)

   # Fisher information and SE
   fisher_total = n * m / (p_hat * (1 - p_hat))
   se = 1.0 / np.sqrt(fisher_total)
   ci_lo = p_hat - 1.96 * se
   ci_hi = p_hat + 1.96 * se

   # Verify score = 0
   score = S / p_hat - (n * m - S) / (1 - p_hat)

   print(f"MLE = {p_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true p = {p_true}")
   print(f"Score at MLE = {score:.2e}")


.. _sec_poisson:

5.3 Poisson Distribution
=========================

Motivating scenario
--------------------

An epidemiologist counts the number of new tuberculosis cases reported per
month in a city over 3 years (36 months).  She needs to estimate the monthly
incidence rate :math:`\lambda`, build a confidence interval, and assess
whether the rate has changed from the historical average of 5.0 cases/month.
The Poisson is the natural model: events arrive independently in fixed time
intervals, and a single parameter :math:`\lambda` controls both the mean and
the variance.

PMF
---

Let :math:`X \sim \text{Poisson}(\lambda)` with :math:`\lambda > 0`:

.. math::

   f(x \mid \lambda) = \frac{\lambda^x e^{-\lambda}}{x!},
   \qquad x = 0, 1, 2, \dots

.. code-block:: python

   # Poisson PMF: compute P(X=k) for several k
   import numpy as np
   from math import factorial

   lam = 4.5
   print(f"{'k':>3}  {'P(X=k)':>12}")
   print("-" * 18)
   for k in range(15):
       pmf_k = lam**k * np.exp(-lam) / factorial(k)
       print(f"{k:3d}  {pmf_k:12.6f}")

   # Verify: sum over enough terms ~ 1
   total = sum(lam**k * np.exp(-lam) / factorial(k) for k in range(50))
   print(f"\nSum P(X=0..49) = {total:.10f}")

Moments: :math:`E[X] = \lambda` and :math:`\text{Var}(X) = \lambda`.
The mean equals the variance --- this is the Poisson's signature.

.. code-block:: python

   # Poisson moments: simulation check
   np.random.seed(42)
   lam_true = 4.5
   samples = np.random.poisson(lam_true, size=100_000)

   print(f"E[X]   = {samples.mean():.4f}  (theory: {lam_true})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {lam_true})")
   print(f"Var/Mean ratio = {samples.var()/samples.mean():.4f}  (Poisson => ~1)")

Log-likelihood
--------------

For :math:`n` i.i.d. observations:

.. math::

   \ell(\lambda)
     = \left(\sum_{i=1}^{n} x_i\right)\ln\lambda - n\lambda
       - \underbrace{\sum_{i=1}^{n}\ln(x_i!)}_{\text{constant in }\lambda}.

.. code-block:: python

   # Poisson log-likelihood: sweep over lambda, find the peak
   import numpy as np

   np.random.seed(42)
   lam_true = 5.2
   n = 36
   data = np.random.poisson(lam_true, size=n)

   lam_grid = np.linspace(1.0, 10.0, 300)
   loglik = data.sum() * np.log(lam_grid) - n * lam_grid

   lam_peak = lam_grid[np.argmax(loglik)]
   print(f"Grid search MLE: lam_hat = {lam_peak:.4f}")
   print(f"Exact MLE:       lam_hat = {data.mean():.4f}")
   print(f"True lambda:               {lam_true}")

Score function
--------------

.. math::

   S(\lambda) = \frac{\sum x_i}{\lambda} - n.

.. code-block:: python

   # Poisson score: verify score = 0 at MLE
   lam_hat = data.mean()
   score_at_mle = data.sum() / lam_hat - n
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

For one observation: :math:`\mathcal{I}(\lambda) = 1/\lambda`.
For :math:`n` observations: :math:`n/\lambda`.

.. admonition:: Intuition

   Larger rates are harder to pin down precisely. When events are rare
   (small :math:`\lambda`), each count is very informative. When events are
   frequent, you need more data for the same relative precision.

MLE and inference
------------------

.. math::

   \hat{\lambda} = \bar{x}, \qquad
   \text{SE} = \sqrt{\hat{\lambda}/n}.

.. code-block:: python

   # Poisson MLE: full summary
   import numpy as np

   np.random.seed(42)
   lam_true = 5.2
   n = 36
   data = np.random.poisson(lam_true, size=n)

   lam_hat = data.mean()
   fisher_total = n / lam_hat
   se = 1.0 / np.sqrt(fisher_total)
   ci_lo = lam_hat - 1.96 * se
   ci_hi = lam_hat + 1.96 * se

   score = data.sum() / lam_hat - n

   print(f"MLE = {lam_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true lam = {lam_true}")
   print(f"Score at MLE = {score:.2e}")


.. _sec_negbin:

5.4 Negative Binomial Distribution
====================================

Motivating scenario
--------------------

An ecologist counts individuals of an insect species at 150 trapping sites.
The counts show far more zeros and far more high values than a Poisson would
predict --- the sample variance is three times the sample mean.  This
*overdispersion* is common when organisms cluster spatially.  The Negative
Binomial adds a second parameter that decouples the mean from the variance,
making it the go-to model when "variance > mean."

PMF
---

Let :math:`X` be the number of failures before the :math:`r`-th success,
:math:`r > 0` known, :math:`p \in (0,1)`:

.. math::

   f(x \mid r, p) = \binom{x + r - 1}{x} p^r (1-p)^x,
   \qquad x = 0, 1, 2, \dots

.. code-block:: python

   # Negative Binomial PMF: compute P(X=k) for several k
   import numpy as np
   from scipy.special import comb

   r, p = 5, 0.4
   print(f"{'k':>3}  {'P(X=k)':>12}")
   print("-" * 18)
   for k in range(20):
       pmf_k = comb(k + r - 1, k, exact=True) * p**r * (1 - p)**k
       print(f"{k:3d}  {pmf_k:12.6f}")

Moments: :math:`E[X] = r(1-p)/p` and
:math:`\text{Var}(X) = r(1-p)/p^2`.
Note :math:`\text{Var}(X) > E[X]` whenever :math:`p < 1` --- this is the
built-in overdispersion.

.. code-block:: python

   # NegBin moments: verify overdispersion
   np.random.seed(42)
   r, p_true = 5, 0.4
   samples = np.random.negative_binomial(r, p_true, size=100_000)

   print(f"E[X]   = {samples.mean():.4f}  (theory: {r*(1-p_true)/p_true:.4f})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {r*(1-p_true)/p_true**2:.4f})")
   print(f"Var/Mean = {samples.var()/samples.mean():.4f}  (>1 = overdispersion)")

Log-likelihood
--------------

For :math:`n` i.i.d. observations with :math:`r` known:

.. math::

   \ell(p) = \text{const} + nr\ln p + \left(\sum_{i=1}^{n} x_i\right)\ln(1-p).

.. code-block:: python

   # NegBin log-likelihood: sweep over p, find the peak
   import numpy as np

   np.random.seed(42)
   r, p_true, n = 5, 0.4, 150
   data = np.random.negative_binomial(r, p_true, size=n)

   p_grid = np.linspace(0.01, 0.99, 300)
   loglik = n * r * np.log(p_grid) + data.sum() * np.log(1 - p_grid)

   p_peak = p_grid[np.argmax(loglik)]
   x_bar = data.mean()
   p_hat_exact = r / (r + x_bar)

   print(f"Grid search MLE: p_hat = {p_peak:.4f}")
   print(f"Exact MLE:       p_hat = {p_hat_exact:.4f}")
   print(f"True p:                  {p_true}")

Score function
--------------

.. math::

   S(p) = \frac{nr}{p} - \frac{\sum x_i}{1-p}.

.. code-block:: python

   # NegBin score: verify score = 0 at MLE
   p_hat = r / (r + data.mean())
   score_at_mle = n * r / p_hat - data.sum() / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

For one observation:

.. math::

   \mathcal{I}(p) = \frac{r}{p^2(1-p)}.

MLE and inference
------------------

.. math::

   \hat{p} = \frac{r}{r + \bar{x}}.

When :math:`r` is also unknown, no closed-form MLE exists and numerical
methods (e.g., Newton--Raphson) are needed.

.. code-block:: python

   # Negative Binomial MLE: full summary
   import numpy as np

   np.random.seed(42)
   r, p_true, n = 5, 0.4, 150
   data = np.random.negative_binomial(r, p_true, size=n)

   x_bar = data.mean()
   p_hat = r / (r + x_bar)

   # Fisher info and SE
   fisher_single = r / (p_hat**2 * (1 - p_hat))
   fisher_total = n * fisher_single
   se = 1.0 / np.sqrt(fisher_total)
   ci_lo = p_hat - 1.96 * se
   ci_hi = p_hat + 1.96 * se

   score = n * r / p_hat - data.sum() / (1 - p_hat)

   print(f"Sample mean = {x_bar:.2f}, Sample var = {data.var():.2f}, "
         f"Var/Mean = {data.var()/x_bar:.2f}")
   print(f"MLE = {p_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true p = {p_true}")
   print(f"Score at MLE = {score:.2e}")


.. _sec_geometric:

5.5 Geometric Distribution
============================

Motivating scenario
--------------------

A reliability engineer tests prototype light bulbs sequentially.  Each test
is pass/fail.  She records how many bulbs fail before the first one passes
all tests.  This "number of failures before the first success" follows a
Geometric distribution --- the special case of the Negative Binomial with
:math:`r = 1`.  The memoryless property makes it the discrete analogue of
the Exponential distribution.

PMF
---

We use the "number of failures before the first success" convention:

.. math::

   f(x \mid p) = (1-p)^x \, p, \qquad x = 0, 1, 2, \dots

.. code-block:: python

   # Geometric PMF: compute P(X=k) for several k
   import numpy as np

   p = 0.3
   print(f"{'k':>3}  {'P(X=k)':>12}  {'CDF':>12}")
   print("-" * 32)
   cdf = 0.0
   for k in range(15):
       pmf_k = (1 - p)**k * p
       cdf += pmf_k
       print(f"{k:3d}  {pmf_k:12.6f}  {cdf:12.6f}")

Moments: :math:`E[X] = (1-p)/p` and :math:`\text{Var}(X) = (1-p)/p^2`.

.. code-block:: python

   # Geometric moments: simulation check
   np.random.seed(42)
   p_true = 0.3
   samples = np.random.geometric(p_true, size=100_000) - 1  # convert to failures

   print(f"E[X]   = {samples.mean():.4f}  (theory: {(1-p_true)/p_true:.4f})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {(1-p_true)/p_true**2:.4f})")

Log-likelihood
--------------

.. math::

   \ell(p) = n\ln p + \left(\sum_{i=1}^{n} x_i\right)\ln(1-p).

.. code-block:: python

   # Geometric log-likelihood: sweep over p
   import numpy as np

   np.random.seed(42)
   p_true, n = 0.3, 200
   data = np.random.geometric(p_true, size=n) - 1

   p_grid = np.linspace(0.01, 0.99, 300)
   loglik = n * np.log(p_grid) + data.sum() * np.log(1 - p_grid)

   p_peak = p_grid[np.argmax(loglik)]
   p_hat_exact = 1 / (1 + data.mean())

   print(f"Grid search MLE: p_hat = {p_peak:.4f}")
   print(f"Exact MLE:       p_hat = {p_hat_exact:.4f}")
   print(f"True p:                  {p_true}")

Score function
--------------

.. math::

   S(p) = \frac{n}{p} - \frac{\sum x_i}{1-p}.

.. code-block:: python

   # Geometric score: verify score = 0 at MLE
   p_hat = 1 / (1 + data.mean())
   score_at_mle = n / p_hat - data.sum() / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

For one observation:

.. math::

   \mathcal{I}(p) = \frac{1}{p^2(1-p)}.

MLE and inference
------------------

.. math::

   \hat{p} = \frac{1}{1 + \bar{x}}.

.. code-block:: python

   # Geometric MLE: full summary
   import numpy as np

   np.random.seed(42)
   p_true, n = 0.3, 200
   data = np.random.geometric(p_true, size=n) - 1

   x_bar = data.mean()
   p_hat = 1 / (1 + x_bar)

   # Fisher info and SE
   fisher_single = 1 / (p_hat**2 * (1 - p_hat))
   fisher_total = n * fisher_single
   se = 1.0 / np.sqrt(fisher_total)
   ci_lo = p_hat - 1.96 * se
   ci_hi = p_hat + 1.96 * se

   score = n / p_hat - data.sum() / (1 - p_hat)

   print(f"MLE = {p_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true p = {p_true}")
   print(f"Score at MLE = {score:.2e}")


.. _sec_hypergeometric:

5.6 Hypergeometric Distribution
=================================

Motivating scenario
--------------------

A fisheries biologist uses capture--recapture to estimate the population of
fish in a lake.  She tags :math:`K` fish, releases them, then draws a
sample of :math:`n` fish and counts how many are tagged.  Because she is
sampling *without replacement* from a finite population of :math:`N` fish,
successive draws are dependent.  The Hypergeometric distribution captures
this dependence exactly --- it is the "finite-population Binomial."

PMF
---

.. math::

   f(x \mid K) = \frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}},
   \qquad \max(0, n+K-N) \le x \le \min(n, K).

.. code-block:: python

   # Hypergeometric PMF: compute P(X=k) for several k
   import numpy as np
   from scipy.stats import hypergeom

   N, K, n_draw = 500, 120, 50
   lo = max(0, n_draw + K - N)
   hi = min(n_draw, K)

   print(f"{'k':>3}  {'P(X=k)':>12}")
   print("-" * 18)
   for k in range(lo, hi + 1):
       print(f"{k:3d}  {hypergeom.pmf(k, N, K, n_draw):12.6f}")

Moments: :math:`E[X] = nK/N` and
:math:`\text{Var}(X) = n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}`.

.. code-block:: python

   # Hypergeometric moments: simulation check
   np.random.seed(42)
   samples = hypergeom.rvs(N, K, n_draw, size=100_000)

   EX_theory = n_draw * K / N
   VarX_theory = n_draw * (K/N) * ((N-K)/N) * ((N-n_draw)/(N-1))

   print(f"E[X]   = {samples.mean():.4f}  (theory: {EX_theory:.4f})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {VarX_theory:.4f})")

MLE
---

Because :math:`K` is an integer parameter, we maximise over the integer
lattice.  The MLE satisfies

.. math::

   \hat{K} = \left\lfloor \frac{(N+1)\,x}{n} \right\rfloor.

.. code-block:: python

   # Hypergeometric MLE: exhaustive search and floor formula
   import numpy as np
   from scipy.stats import hypergeom

   np.random.seed(42)
   N, K_true, n_draw = 500, 120, 50
   x_obs = hypergeom.rvs(N, K_true, n_draw)

   # Exhaustive search over integer K
   K_values = np.arange(max(0, x_obs), N + 1)
   log_likes = [hypergeom.logpmf(x_obs, N, K, n_draw) for K in K_values]
   K_hat_search = K_values[np.argmax(log_likes)]

   # Floor formula
   K_hat_formula = int(np.floor((N + 1) * x_obs / n_draw))

   print(f"Observed tagged in sample x = {x_obs}")
   print(f"MLE K_hat (search)  = {K_hat_search}")
   print(f"MLE K_hat (formula) = {K_hat_formula}")
   print(f"True K = {K_true}")

Fisher information
------------------

Because the parameter space is discrete, the Fisher information is not
defined via second derivatives. For large :math:`N`, the Hypergeometric
approaches the Binomial, and the Fisher information approaches
:math:`n/[p(1-p)]` with :math:`p = K/N`.

.. code-block:: python

   # Hypergeometric: approximate SE via Binomial approximation
   p_hat = K_hat_search / N
   se_approx = np.sqrt(p_hat * (1 - p_hat) / n_draw)  # on p = K/N scale
   K_se = se_approx * N  # transform to K scale

   print(f"MLE K_hat = {K_hat_search}")
   print(f"Approx SE(K_hat) = {K_se:.1f}")
   print(f"Approx 95% CI for K: "
         f"[{K_hat_search - 1.96*K_se:.0f}, {K_hat_search + 1.96*K_se:.0f}]")
   print(f"True K = {K_true}")


.. _sec_multinomial:

5.7 Multinomial Distribution
==============================

Motivating scenario
--------------------

A political pollster surveys :math:`n = 500` likely voters, asking each to
choose among :math:`k = 4` candidates.  She needs to estimate the proportion
supporting each candidate and assess whether a front-runner has a
statistically significant lead.  The Multinomial generalises the Binomial
from two outcomes to :math:`k` outcomes: each trial allocates one unit to
exactly one category.

PMF
---

Let :math:`\mathbf{x} = (x_1, \dots, x_k)` with :math:`\sum_{j=1}^k x_j = m`
and :math:`\mathbf{p} = (p_1, \dots, p_k)` with :math:`\sum_j p_j = 1`:

.. math::

   f(\mathbf{x} \mid \mathbf{p})
     = \frac{m!}{x_1!\cdots x_k!}\,p_1^{x_1}\cdots p_k^{x_k}.

.. code-block:: python

   # Multinomial PMF: compute P(X = x) for a specific observation
   import numpy as np
   from math import factorial
   from functools import reduce

   m = 1  # single trial (categorical)
   p_true = np.array([0.35, 0.30, 0.20, 0.15])
   k = len(p_true)

   # PMF for each single-draw outcome
   print(f"{'Category':>10}  {'p_true':>8}  {'P(X=e_j)':>10}")
   print("-" * 32)
   for j in range(k):
       x = np.zeros(k, dtype=int)
       x[j] = 1
       multinomial_coeff = factorial(m) / reduce(lambda a, b: a * b,
                                                  [factorial(xi) for xi in x])
       pmf = multinomial_coeff * np.prod(p_true**x)
       print(f"{'Cand '+str(j+1):>10}  {p_true[j]:8.2f}  {pmf:10.6f}")

.. code-block:: python

   # Multinomial moments: simulation check (m=1 case = Categorical)
   np.random.seed(42)
   n = 100_000
   samples = np.random.multinomial(1, p_true, size=n)
   means = samples.mean(axis=0)

   print(f"{'Category':>10}  {'E[X_j]':>8}  {'Theory':>8}")
   for j in range(k):
       print(f"{'Cand '+str(j+1):>10}  {means[j]:8.4f}  {p_true[j]:8.4f}")

Log-likelihood
--------------

For :math:`n` independent observations, define :math:`S_j = \sum_{i=1}^{n} x_j^{(i)}`:

.. math::

   \ell(\mathbf{p}) = \text{const} + \sum_{j=1}^{k} S_j \ln p_j.

MLE via Lagrange multipliers
-----------------------------

Subject to :math:`\sum_j p_j = 1`:

.. math::

   \hat{p}_j = \frac{S_j}{nm}.

The MLE for each category probability is the observed relative frequency.

.. code-block:: python

   # Multinomial MLE: estimate from poll data
   import numpy as np

   np.random.seed(42)
   p_true = np.array([0.35, 0.30, 0.20, 0.15])
   k = len(p_true)
   n = 500  # voters
   m = 1    # one vote per person

   data = np.random.multinomial(m, p_true, size=n)
   S = data.sum(axis=0)
   p_hat = S / (n * m)

   print(f"{'Category':>10} | {'True p':>8} | {'MLE':>8} | {'Count':>6} | {'SE':>8} | {'95% CI':>18}")
   print("-" * 72)
   for j in range(k):
       se_j = np.sqrt(p_hat[j] * (1 - p_hat[j]) / (n * m))
       ci_lo = p_hat[j] - 1.96 * se_j
       ci_hi = p_hat[j] + 1.96 * se_j
       print(f"{'Cand '+str(j+1):>10} | {p_true[j]:8.4f} | {p_hat[j]:8.4f} | "
             f"{S[j]:6d} | {se_j:8.4f} | [{ci_lo:.4f}, {ci_hi:.4f}]")

Score function
--------------

The score with respect to :math:`p_j` (before imposing the constraint) is

.. math::

   S_j(p_j) = \frac{S_j}{p_j}.

Fisher information
------------------

The Fisher information matrix for a single multinomial observation
(with :math:`p_k = 1 - \sum_{j<k} p_j` eliminated) has entries

.. math::

   \mathcal{I}_{jl} = \frac{m\,\delta_{jl}}{p_j} + \frac{m}{p_k},
   \qquad j, l = 1, \dots, k-1.

.. code-block:: python

   # Multinomial Fisher information matrix (k-1 x k-1)
   import numpy as np

   p_hat = np.array([0.35, 0.30, 0.20, 0.15])
   k = len(p_hat)
   m = 1

   # Fisher info for (p_1, ..., p_{k-1})
   I_mat = np.zeros((k - 1, k - 1))
   for j in range(k - 1):
       for l in range(k - 1):
           I_mat[j, l] = m * (int(j == l) / p_hat[j] + 1.0 / p_hat[k - 1])

   print("Fisher information matrix:")
   print(np.array2string(I_mat, precision=4))
   print(f"\nCondition number: {np.linalg.cond(I_mat):.2f}")


.. _sec_zip:

5.8 Zero-Inflated Poisson (ZIP) Distribution
===============================================

Motivating scenario
--------------------

An insurance company analyses the number of claims filed per policyholder
per year.  A histogram reveals a massive spike at zero --- far more than a
Poisson model predicts.  Many policyholders never file at all (they are
"structural zeros" who did not experience any insurable event), while those
who do file show Poisson-distributed claim counts.  The ZIP model handles
this two-component structure with a mixture parameter :math:`\pi` (probability
of structural zero) and a Poisson rate :math:`\lambda`.

PMF
---

.. math::

   f(x \mid \pi, \lambda) =
   \begin{cases}
     \pi + (1-\pi)\,e^{-\lambda}, & x = 0, \\[6pt]
     (1-\pi)\,\dfrac{\lambda^x e^{-\lambda}}{x!}, & x = 1, 2, \dots
   \end{cases}

.. code-block:: python

   # ZIP PMF: compute P(X=k) and compare to plain Poisson
   import numpy as np
   from math import factorial

   pi, lam = 0.3, 3.5
   print(f"{'k':>3}  {'ZIP P(X=k)':>12}  {'Poisson P(X=k)':>16}")
   print("-" * 36)
   for k in range(12):
       if k == 0:
           zip_pmf = pi + (1 - pi) * np.exp(-lam)
       else:
           zip_pmf = (1 - pi) * lam**k * np.exp(-lam) / factorial(k)
       pois_pmf = lam**k * np.exp(-lam) / factorial(k)
       print(f"{k:3d}  {zip_pmf:12.6f}  {pois_pmf:16.6f}")

   # The excess zero probability
   pois_zero = np.exp(-lam)
   zip_zero = pi + (1 - pi) * pois_zero
   print(f"\nExcess zeros: ZIP P(0) = {zip_zero:.4f} vs Poisson P(0) = {pois_zero:.4f}")

Moments: :math:`E[X] = (1-\pi)\lambda` and
:math:`\text{Var}(X) = (1-\pi)\lambda(1 + \pi\lambda)`.

.. code-block:: python

   # ZIP moments: simulation check
   np.random.seed(42)
   pi_true, lam_true, n_sim = 0.3, 3.5, 100_000

   is_zero = np.random.binomial(1, pi_true, size=n_sim)
   pois_part = np.random.poisson(lam_true, size=n_sim)
   samples = np.where(is_zero, 0, pois_part)

   EX = (1 - pi_true) * lam_true
   VarX = (1 - pi_true) * lam_true * (1 + pi_true * lam_true)

   print(f"E[X]   = {samples.mean():.4f}  (theory: {EX:.4f})")
   print(f"Var(X) = {samples.var():.4f}  (theory: {VarX:.4f})")
   print(f"Fraction of zeros = {(samples==0).mean():.4f}  "
         f"(theory: {pi_true + (1-pi_true)*np.exp(-lam_true):.4f})")

Log-likelihood
--------------

Partition the data into zeros :math:`Z` (size :math:`n_0`) and positives
:math:`P` (size :math:`n_+`):

.. math::

   \ell(\pi, \lambda)
     = n_0 \ln\!\bigl[\pi + (1-\pi)e^{-\lambda}\bigr]
       + n_+ \ln(1-\pi)
       + \left(\sum_{i \in P} x_i\right)\ln\lambda
       - n_+ \lambda
       - \sum_{i \in P}\ln(x_i!).

Score functions
---------------

Let :math:`A = \pi + (1-\pi)e^{-\lambda}`.

.. math::

   \frac{\partial\ell}{\partial\pi}
     = \frac{n_0(1 - e^{-\lambda})}{A} - \frac{n_+}{1-\pi}.

.. math::

   \frac{\partial\ell}{\partial\lambda}
     = -\frac{n_0(1-\pi)e^{-\lambda}}{A}
       + \frac{\sum_{i \in P} x_i}{\lambda} - n_+.

MLE via EM algorithm
---------------------

No closed-form solution exists. The EM algorithm treats each zero as possibly
structural or Poisson:

**E-step.** Posterior probability that a zero is structural:

.. math::

   w^{(t)} = \frac{\pi^{(t)}}{\pi^{(t)} + (1 - \pi^{(t)})e^{-\lambda^{(t)}}}.

**M-step.** Update:

.. math::

   \pi^{(t+1)} &= \frac{n_0\, w^{(t)}}{n}, \\[4pt]
   \lambda^{(t+1)} &= \frac{\sum_{i \in P} x_i}{n(1 - \pi^{(t+1)})}.

.. code-block:: python

   # ZIP MLE via EM: full implementation with convergence tracking
   import numpy as np

   np.random.seed(42)
   pi_true, lam_true = 0.3, 3.5
   n = 500

   # Simulate ZIP data
   is_structural_zero = np.random.binomial(1, pi_true, size=n)
   poisson_part = np.random.poisson(lam_true, size=n)
   data = np.where(is_structural_zero, 0, poisson_part)

   n0 = (data == 0).sum()
   n_pos = n - n0
   sum_pos = data[data > 0].sum()

   # EM algorithm
   pi_est, lam_est = 0.5, data[data > 0].mean()
   print(f"{'Iter':>4}  {'pi':>8}  {'lambda':>8}  {'loglik':>12}")
   print("-" * 36)

   for t in range(50):
       # E-step
       w = pi_est / (pi_est + (1 - pi_est) * np.exp(-lam_est))
       # M-step
       pi_est = n0 * w / n
       lam_est = sum_pos / (n * (1 - pi_est))
       # Log-likelihood
       A = pi_est + (1 - pi_est) * np.exp(-lam_est)
       ll = (n0 * np.log(A) + n_pos * np.log(1 - pi_est)
             + sum_pos * np.log(lam_est) - n_pos * lam_est)
       if t < 5 or t == 49:
           print(f"{t+1:4d}  {pi_est:8.4f}  {lam_est:8.4f}  {ll:12.2f}")

   print(f"\nMLE: pi = {pi_est:.4f}, lambda = {lam_est:.4f}")
   print(f"True: pi = {pi_true}, lambda = {lam_true}")

Fisher information
------------------

The Fisher information matrix is :math:`2\times 2` and does not simplify to
a clean closed form. We evaluate it numerically at the MLE.

.. code-block:: python

   # ZIP Fisher information: numerical Hessian at the MLE
   import numpy as np

   def zip_loglik(pi_val, lam_val, data):
       n0 = (data == 0).sum()
       n_pos = len(data) - n0
       sum_pos = data[data > 0].sum()
       A = pi_val + (1 - pi_val) * np.exp(-lam_val)
       return (n0 * np.log(A) + n_pos * np.log(1 - pi_val)
               + sum_pos * np.log(lam_val) - n_pos * lam_val)

   # Numerical Hessian via finite differences
   eps = 1e-5
   params = [pi_est, lam_est]
   H = np.zeros((2, 2))
   for i in range(2):
       for j in range(2):
           pp = params.copy(); pp[i] += eps; pp[j] += eps
           pm = params.copy(); pm[i] += eps; pm[j] -= eps
           mp = params.copy(); mp[i] -= eps; mp[j] += eps
           mm = params.copy(); mm[i] -= eps; mm[j] -= eps
           H[i, j] = (zip_loglik(*pp, data) - zip_loglik(*pm, data)
                       - zip_loglik(*mp, data) + zip_loglik(*mm, data)) / (4 * eps**2)

   # Observed information = -H
   obs_info = -H
   cov_matrix = np.linalg.inv(obs_info)
   se_pi = np.sqrt(cov_matrix[0, 0])
   se_lam = np.sqrt(cov_matrix[1, 1])

   print(f"MLE pi = {pi_est:.4f}, SE = {se_pi:.4f}, "
         f"95% CI = [{pi_est - 1.96*se_pi:.4f}, {pi_est + 1.96*se_pi:.4f}]")
   print(f"MLE lam = {lam_est:.4f}, SE = {se_lam:.4f}, "
         f"95% CI = [{lam_est - 1.96*se_lam:.4f}, {lam_est + 1.96*se_lam:.4f}]")
   print(f"True: pi = {pi_true}, lambda = {lam_true}")


.. _sec_discrete_summary:

5.9 Summary and Comparison
============================

The table below summarises the MLE and Fisher information for each
distribution covered in this chapter.

.. list-table:: Discrete Distributions --- MLEs and Fisher Information
   :header-rows: 1
   :widths: 20 30 30

   * - Distribution
     - MLE
     - Fisher information (single obs.)
   * - Bernoulli(:math:`p`)
     - :math:`\hat{p} = \bar{x}`
     - :math:`1/[p(1-p)]`
   * - Binomial(:math:`m,p`)
     - :math:`\hat{p} = \bar{x}/m`
     - :math:`m/[p(1-p)]`
   * - Poisson(:math:`\lambda`)
     - :math:`\hat{\lambda} = \bar{x}`
     - :math:`1/\lambda`
   * - NegBin(:math:`r,p`), :math:`r` known
     - :math:`\hat{p} = r/(r+\bar{x})`
     - :math:`r/[p^2(1-p)]`
   * - Geometric(:math:`p`)
     - :math:`\hat{p} = 1/(1+\bar{x})`
     - :math:`1/[p^2(1-p)]`
   * - Multinomial(:math:`\mathbf{p}`)
     - :math:`\hat{p}_j = S_j/(nm)`
     - see text
   * - ZIP(:math:`\pi,\lambda`)
     - EM algorithm
     - numerical evaluation

Let's verify every MLE and SE in one unified comparison table.

.. code-block:: python

   # Summary: all discrete MLEs side by side
   import numpy as np

   np.random.seed(42)

   results = []

   # Bernoulli
   p_true = 0.7; n = 200
   data = np.random.binomial(1, p_true, size=n)
   p_hat = data.mean()
   se = np.sqrt(p_hat * (1 - p_hat) / n)
   results.append(("Bernoulli", "p", p_true, p_hat, se))

   # Binomial
   m, p_true, n = 10, 0.65, 50
   data = np.random.binomial(m, p_true, size=n)
   p_hat = data.sum() / (n * m)
   se = np.sqrt(p_hat * (1 - p_hat) / (n * m))
   results.append(("Binomial", "p", p_true, p_hat, se))

   # Poisson
   lam_true, n = 4.5, 200
   data = np.random.poisson(lam_true, size=n)
   lam_hat = data.mean()
   se = np.sqrt(lam_hat / n)
   results.append(("Poisson", "lam", lam_true, lam_hat, se))

   # NegBin (r known)
   r, p_true, n = 5, 0.4, 150
   data = np.random.negative_binomial(r, p_true, size=n)
   p_hat = r / (r + data.mean())
   se = 1.0 / np.sqrt(n * r / (p_hat**2 * (1 - p_hat)))
   results.append(("NegBin", "p", p_true, p_hat, se))

   # Geometric
   p_true, n = 0.3, 200
   data = np.random.geometric(p_true, size=n) - 1
   p_hat = 1 / (1 + data.mean())
   se = 1.0 / np.sqrt(n / (p_hat**2 * (1 - p_hat)))
   results.append(("Geometric", "p", p_true, p_hat, se))

   print(f"{'Dist':<12} {'Param':<6} {'True':>6} {'MLE':>8} {'SE':>8} {'95% CI':>20}")
   print("-" * 64)
   for name, param, true, mle, se in results:
       ci = f"[{mle-1.96*se:.4f}, {mle+1.96*se:.4f}]"
       print(f"{name:<12} {param:<6} {true:6.3f} {mle:8.4f} {se:8.4f} {ci:>20}")

Key patterns to notice:

* For single-parameter families the MLE is almost always a simple function of
  the sample mean.
* Fisher information is inversely related to the variance of the sufficient
  statistic.
* The Bernoulli/Binomial/Geometric/Negative-Binomial family shares the same
  algebraic skeleton; the differences arise from how trials are aggregated.

.. admonition:: Common Pitfall

   Do not assume the Poisson is always the right model for count data. If your
   data show more zeros than expected, consider the ZIP. If the variance
   exceeds the mean (overdispersion), consider the Negative Binomial. The
   sample variance-to-mean ratio is a quick diagnostic.

   .. code-block:: python

      # Quick diagnostic: which model fits your count data?
      import numpy as np

      np.random.seed(42)
      counts = np.random.negative_binomial(3, 0.25, size=200)

      mean_x = counts.mean()
      var_x = counts.var()
      frac_zero = (counts == 0).mean()
      poisson_zero = np.exp(-mean_x)

      print(f"Sample mean     = {mean_x:.2f}")
      print(f"Sample variance = {var_x:.2f}")
      print(f"Var/Mean ratio  = {var_x/mean_x:.2f}")
      print(f"Observed zeros  = {frac_zero:.3f}")
      print(f"Poisson zeros   = {poisson_zero:.3f}")
      print()
      if var_x / mean_x > 1.5:
          print("=> Overdispersion detected: consider Negative Binomial")
      if frac_zero > 2 * poisson_zero:
          print("=> Excess zeros detected: consider ZIP model")
