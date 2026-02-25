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
   :class: this-will-duplicate-information-and-it-is-still-useful-here


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

This compact formula is a clever notational trick that packs two cases into
one expression.  When :math:`x = 1` (success), the exponent on :math:`p` is 1
and the exponent on :math:`(1-p)` is 0, so the formula reduces to just
:math:`p`.  When :math:`x = 0` (failure), the exponent on :math:`p` is 0
(making it 1) and the exponent on :math:`(1-p)` is 1, so the formula gives
:math:`1-p`.  The beauty of this trick is that it lets us write a *single*
algebraic expression that we can differentiate, multiply, and take logarithms
of --- exactly what we need for likelihood-based inference.

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

Given :math:`n` i.i.d. observations :math:`x_1, \dots, x_n`, how do we build
the likelihood?  Because the observations are independent, the joint
probability is the product of the individual probabilities:

.. math::

   L(p) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i}
        = p^{\,\sum x_i}\,(1-p)^{\,n - \sum x_i}.

In the second step we used the rule :math:`a^{b_1} \cdot a^{b_2} = a^{b_1+b_2}`
to collect all the powers of :math:`p` and :math:`(1-p)`.

Define :math:`s = \sum_{i=1}^{n} x_i`, the total number of successes.
Taking the natural logarithm (which turns products into sums and is
monotonically increasing, so it does not change where the maximum occurs):

.. math::

   \ell(p) = \ln L(p) = s \ln p + (n - s) \ln(1 - p).

Notice that the entire dataset collapses into a single number :math:`s` --- the
**sufficient statistic**.  Whether you observed 70 ones and 30 zeros, or the
same 70 ones in a completely different order, the log-likelihood is identical.
This is a hallmark of exponential-family distributions.

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

The **score function** is the derivative of the log-likelihood with respect to
the parameter.  It tells us the slope of the log-likelihood curve: when the
score is positive, increasing :math:`p` would increase the log-likelihood;
when negative, we should decrease :math:`p`; and at the maximum, the score is
exactly zero.

Differentiating :math:`\ell(p) = s \ln p + (n - s) \ln(1 - p)` term by term
using the chain rule:

.. math::

   S(p) = \frac{d\ell}{dp}
        = \frac{d}{dp}\bigl[s \ln p\bigr]
          + \frac{d}{dp}\bigl[(n-s) \ln(1-p)\bigr]
        = \frac{s}{p} - \frac{n-s}{1-p}.

The first term :math:`s/p` represents the "pull" of the observed successes ---
it is always positive and pushes the estimate upward.  The second term
:math:`-(n-s)/(1-p)` represents the "pull" of the observed failures --- it is
always negative and pushes the estimate downward.  At the MLE, these two
forces are in perfect balance.

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

The **Fisher information** measures how much information a single observation
carries about the parameter :math:`p`.  It is defined as the negative expected
value of the second derivative of the log-likelihood (the expected curvature
of the log-likelihood curve).

For a single Bernoulli observation, the log-likelihood is
:math:`\ell_1(p) = x \ln p + (1-x)\ln(1-p)`, so the second derivative is:

.. math::

   \frac{d^2\ell_1}{dp^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}.

Taking the expected value (using :math:`E[X] = p`):

.. math::

   \mathcal{I}(p) = -E\!\left[\frac{d^2\ell_1}{dp^2}\right]
   = \frac{p}{p^2} + \frac{1-p}{(1-p)^2}
   = \frac{1}{p} + \frac{1}{1-p}
   = \frac{1}{p(1-p)}.

For :math:`n` independent observations the total information simply scales:
:math:`n\,\mathcal{I}(p) = \dfrac{n}{p(1-p)}`.

.. admonition:: Intuition

   The Fisher information is largest when :math:`p` is near 0 or 1 (extreme
   probabilities) and smallest when :math:`p = 0.5`. When almost every trial
   gives the same result, each observation is highly informative about
   :math:`p`. When outcomes are maximally uncertain, each observation tells
   you relatively less.

MLE and inference
------------------

To find the maximum likelihood estimate, we set the score to zero and solve:

.. math::

   \frac{s}{p} - \frac{n-s}{1-p} = 0.

Cross-multiplying: :math:`s(1-p) = (n-s)p`, which gives :math:`s - sp = np - sp`,
so :math:`s = np`, and therefore:

.. math::

   \hat{p}_{\text{MLE}} = \frac{s}{n} = \bar{x}.

The MLE is simply the sample proportion --- exactly the estimator your
intuition would suggest.  This is reassuring: the formal machinery of
likelihood confirms the common-sense answer.

The **standard error** measures how much :math:`\hat{p}` would vary across
repeated samples.  It comes from the Fisher information via
:math:`\text{SE} = 1/\sqrt{n\,\mathcal{I}(\hat{p})}`:

.. math::

   \text{SE} = \frac{1}{\sqrt{n/({\hat{p}(1-\hat{p})})}}
             = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.

This formula tells us that uncertainty shrinks with more data (:math:`\sqrt{n}`
in the denominator) and is largest when :math:`\hat{p} = 0.5` (maximum
uncertainty about the outcome).

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

Let :math:`X \sim \text{Binomial}(m, p)` where :math:`m` is the (known) number
of trials and :math:`p \in (0,1)` is the probability of success on each trial:

.. math::

   f(x \mid p) = \binom{m}{x} p^x (1-p)^{m-x}, \qquad x = 0, 1, \dots, m.

Reading this formula piece by piece: :math:`p^x` is the probability that the
:math:`x` successes all happened, :math:`(1-p)^{m-x}` is the probability that
the remaining :math:`m-x` trials were all failures, and
:math:`\binom{m}{x} = \frac{m!}{x!(m-x)!}` counts the number of ways to
choose *which* :math:`x` of the :math:`m` trials are the successes.  The
product of these three terms gives the total probability of getting exactly
:math:`x` successes in any order.

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

For :math:`n` independent observations :math:`x_1, \dots, x_n`, each drawn
from :math:`\text{Binomial}(m, p)`, the joint likelihood is the product of
the individual PMFs.  The binomial coefficients :math:`\binom{m}{x_i}` do not
depend on :math:`p`, so they are constant for the purpose of maximization.
Collecting the powers of :math:`p` and :math:`(1-p)` as we did for the
Bernoulli, and letting :math:`S = \sum_{i=1}^{n} x_i` (the total number of
successes across all patients), the parameter-dependent part of the
log-likelihood is

.. math::

   \ell(p) \propto S \ln p + (nm - S)\ln(1-p).

The :math:`\propto` symbol means "equal up to an additive constant that does
not depend on :math:`p`."  Notice this has *exactly* the same form as the
Bernoulli log-likelihood, but with :math:`s \to S` and :math:`n \to nm`.
This makes sense: :math:`n` patients each with :math:`m` sessions gives
:math:`nm` total Bernoulli trials, of which :math:`S` were successes.

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

Differentiating the log-likelihood with respect to :math:`p` (exactly the
same differentiation we performed for the Bernoulli):

.. math::

   S(p) = \frac{d\ell}{dp} = \frac{S}{p} - \frac{nm - S}{1-p}.

This has exactly the same form as the Bernoulli score with
:math:`s \to S` and :math:`n \to nm` --- because the Binomial *is*
a collection of Bernoulli trials.

.. code-block:: python

   # Binomial score: verify score = 0 at MLE
   p_hat = S / (n * m)
   score_at_mle = S / p_hat - (n * m - S) / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

Since each Binomial observation is the sum of :math:`m` independent
Bernoulli trials, its Fisher information is simply :math:`m` times the
Bernoulli information:

.. math::

   \mathcal{I}(p) = \frac{m}{p(1-p)}.

More trials per observation means more information per observation.
For :math:`n` observations: :math:`n\mathcal{I}(p) = \dfrac{nm}{p(1-p)}`.

MLE and inference
------------------

Setting the score to zero and solving (the same algebra as for the Bernoulli,
with :math:`nm` in place of :math:`n`):

.. math::

   \hat{p} = \frac{S}{nm} = \frac{\bar{x}}{m}.

The MLE is the total number of successes divided by the total number of
trials.  Equivalently, it is the average count :math:`\bar{x}` divided by
the number of trials per observation :math:`m`.

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

Reading this formula: :math:`\lambda^x` grows with :math:`x` --- more events
become more likely as the rate increases.  The :math:`e^{-\lambda}` factor
ensures probabilities sum to one (it comes from the Taylor series
:math:`e^{\lambda} = \sum_{x=0}^{\infty} \lambda^x / x!`).  The :math:`x!`
in the denominator accounts for the fact that the order in which events arrive
does not matter.  The single parameter :math:`\lambda` controls both where
the distribution is centred and how spread out it is.

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

For :math:`n` i.i.d. observations, the joint likelihood is the product
:math:`L(\lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}`.
Taking the logarithm and separating terms that depend on :math:`\lambda`
from those that don't:

.. math::

   \ell(\lambda)
     = \left(\sum_{i=1}^{n} x_i\right)\ln\lambda - n\lambda
       - \underbrace{\sum_{i=1}^{n}\ln(x_i!)}_{\text{constant in }\lambda}.

The first term comes from :math:`\ln(\lambda^{x_i}) = x_i \ln\lambda` summed
over all observations.  The second term comes from the :math:`n` factors of
:math:`e^{-\lambda}`, each contributing :math:`-\lambda`.  The last term
involves only the data and disappears when we differentiate.

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

Differentiating with respect to :math:`\lambda` (the :math:`x_i!` term
vanishes since it does not depend on :math:`\lambda`):

.. math::

   S(\lambda) = \frac{d\ell}{d\lambda}
   = \frac{\sum x_i}{\lambda} - n.

The first term is a "pull upward" proportional to the total count; the second
is a constant "pull downward."  When :math:`\lambda` equals the sample mean,
these forces balance.

.. code-block:: python

   # Poisson score: verify score = 0 at MLE
   lam_hat = data.mean()
   score_at_mle = data.sum() / lam_hat - n
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

The second derivative of the single-observation log-likelihood
:math:`\ell_1(\lambda) = x\ln\lambda - \lambda` is
:math:`d^2\ell_1/d\lambda^2 = -x/\lambda^2`.  Taking the negative expected
value (with :math:`E[X] = \lambda`):

.. math::

   \mathcal{I}(\lambda) = -E\!\left[-\frac{X}{\lambda^2}\right]
   = \frac{E[X]}{\lambda^2} = \frac{\lambda}{\lambda^2}
   = \frac{1}{\lambda}.

For :math:`n` observations: :math:`n\mathcal{I}(\lambda) = n/\lambda`.

.. admonition:: Intuition

   Larger rates are harder to pin down precisely. When events are rare
   (small :math:`\lambda`), each count is very informative. When events are
   frequent, you need more data for the same relative precision.

MLE and inference
------------------

Setting the score to zero: :math:`\sum x_i / \lambda = n`, so

.. math::

   \hat{\lambda} = \frac{\sum x_i}{n} = \bar{x}.

The MLE for the Poisson rate is simply the sample mean --- again, the
intuitive estimator.  The standard error follows from
:math:`\text{SE} = 1/\sqrt{n\mathcal{I}(\hat\lambda)}`:

.. math::

   \text{SE} = \sqrt{\frac{\hat{\lambda}}{n}}.

Because the Poisson has mean equal to variance (:math:`E[X] = \text{Var}(X) = \lambda`),
the SE of :math:`\hat{\lambda}` is just the usual "standard deviation divided by
:math:`\sqrt{n}`" formula.

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
where :math:`r > 0` is known and :math:`p \in (0,1)` is the success
probability:

.. math::

   f(x \mid r, p) = \binom{x + r - 1}{x} p^r (1-p)^x,
   \qquad x = 0, 1, 2, \dots

Here :math:`p^r` is the probability of the :math:`r` required successes,
:math:`(1-p)^x` is the probability of :math:`x` failures, and the
binomial coefficient :math:`\binom{x + r - 1}{x}` counts the number of
distinct orderings of :math:`x` failures and :math:`r - 1` non-final
successes (the last trial must be a success, hence :math:`r - 1`).

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

For :math:`n` i.i.d. observations with :math:`r` known, we multiply the PMFs
and take the log.  The binomial coefficients do not depend on :math:`p` and
become part of the constant.  Collecting the powers of :math:`p` and
:math:`(1-p)`:

.. math::

   \ell(p) = \text{const} + nr\ln p + \left(\sum_{i=1}^{n} x_i\right)\ln(1-p).

This has a familiar structure: :math:`\ln p` multiplied by "total successes"
(:math:`nr`, since :math:`r` successes per observation) and :math:`\ln(1-p)`
multiplied by "total failures" (:math:`\sum x_i`).

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

Differentiating the log-likelihood:

.. math::

   S(p) = \frac{d\ell}{dp} = \frac{nr}{p} - \frac{\sum x_i}{1-p}.

Again the same pattern: a term pulling :math:`p` up (from successes) and a
term pulling it down (from failures).

.. code-block:: python

   # NegBin score: verify score = 0 at MLE
   p_hat = r / (r + data.mean())
   score_at_mle = n * r / p_hat - data.sum() / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

Taking the second derivative of the single-observation log-likelihood
:math:`\ell_1 = r\ln p + x\ln(1-p)` gives
:math:`d^2\ell_1/dp^2 = -r/p^2 - x/(1-p)^2`.  Taking the negative
expectation with :math:`E[X] = r(1-p)/p`:

.. math::

   \mathcal{I}(p) = \frac{r}{p^2} + \frac{r(1-p)/p}{(1-p)^2}
   = \frac{r}{p^2} + \frac{r}{p(1-p)}
   = \frac{r}{p^2(1-p)}.

MLE and inference
------------------

Setting the score to zero: :math:`nr/p = \sum x_i/(1-p)`, which gives
:math:`nr(1-p) = p\sum x_i`.  Since :math:`\sum x_i = n\bar{x}`, we get
:math:`r(1-p) = p\bar{x}`, so :math:`r = p(r + \bar{x})`, and therefore:

.. math::

   \hat{p} = \frac{r}{r + \bar{x}}.

This formula makes intuitive sense: if the average number of failures
:math:`\bar{x}` is large relative to :math:`r`, then :math:`p` must be
small (each trial rarely succeeds).

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

The logic is straightforward: to see :math:`x` failures followed by a success,
you need :math:`x` consecutive failures (each with probability :math:`1-p`)
and then one success (probability :math:`p`).  There is no combinatorial
factor because the order is fixed --- all failures come first, then the
success.

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

The likelihood is :math:`L(p) = \prod_i (1-p)^{x_i} p = p^n (1-p)^{\sum x_i}`.
Taking the log:

.. math::

   \ell(p) = n\ln p + \left(\sum_{i=1}^{n} x_i\right)\ln(1-p).

This is the Negative Binomial log-likelihood with :math:`r = 1`: one "success"
per observation, so the total number of successes is just :math:`n`.

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

Differentiating:

.. math::

   S(p) = \frac{n}{p} - \frac{\sum x_i}{1-p}.

This is the Negative Binomial score with :math:`r = 1`.

.. code-block:: python

   # Geometric score: verify score = 0 at MLE
   p_hat = 1 / (1 + data.mean())
   score_at_mle = n / p_hat - data.sum() / (1 - p_hat)
   print(f"Score at MLE = {score_at_mle:.10f}")

Fisher information
------------------

The Negative Binomial Fisher information with :math:`r = 1`:

.. math::

   \mathcal{I}(p) = \frac{1}{p^2(1-p)}.

MLE and inference
------------------

Setting the score to zero gives :math:`n/p = \sum x_i/(1-p)`, so
:math:`n(1-p) = p\sum x_i = np\bar{x}`, hence :math:`1 - p = p\bar{x}`,
and therefore :math:`1 = p(1 + \bar{x})`:

.. math::

   \hat{p} = \frac{1}{1 + \bar{x}}.

If you observe many failures on average before each success (large
:math:`\bar{x}`), the estimated success probability is small, as expected.

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

This formula counts favourable outcomes over total outcomes.  The numerator
says: choose :math:`x` tagged fish from the :math:`K` tagged ones
(:math:`\binom{K}{x}` ways), and choose the remaining :math:`n - x` fish
from the :math:`N - K` untagged ones (:math:`\binom{N-K}{n-x}` ways).
The denominator :math:`\binom{N}{n}` is the total number of ways to draw
:math:`n` fish from the lake.  Unlike the Binomial, this distribution does
*not* assume independence between draws --- each fish you pull out changes
the composition of the remaining population.

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

Because :math:`K` is an integer parameter (a count of tagged fish), we cannot
use calculus in the usual way.  Instead we maximise the PMF over the integer
lattice :math:`K \in \{0, 1, \dots, N\}`.  It can be shown that the PMF
ratio :math:`f(x \mid K) / f(x \mid K-1)` is greater than 1 when
:math:`K < (N+1)x/n` and less than 1 when :math:`K > (N+1)x/n`, so the
peak occurs near the "break-even" point:

.. math::

   \hat{K} = \left\lfloor \frac{(N+1)\,x}{n} \right\rfloor.

This formula is the discrete version of the "proportion times population"
intuition: if :math:`x` out of :math:`n` sampled fish are tagged, the
population should have roughly that same proportion tagged.

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

This generalises the Binomial in a natural way.  The product
:math:`p_1^{x_1}\cdots p_k^{x_k}` gives the probability of one specific
sequence of category assignments.  The multinomial coefficient
:math:`\frac{m!}{x_1!\cdots x_k!}` counts how many distinct orderings
produce the same count vector :math:`\mathbf{x}` --- just as
:math:`\binom{m}{x}` did for two categories.

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

For :math:`n` independent observations, define :math:`S_j = \sum_{i=1}^{n} x_j^{(i)}`
(the total count for category :math:`j` across all observations).  The
multinomial coefficients are constant with respect to :math:`\mathbf{p}`, so:

.. math::

   \ell(\mathbf{p}) = \text{const} + \sum_{j=1}^{k} S_j \ln p_j.

Each category contributes independently to the log-likelihood, weighted by
how many times it was observed.

MLE via Lagrange multipliers
-----------------------------

We want to maximize :math:`\ell(\mathbf{p})` subject to the constraint
:math:`\sum_j p_j = 1`.  Using a Lagrange multiplier :math:`\mu`, we set
:math:`\partial/\partial p_j [S_j \ln p_j - \mu p_j] = 0`, giving
:math:`S_j/p_j = \mu` for all :math:`j`.  This means
:math:`p_j \propto S_j`.  Applying the constraint :math:`\sum p_j = 1`
forces :math:`\mu = \sum S_j = nm`, so:

.. math::

   \hat{p}_j = \frac{S_j}{nm}.

The MLE for each category probability is simply the observed relative
frequency --- the fraction of all votes that went to each candidate.

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
simply the derivative of :math:`S_j \ln p_j`:

.. math::

   S_j(p_j) = \frac{S_j}{p_j}.

This is always positive, reflecting the fact that the unconstrained
maximizer would send each :math:`p_j \to \infty`.  It is only the
constraint :math:`\sum p_j = 1` that forces the solution to a finite value.

Fisher information
------------------

Because of the constraint :math:`\sum p_j = 1`, we can only freely vary
:math:`k - 1` of the :math:`k` probabilities (the last one is determined).
The Fisher information matrix for a single multinomial observation, after
eliminating :math:`p_k = 1 - \sum_{j<k} p_j`, has entries:

.. math::

   \mathcal{I}_{jl} = \frac{m\,\delta_{jl}}{p_j} + \frac{m}{p_k},
   \qquad j, l = 1, \dots, k-1.

The diagonal term :math:`m/p_j` comes from the curvature of
:math:`\ln p_j`, and the :math:`m/p_k` term arises because changing any
:math:`p_j` implicitly changes :math:`p_k`.  The off-diagonal entries are
all equal to :math:`m/p_k`, reflecting the coupling through the constraint.

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

This is a **mixture model**.  With probability :math:`\pi`, the observation
is a "structural zero" (it was never going to produce any events).  With
probability :math:`1 - \pi`, it comes from a standard Poisson process.
The :math:`x = 0` case has two sources: either the observation is a structural
zero (probability :math:`\pi`), or it came from the Poisson component but
happened to produce zero events (probability :math:`(1-\pi)e^{-\lambda}`).
For :math:`x \geq 1`, only the Poisson component can produce positive counts.

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
:math:`P` (size :math:`n_+`).  The zero observations each contribute
:math:`\ln[\pi + (1-\pi)e^{-\lambda}]` to the log-likelihood, while the
positive observations each contribute
:math:`\ln[(1-\pi) \lambda^{x_i} e^{-\lambda}/x_i!]`.  Collecting terms:

.. math::

   \ell(\pi, \lambda)
     = n_0 \ln\!\bigl[\pi + (1-\pi)e^{-\lambda}\bigr]
       + n_+ \ln(1-\pi)
       + \left(\sum_{i \in P} x_i\right)\ln\lambda
       - n_+ \lambda
       - \sum_{i \in P}\ln(x_i!).

The first term is what makes this problem hard: it involves a logarithm of
a *sum*, which does not simplify nicely.  This is why no closed-form MLE
exists and we need the EM algorithm.

Score functions
---------------

Let :math:`A = \pi + (1-\pi)e^{-\lambda}` denote the probability of
observing a zero.  Differentiating :math:`\ell` with respect to each
parameter:

.. math::

   \frac{\partial\ell}{\partial\pi}
     = \frac{n_0(1 - e^{-\lambda})}{A} - \frac{n_+}{1-\pi}.

The first term asks: "given the observed zeros, how much would increasing
the structural-zero probability :math:`\pi` help explain them?"  The second
term is the cost: increasing :math:`\pi` makes it harder to explain the
positive observations.

.. math::

   \frac{\partial\ell}{\partial\lambda}
     = -\frac{n_0(1-\pi)e^{-\lambda}}{A}
       + \frac{\sum_{i \in P} x_i}{\lambda} - n_+.

The first term accounts for zeros that came from the Poisson component
(increasing :math:`\lambda` makes Poisson zeros less likely).  The remaining
terms are the standard Poisson score applied to the positive observations.

MLE via EM algorithm
---------------------

No closed-form solution exists because the :math:`\log(\text{sum})` term
in the log-likelihood cannot be separated.  The EM (Expectation-Maximization)
algorithm sidesteps this by introducing a **latent variable**: for each
zero observation, we pretend we know whether it was a structural zero or
a Poisson zero, then alternate between two steps.

**E-step ("guess the hidden labels").** Given current estimates
:math:`\pi^{(t)}` and :math:`\lambda^{(t)}`, compute the posterior
probability that each zero is structural (using Bayes' theorem):

.. math::

   w^{(t)} = \frac{\pi^{(t)}}{\pi^{(t)} + (1 - \pi^{(t)})e^{-\lambda^{(t)}}}.

This is just Bayes' rule: prior probability of structural zero
(:math:`\pi`) divided by total probability of observing zero (:math:`A`).

**M-step ("re-estimate parameters as if the labels were known").**
Treating :math:`w^{(t)}` as the fraction of zeros that are structural:

.. math::

   \pi^{(t+1)} &= \frac{n_0\, w^{(t)}}{n}, \\[4pt]
   \lambda^{(t+1)} &= \frac{\sum_{i \in P} x_i}{n(1 - \pi^{(t+1)})}.

The first update says: the new :math:`\pi` is the expected number of
structural zeros divided by :math:`n`.  The second says: the Poisson rate
is the total count from positive observations divided by the expected number
of Poisson-component observations.  These two steps are iterated until
convergence.

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
