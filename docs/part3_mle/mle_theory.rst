.. _ch9_mle_theory:

==========================================
Chapter 9 --- MLE Theory
==========================================

This chapter develops the theoretical foundations of maximum likelihood
estimation (MLE). We begin with a precise definition, then establish the
key large-sample properties---consistency, asymptotic normality, and
efficiency---that make MLE the workhorse of parametric inference. Along
the way we prove the invariance property, catalogue the regularity
conditions that underpin the asymptotic results, and discuss the
finite-sample behaviour of MLEs including their potential for bias.

**Running example: estimating vaccine efficacy.**
Throughout this chapter we return to a single concrete problem. A vaccine
has true efficacy :math:`p_0 = 0.75`, meaning 75% of vaccinated
individuals develop immunity. We observe binary outcomes---immune or
not---from a clinical trial and use MLE to learn :math:`p_0`. Every
theorem will be verified against this scenario, so you can see the
theory come alive in one coherent story.

If you take away one message from this chapter, let it be this: the MLE
is not just *a* reasonable estimator---under mild conditions it is, in a
precise sense, the *best* estimator you can construct from the likelihood
alone. The theorems below make that claim rigorous.


9.1 Formal Definition of the Maximum Likelihood Estimator
==========================================================

A pharmaceutical company has just run a clinical trial. They vaccinated
:math:`n` people and recorded, for each person, whether they developed
immunity (1) or not (0). The company needs to estimate the true efficacy
rate :math:`p`. What is the *principled* way to do this?

Suppose we observe data :math:`x_1, x_2, \ldots, x_n` that we model as
realisations of independent, identically distributed (i.i.d.) random
variables :math:`X_1, \ldots, X_n`, each with probability density (or
mass) function :math:`f(x \mid \theta)`, where the parameter
:math:`\theta` lives in a parameter space :math:`\Theta \subseteq
\mathbb{R}^p`.

The **likelihood function** is the joint density viewed as a function of
:math:`\theta` for fixed data:

.. math::

   L(\theta) \;=\; L(\theta \mid x_1, \ldots, x_n)
   \;=\; \prod_{i=1}^{n} f(x_i \mid \theta).

Here is the key idea: the likelihood tells you how "surprised" the model
would be by your data for each possible parameter value. High likelihood
means "this parameter makes the data look plausible."

Because products are unwieldy, we almost always work with the
**log-likelihood**:

.. math::

   \ell(\theta) \;=\; \log L(\theta) \;=\; \sum_{i=1}^{n} \log f(x_i \mid \theta).

Taking the log converts the product into a sum, which is far easier to
differentiate and reason about. Since the logarithm is a monotonically
increasing function, maximising :math:`\ell` is equivalent to maximising
:math:`L`.

.. topic:: Definition (Maximum Likelihood Estimator)

   The **maximum likelihood estimator** (MLE) is any value
   :math:`\hat{\theta}_n` that maximises the likelihood (equivalently,
   the log-likelihood) over the parameter space:

   .. math::

      \hat{\theta}_n
      \;=\; \arg\max_{\theta \in \Theta}\; L(\theta)
      \;=\; \arg\max_{\theta \in \Theta}\; \ell(\theta).

When the log-likelihood is differentiable and the maximum occurs in the
interior of :math:`\Theta`, the MLE satisfies the **likelihood
equation** (or score equation):

.. math::

   S(\theta) \;=\; \frac{\partial \ell(\theta)}{\partial \theta}
   \;=\; \sum_{i=1}^{n} \frac{\partial}{\partial \theta}
   \log f(x_i \mid \theta) \;=\; 0.

The function :math:`S(\theta)` is called the **score function**. It
plays a central role throughout this chapter and in :ref:`ch11_testing`.

.. admonition:: Intuition

   The score function measures the slope of the log-likelihood. At the
   MLE, the slope is zero---you are standing at the peak of the hill.
   Every theorem in this chapter ultimately traces back to the behaviour
   of this slope function.

Let's see MLE in action on our vaccine trial. We vaccinate 50 people and
record whether each develops immunity. The Bernoulli log-likelihood is
:math:`\ell(p) = k \log p + (n - k) \log(1 - p)`, where :math:`k` is
the number of immune individuals. Setting the derivative to zero gives
:math:`\hat{p} = k/n`---the sample proportion.

.. code-block:: python

   # Vaccine trial MLE: the sample proportion maximises the Bernoulli likelihood
   import numpy as np

   np.random.seed(42)
   p_0 = 0.75          # true vaccine efficacy
   n = 50               # trial participants
   data = np.random.binomial(1, p_0, size=n)

   # MLE for Bernoulli: p_hat = sample proportion
   k = np.sum(data)
   mle_p = k / n

   # Verify: score = k/p - (n-k)/(1-p) should be zero at the MLE
   score_at_mle = k / mle_p - (n - k) / (1 - mle_p)

   # Log-likelihood at MLE vs a few other values
   def bernoulli_loglik(p, k, n):
       return k * np.log(p) + (n - k) * np.log(1 - p)

   print(f"Trial data: {k} immune out of {n} vaccinated")
   print(f"MLE of p:   {mle_p:.4f}  (true p = {p_0})")
   print(f"Score at MLE: {score_at_mle:.10f}  (should be ~0)")
   print()
   print("Log-likelihood at various p values:")
   for p_test in [0.50, 0.60, 0.70, mle_p, 0.80, 0.90]:
       ll = bernoulli_loglik(p_test, k, n)
       marker = " <-- MLE" if p_test == mle_p else ""
       print(f"  p = {p_test:.4f}:  ell = {ll:.4f}{marker}")


9.2 Existence and Uniqueness
==============================

The MLE does not always exist, and when it does exist it need not be
unique. Understanding when and why these problems arise is essential for
applied work.

9.2.1 When the MLE May Not Exist
----------------------------------

1. **Unbounded likelihood.** If the likelihood can be made arbitrarily
   large, no finite maximum exists. A classic example is the
   two-component normal mixture

   .. math::

      f(x \mid \mu_1, \mu_2, \sigma_1, \sigma_2, \pi)
      = \pi\,\phi(x;\mu_1,\sigma_1)
        + (1-\pi)\,\phi(x;\mu_2,\sigma_2).

   Setting :math:`\mu_1 = x_1` (one of the data points) and letting
   :math:`\sigma_1 \to 0` sends the likelihood to infinity.

2. **Empty interior.** If the parameter space has no interior points
   (for example, a discrete or lower-dimensional set), a smooth maximum
   may not be well-defined.

3. **Separation in logistic regression.** When a linear combination of
   predictors perfectly separates the two classes, the coefficient
   estimates diverge to :math:`\pm\infty`.

.. admonition:: Common Pitfall

   In practice, the most frequent cause of MLE failure you will encounter
   is *separation* in logistic regression. If your optimiser reports
   "convergence failed" or parameter estimates that seem absurdly large,
   check whether your binary outcome can be perfectly predicted by one or
   more covariates. Firth's penalised likelihood or Bayesian priors are
   standard remedies.

9.2.2 When the MLE May Not Be Unique
--------------------------------------

If the log-likelihood has multiple global maxima, the MLE is not unique.
This happens, for example, with multimodal likelihoods in mixture models
where label switching creates symmetric modes.

9.2.3 Sufficient Conditions for Existence and Uniqueness
---------------------------------------------------------

The most convenient condition is **strict concavity** of the
log-likelihood. If :math:`\ell(\theta)` is strictly concave on
:math:`\Theta` and :math:`\ell(\theta) \to -\infty` as
:math:`\|\theta\| \to \infty`, then there exists exactly one global
maximum. Many exponential-family models enjoy this property. Formally,
if the model belongs to a *full-rank* exponential family with a compact
or appropriately bounded parameter space, the MLE exists and is unique
almost surely once the sample size exceeds the dimension :math:`p`.

For our vaccine problem (Bernoulli), the log-likelihood
:math:`\ell(p) = k \log p + (n-k)\log(1-p)` is strictly concave on
:math:`(0,1)` whenever :math:`0 < k < n`, guaranteeing a unique MLE.

.. code-block:: python

   # Verify strict concavity of the Bernoulli log-likelihood
   import numpy as np

   k, n = 38, 50  # from our vaccine trial
   p_grid = np.linspace(0.01, 0.99, 500)
   loglik = k * np.log(p_grid) + (n - k) * np.log(1 - p_grid)

   # Second derivative: d^2 ell / dp^2 = -k/p^2 - (n-k)/(1-p)^2
   d2ell = -k / p_grid**2 - (n - k) / (1 - p_grid)**2

   print(f"Second derivative is always negative on (0,1)?")
   print(f"  max(d^2 ell / dp^2) = {np.max(d2ell):.4f}")
   print(f"  => Strictly concave, so MLE is unique.")


9.3 Consistency
================

Returning to our vaccine trial: if we vaccinate 20 people, our estimate
of efficacy might be off by quite a bit. But what if we run a trial with
20 000 people? Intuitively, the estimate should be nearly perfect.
**Consistency** is the formal guarantee that this intuition is correct.

Informally, consistency means that the MLE converges to the true
parameter value as the sample size grows. This is the most basic
desirable property of any estimator.

Why does this matter? An estimator that is not consistent is
fundamentally unreliable: no matter how much data you collect, you cannot
trust that it will eventually give you the right answer. Consistency is
the bare minimum we should demand.

9.3.1 Statement
----------------

.. topic:: Theorem (Consistency of MLE)

   Let :math:`X_1, X_2, \ldots` be i.i.d. with density
   :math:`f(x \mid \theta_0)` where :math:`\theta_0 \in \Theta` is the
   true parameter value. Under regularity conditions (see
   Section 9.7), the MLE :math:`\hat{\theta}_n` satisfies

   .. math::

      \hat{\theta}_n \;\xrightarrow{P}\; \theta_0
      \qquad \text{as } n \to \infty.

9.3.2 Proof Sketch via the Law of Large Numbers and KL Divergence
-------------------------------------------------------------------

The key insight is that maximising the log-likelihood is equivalent to
minimising the Kullback--Leibler (KL) divergence from the true
distribution to the model.

.. admonition:: Intuition

   The KL divergence measures how "different" two distributions are. The
   proof works by showing that the log-likelihood ratio, when averaged
   over many samples, converges to a quantity that is uniquely maximised
   at the true parameter. In other words, the data "know" the truth, and
   with enough observations the MLE cannot help but find it.

**Step 1: Normalised log-likelihood ratio.**
Consider the normalised log-likelihood ratio between any candidate
:math:`\theta` and the truth :math:`\theta_0`:

.. math::

   \frac{1}{n}\bigl[\ell(\theta) - \ell(\theta_0)\bigr]
   \;=\;
   \frac{1}{n}\sum_{i=1}^{n}
   \log \frac{f(X_i \mid \theta)}{f(X_i \mid \theta_0)}.

**Step 2: Apply the law of large numbers.**
By the (strong) law of large numbers, this converges almost surely to

.. math::

   E_{\theta_0}\!\left[
     \log \frac{f(X \mid \theta)}{f(X \mid \theta_0)}
   \right]
   \;=\; -\,\operatorname{KL}\!\bigl(f_{\theta_0} \,\|\, f_{\theta}\bigr).

**Step 3: Non-positivity of the limit.**
The KL divergence is always non-negative, so

.. math::

   E_{\theta_0}\!\left[
     \log \frac{f(X \mid \theta)}{f(X \mid \theta_0)}
   \right]
   \;\leq\; 0,

with equality **if and only if** :math:`f(\cdot \mid \theta)
= f(\cdot \mid \theta_0)` almost everywhere, which (under
identifiability) means :math:`\theta = \theta_0`.

**Step 4: Conclusion.**
Because the normalised log-likelihood ratio converges uniformly (under
regularity) to a function uniquely maximised at :math:`\theta_0`, the
maximiser of the finite-sample log-likelihood must converge to
:math:`\theta_0`. More precisely, for any open neighbourhood
:math:`U` of :math:`\theta_0`,

.. math::

   \sup_{\theta \notin U}\;
   \frac{1}{n}\bigl[\ell(\theta) - \ell(\theta_0)\bigr]
   \;\xrightarrow{\text{a.s.}}\;
   \sup_{\theta \notin U}\;
   \bigl(-\operatorname{KL}(f_{\theta_0}\|f_\theta)\bigr) \;<\; 0,

while by definition
:math:`\frac{1}{n}[\ell(\hat\theta_n) - \ell(\theta_0)] \geq 0`.
Hence :math:`\hat\theta_n` must eventually lie inside :math:`U`, giving
convergence in probability.

Now let's verify consistency on our vaccine problem. We grow the trial
from 20 participants to 20 000 and watch the MLE converge to the true
efficacy :math:`p_0 = 0.75`:

.. code-block:: python

   # Consistency: MLE converges to true vaccine efficacy as n grows
   import numpy as np

   np.random.seed(42)
   p_0 = 0.75
   sample_sizes = [20, 50, 100, 500, 1000, 5000, 20000]

   # Generate one long trial; take prefixes
   all_data = np.random.binomial(1, p_0, size=max(sample_sizes))

   print(f"True vaccine efficacy p_0 = {p_0}")
   print(f"{'n':>7s}  {'MLE':>8s}  {'|MLE - p_0|':>12s}")
   print("-" * 32)
   for n in sample_sizes:
       mle_p = np.mean(all_data[:n])
       print(f"{n:7d}  {mle_p:8.4f}  {abs(mle_p - p_0):12.6f}")

So what? Consistency is why clinical trials work at all. It guarantees
that a large enough trial will discover the true vaccine efficacy, no
matter how noisy individual outcomes are.


9.4 Asymptotic Normality
==========================

Consistency tells us the MLE gets close to the truth. But *how close*?
If we repeated our vaccine trial 10 000 times (each with :math:`n`
participants), what would the histogram of MLEs look like? Asymptotic
normality answers: it would be a bell curve centred on :math:`p_0`,
with a precise, predictable spread. This is the foundation for
confidence intervals and hypothesis tests.

.. admonition:: What's the intuition?

   Imagine you repeat your entire experiment many times, each time
   computing the MLE. Asymptotic normality says that the histogram of
   those MLEs will look like a bell curve centred on the true parameter,
   with a spread determined by the Fisher information. The more
   information each observation carries about the parameter, the tighter
   that bell curve becomes.

9.4.1 Statement
----------------

.. topic:: Theorem (Asymptotic Normality of MLE)

   Under regularity conditions (Section 9.7),

   .. math::

      \sqrt{n}\,\bigl(\hat{\theta}_n - \theta_0\bigr)
      \;\xrightarrow{d}\;
      \mathcal{N}\!\bigl(0,\; I(\theta_0)^{-1}\bigr),

   where :math:`I(\theta_0)` is the **Fisher information matrix**
   evaluated at the true parameter:

   .. math::

      I(\theta_0)
      \;=\;
      E_{\theta_0}\!\left[
        -\,\frac{\partial^2}{\partial\theta\,\partial\theta^\top}
        \log f(X \mid \theta)\biggr|_{\theta=\theta_0}
      \right].

   Equivalently, for large :math:`n`,

   .. math::

      \hat{\theta}_n
      \;\stackrel{\text{approx}}{\sim}\;
      \mathcal{N}\!\left(
        \theta_0,\;
        \frac{1}{n}\,I(\theta_0)^{-1}
      \right).

This is a remarkably powerful result. It says that regardless of the
underlying distribution---Poisson, Gamma, Bernoulli, whatever---the MLE
is approximately normally distributed for large enough :math:`n`. This
universality is what allows us to build general-purpose confidence
intervals and tests.

9.4.2 Proof Sketch via Taylor Expansion of the Score
------------------------------------------------------

**Step 1: Score is zero at the MLE.**
Because the MLE :math:`\hat\theta_n` satisfies the score equation,

.. math::

   S(\hat\theta_n) = \sum_{i=1}^{n}
   \frac{\partial}{\partial\theta}\log f(X_i\mid\hat\theta_n) = 0.

**Step 2: Taylor-expand the score around :math:`\theta_0`.**
Expand :math:`S(\hat\theta_n)` in a first-order Taylor series about the
true value :math:`\theta_0`:

.. math::

   0 \;=\; S(\hat\theta_n)
   \;\approx\;
   S(\theta_0) \;+\;
   \frac{\partial S}{\partial\theta^\top}\biggr|_{\theta=\theta_0}
   \!\bigl(\hat\theta_n - \theta_0\bigr).

The matrix :math:`\partial S / \partial\theta^\top` evaluated at
:math:`\theta_0` is the **observed information** (with a sign change):

.. math::

   \frac{\partial S(\theta)}{\partial \theta^\top}\biggr|_{\theta_0}
   \;=\;
   \sum_{i=1}^{n}
   \frac{\partial^2}{\partial\theta\,\partial\theta^\top}
   \log f(X_i \mid \theta)\biggr|_{\theta_0}
   \;=\; -\,J_n(\theta_0),

where :math:`J_n(\theta_0)` is the observed information matrix.

**Step 3: Solve for the MLE deviation.**
Rearranging,

.. math::

   \hat\theta_n - \theta_0
   \;\approx\;
   J_n(\theta_0)^{-1}\; S(\theta_0).

**Step 4: Normalise.**
Multiply both sides by :math:`\sqrt{n}`:

.. math::

   \sqrt{n}\,(\hat\theta_n - \theta_0)
   \;\approx\;
   \left[\frac{1}{n}\,J_n(\theta_0)\right]^{-1}
   \frac{1}{\sqrt{n}}\,S(\theta_0).

**Step 5: Apply LLN and CLT.**

- By the law of large numbers,
  :math:`\frac{1}{n}J_n(\theta_0) \xrightarrow{P} I(\theta_0)`.
- By the central limit theorem,
  :math:`\frac{1}{\sqrt{n}}S(\theta_0) \xrightarrow{d}
  \mathcal{N}(0, I(\theta_0))`, because
  :math:`E[S(\theta_0)] = 0` and
  :math:`\operatorname{Var}\!\left[\frac{\partial}{\partial\theta}
  \log f(X\mid\theta_0)\right] = I(\theta_0)`.

**Step 6: Combine by Slutsky's theorem.**

.. math::

   \sqrt{n}\,(\hat\theta_n - \theta_0)
   \;\xrightarrow{d}\;
   I(\theta_0)^{-1}\;\mathcal{N}\bigl(0,\,I(\theta_0)\bigr)
   \;=\;
   \mathcal{N}\bigl(0,\, I(\theta_0)^{-1}\bigr).

The last equality uses the fact that if :math:`Z \sim \mathcal{N}(0, V)`
then :math:`AZ \sim \mathcal{N}(0, AVA^\top)`, giving covariance
:math:`I^{-1} \cdot I \cdot I^{-1} = I^{-1}`.

This completes the proof sketch.

Now comes the payoff. For our vaccine problem, the Bernoulli Fisher
information is :math:`I(p) = 1 / [p(1-p)]`. So the theorem predicts
:math:`\sqrt{n}(\hat{p} - p_0) \xrightarrow{d} \mathcal{N}(0, p_0(1-p_0))`.
Let's verify this by simulating 10 000 vaccine trials:

.. code-block:: python

   # Asymptotic normality: histogram of MLEs from 10000 vaccine trials
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   p_0 = 0.75
   n = 200              # participants per trial
   n_simulations = 10000

   # Fisher information for Bernoulli: I(p) = 1 / [p(1-p)]
   fisher_info = 1.0 / (p_0 * (1 - p_0))

   # Simulate 10000 trials
   mle_values = np.array([
       np.mean(np.random.binomial(1, p_0, size=n))
       for _ in range(n_simulations)
   ])

   # Scaled deviations: sqrt(n) * (MLE - p_0)
   scaled_devs = np.sqrt(n) * (mle_values - p_0)

   # Theory predicts: Var[sqrt(n)*(MLE - p_0)] = 1/I(p_0) = p_0*(1-p_0)
   theoretical_var = 1.0 / fisher_info  # = p_0 * (1 - p_0) = 0.1875
   empirical_var = np.var(scaled_devs)

   print(f"Vaccine efficacy: p_0 = {p_0}")
   print(f"Trial size:       n = {n}")
   print(f"Simulations:      {n_simulations}")
   print()
   print(f"Var[sqrt(n)*(MLE - p_0)]:")
   print(f"  Theoretical (p_0*(1-p_0)):  {theoretical_var:.4f}")
   print(f"  Empirical from simulation:  {empirical_var:.4f}")
   print(f"  Ratio:                      {empirical_var/theoretical_var:.4f}")
   print()

   # Shapiro-Wilk normality test on a subsample
   _, sw_pval = stats.shapiro(scaled_devs[:500])
   print(f"Shapiro-Wilk p-value (subsample 500): {sw_pval:.4f}")
   print("(Large p-value => consistent with normality)")
   print()

   # Compare empirical quantiles to theoretical normal
   theoretical_std = np.sqrt(theoretical_var)
   for q in [0.025, 0.25, 0.50, 0.75, 0.975]:
       emp_q = np.quantile(scaled_devs, q)
       the_q = stats.norm.ppf(q, loc=0, scale=theoretical_std)
       print(f"  Quantile {q:.3f}: empirical={emp_q:+.4f}, "
             f"theoretical={the_q:+.4f}")

So what? This is the result that makes confidence intervals possible.
Because the MLE is approximately normal, we can say: "Our estimated
vaccine efficacy is :math:`\hat{p} \pm 1.96 \cdot \text{SE}`." The
standard error comes directly from the Fisher information, which we
compute next.


9.5 Efficiency and the Cramer--Rao Lower Bound
================================================

Our vaccine trial produces the MLE :math:`\hat{p} = \bar{X}`. A
colleague proposes an alternative estimator: the sample median, suitably
scaled. Both are consistent. So why should we prefer the MLE? Because
the MLE is **efficient**---it achieves the tightest possible confidence
interval. The Cramer--Rao bound makes this precise.

.. admonition:: Why does this matter?

   Efficiency answers a fundamental question: are we extracting all the
   information that the data contain? If your estimator achieves the
   Cramer--Rao bound, you have squeezed every last drop of information
   from the sample. Any other unbiased estimator is, in a precise sense,
   *wasting* information.

   For a vaccine trial, this translates directly into cost: an efficient
   estimator needs fewer participants to achieve the same precision. With
   each participant costing thousands of dollars, efficiency is not just
   a mathematical curiosity---it is a budget constraint.

9.5.1 The Cramer--Rao Inequality: Full Derivation
----------------------------------------------------

We work in the scalar case (:math:`\theta \in \mathbb{R}`) for
clarity. Let :math:`T = T(X_1,\ldots,X_n)` be any **unbiased**
estimator of :math:`\theta`, so
:math:`E_\theta[T] = \theta` for all :math:`\theta`.

**Step 1: Differentiate the unbiasedness condition.**
Write

.. math::

   \theta \;=\; E_\theta[T]
   \;=\; \int T(x)\,\prod_{i=1}^n f(x_i\mid\theta)\,dx.

Differentiate both sides with respect to :math:`\theta`:

.. math::

   1 \;=\; \frac{d}{d\theta}\int T(x)\,L(\theta\mid x)\,dx
   \;=\; \int T(x)\,\frac{\partial L(\theta\mid x)}{\partial\theta}\,dx.

Using the identity
:math:`\frac{\partial L}{\partial\theta} = L \cdot
\frac{\partial \ell}{\partial\theta}`, we get

.. math::

   1 \;=\; \int T(x)\,
   \frac{\partial\ell(\theta)}{\partial\theta}\,
   L(\theta\mid x)\,dx
   \;=\;
   E_\theta\!\left[T \cdot \frac{\partial\ell}{\partial\theta}\right].

**Step 2: Recall that the score has mean zero.**
Under regularity conditions, interchanging differentiation and
integration gives

.. math::

   E_\theta\!\left[\frac{\partial\ell}{\partial\theta}\right]
   = 0.

Therefore,

.. math::

   1 \;=\;
   E_\theta\!\left[T \cdot \frac{\partial\ell}{\partial\theta}\right]
   - E_\theta[T]\,E_\theta\!\left[\frac{\partial\ell}{\partial\theta}\right]
   \;=\;
   \operatorname{Cov}_\theta\!\left(T,\;
   \frac{\partial\ell}{\partial\theta}\right).

**Step 3: Apply the Cauchy--Schwarz inequality.**
For any two random variables :math:`U, V`,

.. math::

   \bigl[\operatorname{Cov}(U,V)\bigr]^2
   \;\leq\;
   \operatorname{Var}(U)\;\operatorname{Var}(V).

Applying this with :math:`U = T` and
:math:`V = \partial\ell/\partial\theta`:

.. math::

   1 \;\leq\;
   \operatorname{Var}_\theta(T)\;
   \operatorname{Var}_\theta\!\left(\frac{\partial\ell}{\partial\theta}\right).

**Step 4: Identify the Fisher information.**
The variance of the score is precisely the total Fisher information for
the sample:

.. math::

   \operatorname{Var}_\theta\!\left(\frac{\partial\ell}{\partial\theta}\right)
   \;=\; n\,I(\theta).

**Step 5: Rearrange.**

.. math::

   \operatorname{Var}_\theta(T)
   \;\geq\;
   \frac{1}{n\,I(\theta)}.

This is the **Cramer--Rao Lower Bound** (CRLB).

.. topic:: Cramer--Rao Lower Bound

   For any unbiased estimator :math:`T` of :math:`\theta`,

   .. math::

      \operatorname{Var}_\theta(T) \;\geq\; \frac{1}{n\,I(\theta)}.

   Equality holds if and only if

   .. math::

      T - \theta \;=\; a(\theta)\,\frac{\partial\ell}{\partial\theta}

   for some function :math:`a(\theta)`, i.e., the estimator is a linear
   function of the score.

For our Bernoulli vaccine model, :math:`I(p) = 1/[p(1-p)]`, so the
Cramer--Rao bound is :math:`\text{Var}(\hat{p}) \geq p(1-p)/n`. The MLE
:math:`\hat{p} = \bar{X}` has :math:`\text{Var}(\bar{X}) = p(1-p)/n`
exactly, so it *achieves* the bound---it is efficient.

Now let's compute the CRLB numerically and verify that the MLE
achieves it, while an alternative estimator (based on the sample median)
does not:

.. code-block:: python

   # Cramer-Rao bound: MLE achieves it, median-based estimator does not
   import numpy as np

   np.random.seed(42)
   p_0 = 0.75
   n = 200
   n_sims = 50000

   # CRLB for Bernoulli: Var >= p(1-p)/n
   crlb = p_0 * (1 - p_0) / n

   mle_estimates = []
   median_estimates = []

   for _ in range(n_sims):
       data = np.random.binomial(1, p_0, size=n)
       # MLE: sample proportion
       mle_estimates.append(np.mean(data))
       # Median-based estimator: not great for binary data, but illustrative
       # Use smoothed median: median of batch means (batches of 10)
       batch_means = data.reshape(-1, 10).mean(axis=1)
       median_estimates.append(np.median(batch_means))

   mle_var = np.var(mle_estimates, ddof=0)
   med_var = np.var(median_estimates, ddof=0)

   print(f"True p = {p_0}, n = {n}")
   print(f"Cramer-Rao Lower Bound:      {crlb:.6f}")
   print(f"Variance of MLE:             {mle_var:.6f}")
   print(f"Variance of median estimator:{med_var:.6f}")
   print(f"Ratio MLE / CRLB:            {mle_var / crlb:.4f}  (should be ~1.0)")
   print(f"Ratio median / CRLB:         {med_var / crlb:.4f}  (should be > 1.0)")
   print()
   print("=> The MLE achieves the bound; the median estimator wastes information.")

So what? The Cramer--Rao bound answers the question: "What is the
tightest 95% confidence interval we can possibly construct for vaccine
efficacy with 200 participants?" The answer is
:math:`\hat{p} \pm 1.96\sqrt{p(1-p)/n} = \hat{p} \pm 0.060`. No
unbiased estimator can do better.

.. code-block:: python

   # The tightest possible 95% CI for vaccine efficacy
   import numpy as np

   p_0 = 0.75
   n = 200

   # Cramer-Rao bound gives the minimum standard error
   min_se = np.sqrt(p_0 * (1 - p_0) / n)
   half_width = 1.96 * min_se

   print(f"With n = {n} participants and true p = {p_0}:")
   print(f"  Minimum achievable SE: {min_se:.4f}")
   print(f"  Tightest 95% CI half-width: +/- {half_width:.4f}")
   print(f"  CI: ({p_0 - half_width:.4f}, {p_0 + half_width:.4f})")
   print()
   print(f"To halve the CI width, you need 4x the participants:")
   for n_trial in [50, 200, 800, 3200]:
       se = np.sqrt(p_0 * (1 - p_0) / n_trial)
       hw = 1.96 * se
       print(f"  n = {n_trial:5d}: 95% CI half-width = {hw:.4f}")

9.5.2 What the Bound Means
----------------------------

The CRLB is the smallest variance any unbiased estimator can achieve.
An estimator that achieves this bound is called **efficient**. In
general, efficient estimators exist only for exponential-family models,
where the MLE achieves the CRLB exactly at every sample size.

For non-exponential families, the MLE is **asymptotically efficient**:
its variance approaches the CRLB as :math:`n \to \infty`. This follows
directly from the asymptotic normality result, since
:math:`\operatorname{Var}(\hat\theta_n) \approx 1/(nI(\theta_0))`.

9.5.3 Multiparameter Extension
--------------------------------

For a :math:`p`-dimensional parameter :math:`\theta`, the CRLB
generalises to a matrix inequality:

.. math::

   \operatorname{Cov}_\theta(T)
   \;-\;
   \frac{1}{n}\,I(\theta)^{-1}

is positive semi-definite for any unbiased estimator :math:`T`. This
means that the variance of any linear combination
:math:`a^\top T` is at least :math:`a^\top I(\theta)^{-1} a / n`.


9.6 Invariance Property
=========================

The vaccine trial gives us :math:`\hat{p} = 0.76`. A regulator asks:
"What is the MLE of the *odds ratio*
:math:`p / (1-p)`?" Do we need to re-derive the MLE from scratch?
No---invariance says we just plug in: :math:`\widehat{p/(1-p)} = \hat{p}/(1-\hat{p})`.

9.6.1 Statement
----------------

.. topic:: Theorem (Invariance / Equivariance of MLE)

   Let :math:`\hat\theta_n` be the MLE of :math:`\theta`. For any
   function :math:`g : \Theta \to \mathbb{R}^q`, the MLE of
   :math:`\eta = g(\theta)` is

   .. math::

      \hat\eta_n \;=\; g(\hat\theta_n).

In words: to estimate a function of the parameter, just plug the MLE
into that function. This is an enormous convenience---you never need to
re-derive or re-optimise.

9.6.2 Proof
-------------

Define the reparameterised likelihood:

.. math::

   L^*(\eta) \;=\; \sup_{\{\theta : g(\theta)=\eta\}} L(\theta).

We need to show that :math:`L^*` is maximised at
:math:`\hat\eta = g(\hat\theta_n)`.

For any :math:`\eta` in the range of :math:`g`,

.. math::

   L^*(g(\hat\theta_n))
   \;=\;
   \sup_{\{\theta : g(\theta)=g(\hat\theta_n)\}} L(\theta)
   \;\geq\; L(\hat\theta_n).

For any other :math:`\eta'`,

.. math::

   L^*(\eta')
   \;=\;
   \sup_{\{\theta : g(\theta)=\eta'\}} L(\theta)
   \;\leq\;
   \sup_{\theta \in \Theta} L(\theta)
   \;=\; L(\hat\theta_n)
   \;\leq\; L^*(g(\hat\theta_n)).

Hence :math:`g(\hat\theta_n)` maximises :math:`L^*(\eta)`.

9.6.3 Examples
---------------

- If :math:`\hat\lambda` is the MLE of the Poisson rate, then
  :math:`e^{-\hat\lambda}` is the MLE of :math:`P(X=0) = e^{-\lambda}`.

- If :math:`\hat\mu` and :math:`\hat\sigma^2` are the MLEs of the
  normal parameters, then :math:`\hat\mu + 1.96\hat\sigma` is the MLE
  of the 97.5th percentile.

.. note::

   Invariance does **not** preserve unbiasedness. Even if
   :math:`\hat\theta` is unbiased for :math:`\theta`,
   :math:`g(\hat\theta)` may be biased for :math:`g(\theta)` due to
   Jensen's inequality (whenever :math:`g` is nonlinear).

Let's see invariance applied to our vaccine problem. The regulator wants
several transformed quantities: odds, log-odds, and the number needed to
vaccinate (NNV = :math:`1/p`).

.. code-block:: python

   # Invariance property: MLE of g(p) = g(MLE of p)
   import numpy as np

   np.random.seed(42)
   p_0 = 0.75
   n = 200
   data = np.random.binomial(1, p_0, size=n)

   mle_p = np.mean(data)

   # Various transformations --- all via invariance
   mle_odds = mle_p / (1 - mle_p)          # odds
   mle_log_odds = np.log(mle_odds)          # log-odds
   mle_nnv = 1.0 / mle_p                   # number needed to vaccinate

   true_odds = p_0 / (1 - p_0)
   true_log_odds = np.log(true_odds)
   true_nnv = 1.0 / p_0

   print(f"MLE of p:          {mle_p:.4f}  (true: {p_0})")
   print(f"MLE of odds:       {mle_odds:.4f}  (true: {true_odds:.4f})")
   print(f"MLE of log-odds:   {mle_log_odds:.4f}  (true: {true_log_odds:.4f})")
   print(f"MLE of NNV (1/p):  {mle_nnv:.4f}  (true: {true_nnv:.4f})")
   print()
   print("Each is obtained by simply plugging the MLE into the function.")
   print("No re-optimisation needed.")

.. code-block:: python

   # Verify invariance by brute-force optimisation of the transformed likelihood
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   p_0 = 0.75
   n = 200
   data = np.random.binomial(1, p_0, size=n)
   k = np.sum(data)
   mle_p = k / n

   # MLE of odds = p/(1-p) via invariance
   mle_odds_invariance = mle_p / (1 - mle_p)

   # MLE of odds by directly maximising the likelihood reparameterised in odds
   # odds = p/(1-p), so p = odds/(1+odds)
   def neg_loglik_odds(odds):
       if odds <= 0:
           return 1e10
       p = odds / (1 + odds)
       return -(k * np.log(p) + (n - k) * np.log(1 - p))

   result = minimize_scalar(neg_loglik_odds, bounds=(0.01, 100), method='bounded')
   mle_odds_direct = result.x

   print(f"MLE of odds via invariance:      {mle_odds_invariance:.6f}")
   print(f"MLE of odds via direct optimisation: {mle_odds_direct:.6f}")
   print(f"Difference:                      {abs(mle_odds_invariance - mle_odds_direct):.2e}")


9.7 Regularity Conditions
===========================

The asymptotic results in Sections 9.3--9.5 all require certain
**regularity conditions**. These ensure the likelihood is sufficiently
well-behaved for the Taylor-expansion and interchange-of-limits
arguments to go through.

Think of regularity conditions as the "fine print" of MLE theory. Most
standard parametric models satisfy them automatically, but it pays to
know what they are---because the cases where they fail are exactly the
cases where the MLE behaves unexpectedly.

Below we list and explain the standard regularity conditions. Throughout,
:math:`f(x\mid\theta)` denotes the model density and :math:`\Theta` the
parameter space.

1. **Identifiability.**
   :math:`\theta_1 \neq \theta_2 \;\Longrightarrow\;
   f(\cdot\mid\theta_1) \neq f(\cdot\mid\theta_2)` as functions.
   *Without identifiability, the true parameter is not uniquely defined,
   so "convergence to the true value" is meaningless.*

2. **Common support.**
   The support :math:`\{x : f(x\mid\theta) > 0\}` does not depend on
   :math:`\theta`.
   *This rules out models like the Uniform(0, :math:`\theta`)
   distribution, whose support changes with :math:`\theta`. (The MLE
   still exists for the Uniform, but the standard asymptotics do not
   apply.)*

3. **Interior true parameter.**
   The true value :math:`\theta_0` is an interior point of
   :math:`\Theta`.
   *If :math:`\theta_0` is on the boundary, the score equation may not
   hold at the MLE, and the limiting distribution changes.*

4. **Smoothness.**
   :math:`\log f(x\mid\theta)` is at least three times continuously
   differentiable with respect to :math:`\theta` for all :math:`x` in
   the support.
   *This ensures the Taylor expansion in the asymptotic-normality proof
   is valid.*

5. **Interchange of differentiation and integration.**
   We can differentiate under the integral sign:

   .. math::

      \frac{\partial}{\partial\theta}
      \int f(x\mid\theta)\,dx
      \;=\;
      \int \frac{\partial f(x\mid\theta)}{\partial\theta}\,dx.

   *This is what gives :math:`E[S(\theta)] = 0` and the information
   identity. It typically holds whenever there exists a dominating
   function (Leibniz's rule / dominated convergence).*

6. **Positive-definite Fisher information.**
   :math:`0 < I(\theta) < \infty` for all :math:`\theta` in a
   neighbourhood of :math:`\theta_0`.
   *If the Fisher information is zero, the model is locally flat and the
   standard :math:`\sqrt{n}` rate breaks down.*

7. **Dominance / uniform integrability.**
   There exists a function :math:`M(x)` with
   :math:`E_{\theta_0}[M(X)] < \infty` such that

   .. math::

      \left|\frac{\partial^3}{\partial\theta_j\,\partial\theta_k\,
      \partial\theta_l} \log f(x\mid\theta)\right|
      \;\leq\; M(x)

   for all :math:`\theta` in a neighbourhood of :math:`\theta_0`.
   *This bounds the remainder term in the Taylor expansion, ensuring it
   is negligible.*

Our Bernoulli vaccine model satisfies all seven conditions (for
:math:`p_0 \in (0,1)`), which is why the asymptotic theory works
beautifully in that setting. Let's verify the key ones numerically:

.. code-block:: python

   # Verify regularity conditions for the Bernoulli model
   import numpy as np

   p_0 = 0.75

   # Condition 6: Fisher information is finite and positive
   fisher_info = 1.0 / (p_0 * (1 - p_0))
   print(f"Fisher information I(p_0) = 1/[p(1-p)] = {fisher_info:.4f}")
   print(f"  Positive and finite? Yes" if 0 < fisher_info < np.inf else "  Problem!")

   # Condition 5: E[score] = 0
   # Score for Bernoulli: s(x, p) = x/p - (1-x)/(1-p)
   # E[s] = p * (1/p) + (1-p) * (-(1/(1-p))) = 1 - 1 = 0
   n_check = 100000
   data = np.random.binomial(1, p_0, size=n_check)
   scores = data / p_0 - (1 - data) / (1 - p_0)
   print(f"\nE[score] (should be 0): {np.mean(scores):.6f}")

   # Condition 6 again: Var[score] = I(p)
   print(f"Var[score] (should be {fisher_info:.4f}): {np.var(scores):.4f}")

   # Information identity: E[-d^2 ell / dp^2] = I(p)
   # Second derivative: -x/p^2 - (1-x)/(1-p)^2
   neg_d2ell = data / p_0**2 + (1 - data) / (1 - p_0)**2
   print(f"E[-d^2 logf/dp^2] (should be {fisher_info:.4f}): "
         f"{np.mean(neg_d2ell):.4f}")

When any of these conditions fails, the MLE may still be consistent and
asymptotically normal, but the proofs require more specialised
arguments. The Uniform distribution (condition 2 violation) and models
with boundary parameters (condition 3 violation) are treated in
:ref:`ch10_analytical_mle`.


9.8 Finite-Sample Properties: Bias of the MLE
================================================

Although the MLE is consistent and asymptotically efficient, it is
**not** in general unbiased in finite samples. This is a subtle but
important point: the MLE is the best thing going for large samples, but
for any *fixed* :math:`n` it may systematically over- or under-estimate
the true parameter.

For our vaccine trial, the MLE :math:`\hat{p} = \bar{X}` happens to be
unbiased (because the Bernoulli is an exponential family). But this is
the exception, not the rule.

9.8.1 Why MLE Can Be Biased
-----------------------------

The score equation
:math:`E\bigl[\partial\ell/\partial\theta\bigr] = 0` guarantees that
the score is centred at zero, but it does **not** imply that the
solution to the score equation is centred at the true parameter. In
general, solving a nonlinear equation introduces bias through the
curvature of the log-likelihood.

The bias of the MLE is typically of order :math:`O(1/n)`:

.. math::

   E_{\theta_0}[\hat\theta_n] \;=\; \theta_0 + \frac{b(\theta_0)}{n}
   + O(n^{-2}),

where :math:`b(\theta_0)` depends on the third derivative of the
log-likelihood and can be computed via a formula due to Cox and Snell
(1968).

9.8.2 Example: Normal Variance
--------------------------------

The most well-known example is the MLE of the normal variance. For
i.i.d. :math:`X_i \sim \mathcal{N}(\mu, \sigma^2)`:

.. math::

   \hat\sigma^2_{\text{MLE}}
   \;=\; \frac{1}{n}\sum_{i=1}^{n}(X_i - \bar X)^2.

Taking expectations:

.. math::

   E\bigl[\hat\sigma^2_{\text{MLE}}\bigr]
   \;=\; \frac{n-1}{n}\,\sigma^2
   \;\neq\; \sigma^2.

The MLE **underestimates** the true variance on average. The bias is
:math:`-\sigma^2/n`, which vanishes as :math:`n \to \infty` (consistent
with the :math:`O(1/n)` rate). The unbiased corrected estimator divides
by :math:`n-1` instead.

A full derivation of this result appears in :ref:`ch10_analytical_mle`.

9.8.3 Example: Exponential Distribution
-----------------------------------------

For :math:`X_1, \ldots, X_n \sim \text{Exp}(\lambda)`, the MLE is
:math:`\hat\lambda = 1/\bar{X}`. The expectation of :math:`1/\bar{X}`
for exponential data is

.. math::

   E\!\left[\frac{1}{\bar X}\right]
   \;=\; \frac{n\lambda}{n - 1},

so the MLE is biased upward with bias :math:`\lambda/(n-1)`.

Let's verify the bias formulas numerically for both cases and contrast
with our unbiased Bernoulli MLE:

.. code-block:: python

   # MLE bias: Bernoulli (unbiased), Normal variance (biased), Exponential (biased)
   import numpy as np

   np.random.seed(42)
   n = 10
   n_sims = 100000

   # --- Bernoulli MLE (unbiased) ---
   p_0 = 0.75
   mle_p_vals = [np.mean(np.random.binomial(1, p_0, size=n)) for _ in range(n_sims)]
   print("=== Bernoulli p MLE (vaccine example) ===")
   print(f"True p:        {p_0}")
   print(f"E[MLE]:        {np.mean(mle_p_vals):.4f}")
   print(f"Bias:          {np.mean(mle_p_vals) - p_0:.4f}")
   print(f"(Bernoulli MLE is unbiased at every n)")

   # --- Normal variance MLE (biased downward) ---
   true_sigma2 = 4.0
   mle_sigma2_vals = []
   for _ in range(n_sims):
       data = np.random.normal(0, np.sqrt(true_sigma2), size=n)
       mle_sigma2_vals.append(np.mean((data - np.mean(data))**2))

   print(f"\n=== Normal variance MLE ===")
   print(f"True sigma^2:      {true_sigma2}")
   print(f"E[MLE sigma^2]:    {np.mean(mle_sigma2_vals):.4f}")
   print(f"Theoretical E[MLE]:{(n-1)/n * true_sigma2:.4f}")
   print(f"Bias:              {np.mean(mle_sigma2_vals) - true_sigma2:.4f}")
   print(f"Theoretical bias:  {-true_sigma2/n:.4f}")

   # --- Exponential rate MLE (biased upward) ---
   true_lambda = 2.0
   mle_lambda_vals = []
   for _ in range(n_sims):
       data = np.random.exponential(1/true_lambda, size=n)
       mle_lambda_vals.append(1.0 / np.mean(data))

   print(f"\n=== Exponential rate MLE ===")
   print(f"True lambda:       {true_lambda}")
   print(f"E[MLE lambda]:     {np.mean(mle_lambda_vals):.4f}")
   print(f"Theoretical E[MLE]:{n*true_lambda/(n-1):.4f}")
   print(f"Bias:              {np.mean(mle_lambda_vals) - true_lambda:.4f}")
   print(f"Theoretical bias:  {true_lambda/(n-1):.4f}")

9.8.4 Bias Correction
-----------------------

Several approaches can reduce MLE bias:

- **Analytical correction:** Subtract the leading bias term
  :math:`b(\theta_0)/n`, replacing :math:`\theta_0` with
  :math:`\hat\theta_n` to get a *bias-corrected* MLE.
- **Bootstrap bias correction:** Estimate the bias via the bootstrap and
  subtract it.
- **Jackknife:** The delete-one jackknife estimator automatically
  removes :math:`O(1/n)` bias.
- **Firth's penalised likelihood (1993):** Adds a penalty proportional
  to :math:`\log|I(\theta)|^{1/2}` to the log-likelihood, producing an
  estimator with :math:`O(1/n^2)` bias.

Let's demonstrate bootstrap bias correction on the exponential rate
MLE:

.. code-block:: python

   # Bootstrap bias correction for exponential rate MLE
   import numpy as np

   np.random.seed(42)
   true_lambda = 2.0
   n = 15
   data = np.random.exponential(1/true_lambda, size=n)

   mle_lambda = 1.0 / np.mean(data)

   # Bootstrap: resample and compute MLE many times
   n_boot = 10000
   boot_mles = []
   for _ in range(n_boot):
       boot_sample = np.random.choice(data, size=n, replace=True)
       boot_mles.append(1.0 / np.mean(boot_sample))

   boot_bias = np.mean(boot_mles) - mle_lambda
   mle_corrected = mle_lambda - boot_bias

   # Analytical correction: (n-1)/n * MLE
   analytical_corrected = (n - 1) / n * mle_lambda

   print(f"True lambda:             {true_lambda}")
   print(f"MLE:                     {mle_lambda:.4f}")
   print(f"Bootstrap bias estimate: {boot_bias:.4f}")
   print(f"Bias-corrected MLE:      {mle_corrected:.4f}")
   print(f"Analytical correction:   {analytical_corrected:.4f}")
   print(f"  (multiply MLE by (n-1)/n = {(n-1)/n:.4f})")


9.9 Summary
=============

The MLE is the dominant estimation paradigm in parametric statistics for
good reason:

- **Consistency:** :math:`\hat\theta_n \to \theta_0` in probability.
- **Asymptotic normality:**
  :math:`\sqrt{n}(\hat\theta_n - \theta_0) \to \mathcal{N}(0, I^{-1})`.
- **Efficiency:** It achieves the Cramer--Rao lower bound
  asymptotically.
- **Invariance:** :math:`g(\hat\theta)` is the MLE of :math:`g(\theta)`.

We verified every one of these properties on our running vaccine
efficacy example: consistency showed the MLE converging to
:math:`p_0 = 0.75` as trial size grew from 20 to 20 000; asymptotic
normality was confirmed by the bell-shaped histogram from 10 000
simulated trials; efficiency was demonstrated by comparing the MLE's
variance to the Cramer--Rao bound; and invariance let us estimate odds,
log-odds, and NNV by simple plug-in.

These properties hold under regularity conditions that are satisfied by
most standard parametric models. When regularity fails---boundary
parameters, non-smooth likelihoods, non-identifiable models---special
care is needed, and the standard :math:`\sqrt{n}` convergence rate or
Gaussian limiting distribution may no longer apply.

The next chapter applies the theory developed here to obtain closed-form
MLE solutions for the most important parametric families.
