.. _ch11_testing:

===================================================================
Chapter 11 --- Confidence Intervals and Hypothesis Testing
===================================================================

Once we have a maximum likelihood estimate :math:`\hat\theta`, two
natural questions arise: *How certain are we about this estimate?* and
*Is a hypothesised parameter value compatible with the data?* This
chapter develops the classical answers to both questions, all grounded in
the likelihood function.

**Running example: A/B testing at a tech company.**
A product team runs an A/B test on a checkout page. The control group
(1000 users) sees the old design; the treatment group (1000 users) sees a
redesigned page.

- Control: 120 out of 1000 users convert (12.0%)
- Treatment: 145 out of 1000 users convert (14.5%)

Is the treatment really better, or is this just noise? We will apply
every method in this chapter---LRT, Wald, score test, Wald CI, profile
likelihood CI---to this single dataset. By the end, you will see how all
three tests and all three CI methods converge on the same conclusion,
and understand when they might disagree.

We derive three families of tests---likelihood ratio, Wald, and
score---show how each gives rise to confidence intervals, and discuss
profile-likelihood methods and multiple-testing corrections.

The asymptotic theory developed in :ref:`ch9_mle_theory` (in particular,
asymptotic normality and the Fisher information) is used throughout.

Let's begin by setting up the data that we will use throughout the
chapter:

.. code-block:: python

   # A/B test data: used throughout this chapter
   import numpy as np
   from scipy import stats

   # Observed data
   n_ctrl, x_ctrl = 1000, 120    # control: 120/1000 convert
   n_trt, x_trt = 1000, 145      # treatment: 145/1000 convert

   p_hat_ctrl = x_ctrl / n_ctrl   # 0.120
   p_hat_trt = x_trt / n_trt      # 0.145
   diff = p_hat_trt - p_hat_ctrl   # 0.025

   print("=== A/B Test: Checkout Page Redesign ===")
   print(f"Control:    {x_ctrl}/{n_ctrl} conversions (rate = {p_hat_ctrl:.3f})")
   print(f"Treatment:  {x_trt}/{n_trt}  conversions (rate = {p_hat_trt:.3f})")
   print(f"Difference: {diff:.3f}  (treatment - control)")
   print(f"\nQuestion: Is this 2.5 percentage point lift real or noise?")


11.1 The Likelihood Ratio Test
================================

The likelihood ratio test (LRT) is arguably the most fundamental of the
three classical tests. Here is the key idea: compare the likelihood at
the MLE to the likelihood at the hypothesised value. If the data are
much more likely under the MLE than under the null, we reject the null.

.. admonition:: What's the intuition?

   Imagine the log-likelihood as a mountain. The MLE sits at the peak.
   If you walk downhill to the null hypothesis value :math:`\theta_0` and
   the drop in elevation is large, the data are telling you that
   :math:`\theta_0` is a poor explanation compared to the MLE. The LRT
   measures exactly this vertical drop.

11.1.1 Setup
--------------

We test the null hypothesis

.. math::

   H_0: \theta = \theta_0

against the alternative :math:`H_1: \theta \neq \theta_0`, where
:math:`\theta_0` is a specific (known) value.

11.1.2 Derivation of the Test Statistic
-----------------------------------------

Define the **likelihood ratio**:

.. math::

   \Lambda = \frac{L(\theta_0)}{L(\hat\theta)},

where :math:`L(\hat\theta)` is the maximum of the likelihood (achieved
at the MLE). Since :math:`\hat\theta` maximises :math:`L`, we always
have :math:`0 \leq \Lambda \leq 1`. Small values of :math:`\Lambda`
indicate that the data are much more likely under the MLE than under the
null, providing evidence against :math:`H_0`.

It is more convenient to work on the log scale. The **likelihood ratio
test statistic** is

.. math::

   W_{\text{LR}}
   = -2\log\Lambda
   = -2\bigl[\ell(\theta_0) - \ell(\hat\theta)\bigr]
   = 2\bigl[\ell(\hat\theta) - \ell(\theta_0)\bigr].

Note that :math:`W_{\text{LR}} \geq 0`, with larger values providing
stronger evidence against :math:`H_0`.

11.1.3 Wilks' Theorem
-----------------------

The key result that makes the LRT practical is Wilks' theorem (1938),
which provides the null distribution of the test statistic.

.. topic:: Theorem (Wilks)

   Under :math:`H_0: \theta = \theta_0` and the regularity conditions
   from :ref:`ch9_mle_theory`,

   .. math::

      W_{\text{LR}}
      = 2\bigl[\ell(\hat\theta) - \ell(\theta_0)\bigr]
      \;\xrightarrow{d}\; \chi^2_r
      \qquad \text{as } n \to \infty,

   where :math:`r` is the number of parameters constrained (set to
   specific values) under :math:`H_0`. For testing a single parameter
   :math:`\theta = \theta_0`, we have :math:`r = 1`.

**Proof sketch.** Taylor-expand :math:`\ell(\theta_0)` around the MLE
:math:`\hat\theta`:

.. math::

   \ell(\theta_0)
   \approx \ell(\hat\theta)
   + \underbrace{\ell'(\hat\theta)}_{=\,0}(\theta_0 - \hat\theta)
   + \frac{1}{2}\,\ell''(\hat\theta)\,(\theta_0 - \hat\theta)^2.

The first-derivative term vanishes because the score is zero at the MLE.
Therefore:

.. math::

   W_{\text{LR}}
   = 2[\ell(\hat\theta) - \ell(\theta_0)]
   \approx -\ell''(\hat\theta)\,(\hat\theta - \theta_0)^2.

Now, :math:`-\ell''(\hat\theta)/n \xrightarrow{P} I(\theta_0)` by the
law of large numbers, and
:math:`\sqrt{n}(\hat\theta - \theta_0) \xrightarrow{d}
\mathcal{N}(0, I(\theta_0)^{-1})` by asymptotic normality. Combining:

.. math::

   W_{\text{LR}}
   \approx n\,I(\theta_0)\,(\hat\theta - \theta_0)^2
   = \left[\sqrt{n\,I(\theta_0)}\,(\hat\theta - \theta_0)\right]^2,

which is the square of a quantity converging to a standard normal---hence
it converges to :math:`\chi^2_1`.

For a :math:`p`-dimensional parameter with :math:`r` constraints under
the null (so the null fixes :math:`r` components), the same argument
generalises using a quadratic form, yielding :math:`\chi^2_r`.

11.1.4 Decision Rule
----------------------

At significance level :math:`\alpha`:

- Reject :math:`H_0` if :math:`W_{\text{LR}} > \chi^2_{r,\,1-\alpha}`,
  the :math:`(1-\alpha)` quantile of :math:`\chi^2_r`.
- Equivalently, the *p*-value is
  :math:`P(\chi^2_r \geq W_{\text{LR}})`.

11.1.5 Composite Null Hypotheses
----------------------------------

In many real problems, the null hypothesis does not pin down all
parameters. For example, when comparing two groups we may hypothesise
that the treatment effect is zero, but the baseline rate is still
unknown. This is a *composite* null: some parameters are fixed, others
are free.

More generally, if :math:`\theta = (\psi, \lambda)` where :math:`\psi`
is the parameter of interest and :math:`\lambda` is a nuisance
parameter, we test :math:`H_0: \psi = \psi_0` (with :math:`\lambda`
unspecified) using

.. math::

   W_{\text{LR}}
   = 2\bigl[\ell(\hat\psi, \hat\lambda)
     - \ell(\psi_0, \hat\lambda_0)\bigr],

where :math:`(\hat\psi, \hat\lambda)` is the unrestricted MLE and
:math:`\hat\lambda_0` is the MLE of :math:`\lambda` under the
constraint :math:`\psi = \psi_0`.

In words: we compare the fully optimised log-likelihood (both parameters
free) to the *partially* optimised log-likelihood (the parameter of
interest pinned at the null value, the nuisance parameter re-optimised
subject to that constraint). The re-optimisation of :math:`\lambda`
under the null is important---it gives the null hypothesis its "best
shot" by choosing the nuisance parameter that makes the null fit as well
as possible.

By Wilks' theorem, this has a
:math:`\chi^2_r` distribution where :math:`r = \dim(\psi)`.

Now let's apply the LRT to our A/B test. Under :math:`H_0`, both groups
have the same conversion rate :math:`p`, so we pool the data. Under
:math:`H_1`, each group has its own rate. The LRT compares the pooled
log-likelihood to the separate-rates log-likelihood.

.. code-block:: python

   # LRT for A/B test: is treatment conversion rate different from control?
   import numpy as np
   from scipy import stats

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   # Binomial log-likelihood helper
   def binom_ll(x, n, p):
       p = np.clip(p, 1e-15, 1 - 1e-15)
       return x * np.log(p) + (n - x) * np.log(1 - p)

   # Under H0: common rate (pooled)
   p_pooled = (x_ctrl + x_trt) / (n_ctrl + n_trt)
   ll_null = binom_ll(x_ctrl, n_ctrl, p_pooled) + binom_ll(x_trt, n_trt, p_pooled)

   # Under H1: separate rates
   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   ll_alt = binom_ll(x_ctrl, n_ctrl, p_hat_ctrl) + binom_ll(x_trt, n_trt, p_hat_trt)

   # LRT statistic
   W_LR = 2 * (ll_alt - ll_null)
   p_LR = 1 - stats.chi2.cdf(W_LR, df=1)

   print("=== Likelihood Ratio Test ===")
   print(f"H0: p_ctrl = p_trt (pooled rate = {p_pooled:.4f})")
   print(f"H1: p_ctrl != p_trt (ctrl={p_hat_ctrl:.3f}, trt={p_hat_trt:.3f})")
   print(f"Log-lik under H0: {ll_null:.4f}")
   print(f"Log-lik under H1: {ll_alt:.4f}")
   print(f"LRT statistic:    {W_LR:.4f}")
   print(f"p-value:          {p_LR:.4f}")
   print(f"Reject H0 at 5%?  {'Yes' if p_LR < 0.05 else 'No'}")


11.2 The Wald Test
====================

The Wald test derives directly from the asymptotic normality of the MLE.
Its key advantage is that it only requires fitting the model once (under
the alternative), without re-fitting under the null.

.. admonition:: Intuition

   While the LRT asks "how much does the likelihood drop when we move
   from the MLE to the null?", the Wald test asks "how far is the MLE
   from the null value, measured in standard-error units?" If the MLE is
   many standard errors away from :math:`\theta_0`, we reject the null.

11.2.1 Derivation
-------------------

From :ref:`ch9_mle_theory`, the MLE satisfies

.. math::

   \hat\theta \;\stackrel{\text{approx}}{\sim}\;
   \mathcal{N}\!\left(\theta_0,\;
   \frac{1}{n\,I(\theta_0)}\right)

under :math:`H_0: \theta = \theta_0`. Therefore, the standardised
estimator

.. math::

   Z = \frac{\hat\theta - \theta_0}
            {\sqrt{1/(n\,I(\theta_0))}}
   = \sqrt{n\,I(\theta_0)}\;(\hat\theta - \theta_0)

is approximately standard normal, and the **Wald statistic** is

.. math::

   W_{\text{W}} = Z^2
   = n\,I(\theta_0)\,(\hat\theta - \theta_0)^2.

In practice, the true Fisher information :math:`I(\theta_0)` is unknown
(since :math:`\theta_0` may not be the true value in general settings).
Two common replacements are:

- **Expected information at the MLE:**
  :math:`I(\hat\theta)`.
- **Observed information:** :math:`J_n(\hat\theta)/n`, where
  :math:`J_n = -\ell''(\hat\theta)`.

This gives the practical Wald statistic:

.. math::

   W_{\text{W}} = \frac{(\hat\theta - \theta_0)^2}
                       {\widehat{\operatorname{Var}}(\hat\theta)},

where
:math:`\widehat{\operatorname{Var}}(\hat\theta) = 1/(n\,I(\hat\theta))`
or :math:`1/J_n(\hat\theta)`.

Now let's apply the Wald test to our A/B test. We test whether the
difference :math:`\delta = p_{\text{trt}} - p_{\text{ctrl}}` is zero.

.. code-block:: python

   # Wald Test for A/B test
   import numpy as np
   from scipy import stats

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   diff = p_hat_trt - p_hat_ctrl

   # Under H0: delta = 0
   # SE of difference = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
   se_diff = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_ctrl
                     + p_hat_trt * (1 - p_hat_trt) / n_trt)

   Z_W = diff / se_diff
   W_W = Z_W**2
   p_W = 2 * (1 - stats.norm.cdf(abs(Z_W)))  # two-sided

   print("=== Wald Test ===")
   print(f"H0: p_trt - p_ctrl = 0")
   print(f"Estimated difference: {diff:.4f}")
   print(f"Standard error:       {se_diff:.4f}")
   print(f"Z statistic:          {Z_W:.4f}")
   print(f"Wald statistic (Z^2): {W_W:.4f}")
   print(f"p-value (two-sided):  {p_W:.4f}")
   print(f"Reject H0 at 5%?     {'Yes' if p_W < 0.05 else 'No'}")

11.2.2 Connection to Confidence Intervals
--------------------------------------------

The Wald test and confidence intervals are two sides of the same coin.
If we do **not** reject :math:`H_0: \theta = \theta_0` at level
:math:`\alpha`, then :math:`\theta_0` lies inside the Wald confidence
interval:

.. math::

   \hat\theta \;\pm\; z_{\alpha/2}\,
   \sqrt{\widehat{\operatorname{Var}}(\hat\theta)}.

Conversely, the set of all :math:`\theta_0` values for which the Wald
test does not reject at level :math:`\alpha` defines the
:math:`100(1-\alpha)\%` confidence interval.

.. code-block:: python

   # Wald 95% CI for the A/B test difference
   import numpy as np
   from scipy import stats

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145
   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   diff = p_hat_trt - p_hat_ctrl

   se_diff = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_ctrl
                     + p_hat_trt * (1 - p_hat_trt) / n_trt)
   z = stats.norm.ppf(0.975)
   ci_low = diff - z * se_diff
   ci_high = diff + z * se_diff

   print("=== Wald 95% CI for Treatment Effect ===")
   print(f"Difference (trt - ctrl): {diff:.4f}")
   print(f"SE:                      {se_diff:.4f}")
   print(f"95% CI:                  ({ci_low:.4f}, {ci_high:.4f})")
   print(f"CI contains 0?           {'Yes' if ci_low <= 0 <= ci_high else 'No'}")

11.2.3 Multiparameter Wald Test
---------------------------------

When there are multiple parameters, we cannot simply compute a single
"distance divided by standard error." Instead, we need to account for
the correlations between the different parameter estimates. The Fisher
information matrix :math:`I(\hat\theta)` captures both the precision of
each estimate (diagonal entries) and how the estimates co-vary
(off-diagonal entries).

For a :math:`p`-dimensional parameter, testing
:math:`H_0: \theta = \theta_0` uses

.. math::

   W_{\text{W}}
   = n\,(\hat\theta - \theta_0)^\top\,
     I(\hat\theta)\,(\hat\theta - \theta_0)
   \;\xrightarrow{d}\; \chi^2_p.

This is a *quadratic form*---the multivariate generalisation of "squared
distance in standard-error units." The information matrix acts as a
metric that stretches and rotates the distance to account for the shape
of the likelihood surface. When the parameters are uncorrelated,
:math:`I` is diagonal and this reduces to a sum of individual Wald
statistics, one per parameter.


11.3 The Score (Rao) Test
===========================

The score test, proposed by C. R. Rao (1948), is the third member of the
classical trinity. Its distinctive advantage is that it requires fitting
the model **only under the null hypothesis**, making it especially
useful when the alternative model is expensive to fit.

.. admonition:: Intuition

   The score test asks: "Standing at the null value :math:`\theta_0`, is
   the slope of the log-likelihood steep enough to suggest we should
   move to a different value?" If the log-likelihood is climbing steeply
   away from :math:`\theta_0`, the null is a poor fit and should be
   rejected.

11.3.1 Derivation
-------------------

The key insight is that the score function :math:`S(\theta)` is zero at
the MLE but typically nonzero at the null value :math:`\theta_0`. If the
null is true, the score at :math:`\theta_0` should be "small" (close to
zero); if the null is false, the score at :math:`\theta_0` should be
"large".

Under :math:`H_0`, we know from :ref:`ch9_mle_theory` that

.. math::

   \frac{1}{\sqrt{n}}\,S(\theta_0)
   \;\xrightarrow{d}\;
   \mathcal{N}\bigl(0,\; I(\theta_0)\bigr).

To turn this into a test statistic, we standardise the score by its
variance. The variance of the score under the null is exactly
:math:`n\,I(\theta_0)` (this is, in fact, one of the definitions of
Fisher information). Therefore, the **score statistic** (sometimes called
the Rao statistic or Lagrange multiplier statistic) is

.. math::

   W_{\text{S}}
   = \frac{S(\theta_0)^2}{n\,I(\theta_0)}
   = \frac{1}{n}\,S(\theta_0)^\top\,I(\theta_0)^{-1}\,S(\theta_0)
   \;\xrightarrow{d}\; \chi^2_r

under :math:`H_0`, where :math:`r` is the number of constraints. The
logic is the same as any z-test: divide a quantity by its standard
deviation to get something approximately standard normal, then square it
to get a chi-squared. Here, the "quantity" is the score and its
"standard deviation" comes from the Fisher information.

In the scalar case (one parameter, one constraint), the multivariate
formula simplifies to

.. math::

   W_{\text{S}}
   = \frac{\left[\sum_{i=1}^n
       \frac{\partial}{\partial\theta}\log f(x_i\mid\theta_0)
     \right]^2}
     {n\,I(\theta_0)}.

The numerator is the squared total score (the slope of the
log-likelihood at :math:`\theta_0`), and the denominator normalises it
by the expected variability of the score under the null.

Now let's apply the score test to our A/B test. Under :math:`H_0`,
both groups share rate :math:`p_0 = p_{\text{pooled}}`. The score for
the treatment group's rate, evaluated at the pooled rate, tells us
whether the data want to "move away" from the common rate.

.. code-block:: python

   # Score (Rao) Test for A/B test
   import numpy as np
   from scipy import stats

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   # Under H0: common rate
   p_0 = (x_ctrl + x_trt) / (n_ctrl + n_trt)  # pooled rate

   # Score for comparing two binomials under H0:
   # The score for the difference parameter, evaluated at H0, is:
   # S = x_trt/p_0 - (n_trt - x_trt)/(1-p_0) - [x_ctrl/p_0 - (n_ctrl - x_ctrl)/(1-p_0)]
   # Simplifies to: (x_trt - n_trt*p_0)/[p_0*(1-p_0)] - (x_ctrl - n_ctrl*p_0)/[p_0*(1-p_0)]
   # which equals (x_trt/n_trt - x_ctrl/n_ctrl) * n / [p_0*(1-p_0)] when n_trt = n_ctrl = n/2

   # More directly: use the pooled SE (evaluated under H0)
   se_pooled = np.sqrt(p_0 * (1 - p_0) * (1/n_ctrl + 1/n_trt))
   diff = x_trt/n_trt - x_ctrl/n_ctrl
   Z_S = diff / se_pooled
   W_S = Z_S**2
   p_S = 2 * (1 - stats.norm.cdf(abs(Z_S)))

   print("=== Score (Rao) Test ===")
   print(f"H0: p_trt = p_ctrl = {p_0:.4f} (pooled)")
   print(f"Score-based Z:        {Z_S:.4f}")
   print(f"Score statistic (Z^2):{W_S:.4f}")
   print(f"p-value (two-sided):  {p_S:.4f}")
   print(f"Reject H0 at 5%?     {'Yes' if p_S < 0.05 else 'No'}")

11.3.2 Practical Computation
------------------------------

The score test requires:

1. The score :math:`S(\theta_0)`, evaluated at the null value.
2. The Fisher information :math:`I(\theta_0)` at the null value.

Neither quantity requires the MLE :math:`\hat\theta`. This is a major
computational advantage when fitting under the alternative is difficult
(e.g., in large generalised linear mixed models or complex latent
variable models).

11.3.3 Example: Testing a Normal Mean
----------------------------------------

For :math:`X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2_0)` with
known variance, testing :math:`H_0: \mu = \mu_0`:

- Score: the derivative of the log-likelihood evaluated at :math:`\mu_0`
  is :math:`S(\mu_0) = \frac{n(\bar x - \mu_0)}{\sigma^2_0}`. This
  measures how steeply the log-likelihood is rising at the null value.
  If :math:`\bar x` is far from :math:`\mu_0`, the slope is large.
- Fisher information:
  :math:`I(\mu_0) = 1/\sigma^2_0`. (Recall from the normal MLE that the
  information about the mean is inversely proportional to the variance.)

Plugging into the score statistic formula:

.. math::

   W_{\text{S}}
   = \frac{[n(\bar x - \mu_0)/\sigma^2_0]^2}
          {n/\sigma^2_0}
   = \frac{n(\bar x - \mu_0)^2}{\sigma^2_0}.

This is exactly the square of the familiar :math:`z`-test statistic
:math:`Z = \sqrt{n}(\bar x - \mu_0)/\sigma_0`. In other words, for the
normal distribution with known variance, the score test, the Wald test,
and the LRT all give the same answer---they coincide exactly, not just
asymptotically.

Now let's put all three tests side by side on our A/B test data, so you
can see how they compare:

.. code-block:: python

   # All three tests side by side on the A/B test data
   import numpy as np
   from scipy import stats

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   diff = p_hat_trt - p_hat_ctrl
   p_pooled = (x_ctrl + x_trt) / (n_ctrl + n_trt)

   def binom_ll(x, n, p):
       p = np.clip(p, 1e-15, 1 - 1e-15)
       return x * np.log(p) + (n - x) * np.log(1 - p)

   # --- LRT ---
   ll_null = binom_ll(x_ctrl, n_ctrl, p_pooled) + binom_ll(x_trt, n_trt, p_pooled)
   ll_alt = binom_ll(x_ctrl, n_ctrl, p_hat_ctrl) + binom_ll(x_trt, n_trt, p_hat_trt)
   W_LR = 2 * (ll_alt - ll_null)
   p_LR = 1 - stats.chi2.cdf(W_LR, df=1)

   # --- Wald ---
   se_wald = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_ctrl
                     + p_hat_trt * (1 - p_hat_trt) / n_trt)
   W_W = (diff / se_wald)**2
   p_W = 1 - stats.chi2.cdf(W_W, df=1)

   # --- Score ---
   se_score = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_ctrl + 1/n_trt))
   W_S = (diff / se_score)**2
   p_S = 1 - stats.chi2.cdf(W_S, df=1)

   print("=" * 55)
   print("ALL THREE TESTS ON THE SAME A/B TEST DATA")
   print("=" * 55)
   print(f"Control: {x_ctrl}/{n_ctrl} = {p_hat_ctrl:.3f}")
   print(f"Treatment: {x_trt}/{n_trt} = {p_hat_trt:.3f}")
   print(f"Pooled: {p_pooled:.4f}")
   print()
   print(f"{'Test':<12s} {'Statistic':>10s} {'p-value':>10s} {'Reject?':>10s}")
   print("-" * 45)
   print(f"{'LRT':<12s} {W_LR:>10.4f} {p_LR:>10.4f} "
         f"{'Yes' if p_LR < 0.05 else 'No':>10s}")
   print(f"{'Wald':<12s} {W_W:>10.4f} {p_W:>10.4f} "
         f"{'Yes' if p_W < 0.05 else 'No':>10s}")
   print(f"{'Score':<12s} {W_S:>10.4f} {p_S:>10.4f} "
         f"{'Yes' if p_S < 0.05 else 'No':>10s}")
   print()
   print("Note: All three tests are asymptotically equivalent.")
   print("Typical finite-sample ordering: W_S <= W_LR <= W_W")
   print(f"Actual: {W_S:.4f} <= {W_LR:.4f} <= {W_W:.4f}? "
         f"{'Yes' if W_S <= W_LR <= W_W else 'Approximate'}")


11.4 Comparison of the Three Tests
=====================================

The LRT, Wald, and score tests are asymptotically equivalent---they all
converge to :math:`\chi^2_r` under the null and give identical results
in the limit. In finite samples, however, they can disagree.

11.4.1 Geometric Interpretation
---------------------------------

All three tests measure the "distance" between the null and the MLE, but
each measures a different aspect of the likelihood surface:

- **LRT:** measures the **vertical** drop in log-likelihood from MLE to
  null.
- **Wald:** measures the **horizontal** distance from MLE to null,
  scaled by curvature at the MLE.
- **Score:** measures the **slope** of the log-likelihood at the null.

.. admonition:: Intuition

   Picture the log-likelihood as a hill. You are standing at the MLE (the
   summit). The LRT looks *down* and asks how far the null is below you.
   The Wald test looks *sideways* and asks how far the null is from you
   horizontally. The score test stands at the null and looks at how steep
   the ground is---if the ground is steep, you are probably not at the
   top, so the null is likely wrong.

Let's visualise this on a simple one-parameter example, then return to
the A/B test:

.. code-block:: python

   # Geometric comparison: the three tests measure different things
   import numpy as np
   from scipy import stats

   # Simple Poisson example for clear visualisation
   n, x_bar = 60, 5.2
   total = n * x_bar
   lambda_0 = 4.0  # null hypothesis

   # Log-likelihood function (up to constant)
   def poisson_ll(lam):
       return total * np.log(lam) - n * lam

   # LRT: vertical drop
   ll_mle = poisson_ll(x_bar)
   ll_null = poisson_ll(lambda_0)
   vertical_drop = ll_mle - ll_null

   # Wald: horizontal distance / SE
   se = np.sqrt(x_bar / n)
   horizontal_dist = (x_bar - lambda_0) / se

   # Score: slope at null
   score_at_null = total / lambda_0 - n
   fisher_at_null = 1.0 / lambda_0
   score_standardised = score_at_null / np.sqrt(n * fisher_at_null)

   print("Geometric interpretation (Poisson, lambda_0 = 4.0, MLE = 5.2):")
   print(f"  LRT  - vertical drop in log-lik:     {vertical_drop:.4f}")
   print(f"  Wald - distance in SE units:          {horizontal_dist:.4f}")
   print(f"  Score - standardised slope at null:   {score_standardised:.4f}")
   print()
   print(f"  LRT statistic  = 2 * drop    = {2*vertical_drop:.4f}")
   print(f"  Wald statistic = dist^2      = {horizontal_dist**2:.4f}")
   print(f"  Score statistic= slope^2     = {score_standardised**2:.4f}")

11.4.2 Ordering in Finite Samples
------------------------------------

For many common models (especially those in the exponential family), the
test statistics satisfy the inequality

.. math::

   W_{\text{S}} \;\leq\; W_{\text{LR}} \;\leq\; W_{\text{W}}.

Why does this ordering arise? Recall the geometric picture: the Wald
test uses the curvature at the MLE to measure the distance to the null,
while the score test uses the curvature at the null. Typically the
log-likelihood is flatter at the MLE (the peak) than at the null
(further from the peak), so the Wald test "under-corrects" for
curvature and produces a larger statistic. The LRT, which uses the
actual height difference, naturally falls between these two curvature-based
approximations.

This means the Wald test tends to reject most often (is most liberal),
the score test rejects least often (is most conservative), and the LRT
is in between. However, this ordering is not universal---it can be
violated when the log-likelihood is highly non-quadratic or when
parameters lie near boundaries.

11.4.3 Relative Merits
------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 30

   * - Test
     - Advantages
     - Disadvantages
   * - LRT
     - Intuitive; good finite-sample performance; naturally handles
       composite nulls
     - Requires fitting model under both null and alternative
   * - Wald
     - Only requires fitting under alternative; directly yields
       confidence intervals; easy to compute
     - Can be unreliable for nonlinear parameters or small samples;
       not invariant to reparameterisation
   * - Score
     - Only requires fitting under null; invariant to
       reparameterisation; useful for model building (adding terms)
     - Less intuitive; requires Fisher information at the null

A notable deficiency of the Wald test is that it is **not invariant
under reparameterisation**. If we test :math:`H_0: \theta = \theta_0`
versus :math:`H_0: g(\theta) = g(\theta_0)` for a nonlinear :math:`g`,
the Wald test can give different results. The LRT and score test do not
suffer from this problem.

.. admonition:: Common Pitfall

   The Wald test is the default output of most statistical software
   packages because it only requires fitting one model. However, for
   parameters near boundaries (e.g., testing whether a variance
   component is zero) or with highly nonlinear parameterisations, the
   Wald test can be badly misleading. In such situations, always compute
   the LRT as a check.

Let's demonstrate reparameterisation non-invariance of the Wald test:

.. code-block:: python

   # Wald test is NOT invariant to reparameterisation
   import numpy as np
   from scipy import stats

   # Test H0: lambda = 4 for Poisson data
   n = 30
   np.random.seed(42)
   data = np.random.poisson(5.0, size=n)
   x_bar = np.mean(data)
   lambda_0 = 4.0

   # Wald test on lambda scale
   se_lambda = np.sqrt(x_bar / n)
   W_lambda = ((x_bar - lambda_0) / se_lambda)**2
   p_lambda = 1 - stats.chi2.cdf(W_lambda, df=1)

   # Wald test on log(lambda) scale (reparameterised)
   # MLE of log(lambda) = log(x_bar), H0: log(lambda) = log(4)
   # SE via delta method: SE(log(lambda_hat)) = SE(lambda_hat) / lambda_hat
   mle_log = np.log(x_bar)
   null_log = np.log(lambda_0)
   se_log = se_lambda / x_bar
   W_log = ((mle_log - null_log) / se_log)**2
   p_log = 1 - stats.chi2.cdf(W_log, df=1)

   # LRT is the same regardless of parameterisation
   ll_mle = np.sum(data) * np.log(x_bar) - n * x_bar
   ll_null = np.sum(data) * np.log(lambda_0) - n * lambda_0
   W_LR = 2 * (ll_mle - ll_null)
   p_LR = 1 - stats.chi2.cdf(W_LR, df=1)

   print("Reparameterisation non-invariance of the Wald test:")
   print(f"  Wald on lambda scale:      W = {W_lambda:.4f}, p = {p_lambda:.4f}")
   print(f"  Wald on log(lambda) scale: W = {W_log:.4f}, p = {p_log:.4f}")
   print(f"  LRT (invariant):           W = {W_LR:.4f}, p = {p_LR:.4f}")
   print(f"\n  The two Wald statistics differ by "
         f"{abs(W_lambda - W_log):.4f} -- that's the problem.")


11.5 Confidence Intervals from Fisher Information (Wald-Type CIs)
===================================================================

The most common confidence intervals in practice are Wald-type CIs,
derived directly from the asymptotic normality of the MLE.

11.5.1 Construction
---------------------

From the asymptotic result in :ref:`ch9_mle_theory`:

.. math::

   \hat\theta
   \;\stackrel{\text{approx}}{\sim}\;
   \mathcal{N}\!\left(\theta,\;
     \frac{1}{n\,I(\theta)}\right).

This implies

.. math::

   P\!\left(
     \hat\theta - z_{\alpha/2}\,\text{se}(\hat\theta)
     \;\leq\; \theta \;\leq\;
     \hat\theta + z_{\alpha/2}\,\text{se}(\hat\theta)
   \right) \approx 1 - \alpha,

where the standard error is

.. math::

   \text{se}(\hat\theta)
   = \sqrt{\frac{1}{n\,I(\hat\theta)}}

and :math:`z_{\alpha/2}` is the :math:`(1-\alpha/2)` quantile of the
standard normal (for a 95% CI, this is 1.96).

The standard error shrinks with :math:`\sqrt{n}` (more data means
more precision) and grows when the Fisher information
:math:`I(\hat\theta)` is small (parameters that are hard to pin down
have wider intervals). Plugging in the MLE :math:`\hat\theta` for the
unknown true parameter in the Fisher information is justified by the
consistency of the MLE: as :math:`n` grows,
:math:`I(\hat\theta) \to I(\theta)`.

.. topic:: Wald Confidence Interval

   An approximate :math:`100(1-\alpha)\%` confidence interval for
   :math:`\theta` is

   .. math::

      \hat\theta \;\pm\; z_{\alpha/2}\;
      \sqrt{\frac{1}{n\,I(\hat\theta)}}.

   For a 95% interval, :math:`z_{0.025} = 1.96`.

11.5.2 Using Observed Information
-----------------------------------

An alternative replaces the expected information (an average over all
possible datasets) with the *observed* information (computed from the
actual dataset at hand):

.. math::

   \text{se}(\hat\theta)
   = \frac{1}{\sqrt{J_n(\hat\theta)}},
   \qquad
   J_n(\hat\theta) = -\ell''(\hat\theta).

This is often preferred because it captures the actual curvature of the
log-likelihood in the particular dataset, rather than an expectation over
datasets.

11.5.3 Limitations of Wald CIs
---------------------------------

Wald CIs rely on a local quadratic (Gaussian) approximation to the
log-likelihood. This approximation can be poor when:

- The sample size is small.
- The log-likelihood is markedly asymmetric (skewed).
- The parameter is near a boundary (e.g., a variance near zero, a
  probability near 0 or 1).
- The parameterisation is "unnatural" (e.g., using :math:`\sigma`
  instead of :math:`\log\sigma`).

In these situations, profile-likelihood or likelihood-ratio CIs (next
sections) are preferred.


11.6 Profile Likelihood Confidence Intervals
===============================================

Profile-likelihood CIs overcome many of the limitations of Wald CIs by
working directly with the shape of the likelihood surface. For our A/B
test, the profile CI will respect the natural asymmetry of the binomial
likelihood near boundary values.

11.6.1 Definition: Profile Likelihood
---------------------------------------

When the full parameter is :math:`\theta = (\psi, \lambda)` with
:math:`\psi` the parameter of interest and :math:`\lambda` a nuisance
parameter, the **profile likelihood** for :math:`\psi` is

.. math::

   L_P(\psi) = \max_{\lambda}\; L(\psi, \lambda),

and the **profile log-likelihood** is

.. math::

   \ell_P(\psi) = \max_{\lambda}\; \ell(\psi, \lambda)
   = \ell(\psi, \hat\lambda_\psi),

where :math:`\hat\lambda_\psi` is the MLE of :math:`\lambda` for fixed
:math:`\psi`.

Even when there is only a single parameter, the profile likelihood
concept is useful: it is simply :math:`\ell_P(\theta) = \ell(\theta)`.

.. admonition:: What's the intuition?

   The profile likelihood answers the question: "For each possible value
   of the parameter of interest :math:`\psi`, what is the best-case
   log-likelihood after optimally choosing all the nuisance parameters?"
   It gives you a one-dimensional "slice" through a potentially
   high-dimensional likelihood surface, preserving the shape that matters
   for inference about :math:`\psi`.

11.6.2 Construction of Profile-Likelihood CIs
------------------------------------------------

The profile-likelihood confidence interval is obtained by inverting the
likelihood ratio test. The key idea is: *a confidence interval is the
set of all parameter values that would not be rejected by a hypothesis
test*. For each candidate value :math:`\psi`, we run a likelihood ratio
test of :math:`H_0: \psi = \psi`; the values that pass form the CI.

Formally, a :math:`100(1-\alpha)\%` confidence region for
:math:`\psi` is the set

.. math::

   \bigl\{\psi : 2[\ell(\hat\psi, \hat\lambda)
     - \ell_P(\psi)] \leq \chi^2_{1,\,1-\alpha}\bigr\}.

Equivalently, find all values of :math:`\psi` for which the profile
log-likelihood does not drop by more than :math:`\chi^2_{1,1-\alpha}/2`
from its maximum:

.. math::

   \bigl\{\psi : \ell_P(\psi) \geq \ell_P(\hat\psi)
     - \tfrac{1}{2}\chi^2_{1,\,1-\alpha}\bigr\}.

For a 95% CI, the cutoff is :math:`\chi^2_{1,0.95}/2 = 3.84/2 = 1.92`.

In the mountain analogy: draw a horizontal line 1.92 units below the
summit of the profile log-likelihood. Every parameter value whose profile
log-likelihood sits above this line is in the confidence interval. The
endpoints of the CI are the two points where the profile log-likelihood
crosses this threshold.

Let's build a profile-likelihood CI for the treatment effect in our A/B
test. We profile over the difference :math:`\delta = p_{\text{trt}} -
p_{\text{ctrl}}`:

.. code-block:: python

   # Profile Likelihood CI for A/B test treatment effect
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize_scalar, brentq

   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   def binom_ll(x, n, p):
       p = np.clip(p, 1e-15, 1 - 1e-15)
       return x * np.log(p) + (n - x) * np.log(1 - p)

   # Profile log-likelihood as a function of delta = p_trt - p_ctrl
   # For each delta, maximise over p_ctrl (nuisance parameter)
   def profile_loglik(delta):
       def neg_ll(p_ctrl):
           p_trt = p_ctrl + delta
           if p_ctrl <= 0 or p_ctrl >= 1 or p_trt <= 0 or p_trt >= 1:
               return 1e10
           return -(binom_ll(x_ctrl, n_ctrl, p_ctrl) +
                    binom_ll(x_trt, n_trt, p_trt))
       result = minimize_scalar(neg_ll, bounds=(0.001, 1 - delta - 0.001 if delta > 0
                                                 else 0.001),
                                method='bounded')
       return -result.fun

   # MLE of delta
   mle_delta = x_trt/n_trt - x_ctrl/n_ctrl
   ll_max = profile_loglik(mle_delta)

   # Cutoff for 95% CI
   chi2_cutoff = stats.chi2.ppf(0.95, df=1)
   threshold = ll_max - chi2_cutoff / 2

   # Find CI endpoints by root-finding
   def objective(delta):
       return profile_loglik(delta) - threshold

   pl_low = brentq(objective, -0.10, mle_delta)
   pl_high = brentq(objective, mle_delta, 0.15)

   # Compare to Wald CI
   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   se_wald = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_ctrl
                     + p_hat_trt * (1 - p_hat_trt) / n_trt)
   z = stats.norm.ppf(0.975)
   wald_low = mle_delta - z * se_wald
   wald_high = mle_delta + z * se_wald

   print("=== Profile Likelihood CI vs Wald CI ===")
   print(f"MLE of delta (trt - ctrl): {mle_delta:.4f}")
   print()
   print(f"Wald 95% CI:    ({wald_low:.4f}, {wald_high:.4f})")
   print(f"  Width:        {wald_high - wald_low:.4f}")
   print(f"  Symmetric:    Yes (by construction)")
   print()
   print(f"Profile 95% CI: ({pl_low:.4f}, {pl_high:.4f})")
   print(f"  Width:        {pl_high - pl_low:.4f}")
   print(f"  Symmetric:    {'~Yes' if abs((pl_high - mle_delta) - (mle_delta - pl_low)) < 0.001 else 'No'}")
   print()
   print("Note: For large n, both CIs are very similar.")
   print("Differences emerge with small n or parameters near boundaries.")

11.6.3 Advantages Over Wald CIs
----------------------------------

1. **Respects likelihood shape.** Profile CIs capture asymmetry in the
   likelihood; if the log-likelihood is skewed, the CI will be
   asymmetric rather than forcibly symmetric around the MLE.

2. **Respects parameter boundaries.** If :math:`\psi` is constrained
   (e.g., :math:`\sigma^2 > 0`), the profile CI will not extend beyond
   the boundary.

3. **Transformation invariant.** Because the likelihood is invariant
   under reparameterisation, profile CIs have the same coverage under
   any monotone transformation of :math:`\psi`.

4. **Better finite-sample coverage.** Empirical studies consistently
   show that profile-likelihood CIs have coverage closer to the nominal
   level than Wald CIs, especially for small to moderate :math:`n`.

Let's verify coverage properties with a simulation. We generate many
A/B tests and check how often each CI type contains the true difference:

.. code-block:: python

   # Coverage comparison: Wald CI vs Profile CI (small sample regime)
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize_scalar, brentq

   np.random.seed(42)
   true_p_ctrl = 0.10
   true_p_trt = 0.14
   true_delta = true_p_trt - true_p_ctrl
   n_per_group = 80   # small sample to highlight differences
   n_sims = 2000

   def binom_ll(x, n, p):
       p = np.clip(p, 1e-15, 1 - 1e-15)
       return x * np.log(p) + (n - x) * np.log(1 - p)

   wald_covers = 0
   profile_covers = 0

   for _ in range(n_sims):
       x_c = np.random.binomial(n_per_group, true_p_ctrl)
       x_t = np.random.binomial(n_per_group, true_p_trt)
       if x_c == 0 or x_t == 0 or x_c == n_per_group or x_t == n_per_group:
           continue

       p_c = x_c / n_per_group
       p_t = x_t / n_per_group
       delta_hat = p_t - p_c

       # Wald CI
       se = np.sqrt(p_c * (1 - p_c) / n_per_group +
                    p_t * (1 - p_t) / n_per_group)
       z = 1.96
       if delta_hat - z*se <= true_delta <= delta_hat + z*se:
           wald_covers += 1

       # Profile CI (simplified: single-parameter profile)
       def prof_ll(delta):
           def neg_ll(pc):
               pt = pc + delta
               if pc <= 0 or pc >= 1 or pt <= 0 or pt >= 1:
                   return 1e10
               return -(binom_ll(x_c, n_per_group, pc) +
                        binom_ll(x_t, n_per_group, pt))
           res = minimize_scalar(neg_ll, bounds=(0.001, 0.999), method='bounded')
           return -res.fun

       ll_max = prof_ll(delta_hat)
       cutoff = ll_max - stats.chi2.ppf(0.95, 1) / 2

       try:
           pl_lo = brentq(lambda d: prof_ll(d) - cutoff, -0.5, delta_hat)
           pl_hi = brentq(lambda d: prof_ll(d) - cutoff, delta_hat, 0.5)
           if pl_lo <= true_delta <= pl_hi:
               profile_covers += 1
       except (ValueError, RuntimeError):
           pass  # skip problematic cases

   print(f"=== Coverage Comparison (n = {n_per_group} per group) ===")
   print(f"True delta = {true_delta:.3f}")
   print(f"Simulations: {n_sims}")
   print(f"Wald CI coverage:    {wald_covers/n_sims:.3f}  (nominal: 0.950)")
   print(f"Profile CI coverage: {profile_covers/n_sims:.3f}  (nominal: 0.950)")
   print("(Profile CI typically has coverage closer to 0.95)")

11.6.4 Computational Procedure
--------------------------------

To compute a profile-likelihood CI numerically:

1. Compute the MLE :math:`(\hat\psi, \hat\lambda)` and record
   :math:`\ell_P(\hat\psi) = \ell(\hat\psi, \hat\lambda)`.

2. For a grid of :math:`\psi` values (or using root-finding), compute
   :math:`\ell_P(\psi) = \max_\lambda \ell(\psi, \lambda)`.

3. Find the two roots :math:`\psi_L` and :math:`\psi_U` of

   .. math::

      \ell_P(\psi) = \ell_P(\hat\psi) - \tfrac{1}{2}\chi^2_{1,1-\alpha}.

   These are the endpoints of the CI.


11.7 Likelihood Ratio Confidence Regions
==========================================

The profile-likelihood CI is a special case of a broader technique:
**inverting the LRT** to obtain confidence regions.

11.7.1 General Construction
-----------------------------

The same "inversion" logic from profile-likelihood CIs extends to
the full parameter vector. A :math:`100(1-\alpha)\%` confidence region
for the full :math:`p`-dimensional parameter :math:`\theta` is

.. math::

   C_{1-\alpha}
   = \bigl\{\theta :
     2[\ell(\hat\theta) - \ell(\theta)] \leq \chi^2_{p,\,1-\alpha}
   \bigr\}.

This is the set of all parameter values that, if tested as
:math:`H_0: \theta = \theta_0`, would **not** be rejected by the LRT at
level :math:`\alpha`. Geometrically, it is the set of all points on the
log-likelihood surface that lie within a certain "height" of the summit.
The chi-squared quantile :math:`\chi^2_{p,1-\alpha}` controls how far
below the peak we are willing to go. With more parameters (:math:`p`
larger), the critical value is larger, reflecting the need for a wider
region in higher dimensions.

11.7.2 Geometry
-----------------

In two dimensions, the confidence region is bounded by a contour of the
log-likelihood surface at height
:math:`\ell(\hat\theta) - \chi^2_{2,1-\alpha}/2`. If the log-likelihood
is well approximated by a quadratic, this contour is an ellipse. If the
quadratic approximation is poor, the region can be non-ellipsoidal,
reflecting the true shape of the likelihood.

11.7.3 Relationship to Wald Ellipses
---------------------------------------

The Wald confidence region is

.. math::

   \bigl\{\theta :
     (\hat\theta - \theta)^\top\,
     n\,I(\hat\theta)\,
     (\hat\theta - \theta) \leq \chi^2_{p,\,1-\alpha}
   \bigr\}.

This is a quadratic form in :math:`(\hat\theta - \theta)`, which
defines an **ellipse** centred at :math:`\hat\theta`. The shape and
orientation of the ellipse are determined by the Fisher information
matrix: directions in which the likelihood is sharply curved (high
information) produce short axes, while directions of low curvature
produce long axes. Correlated parameters tilt the ellipse off-axis.

This is always an ellipse centred at :math:`\hat\theta`. The LRT region
equals the Wald region when the log-likelihood is exactly quadratic
(which happens in exponential families with canonical sufficient
statistics). In general, they differ, and the LRT region is preferred
for its better coverage properties because it follows the actual
contours of the log-likelihood rather than a quadratic approximation.


11.8 Complete A/B Test Analysis: All Methods Together
======================================================

Now let's bring every method from this chapter together on our A/B test
data. This is the comprehensive analysis that a data scientist would
present to stakeholders.

.. code-block:: python

   # COMPREHENSIVE A/B TEST ANALYSIS
   # All three tests + all three CI methods + summary table
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize_scalar, brentq

   # === DATA ===
   n_ctrl, x_ctrl = 1000, 120
   n_trt, x_trt = 1000, 145

   p_hat_ctrl = x_ctrl / n_ctrl
   p_hat_trt = x_trt / n_trt
   diff = p_hat_trt - p_hat_ctrl
   p_pooled = (x_ctrl + x_trt) / (n_ctrl + n_trt)

   def binom_ll(x, n, p):
       p = np.clip(p, 1e-15, 1 - 1e-15)
       return x * np.log(p) + (n - x) * np.log(1 - p)

   # === THREE TESTS ===
   # LRT
   ll_null = binom_ll(x_ctrl, n_ctrl, p_pooled) + binom_ll(x_trt, n_trt, p_pooled)
   ll_alt = binom_ll(x_ctrl, n_ctrl, p_hat_ctrl) + binom_ll(x_trt, n_trt, p_hat_trt)
   W_LR = 2 * (ll_alt - ll_null)
   p_LR = 1 - stats.chi2.cdf(W_LR, df=1)

   # Wald
   se_wald = np.sqrt(p_hat_ctrl * (1 - p_hat_ctrl) / n_ctrl
                     + p_hat_trt * (1 - p_hat_trt) / n_trt)
   W_W = (diff / se_wald)**2
   p_W = 1 - stats.chi2.cdf(W_W, df=1)

   # Score
   se_score = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_ctrl + 1/n_trt))
   W_S = (diff / se_score)**2
   p_S = 1 - stats.chi2.cdf(W_S, df=1)

   # === THREE CI METHODS ===
   z = stats.norm.ppf(0.975)

   # Wald CI
   wald_ci = (diff - z * se_wald, diff + z * se_wald)

   # Score CI (using pooled SE)
   score_ci = (diff - z * se_score, diff + z * se_score)

   # Profile Likelihood CI
   def profile_loglik(delta):
       def neg_ll(p_ctrl):
           p_trt = p_ctrl + delta
           if p_ctrl <= 0 or p_ctrl >= 1 or p_trt <= 0 or p_trt >= 1:
               return 1e10
           return -(binom_ll(x_ctrl, n_ctrl, p_ctrl) +
                    binom_ll(x_trt, n_trt, p_trt))
       result = minimize_scalar(neg_ll, bounds=(0.001, 0.998), method='bounded')
       return -result.fun

   ll_max_profile = profile_loglik(diff)
   chi2_cut = stats.chi2.ppf(0.95, df=1)
   threshold = ll_max_profile - chi2_cut / 2

   pl_low = brentq(lambda d: profile_loglik(d) - threshold, -0.10, diff)
   pl_high = brentq(lambda d: profile_loglik(d) - threshold, diff, 0.15)
   profile_ci = (pl_low, pl_high)

   # === PRINT COMPREHENSIVE SUMMARY ===
   print("=" * 65)
   print("       COMPREHENSIVE A/B TEST ANALYSIS")
   print("=" * 65)
   print(f"\nData:")
   print(f"  Control:   {x_ctrl}/{n_ctrl} = {p_hat_ctrl:.1%} conversion rate")
   print(f"  Treatment: {x_trt}/{n_trt}  = {p_hat_trt:.1%} conversion rate")
   print(f"  Lift:      {diff:.1%} absolute ({diff/p_hat_ctrl:.1%} relative)")

   print(f"\n{'='*65}")
   print("HYPOTHESIS TESTS  (H0: treatment effect = 0)")
   print(f"{'='*65}")
   print(f"  {'Test':<12s} {'Statistic':>10s} {'p-value':>10s} {'Decision':>12s}")
   print(f"  {'-'*46}")
   print(f"  {'LRT':<12s} {W_LR:>10.4f} {p_LR:>10.4f} "
         f"{'Reject H0' if p_LR < 0.05 else 'Fail to reject':>12s}")
   print(f"  {'Wald':<12s} {W_W:>10.4f} {p_W:>10.4f} "
         f"{'Reject H0' if p_W < 0.05 else 'Fail to reject':>12s}")
   print(f"  {'Score':<12s} {W_S:>10.4f} {p_S:>10.4f} "
         f"{'Reject H0' if p_S < 0.05 else 'Fail to reject':>12s}")

   print(f"\n{'='*65}")
   print("CONFIDENCE INTERVALS  (95% CI for treatment effect)")
   print(f"{'='*65}")
   print(f"  {'Method':<12s} {'Lower':>10s} {'Upper':>10s} {'Width':>10s} {'Contains 0?':>12s}")
   print(f"  {'-'*56}")
   for name, ci in [("Wald", wald_ci), ("Score", score_ci), ("Profile LR", profile_ci)]:
       contains_zero = "Yes" if ci[0] <= 0 <= ci[1] else "No"
       print(f"  {name:<12s} {ci[0]:>10.4f} {ci[1]:>10.4f} "
             f"{ci[1]-ci[0]:>10.4f} {contains_zero:>12s}")

   print(f"\n{'='*65}")
   print("CONCLUSION")
   print(f"{'='*65}")
   all_reject = p_LR < 0.05 and p_W < 0.05 and p_S < 0.05
   if all_reject:
       print(f"  All three tests reject H0 at the 5% level.")
       print(f"  The treatment effect of {diff:.1%} is statistically significant.")
       print(f"  Estimated lift: {diff:.4f} (95% Wald CI: "
             f"{wald_ci[0]:.4f} to {wald_ci[1]:.4f})")
   else:
       no_reject = p_LR >= 0.05 and p_W >= 0.05 and p_S >= 0.05
       if no_reject:
           print(f"  None of the three tests reject H0 at the 5% level.")
           print(f"  The observed lift of {diff:.1%} is not statistically significant.")
           print(f"  We cannot conclude that the treatment is better than control.")
       else:
           print(f"  Tests disagree -- borderline result. Consider collecting more data.")


11.9 Multiple Testing Corrections
====================================

When performing many simultaneous hypothesis tests, the probability of
at least one false rejection (the **family-wise error rate**, FWER)
increases rapidly. Multiple testing corrections control this inflation.

Suppose the product team is not just testing one page redesign, but ten
different variations of the checkout page simultaneously. Now the risk
of a false positive is much higher.

11.9.1 The Problem
--------------------

If we perform :math:`m` independent tests, each at level :math:`\alpha`,
the probability of at least one false rejection under the global null
is

.. math::

   \text{FWER}
   = 1 - (1-\alpha)^m
   \;\approx\; m\alpha
   \quad\text{for small } \alpha.

The exact formula follows from independence: the probability that a
single test does *not* falsely reject is :math:`1-\alpha`, so the
probability that *all* :math:`m` tests avoid a false rejection is
:math:`(1-\alpha)^m`. One minus this gives the probability of at least
one false rejection. The approximation :math:`\approx m\alpha` comes
from the first-order Taylor expansion of :math:`(1-\alpha)^m` when
:math:`\alpha` is small.

With :math:`m = 20` tests at :math:`\alpha = 0.05`, the FWER is about
:math:`1 - 0.95^{20} \approx 0.64`---we have a 64% chance of at least
one false positive.

.. admonition:: Why does this matter?

   Imagine a tech company testing 10 different page designs against the
   current design, each at the 5% level. Even if *none* of the new
   designs are better, the company has a
   :math:`1 - 0.95^{10} \approx 0.40` chance of declaring at least one
   design "significantly better." Without correction, the company wastes
   engineering effort implementing a change that does not actually help.

.. code-block:: python

   # FWER inflation: probability of at least one false positive
   import numpy as np

   alpha = 0.05
   print(f"FWER for m simultaneous tests at alpha = {alpha}:")
   print(f"{'m':>4s}  {'FWER':>8s}")
   print("-" * 16)
   for m in [1, 2, 5, 10, 20, 50, 100]:
       fwer = 1 - (1 - alpha)**m
       print(f"{m:4d}  {fwer:8.3f}")

11.9.2 Bonferroni Correction
-------------------------------

The simplest and most conservative approach. To control the FWER at
level :math:`\alpha` across :math:`m` tests, conduct each individual
test at level :math:`\alpha/m`.

**Why it works.** By the union bound (Boole's inequality):

.. math::

   \text{FWER}
   = P\!\left(\bigcup_{j=1}^m \{\text{reject } H_{0j}\}\right)
   \leq \sum_{j=1}^m P(\text{reject } H_{0j})
   = m \cdot \frac{\alpha}{m}
   = \alpha.

Equivalently, multiply each *p*-value by :math:`m` and reject if the
adjusted *p*-value is below :math:`\alpha`:

.. math::

   p_j^{\text{adj}} = \min(m \cdot p_j,\; 1).

The :math:`\min(\cdot, 1)` ensures the adjusted *p*-value stays a valid
probability. The intuition for multiplying by :math:`m` is simple: if
you are running :math:`m` tests, the bar for significance should be
:math:`m` times higher than for a single test. A *p*-value of 0.04
might look significant on its own, but after running 20 tests, the
adjusted *p*-value is :math:`20 \times 0.04 = 0.80`---far from
significant.

**Strengths:** Simple, valid for any dependence structure among tests
(the union bound does not require independence).
**Weakness:** Very conservative when :math:`m` is large, leading to low
power (many truly significant effects get missed).

11.9.3 Holm's Step-Down Procedure
------------------------------------

Holm's method (1979) is a **uniformly more powerful** refinement of the
Bonferroni correction that is still valid under arbitrary dependence.

**Procedure:**

1. Order the :math:`m` *p*-values:
   :math:`p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}`.

2. For :math:`j = 1, 2, \ldots, m`:

   - If :math:`p_{(j)} > \alpha / (m - j + 1)`, stop and do **not**
     reject :math:`H_{0(j)}` or any subsequent hypothesis.
   - Otherwise, reject :math:`H_{0(j)}` and continue to :math:`j+1`.

**Why it is more powerful than Bonferroni.** The Bonferroni uses the
same threshold :math:`\alpha/m` for every test. Holm uses the threshold
:math:`\alpha/m` only for the smallest *p*-value, then
:math:`\alpha/(m-1)` for the next, and so on---larger (less stringent)
thresholds for the later tests.

**Adjusted *p*-values.** Rather than comparing each *p*-value against a
changing threshold, it is often more convenient to compute *adjusted*
*p*-values that can be compared directly against :math:`\alpha`.
Holm's adjusted *p*-values are computed as

.. math::

   \tilde p_{(j)}
   = \max_{k \leq j}\bigl[(m - k + 1)\,p_{(k)}\bigr],

capped at 1. Each raw *p*-value :math:`p_{(k)}` is multiplied by the
number of hypotheses still in play at step :math:`k`, namely
:math:`m - k + 1`. The running maximum ensures that the adjusted
*p*-values remain in sorted order (a requirement for the step-down logic
to be consistent). You then simply reject all hypotheses whose adjusted
*p*-value falls below :math:`\alpha`.

.. topic:: Example

   Suppose :math:`m = 4` tests yield *p*-values 0.001, 0.013, 0.040,
   0.520. At :math:`\alpha = 0.05`:

   - :math:`j=1`: Compare :math:`p_{(1)} = 0.001` to
     :math:`0.05/4 = 0.0125`. Reject.
   - :math:`j=2`: Compare :math:`p_{(2)} = 0.013` to
     :math:`0.05/3 = 0.0167`. Reject.
   - :math:`j=3`: Compare :math:`p_{(3)} = 0.040` to
     :math:`0.05/2 = 0.025`. Fail to reject; stop.

   We reject the first two hypotheses. Bonferroni (threshold 0.0125 for
   all) would only reject the first.

Let's apply both corrections to a realistic multi-variant A/B test
scenario: 10 page variations tested against the control.

.. code-block:: python

   # Multiple A/B tests: Bonferroni and Holm corrections
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Simulate 10 A/B tests against a control
   # Only variations 1 and 2 actually have a real effect
   n_per_group = 1000
   p_ctrl = 0.12
   true_effects = [0.03, 0.025, 0, 0, 0, 0, 0, 0, 0, 0]  # only 2 real effects
   m = len(true_effects)
   alpha = 0.05

   p_values = []
   for i, effect in enumerate(true_effects):
       x_c = np.random.binomial(n_per_group, p_ctrl)
       x_t = np.random.binomial(n_per_group, p_ctrl + effect)
       # Two-proportion z-test
       p_c = x_c / n_per_group
       p_t = x_t / n_per_group
       p_pool = (x_c + x_t) / (2 * n_per_group)
       se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per_group)
       z = (p_t - p_c) / se if se > 0 else 0
       pval = 2 * (1 - stats.norm.cdf(abs(z)))
       p_values.append(pval)

   p_values = np.array(p_values)

   # Bonferroni
   bonf_adj = np.minimum(p_values * m, 1.0)
   bonf_reject = bonf_adj < alpha

   # Holm's step-down
   sorted_idx = np.argsort(p_values)
   sorted_p = p_values[sorted_idx]
   holm_adj = np.zeros(m)
   running_max = 0.0
   for j in range(m):
       val = (m - j) * sorted_p[j]
       running_max = max(running_max, val)
       holm_adj[j] = min(running_max, 1.0)
   holm_adj_orig = np.zeros(m)
   holm_adj_orig[sorted_idx] = holm_adj
   holm_reject = holm_adj_orig < alpha

   # Raw (uncorrected)
   raw_reject = p_values < alpha

   print("=== Multi-Variant A/B Test: 10 Variations vs Control ===")
   print(f"(Variations 1-2 have real effects; 3-10 are null)\n")
   print(f"{'Var':>4s} {'True eff':>9s} {'p-value':>8s} {'Raw':>5s} "
         f"{'Bonf':>5s} {'Holm':>5s}")
   print("-" * 42)
   for j in range(m):
       effect_str = f"{true_effects[j]:.3f}" if true_effects[j] > 0 else "  0   "
       print(f"{j+1:4d} {effect_str:>9s} {p_values[j]:8.4f} "
             f"{'*' if raw_reject[j] else ' ':>5s} "
             f"{'*' if bonf_reject[j] else ' ':>5s} "
             f"{'*' if holm_reject[j] else ' ':>5s}")

   print(f"\nRaw rejections:        {raw_reject.sum()} "
         f"(may include false positives!)")
   print(f"Bonferroni rejections: {bonf_reject.sum()}")
   print(f"Holm rejections:       {holm_reject.sum()}")

   # Count false positives among null variations (3-10)
   raw_fp = raw_reject[2:].sum()
   bonf_fp = bonf_reject[2:].sum()
   holm_fp = holm_reject[2:].sum()
   print(f"\nFalse positives (among 8 null variations):")
   print(f"  Raw: {raw_fp}, Bonferroni: {bonf_fp}, Holm: {holm_fp}")


11.10 Practical Guidance: Choosing a Method
============================================

11.10.1 Which Test to Use
--------------------------

- **Default recommendation:** Use the LRT when computationally feasible.
  It has the best finite-sample properties in most settings.

- **Wald test:** Convenient when you already have the MLE and standard
  errors (e.g., output from most software packages). Be cautious with
  parameters near boundaries or highly nonlinear reparameterisations.

- **Score test:** Use when you want to test whether adding a parameter
  to a model is worthwhile, without fitting the larger model. Common in
  sequential model-building strategies.

11.10.2 Which Confidence Interval to Use
-----------------------------------------

- For routine use with moderate to large :math:`n`: Wald CIs are fast
  and simple.
- For small :math:`n`, skewed likelihoods, or parameters near
  boundaries: use profile-likelihood CIs.
- For simultaneous inference on multiple parameters: use LRT confidence
  regions.

11.10.3 When to Correct for Multiple Testing
----------------------------------------------

- If you are testing a pre-specified set of hypotheses and want to
  control the probability of **any** false rejection: use Holm (preferred
  over Bonferroni due to uniformly greater power).
- If you have thousands of tests (e.g., genomics) and can tolerate a
  controlled proportion of false discoveries: consider the
  Benjamini--Hochberg procedure (controls the false discovery rate
  rather than the FWER), though a full treatment is beyond this chapter.


11.11 Summary
===============

This chapter developed three interconnected frameworks for likelihood-
based inference, all applied to our A/B testing scenario:

1. **Hypothesis tests** (LRT, Wald, Score) all test
   :math:`H_0: \theta = \theta_0` using different aspects of the
   likelihood surface. All three converge to :math:`\chi^2` distributions
   under the null and are asymptotically equivalent. We applied all three
   to the same A/B test data and saw they agreed.

2. **Confidence intervals** can be derived from each test by inversion:

   - Wald CIs:
     :math:`\hat\theta \pm z_{\alpha/2}\,\text{se}(\hat\theta)`.
   - Profile-likelihood CIs: contour of the profile log-likelihood.
   - LRT regions: all :math:`\theta` not rejected by the LRT.

3. **Multiple testing corrections** (Bonferroni, Holm) control the
   family-wise error rate when many tests are performed simultaneously.
   Holm is uniformly more powerful than Bonferroni and should be the
   default choice.

Together with the MLE theory from :ref:`ch9_mle_theory` and the
analytical solutions from :ref:`ch10_analytical_mle`, these tools form a
complete toolkit for parametric inference using likelihood methods.
