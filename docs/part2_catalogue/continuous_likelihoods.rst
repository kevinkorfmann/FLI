.. _ch6_continuous:

==========================================
Chapter 6 --- Continuous Likelihoods
==========================================

This chapter extends the programme of :ref:`ch5_discrete` to continuous
distributions. For each family we state the probability density function (PDF),
construct the likelihood and log-likelihood for *n* i.i.d. observations,
derive the score function and Fisher information, and solve for the MLE.
When a closed-form MLE does not exist we explain why and outline the
numerical strategy.

The transition from discrete to continuous brings one conceptual shift: the
PMF becomes a PDF, and likelihood values are no longer bounded above by 1.
But the machinery---log-likelihood, score, information, MLE---works exactly the
same way. Let's dive in.

.. contents:: Distributions in this chapter
   :local:
   :depth: 1
   :class: this-will-duplicate-information-and-it-is-still-useful-here


.. _sec_normal:

6.1 Normal (Gaussian) Distribution
====================================

Motivation
----------

The Normal distribution is the workhorse of statistics. The Central Limit
Theorem guarantees that sums and averages of independent random variables
converge to it, making it a natural model for measurement error, biological
variation, and financial returns over short horizons. It is parametrised by
its mean :math:`\mu \in \mathbb{R}` and variance
:math:`\sigma^2 > 0`.

If you could only learn one distribution, this would be the one. Its
mathematical tractability is unmatched, and its MLE derivation serves as
the template for everything that follows.

PDF
---

.. math::

   f(x \mid \mu, \sigma^2)
     = \frac{1}{\sqrt{2\pi\sigma^2}}
       \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
   \qquad x \in \mathbb{R}.

Reading this formula: the factor :math:`\frac{1}{\sqrt{2\pi\sigma^2}}` is
a normalizing constant that ensures the density integrates to 1.  The
exponential term :math:`\exp(-(x-\mu)^2/(2\sigma^2))` is a *Gaussian bell
curve* centred at :math:`\mu`.  The larger the deviation :math:`|x - \mu|`,
the smaller the density.  The parameter :math:`\sigma^2` controls the width:
large :math:`\sigma^2` gives a wide, flat bell; small :math:`\sigma^2` gives
a narrow, tall one.

Likelihood
----------

For :math:`n` i.i.d. observations :math:`x_1, \dots, x_n`, the likelihood is
the product of the individual densities (just as for discrete distributions,
but now using the PDF rather than the PMF):

.. math::

   L(\mu, \sigma^2)
     = \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}
       \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
     = (2\pi\sigma^2)^{-n/2}
       \exp\!\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2\right).

Log-likelihood
--------------

Taking the logarithm of the likelihood, each factor contributes three terms:
:math:`-\frac{1}{2}\ln(2\pi)` from the normalizing constant,
:math:`-\frac{1}{2}\ln\sigma^2` from the :math:`\sigma` in the denominator,
and :math:`-(x_i-\mu)^2/(2\sigma^2)` from the exponential.  Summing over
all :math:`n` observations:

.. math::

   \ell(\mu,\sigma^2)
     = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2
       - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2.

Here's the key idea: the log-likelihood is a *downward-opening parabola* in
:math:`\mu` (for fixed :math:`\sigma^2`). This means it has a unique, global
maximum---which is exactly what makes the Normal so tractable.

Score functions
---------------

The score functions are the partial derivatives of the log-likelihood with
respect to each parameter.  They tell us how the log-likelihood changes as
we adjust :math:`\mu` or :math:`\sigma^2`.

**With respect to** :math:`\mu` (differentiating the quadratic term
:math:`-(x_i - \mu)^2/(2\sigma^2)` and using the chain rule):

.. math::

   \frac{\partial\ell}{\partial\mu}
     = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu).

This says: each observation :math:`x_i` exerts a "pull" on :math:`\mu`
proportional to the residual :math:`x_i - \mu`.  Points above the current
:math:`\mu` pull it up; points below pull it down.  At the MLE, these pulls
balance perfectly.

**With respect to** :math:`\sigma^2` (treating :math:`\tau = \sigma^2` as the
parameter, and differentiating both the :math:`\ln\sigma^2` and the
:math:`1/\sigma^2` terms):

.. math::

   \frac{\partial\ell}{\partial\tau}
     = -\frac{n}{2\tau} + \frac{1}{2\tau^2}\sum_{i=1}^{n}(x_i - \mu)^2.

The first term comes from :math:`-\frac{n}{2}\ln\tau` and penalizes large
variance (the density spreads out, getting shorter).  The second term rewards
large variance when the data are spread far from :math:`\mu`.  The MLE
balances these two effects.

Fisher information
------------------

The Fisher information matrix is found by taking the negative expectation
of the second derivatives of the log-likelihood.  For the Normal, the
second derivative with respect to :math:`\mu` is :math:`-n/\sigma^2`,
which does not depend on the data, so the expected value is just itself.
For a single observation:

.. math::

   \mathcal{I}(\mu, \sigma^2)
     = \begin{pmatrix}
         1/\sigma^2 & 0 \\
         0 & 1/(2\sigma^4)
       \end{pmatrix}.

The off-diagonal is zero, reflecting the remarkable fact that :math:`\bar{X}`
and :math:`S^2` are independent statistics --- a property unique to the
Normal family.  The :math:`1/\sigma^2` entry says that each observation
carries more information about :math:`\mu` when the variance is small
(data are tightly clustered, so the mean is easier to pin down).

.. admonition:: Why does this matter?

   The diagonal Fisher information matrix means that estimating the mean and
   estimating the variance are *independent* problems. Learning about one does
   not help you learn about the other. This orthogonality is one of the Normal
   distribution's most remarkable features.

MLE
---

Setting :math:`\partial\ell/\partial\mu = 0`:

.. math::

   \sum_{i=1}^{n}(x_i - \mu) = 0
   \;\Longrightarrow\;
   \hat{\mu} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.

Setting :math:`\partial\ell/\partial\tau = 0`:

.. math::

   \frac{n}{2\tau} = \frac{1}{2\tau^2}\sum_{i=1}^{n}(x_i - \hat\mu)^2
   \;\Longrightarrow\;
   \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2.

Note that the MLE for the variance divides by :math:`n`, not :math:`n-1`, and
is therefore biased (though consistent).

.. code-block:: python

   # Normal MLE: simulate data, compute MLE, verify score = 0
   import numpy as np

   np.random.seed(42)
   mu_true, sigma2_true = 5.0, 4.0
   n = 100
   data = np.random.normal(mu_true, np.sqrt(sigma2_true), size=n)

   mu_hat = data.mean()
   sigma2_hat = np.mean((data - mu_hat)**2)  # MLE divides by n

   # Scores at MLE
   score_mu = np.sum(data - mu_hat) / sigma2_hat
   score_sigma2 = -n / (2 * sigma2_hat) + np.sum((data - mu_hat)**2) / (2 * sigma2_hat**2)

   print(f"True: mu={mu_true}, sigma^2={sigma2_true}")
   print(f"MLE:  mu_hat={mu_hat:.4f}, sigma^2_hat={sigma2_hat:.4f}")
   print(f"Score at MLE: score_mu={score_mu:.10f}, score_sigma2={score_sigma2:.10f}")


.. _sec_exponential:

6.2 Exponential Distribution
==============================

Motivation
----------

The Exponential distribution models the time between events in a Poisson
process. If events occur at rate :math:`\lambda`, the waiting time until the
next event is :math:`\text{Exp}(\lambda)`. It is the only continuous
distribution with the **memoryless property**: knowing that you have already
waited :math:`t` units does not change the distribution of the remaining
wait.

The Exponential is to continuous distributions what the Geometric is to
discrete ones---it is the simplest waiting-time model, and a building block
for more complex lifetime distributions.

PDF
---

.. math::

   f(x \mid \lambda) = \lambda\, e^{-\lambda x},
   \qquad x \ge 0, \quad \lambda > 0.

The factor :math:`\lambda` ensures the density integrates to 1, and the
exponential decay :math:`e^{-\lambda x}` captures the idea that longer waits
are less probable.  Higher :math:`\lambda` means events happen more frequently,
so the density is concentrated near zero (short waiting times).

Likelihood
----------

Because the observations are independent, the joint density is the product
of the individual densities.  The :math:`n` factors of :math:`\lambda`
combine into :math:`\lambda^n`, and the exponentials multiply via
:math:`e^a \cdot e^b = e^{a+b}`:

.. math::

   L(\lambda) = \prod_{i=1}^{n}\lambda\,e^{-\lambda x_i}
              = \lambda^n \exp\!\left(-\lambda\sum_{i=1}^{n} x_i\right).

Log-likelihood
--------------

Taking the logarithm:

.. math::

   \ell(\lambda) = n\ln\lambda - \lambda\sum_{i=1}^{n} x_i.

The first term :math:`n\ln\lambda` increases without bound as
:math:`\lambda \to \infty` (more events per unit time is always "better" at
explaining having observed events at all).  The second term
:math:`-\lambda\sum x_i` provides the counterbalancing penalty: a high rate
makes the observed (long) waiting times unlikely.  The maximum occurs where
these two effects balance.

Score function
--------------

Differentiating:

.. math::

   S(\lambda) = \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} x_i.

This has exactly the same "pull up / pull down" structure we have seen
throughout.  The first term pulls :math:`\lambda` up (the data consist of
:math:`n` events, each providing evidence for a positive rate).  The second
term pulls it down (the total waiting time :math:`\sum x_i` limits how large
the rate can be).

Fisher information
------------------

The second derivative of the single-observation log-likelihood
:math:`\ell_1 = \ln\lambda - \lambda x` is :math:`-1/\lambda^2`, which does
not depend on the data:

.. math::

   \mathcal{I}(\lambda) = -E\!\left[-\frac{1}{\lambda^2}\right]
   = \frac{1}{\lambda^2}.

Higher rates are estimated more precisely (larger information), because the
relative precision :math:`\text{SE}/\hat{\lambda} = 1/\sqrt{n}` is constant.

MLE
---

Setting the score to zero and solving:

.. math::

   \frac{n}{\lambda} = \sum x_i
   \;\Longrightarrow\;
   \hat{\lambda} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}.

The MLE is the reciprocal of the sample mean---the natural estimate for a rate
when your data are waiting times. If the average wait is 5 minutes, you
estimate 1/5 events per minute.

.. code-block:: python

   # Exponential MLE: simulate, compute, verify
   import numpy as np

   np.random.seed(42)
   lam_true = 0.5
   n = 150
   data = np.random.exponential(1 / lam_true, size=n)

   lam_hat = 1 / data.mean()
   score_at_mle = n / lam_hat - data.sum()

   print(f"True lambda = {lam_true}")
   print(f"MLE lambda_hat = {lam_hat:.4f}")
   print(f"Score at MLE = {score_at_mle:.10f}")


.. _sec_gamma:

6.3 Gamma Distribution
========================

Motivation
----------

The Gamma distribution generalises the Exponential. While the Exponential
models the wait until one event, the Gamma models the wait until the
:math:`\alpha`-th event (when :math:`\alpha` is a positive integer this is
the Erlang distribution). More broadly, the Gamma is a flexible two-parameter
family for non-negative continuous data, widely used in survival analysis,
queueing theory, and Bayesian statistics as a conjugate prior for the Poisson
rate.

Think of it this way: the Exponential is the Gamma with :math:`\alpha = 1`.
By allowing :math:`\alpha` to vary, you get a family that can model both
strongly right-skewed data (small :math:`\alpha`) and nearly symmetric data
(large :math:`\alpha`).

PDF
---

With shape :math:`\alpha > 0` and rate :math:`\beta > 0`:

.. math::

   f(x \mid \alpha, \beta)
     = \frac{\beta^\alpha}{\Gamma(\alpha)}\,x^{\alpha-1}e^{-\beta x},
   \qquad x > 0.

Here :math:`\Gamma(\alpha) = \int_0^\infty t^{\alpha-1}e^{-t}\,dt` is the
Gamma function (the continuous generalization of the factorial:
:math:`\Gamma(n) = (n-1)!` for positive integers).

The term :math:`x^{\alpha-1}` controls the shape near zero: when
:math:`\alpha > 1`, the density starts at zero and rises to a mode before
falling; when :math:`\alpha < 1`, it shoots up to infinity at zero (a
J-shape); when :math:`\alpha = 1` the density starts at :math:`\beta` and
decays --- the Exponential.  The :math:`e^{-\beta x}` term governs the
exponential tail decay, with larger :math:`\beta` producing a steeper
drop-off.

Likelihood
----------

Multiplying the :math:`n` individual PDFs and collecting like terms (using
:math:`\prod x_i^{\alpha-1} = (\prod x_i)^{\alpha-1}`):

.. math::

   L(\alpha,\beta)
     = \frac{\beta^{n\alpha}}{\Gamma(\alpha)^n}
       \left(\prod_{i=1}^{n} x_i\right)^{\alpha-1}
       \exp\!\left(-\beta\sum_{i=1}^{n} x_i\right).

Log-likelihood
--------------

Taking the log of each factor:

.. math::

   \ell(\alpha,\beta)
     = n\alpha\ln\beta - n\ln\Gamma(\alpha)
       + (\alpha - 1)\sum_{i=1}^{n}\ln x_i
       - \beta\sum_{i=1}^{n} x_i.

The log-likelihood depends on the data only through two summaries:
:math:`\sum x_i` (the total) and :math:`\sum \ln x_i` (the sum of logs).
These are the **sufficient statistics** for the Gamma family.

Score functions
---------------

**With respect to** :math:`\beta` (differentiating
:math:`n\alpha\ln\beta - \beta\sum x_i`):

.. math::

   \frac{\partial\ell}{\partial\beta}
     = \frac{n\alpha}{\beta} - \sum_{i=1}^{n} x_i.

This has the familiar "pull up versus pull down" structure.

**With respect to** :math:`\alpha` (differentiating
:math:`n\alpha\ln\beta - n\ln\Gamma(\alpha) + (\alpha-1)\sum\ln x_i`):

.. math::

   \frac{\partial\ell}{\partial\alpha}
     = n\ln\beta - n\psi(\alpha) + \sum_{i=1}^{n}\ln x_i,

where :math:`\psi(\alpha) = \Gamma'(\alpha)/\Gamma(\alpha)` is the **digamma
function** --- the derivative of :math:`\ln\Gamma(\alpha)`.  This function
appears whenever we differentiate a log-likelihood involving the Gamma
function, and it has no simple closed-form inverse, which is why the Gamma
MLE requires numerical methods.

Fisher information
------------------

The Fisher information matrix for a single observation is

.. math::

   \mathcal{I}(\alpha,\beta)
     = \begin{pmatrix}
         \psi'(\alpha) & -1/\beta \\
         -1/\beta & \alpha/\beta^2
       \end{pmatrix},

where :math:`\psi'(\alpha)` is the **trigamma function** (the second
derivative of :math:`\ln\Gamma`).  The off-diagonal entry :math:`-1/\beta`
means the parameters are *not* orthogonal: estimating :math:`\alpha` and
:math:`\beta` jointly creates some correlation between the estimates.

MLE
---

The :math:`\beta` score equation is easy to solve.  Setting it to zero:

.. math::

   \hat{\beta} = \frac{n\alpha}{\sum x_i} = \frac{\alpha}{\bar{x}}.

This says: given the shape :math:`\alpha`, the rate is just
:math:`\alpha/\bar{x}` --- a natural relationship since
:math:`E[X] = \alpha/\beta`.

Substituting this expression for :math:`\beta` into the :math:`\alpha` score
equation eliminates :math:`\beta` and leaves a single equation in
:math:`\alpha`:

.. math::

   \ln\hat{\alpha} - \psi(\hat{\alpha})
     = \ln\bar{x} - \frac{1}{n}\sum_{i=1}^{n}\ln x_i
     = \ln\bar{x} - \overline{\ln x}.

The right-hand side is a fixed quantity computed from the data (note that by
Jensen's inequality, :math:`\ln\bar{x} \geq \overline{\ln x}`, so the
right-hand side is always non-negative). The left-hand side is a monotone
decreasing function of :math:`\alpha`, so a unique solution exists.  It can be
found by Newton's method, or by using the rough starting approximation:

.. math::

   \hat{\alpha} \approx \frac{0.5}{
     \ln\bar{x} - \overline{\ln x}
   }.

Once :math:`\hat{\alpha}` is found, :math:`\hat{\beta} = \hat{\alpha}/\bar{x}`.

.. code-block:: python

   # Gamma MLE using scipy.stats.fit and verifying with Newton's method
   import numpy as np
   from scipy.special import digamma, polygamma

   np.random.seed(42)
   alpha_true, beta_true = 3.0, 2.0
   n = 300
   data = np.random.gamma(alpha_true, 1 / beta_true, size=n)

   x_bar = data.mean()
   log_x_bar = np.log(x_bar)
   mean_log_x = np.log(data).mean()
   s = log_x_bar - mean_log_x  # key data summary

   # Newton's method for alpha
   alpha_hat = 0.5 / s  # initial approximation
   for _ in range(20):
       alpha_hat -= (np.log(alpha_hat) - digamma(alpha_hat) - s) / \
                    (1 / alpha_hat - polygamma(1, alpha_hat))
   beta_hat = alpha_hat / x_bar

   print(f"True: alpha={alpha_true}, beta={beta_true}")
   print(f"MLE:  alpha_hat={alpha_hat:.4f}, beta_hat={beta_hat:.4f}")

.. admonition:: Real-World Example

   **Rainfall modelling.** Hydrologists commonly model daily rainfall amounts
   (on days when it rains) using a Gamma distribution. The shape parameter
   captures how skewed the rainfall distribution is, while the rate controls
   the overall scale.

   .. code-block:: text

      Scenario: Daily rainfall amounts (mm) on 200 rainy days
      Data: positive values like 2.1, 0.3, 15.7, 8.2, ...
      Goal: estimate shape (alpha) and rate (beta)
      Approach: Gamma(alpha, beta) MLE via Newton on digamma equation

   .. code-block:: python

      # Rainfall modelling with Gamma distribution
      import numpy as np
      from scipy.stats import gamma as gamma_dist

      np.random.seed(42)
      alpha_true, beta_true = 2.0, 0.15  # shape, rate
      n_days = 200

      # Simulate daily rainfall amounts (mm) on rainy days
      rainfall = np.random.gamma(alpha_true, 1 / beta_true, size=n_days)

      # Fit using scipy (shape, loc, scale parametrisation)
      a_fit, loc_fit, scale_fit = gamma_dist.fit(rainfall, floc=0)
      beta_fit = 1 / scale_fit

      print(f"Rainfall stats: mean={rainfall.mean():.1f}mm, "
            f"std={rainfall.std():.1f}mm")
      print(f"True:  alpha={alpha_true}, beta={beta_true}")
      print(f"MLE:   alpha={a_fit:.4f}, beta={beta_fit:.4f}")
      print(f"Mean rainfall (estimated): {a_fit/beta_fit:.1f}mm")


.. _sec_beta:

6.4 Beta Distribution
=======================

Motivation
----------

The Beta distribution is defined on the interval :math:`(0,1)` and is the
standard model for proportions, probabilities, and rates. It is the conjugate
prior for the Bernoulli and Binomial parameters in Bayesian inference, and
its two shape parameters give it remarkable flexibility: it can be uniform,
U-shaped, J-shaped, or bell-shaped.

If your data are naturally bounded between 0 and 1---batting averages,
completion rates, proportions---the Beta is the distribution to reach for.

PDF
---

With shape parameters :math:`\alpha > 0` and :math:`\beta > 0`:

.. math::

   f(x \mid \alpha, \beta)
     = \frac{1}{B(\alpha,\beta)}\,x^{\alpha-1}(1-x)^{\beta-1},
   \qquad 0 < x < 1,

where :math:`B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)`
is the Beta function (a normalizing constant).

The structure is reminiscent of the Bernoulli PMF:
:math:`x^{\alpha-1}` "rewards" values near 1 (when :math:`\alpha > 1`), and
:math:`(1-x)^{\beta-1}` rewards values near 0 (when :math:`\beta > 1`).
When :math:`\alpha = \beta`, the density is symmetric around 0.5.  When
:math:`\alpha > \beta`, the distribution is skewed toward 1; when
:math:`\alpha < \beta`, toward 0.  The special case
:math:`\alpha = \beta = 1` is the Uniform on :math:`(0,1)`.

Likelihood
----------

.. math::

   L(\alpha,\beta)
     = \frac{1}{B(\alpha,\beta)^n}
       \prod_{i=1}^{n} x_i^{\alpha-1}(1-x_i)^{\beta-1}.

Log-likelihood
--------------

Taking the logarithm:

.. math::

   \ell(\alpha,\beta)
     = -n\ln B(\alpha,\beta)
       + (\alpha-1)\sum_{i=1}^{n}\ln x_i
       + (\beta-1)\sum_{i=1}^{n}\ln(1-x_i).

Just as with the Gamma, the data enter only through two sufficient statistics:
:math:`\sum \ln x_i` and :math:`\sum \ln(1 - x_i)`.

Expanding :math:`\ln B(\alpha,\beta) = \ln\Gamma(\alpha) + \ln\Gamma(\beta) - \ln\Gamma(\alpha+\beta)`:

.. math::

   \ell = n\ln\Gamma(\alpha+\beta) - n\ln\Gamma(\alpha) - n\ln\Gamma(\beta)
          + (\alpha-1)\sum\ln x_i + (\beta-1)\sum\ln(1-x_i).

Score functions
---------------

Differentiating with respect to :math:`\alpha` (noting that
:math:`d\ln\Gamma(z)/dz = \psi(z)`, the digamma function):

.. math::

   \frac{\partial\ell}{\partial\alpha}
     &= n\bigl[\psi(\alpha+\beta) - \psi(\alpha)\bigr] + \sum_{i=1}^{n}\ln x_i, \\[6pt]
   \frac{\partial\ell}{\partial\beta}
     &= n\bigl[\psi(\alpha+\beta) - \psi(\beta)\bigr] + \sum_{i=1}^{n}\ln(1-x_i).

Notice the symmetry: each score has a digamma difference (from the
normalizing constant) plus a data term.  The :math:`\alpha` score uses
:math:`\ln x_i`; the :math:`\beta` score uses :math:`\ln(1-x_i)`.

Fisher information
------------------

The Fisher information matrix involves trigamma functions
(:math:`\psi' = d\psi/d\alpha`):

.. math::

   \mathcal{I}(\alpha,\beta)
     = \begin{pmatrix}
         \psi'(\alpha) - \psi'(\alpha+\beta) & -\psi'(\alpha+\beta) \\
         -\psi'(\alpha+\beta) & \psi'(\beta) - \psi'(\alpha+\beta)
       \end{pmatrix}.

The :math:`-\psi'(\alpha+\beta)` off-diagonal means the two parameters are
coupled: changing :math:`\alpha` affects how well we can estimate
:math:`\beta`, and vice versa.

MLE
---

No closed-form solution exists because the digamma function has no algebraic
inverse.  The two score equations form a system of nonlinear equations that
must be solved iteratively.  Newton--Raphson, using the Fisher information
matrix as the Hessian approximation, converges quickly.

Good starting values can be obtained from the **method of moments**.  The
Beta distribution has :math:`E[X] = \alpha/(\alpha+\beta)` and
:math:`\text{Var}(X) = \alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]`.
Equating these to the sample mean :math:`\bar{x}` and sample variance
:math:`s^2` and solving:

.. math::

   \hat\alpha_0 = \bar{x}\!\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right),
   \qquad
   \hat\beta_0 = (1-\bar{x})\!\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right).

These moment-based estimates are usually close enough for Newton's method
to converge in a few iterations.

.. code-block:: python

   # Beta MLE: method of moments start + scipy optimization
   import numpy as np
   from scipy.stats import beta as beta_dist

   np.random.seed(42)
   alpha_true, beta_true = 2.5, 5.0
   n = 200
   data = np.random.beta(alpha_true, beta_true, size=n)

   # Method of moments starting values
   x_bar = data.mean()
   s2 = data.var()
   common = x_bar * (1 - x_bar) / s2 - 1
   alpha_mom = x_bar * common
   beta_mom = (1 - x_bar) * common

   # MLE via scipy
   a_fit, b_fit, loc_fit, scale_fit = beta_dist.fit(data, floc=0, fscale=1)

   print(f"True: alpha={alpha_true}, beta={beta_true}")
   print(f"MoM:  alpha={alpha_mom:.4f}, beta={beta_mom:.4f}")
   print(f"MLE:  alpha={a_fit:.4f}, beta={b_fit:.4f}")


.. _sec_lognormal:

6.5 Log-Normal Distribution
=============================

Motivation
----------

If :math:`Y = \ln X` is normally distributed, then :math:`X` has a Log-Normal
distribution. This arises when a quantity is the product of many small positive
random factors (by the multiplicative CLT). Incomes, stock prices, and
particle sizes are often modelled as Log-Normal.

The Log-Normal is a great example of how a simple transformation can turn a
familiar distribution into something new and useful. Everything you know about
the Normal carries over, but applied to the log-transformed data.

PDF
---

.. math::

   f(x \mid \mu, \sigma^2)
     = \frac{1}{x\sqrt{2\pi\sigma^2}}
       \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right),
   \qquad x > 0.

This is a Normal density applied to :math:`\ln x`, with an extra factor of
:math:`1/x` from the change-of-variables formula (the Jacobian of the log
transformation).  The parameters :math:`\mu` and :math:`\sigma^2` are the
mean and variance of :math:`\ln X`, not of :math:`X` itself.

.. admonition:: Common Pitfall

   A frequent source of confusion: the parameters :math:`\mu` and
   :math:`\sigma^2` of the Log-Normal are *not* the mean and variance of
   :math:`X`. The actual mean of :math:`X` is :math:`e^{\mu + \sigma^2/2}`,
   which can be much larger than :math:`\mu`.

Likelihood
----------

.. math::

   L(\mu,\sigma^2)
     = \prod_{i=1}^{n}\frac{1}{x_i\sqrt{2\pi\sigma^2}}
       \exp\!\left(-\frac{(\ln x_i - \mu)^2}{2\sigma^2}\right).

Log-likelihood
--------------

.. math::

   \ell(\mu,\sigma^2)
     = -\frac{n}{2}\ln(2\pi\sigma^2)
       - \sum_{i=1}^{n}\ln x_i
       - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(\ln x_i - \mu)^2.

Score functions
---------------

Setting :math:`y_i = \ln x_i`:

.. math::

   \frac{\partial\ell}{\partial\mu}
     &= \frac{1}{\sigma^2}\sum_{i=1}^{n}(y_i - \mu), \\[6pt]
   \frac{\partial\ell}{\partial\sigma^2}
     &= -\frac{n}{2\sigma^2}
        + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(y_i - \mu)^2.

Fisher information
------------------

Identical to the Normal Fisher information (see :ref:`sec_normal`) applied to
the log-transformed data:

.. math::

   \mathcal{I}(\mu,\sigma^2) = \begin{pmatrix}
     1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4)
   \end{pmatrix}.

MLE
---

The key insight is that if we define :math:`y_i = \ln x_i`, the
:math:`y_i` are i.i.d. Normal, so the Normal MLEs apply directly:

.. math::

   \hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}\ln x_i = \bar{y},
   \qquad
   \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(\ln x_i - \bar{y})^2.

This is a common pattern in statistics: when a transformation reduces a
problem to one we have already solved, we should use it.

.. code-block:: python

   # Log-Normal MLE: just take logs and use Normal MLE
   import numpy as np

   np.random.seed(42)
   mu_true, sigma2_true = 2.0, 0.5
   n = 200
   data = np.random.lognormal(mu_true, np.sqrt(sigma2_true), size=n)

   log_data = np.log(data)
   mu_hat = log_data.mean()
   sigma2_hat = np.mean((log_data - mu_hat)**2)

   print(f"True: mu={mu_true}, sigma^2={sigma2_true}")
   print(f"MLE:  mu_hat={mu_hat:.4f}, sigma^2_hat={sigma2_hat:.4f}")
   print(f"Mean of X (estimated): {np.exp(mu_hat + sigma2_hat/2):.4f}")
   print(f"Mean of X (sample):    {data.mean():.4f}")

.. admonition:: Real-World Example

   **Income distribution.** Household incomes in many countries are well
   described by a Log-Normal distribution. The log-transform converts the
   heavily right-skewed income data into something approximately Normal.

   .. code-block:: text

      Scenario: Annual household incomes (in thousands) for 500 households
      Data: positive values like 35.2, 72.1, 120.5, 28.0, ...
      Goal: estimate parameters of the log-income distribution
      Approach: Log-Normal(mu, sigma^2) -> MLE on log-transformed data

   .. code-block:: python

      # Income distribution: Log-Normal MLE
      import numpy as np

      np.random.seed(42)
      mu_true = 3.8    # log-scale mean (approx $45k median income)
      sigma2_true = 0.6
      n_households = 500

      incomes = np.random.lognormal(mu_true, np.sqrt(sigma2_true),
                                     size=n_households)

      log_inc = np.log(incomes)
      mu_hat = log_inc.mean()
      sigma2_hat = np.mean((log_inc - mu_hat)**2)

      print(f"Income stats: mean=${incomes.mean()*1000:.0f}, "
            f"median=${np.median(incomes)*1000:.0f}")
      print(f"MLE: mu_hat={mu_hat:.4f}, sigma^2_hat={sigma2_hat:.4f}")
      print(f"Estimated median income: ${np.exp(mu_hat)*1000:.0f}")


.. _sec_weibull:

6.6 Weibull Distribution
==========================

Motivation
----------

The Weibull distribution is the most widely used lifetime distribution in
reliability engineering. It generalises the Exponential by adding a shape
parameter that allows the hazard rate to increase, decrease, or remain
constant over time.

Here is the key idea: when :math:`k < 1`, the failure rate *decreases* over
time (infant mortality), when :math:`k = 1` it is constant (Exponential), and
when :math:`k > 1` it *increases* (wear-out). This single parameter lets you
model a wide range of lifetime behaviours.

PDF
---

With shape :math:`k > 0` and scale :math:`\lambda > 0`:

.. math::

   f(x \mid k, \lambda)
     = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
       \exp\!\left[-\left(\frac{x}{\lambda}\right)^k\right],
   \qquad x > 0.

The factor :math:`(x/\lambda)^{k-1}` controls the shape near zero (just as
:math:`x^{\alpha-1}` does for the Gamma).  The exponential term
:math:`\exp[-(x/\lambda)^k]` governs the tail: when :math:`k > 1`, the
exponent :math:`(x/\lambda)^k` grows faster than linearly, making the tail
thinner than the Exponential (wear-out); when :math:`k < 1`, it grows
slower, producing a heavier tail (infant mortality).  When :math:`k = 1`
this reduces to :math:`\text{Exp}(1/\lambda)`.

Likelihood
----------

.. math::

   L(k,\lambda)
     = \frac{k^n}{\lambda^{nk}}
       \left(\prod_{i=1}^{n} x_i\right)^{k-1}
       \exp\!\left[-\frac{1}{\lambda^k}\sum_{i=1}^{n} x_i^k\right].

Log-likelihood
--------------

.. math::

   \ell(k,\lambda)
     = n\ln k - nk\ln\lambda + (k-1)\sum_{i=1}^{n}\ln x_i
       - \frac{1}{\lambda^k}\sum_{i=1}^{n} x_i^k.

Score functions
---------------

**With respect to** :math:`\lambda`:

.. math::

   \frac{\partial\ell}{\partial\lambda}
     = -\frac{nk}{\lambda}
       + \frac{k}{\lambda^{k+1}}\sum_{i=1}^{n} x_i^k.

**With respect to** :math:`k`:

.. math::

   \frac{\partial\ell}{\partial k}
     = \frac{n}{k} - n\ln\lambda + \sum_{i=1}^{n}\ln x_i
       + \frac{\ln\lambda}{\lambda^k}\sum_{i=1}^{n}x_i^k
       - \frac{1}{\lambda^k}\sum_{i=1}^{n} x_i^k \ln x_i.

MLE
---

From the :math:`\lambda` score:

.. math::

   \hat{\lambda}^k = \frac{1}{n}\sum_{i=1}^{n} x_i^k
   \;\Longrightarrow\;
   \hat{\lambda} = \left(\frac{1}{n}\sum_{i=1}^{n} x_i^k\right)^{1/k}.

Substituting into the :math:`k` score equation yields a single nonlinear
equation in :math:`k`:

.. math::

   \frac{1}{k}
     + \frac{1}{n}\sum_{i=1}^{n}\ln x_i
     - \frac{\sum_{i=1}^{n} x_i^k \ln x_i}{\sum_{i=1}^{n} x_i^k}
     = 0.

This must be solved numerically (e.g., Newton's method).

.. code-block:: python

   # Weibull MLE using scipy
   import numpy as np
   from scipy.stats import weibull_min

   np.random.seed(42)
   k_true = 1.8    # shape
   lam_true = 10.0 # scale
   n = 200

   data = weibull_min.rvs(k_true, scale=lam_true, size=n)

   # Fit Weibull
   k_hat, loc_hat, lam_hat = weibull_min.fit(data, floc=0)

   print(f"True: k={k_true}, lambda={lam_true}")
   print(f"MLE:  k_hat={k_hat:.4f}, lambda_hat={lam_hat:.4f}")

.. admonition:: Real-World Example

   **Component lifetime analysis.** A manufacturer tests 150 electronic
   components until failure to determine the lifetime distribution and plan
   warranty periods. The Weibull model reveals whether components suffer from
   infant mortality (:math:`k < 1`) or wear-out (:math:`k > 1`).

   .. code-block:: text

      Scenario: 150 component lifetimes (in thousands of hours)
      Data: positive failure times
      Goal: estimate shape k and scale lambda; determine failure mode
      Approach: Weibull(k, lambda) MLE; interpret k

   .. code-block:: python

      # Component lifetime analysis with Weibull
      import numpy as np
      from scipy.stats import weibull_min

      np.random.seed(42)
      k_true = 2.5      # shape > 1 => wear-out
      lam_true = 8.0     # scale (thousands of hours)
      n_components = 150

      lifetimes = weibull_min.rvs(k_true, scale=lam_true, size=n_components)

      k_hat, _, lam_hat = weibull_min.fit(lifetimes, floc=0)

      failure_mode = ("infant mortality" if k_hat < 1
                      else "constant rate" if abs(k_hat - 1) < 0.1
                      else "wear-out")

      print(f"Lifetime stats: mean={lifetimes.mean():.1f}k hrs, "
            f"median={np.median(lifetimes):.1f}k hrs")
      print(f"MLE: k_hat={k_hat:.4f}, lambda_hat={lam_hat:.4f}")
      print(f"Failure mode: {failure_mode} (k {'>' if k_hat > 1 else '<'} 1)")
      print(f"Estimated warranty (5th percentile): "
            f"{weibull_min.ppf(0.05, k_hat, scale=lam_hat):.1f}k hrs")

Fisher information
------------------

The Fisher information matrix for a single Weibull observation is

.. math::

   \mathcal{I}(k,\lambda)
     = \begin{pmatrix}
         (1 + \gamma_E)^2 + \pi^2/6 & (\gamma_E + 1 - 1)/\lambda \\[2pt]
         (\gamma_E + 1 - 1)/\lambda & k^2/\lambda^2
       \end{pmatrix}
     \cdot \frac{1}{k^2},

where :math:`\gamma_E \approx 0.5772` is the Euler--Mascheroni constant.
(The exact form is somewhat involved; the diagonal entry for :math:`k` involves
:math:`[\psi(1)]^2 + \psi'(1) = \gamma_E^2 + \pi^2/6`.)


.. _sec_pareto:

6.7 Pareto Distribution
=========================

Motivation
----------

The Pareto distribution models phenomena that obey a power law: city
populations, wealth distributions, file sizes on the internet, and word
frequencies in natural language. It is characterised by a minimum value
:math:`x_m > 0` and a shape (tail index) :math:`\alpha > 0`.

The Pareto is famous for the "80-20 rule": in many real-world situations,
roughly 80% of the effects come from 20% of the causes. When you see an
extremely long right tail---a few values vastly larger than the rest---the
Pareto is often a good fit.

PDF
---

.. math::

   f(x \mid x_m, \alpha)
     = \frac{\alpha\, x_m^\alpha}{x^{\alpha+1}},
   \qquad x \ge x_m.

The density is a **power law**: it decays as :math:`x^{-(\alpha+1)}` for
large :math:`x`.  The tail index :math:`\alpha` controls how quickly the tail
falls off --- larger :math:`\alpha` means a thinner tail (less extreme
inequality).  The minimum :math:`x_m` is the smallest possible value.

Likelihood
----------

Multiplying the :math:`n` individual densities:

.. math::

   L(x_m, \alpha)
     = \prod_{i=1}^{n}\frac{\alpha\, x_m^\alpha}{x_i^{\alpha+1}}
     = \alpha^n\, x_m^{n\alpha}\,
       \left(\prod_{i=1}^{n} x_i\right)^{-(\alpha+1)},

valid only when :math:`x_m \le x_{(1)} = \min_i x_i`.  This constraint is
crucial: if we set :math:`x_m` above the smallest observation, that
observation would have zero density and the likelihood would be zero.

Log-likelihood
--------------

.. math::

   \ell(x_m, \alpha)
     = n\ln\alpha + n\alpha\ln x_m
       - (\alpha+1)\sum_{i=1}^{n}\ln x_i.

Score functions
---------------

**With respect to** :math:`\alpha`:

.. math::

   \frac{\partial\ell}{\partial\alpha}
     = \frac{n}{\alpha} + n\ln x_m - \sum_{i=1}^{n}\ln x_i.

**With respect to** :math:`x_m`:

.. math::

   \frac{\partial\ell}{\partial x_m}
     = \frac{n\alpha}{x_m}.

This is positive for all :math:`x_m > 0`, so the log-likelihood is increasing
in :math:`x_m`. The MLE is therefore the largest value of :math:`x_m` that is
still consistent with the data:

.. math::

   \hat{x}_m = x_{(1)} = \min(x_1, \dots, x_n).

MLE
---

With :math:`\hat{x}_m = x_{(1)}`, setting the :math:`\alpha` score to zero:

.. math::

   \frac{n}{\alpha} = \sum_{i=1}^{n}\ln x_i - n\ln x_{(1)}
                     = \sum_{i=1}^{n}\ln\frac{x_i}{x_{(1)}}.

Hence

.. math::

   \hat{\alpha} = \frac{n}{\sum_{i=1}^{n}\ln(x_i / x_{(1)})}.

.. code-block:: python

   # Pareto MLE: estimate minimum and tail index
   import numpy as np

   np.random.seed(42)
   xm_true = 1.0
   alpha_true = 2.5
   n = 300

   data = (np.random.pareto(alpha_true, size=n) + 1) * xm_true

   xm_hat = data.min()
   alpha_hat = n / np.sum(np.log(data / xm_hat))

   # Score at MLE
   score_alpha = n / alpha_hat + n * np.log(xm_hat) - np.sum(np.log(data))

   print(f"True: x_m={xm_true}, alpha={alpha_true}")
   print(f"MLE:  x_m_hat={xm_hat:.4f}, alpha_hat={alpha_hat:.4f}")
   print(f"Score at MLE (alpha) = {score_alpha:.10f}")

Fisher information
------------------

For :math:`\alpha` (with :math:`x_m` known):

.. math::

   \mathcal{I}(\alpha) = \frac{1}{\alpha^2}.


.. _sec_student_t:

6.8 Student's *t*-Distribution
================================

Motivation
----------

The Student's *t*-distribution arises when estimating the mean of a Normal
population with unknown variance using a small sample. More broadly, it is
used as a robust alternative to the Normal for heavy-tailed data because its
tails decay as a power law rather than exponentially.

If your data has occasional extreme values that a Normal model would consider
impossibly unlikely, the *t*-distribution may be a better choice. It
down-weights outliers naturally through its heavy tails.

PDF
---

With :math:`\nu > 0` degrees of freedom, location :math:`\mu`, and scale
:math:`\sigma > 0`:

.. math::

   f(x \mid \nu, \mu, \sigma)
     = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
            {\sigma\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)}
       \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}.

Compare this with the Normal: the Normal has :math:`\exp(-z^2/2)`, which
decays exponentially in :math:`z^2`.  The *t* replaces this with
:math:`(1 + z^2/\nu)^{-(\nu+1)/2}`, which decays only as a power law.
This means extreme values are much more probable under the *t* than the
Normal.  As :math:`\nu \to \infty`, the power-law term converges to the
Gaussian exponential (by the identity
:math:`(1 + z^2/\nu)^{-\nu/2} \to e^{-z^2/2}`).

When :math:`\mu = 0` and :math:`\sigma = 1` this is the standard
*t*-distribution.

Log-likelihood
--------------

For :math:`n` observations with known :math:`\nu`:

.. math::

   \ell(\mu, \sigma)
     = n\ln\Gamma\!\left(\tfrac{\nu+1}{2}\right)
       - n\ln\Gamma\!\left(\tfrac{\nu}{2}\right)
       - \frac{n}{2}\ln(\nu\pi)
       - n\ln\sigma
       - \frac{\nu+1}{2}\sum_{i=1}^{n}
         \ln\!\left(1 + \frac{(x_i-\mu)^2}{\nu\sigma^2}\right).

Score functions
---------------

**With respect to** :math:`\mu`:

.. math::

   \frac{\partial\ell}{\partial\mu}
     = (\nu+1)\sum_{i=1}^{n}
       \frac{x_i - \mu}{\nu\sigma^2 + (x_i - \mu)^2}.

**With respect to** :math:`\sigma`:

.. math::

   \frac{\partial\ell}{\partial\sigma}
     = -\frac{n}{\sigma}
       + \frac{\nu+1}{\sigma}\sum_{i=1}^{n}
         \frac{(x_i - \mu)^2}{\nu\sigma^2 + (x_i - \mu)^2}.

MLE
---

No closed-form solutions exist. The score equations must be solved
iteratively. A common and elegant approach uses the EM algorithm, based on
the insight that the *t*-distribution can be written as a **scale mixture of
Normals**: each observation :math:`x_i` is Normal with a data-point-specific
precision that is itself Gamma-distributed.

In the **E-step**, one computes a weight for each observation that measures
how "typical" it is under the current parameter estimates:

.. math::

   w_i = \frac{\nu + 1}{\nu + (x_i - \mu)^2/\sigma^2}.

Points close to :math:`\mu` get weights near 1 (they look Normal); outliers
get down-weighted (their large :math:`(x_i - \mu)^2` inflates the
denominator).  This automatic down-weighting of outliers is why the
*t*-distribution is robust.

In the **M-step**, one solves weighted Normal equations --- exactly like
Normal MLE but with the weights from the E-step:

.. math::

   \hat\mu = \frac{\sum w_i x_i}{\sum w_i},
   \qquad
   \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} w_i(x_i - \hat\mu)^2.

When :math:`\nu` is also unknown, a profile likelihood over :math:`\nu`
(often restricted to a grid) is used.

.. code-block:: python

   # Student-t MLE using EM algorithm (known nu)
   import numpy as np

   np.random.seed(42)
   nu = 4
   mu_true, sigma_true = 3.0, 2.0
   n = 200

   data = mu_true + sigma_true * np.random.standard_t(nu, size=n)

   # EM algorithm
   mu_hat = np.median(data)  # robust start
   sigma2_hat = np.mean((data - mu_hat)**2)

   for iteration in range(50):
       # E-step: compute weights
       w = (nu + 1) / (nu + (data - mu_hat)**2 / sigma2_hat)
       # M-step
       mu_hat = np.sum(w * data) / np.sum(w)
       sigma2_hat = np.sum(w * (data - mu_hat)**2) / n

   print(f"True: mu={mu_true}, sigma={sigma_true}")
   print(f"MLE:  mu_hat={mu_hat:.4f}, sigma_hat={np.sqrt(sigma2_hat):.4f}")


.. _sec_chisq:

6.9 Chi-Squared Distribution
==============================

Motivation
----------

The :math:`\chi^2` distribution with :math:`k` degrees of freedom is the
distribution of :math:`Z_1^2 + \cdots + Z_k^2` where the :math:`Z_i` are
i.i.d. standard normal. It is central to goodness-of-fit tests, analysis of
variance, and the distribution of the sample variance. It is a special case of
the Gamma distribution with :math:`\alpha = k/2` and :math:`\beta = 1/2`.

PDF
---

.. math::

   f(x \mid k)
     = \frac{1}{2^{k/2}\Gamma(k/2)}\,x^{k/2 - 1}\,e^{-x/2},
   \qquad x > 0.

Log-likelihood
--------------

.. math::

   \ell(k)
     = -\frac{nk}{2}\ln 2 - n\ln\Gamma(k/2)
       + \left(\frac{k}{2} - 1\right)\sum_{i=1}^{n}\ln x_i
       - \frac{1}{2}\sum_{i=1}^{n} x_i.

Score function
--------------

.. math::

   \frac{d\ell}{dk}
     = -\frac{n}{2}\ln 2 - \frac{n}{2}\psi(k/2)
       + \frac{1}{2}\sum_{i=1}^{n}\ln x_i.

MLE
---

Setting the score to zero:

.. math::

   \psi(k/2) = \frac{1}{n}\sum_{i=1}^{n}\ln x_i - \ln 2.

This must be solved numerically for :math:`k` (e.g., by Newton's method or
bisection on the monotone left-hand side).


.. _sec_f_dist:

6.10 *F*-Distribution
======================

Motivation
----------

The *F*-distribution arises as the ratio of two independent Chi-squared
random variables, each divided by its degrees of freedom. It is the
foundation of analysis-of-variance (ANOVA) *F*-tests and tests comparing two
variances.

PDF
---

With :math:`d_1, d_2 > 0` degrees of freedom:

.. math::

   f(x \mid d_1, d_2)
     = \frac{1}{B(d_1/2,\, d_2/2)}
       \left(\frac{d_1}{d_2}\right)^{d_1/2}
       \frac{x^{d_1/2 - 1}}{(1 + d_1 x / d_2)^{(d_1+d_2)/2}},
   \qquad x > 0.

Log-likelihood
--------------

.. math::

   \ell(d_1, d_2)
     &= -n\ln B(d_1/2, d_2/2)
        + \frac{n d_1}{2}\ln\frac{d_1}{d_2}
        + \left(\frac{d_1}{2}-1\right)\sum\ln x_i \\
     &\quad - \frac{d_1+d_2}{2}\sum_{i=1}^{n}\ln\!\left(1 + \frac{d_1 x_i}{d_2}\right).

MLE
---

The score equations involve digamma functions and have no closed form. The
degrees-of-freedom parameters are estimated numerically. In practice the
*F*-distribution is almost always used in testing with known degrees of
freedom, so MLE estimation of :math:`d_1, d_2` is relatively rare.


.. _sec_uniform:

6.11 Uniform Distribution
===========================

Motivation
----------

The Uniform distribution assigns equal density to every point in an interval
:math:`[a, b]`. It is the maximum-entropy distribution for bounded support.
Rounding errors and random number generators are modelled as Uniform.

The Uniform is special because it is a **non-regular** family: the support of
the distribution depends on the parameters. This leads to unusual MLE
behaviour that is worth understanding.

PDF
---

.. math::

   f(x \mid a, b)
     = \frac{1}{b - a},
   \qquad a \le x \le b.

Likelihood
----------

.. math::

   L(a, b)
     = \frac{1}{(b-a)^n}\,\prod_{i=1}^{n}\mathbf{1}(a \le x_i \le b)
     = \frac{1}{(b-a)^n}\,\mathbf{1}\!\bigl(a \le x_{(1)}\bigr)
       \,\mathbf{1}\!\bigl(x_{(n)} \le b\bigr),

where :math:`x_{(1)} = \min x_i` and :math:`x_{(n)} = \max x_i`.  The
indicator functions :math:`\mathbf{1}(\cdot)` enforce the constraint that all
data must lie within :math:`[a, b]`: the likelihood is exactly zero if any
observation falls outside.  Among all valid :math:`(a, b)`, the likelihood
decreases as the interval gets wider (because :math:`1/(b-a)^n` shrinks),
so we want the *tightest possible* interval that still contains all the data.

Log-likelihood
--------------

.. math::

   \ell(a,b) = -n\ln(b - a),

subject to :math:`a \le x_{(1)}` and :math:`b \ge x_{(n)}`.

MLE
---

The log-likelihood is decreasing in :math:`(b-a)`, so we want to make the
interval as small as possible while still containing all data:

.. math::

   \hat{a} = x_{(1)}, \qquad \hat{b} = x_{(n)}.

This MLE is **biased** (the interval is too narrow on average) but
**consistent**. The score function is not defined in the usual sense because
the support depends on the parameters; this is a non-regular problem where
the MLE converges at rate :math:`1/n` rather than :math:`1/\sqrt{n}`.

.. code-block:: python

   # Uniform MLE: endpoints are min and max of data
   import numpy as np

   np.random.seed(42)
   a_true, b_true = 2.0, 7.0
   n = 100
   data = np.random.uniform(a_true, b_true, size=n)

   a_hat = data.min()
   b_hat = data.max()

   print(f"True: a={a_true}, b={b_true}, range={b_true - a_true}")
   print(f"MLE:  a_hat={a_hat:.4f}, b_hat={b_hat:.4f}, "
         f"range={b_hat - a_hat:.4f}")
   print(f"Bias in range: {(b_hat - a_hat) - (b_true - a_true):.4f} "
         f"(expected negative)")

Fisher information
------------------

Because the support depends on the parameters, the regularity conditions for
the standard Fisher information do not hold. The Uniform is a classic example
of a **non-regular** family for which the Cram\'{e}r--Rao bound does not
apply.


.. _sec_cauchy:

6.12 Cauchy Distribution
==========================

Motivation
----------

The Cauchy distribution has such heavy tails that it has no finite mean or
variance. It arises as the ratio of two independent standard normals and as
the Student-*t* with one degree of freedom. Despite (or because of) its
pathological moments, it is important as a stress test for estimation
procedures: the sample mean is a terrible estimator of the location parameter,
but the MLE works well.

.. admonition:: Common Pitfall

   Never use the sample mean to estimate the location of a Cauchy distribution.
   The sample mean does not converge---it has the same distribution regardless of
   sample size! The MLE, by contrast, is consistent and asymptotically efficient.

PDF
---

With location :math:`\mu` and scale :math:`\gamma > 0`:

.. math::

   f(x \mid \mu, \gamma)
     = \frac{1}{\pi\gamma\!\left[1 + \left(\frac{x - \mu}{\gamma}\right)^2\right]},
   \qquad x \in \mathbb{R}.

Log-likelihood
--------------

.. math::

   \ell(\mu, \gamma)
     = -n\ln\pi - n\ln\gamma
       - \sum_{i=1}^{n}\ln\!\left[1 + \left(\frac{x_i - \mu}{\gamma}\right)^2\right].

Score functions
---------------

**With respect to** :math:`\mu`:

.. math::

   \frac{\partial\ell}{\partial\mu}
     = \sum_{i=1}^{n}
       \frac{2(x_i - \mu)/\gamma^2}{1 + (x_i - \mu)^2/\gamma^2}
     = \sum_{i=1}^{n}\frac{2(x_i - \mu)}{\gamma^2 + (x_i - \mu)^2}.

**With respect to** :math:`\gamma`:

.. math::

   \frac{\partial\ell}{\partial\gamma}
     = -\frac{n}{\gamma}
       + \frac{2}{\gamma}\sum_{i=1}^{n}
         \frac{(x_i - \mu)^2}{\gamma^2 + (x_i - \mu)^2}.

Fisher information
------------------

For a single observation:

.. math::

   \mathcal{I}(\mu, \gamma)
     = \begin{pmatrix}
         1/(2\gamma^2) & 0 \\
         0 & 1/(2\gamma^2)
       \end{pmatrix}.

The off-diagonal is zero by symmetry.

MLE
---

There is no closed-form MLE. The score equations must be solved iteratively.
Newton--Raphson works but may require good starting values; the sample median
is a robust starting point for :math:`\mu` and the interquartile range for
:math:`\gamma`. Despite the lack of moments, the MLE is consistent and
asymptotically efficient at the Cauchy model.

.. code-block:: python

   # Cauchy MLE: Newton-Raphson from robust starting values
   import numpy as np
   from scipy.stats import cauchy

   np.random.seed(42)
   mu_true, gamma_true = 3.0, 2.0
   n = 200
   data = cauchy.rvs(loc=mu_true, scale=gamma_true, size=n)

   # Robust starting values
   mu_hat = np.median(data)
   gamma_hat = (np.percentile(data, 75) - np.percentile(data, 25)) / 2

   # MLE via scipy
   mu_mle, gamma_mle = cauchy.fit(data)

   print(f"True: mu={mu_true}, gamma={gamma_true}")
   print(f"Sample mean: {data.mean():.4f} (unreliable!)")
   print(f"Sample median: {np.median(data):.4f} (robust)")
   print(f"MLE: mu={mu_mle:.4f}, gamma={gamma_mle:.4f}")


.. _sec_continuous_summary:

6.13 Summary Table
===================

.. list-table:: Continuous Distributions --- MLEs at a glance
   :header-rows: 1
   :widths: 22 38 20

   * - Distribution
     - MLE
     - Closed form?
   * - Normal(:math:`\mu,\sigma^2`)
     - :math:`\hat\mu=\bar{x}`,
       :math:`\hat\sigma^2=n^{-1}\sum(x_i-\bar{x})^2`
     - Yes
   * - Exponential(:math:`\lambda`)
     - :math:`\hat\lambda = 1/\bar{x}`
     - Yes
   * - Gamma(:math:`\alpha,\beta`)
     - :math:`\hat\beta=\hat\alpha/\bar{x}`;
       :math:`\hat\alpha` via digamma eq.
     - Partial
   * - Beta(:math:`\alpha,\beta`)
     - Newton--Raphson on digamma system
     - No
   * - Log-Normal(:math:`\mu,\sigma^2`)
     - Normal MLE on :math:`\ln x_i`
     - Yes
   * - Weibull(:math:`k,\lambda`)
     - :math:`\hat\lambda` in closed form given :math:`\hat{k}`;
       :math:`\hat{k}` numerical
     - Partial
   * - Pareto(:math:`x_m,\alpha`)
     - :math:`\hat{x}_m = x_{(1)}`,
       :math:`\hat\alpha = n/\sum\ln(x_i/x_{(1)})`
     - Yes
   * - Student-t(:math:`\mu,\sigma`)
     - EM or Newton
     - No
   * - :math:`\chi^2(k)`
     - Digamma equation
     - No
   * - F(:math:`d_1,d_2`)
     - Numerical
     - No
   * - Uniform(:math:`a,b`)
     - :math:`\hat{a}=x_{(1)}`, :math:`\hat{b}=x_{(n)}`
     - Yes (non-regular)
   * - Cauchy(:math:`\mu,\gamma`)
     - Newton from sample median
     - No

Key observations:

* Exponential-family distributions (Normal, Exponential, Gamma, Poisson)
  always have sufficient statistics of fixed dimension and often yield
  closed-form MLEs.
* Distributions with digamma functions in the score (Gamma, Beta,
  :math:`\chi^2`) require numerical methods because the digamma equation has
  no algebraic inverse.
* Non-regular families (Uniform, Pareto for :math:`x_m`) have MLEs that
  converge faster than :math:`1/\sqrt{n}` but violate the usual regularity
  conditions.

.. admonition:: Real-World Connection

   In practice, you rarely fit these distributions in isolation. More often,
   they appear as building blocks inside larger models: the Normal in linear
   regression, the Gamma as a prior in Bayesian hierarchical models, the
   Weibull in survival analysis. Understanding the MLE for each family gives
   you the foundation for working with these more complex models.
