.. _ch10_analytical_mle:

==========================================
Chapter 10 --- Analytical MLE Solutions
==========================================

This chapter works through complete, step-by-step maximum likelihood
derivations for the most important parametric families. For every
distribution we follow a uniform procedure:

1. Write down the likelihood and log-likelihood.
2. Compute the first derivative (score) and set it to zero.
3. Solve the resulting equation for the parameter(s).
4. Verify via the second derivative that the solution is indeed a
   maximum.

Where closed-form solutions are not available (Gamma shape, Beta
distribution), we show *why* the equations cannot be solved analytically
and describe the standard numerical approaches.

The theoretical justification for this procedure---and the conditions
under which the resulting estimator is consistent, asymptotically
normal, and efficient---was developed in :ref:`ch9_mle_theory`.

For each distribution below we generate realistic data from a concrete
scenario, derive the MLE analytically, compare to ``scipy``, verify the
score is zero at the MLE, compute the Fisher information and standard
error, and print a summary with a 95% confidence interval.


10.1 Normal Distribution --- :math:`\mathcal{N}(\mu, \sigma^2)`
=================================================================

**Motivating problem.** A factory produces bolts whose diameter is
specified as 10.0 mm. Quality control draws a random sample of bolts
from the production line and measures each one. The diameters vary due to
manufacturing noise. The engineers need to estimate the true mean
diameter (is the machine centred correctly?) and the variability (are the
tolerances acceptable?). The normal distribution is the natural model.

**Setup.** Let :math:`x_1, \ldots, x_n` be i.i.d. realisations from
:math:`\mathcal{N}(\mu, \sigma^2)` with density

.. math::

   f(x \mid \mu, \sigma^2)
   = \frac{1}{\sqrt{2\pi\sigma^2}}
     \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).

10.1.1 Log-Likelihood
-----------------------

Here is the key idea: we take the product of :math:`n` normal densities
and convert it into a sum by taking the logarithm. Every term separates
cleanly into a piece involving :math:`\mu` and a piece involving
:math:`\sigma^2`, which is what allows us to optimise over each
parameter one at a time.

.. math::

   \ell(\mu, \sigma^2)
   &= \sum_{i=1}^n \log f(x_i \mid \mu, \sigma^2) \\
   &= \sum_{i=1}^n \left[
       -\frac{1}{2}\log(2\pi)
       - \frac{1}{2}\log(\sigma^2)
       - \frac{(x_i - \mu)^2}{2\sigma^2}
     \right] \\
   &= -\frac{n}{2}\log(2\pi)
      - \frac{n}{2}\log(\sigma^2)
      - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2.

10.1.2 Derivative with Respect to :math:`\mu`
-----------------------------------------------

The only term involving :math:`\mu` is the sum of squared deviations.
Differentiating:

.. math::

   \frac{\partial\ell}{\partial\mu}
   = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu).

Setting this to zero:

.. math::

   \sum_{i=1}^n (x_i - \mu) = 0
   \quad\Longrightarrow\quad
   n\mu = \sum_{i=1}^n x_i
   \quad\Longrightarrow\quad
   \boxed{\hat\mu = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i.}

The MLE of the mean is simply the sample mean. This should feel
intuitive: the sample mean is the value that minimises the total squared
distance from all observations.

10.1.3 Derivative with Respect to :math:`\sigma^2`
-----------------------------------------------------

Let :math:`\tau = \sigma^2` for cleaner notation. Differentiating the
log-likelihood with respect to :math:`\tau`:

.. math::

   \frac{\partial\ell}{\partial\tau}
   = -\frac{n}{2\tau}
     + \frac{1}{2\tau^2}\sum_{i=1}^n (x_i - \mu)^2.

Setting to zero and substituting :math:`\hat\mu = \bar x`:

.. math::

   \frac{n}{2\tau}
   = \frac{1}{2\tau^2}\sum_{i=1}^n (x_i - \bar x)^2
   \quad\Longrightarrow\quad
   \tau = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2.

Therefore:

.. math::

   \boxed{
   \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2.
   }

Notice the divisor is :math:`n`, not :math:`n-1`. We will return to this
subtle but important distinction when we discuss bias below.

Now let's carry out the full procedure on bolt diameter data. We
generate realistic measurements, compute the MLE, compare to ``scipy``,
verify the score, and produce a confidence interval.

.. code-block:: python

   # Normal MLE: bolt diameter quality control
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   true_mu, true_sigma = 10.0, 0.15  # mm
   n = 60
   data = np.random.normal(true_mu, true_sigma, size=n)

   # Step 1: Analytical MLE
   mle_mu = np.mean(data)
   mle_sigma2 = np.mean((data - mle_mu)**2)
   mle_sigma = np.sqrt(mle_sigma2)

   # Step 2: Compare to scipy
   sp_mu, sp_sigma = stats.norm.fit(data)

   # Step 3: Verify the score is zero at the MLE
   score_mu = np.sum(data - mle_mu) / mle_sigma2
   score_sigma2 = -n / (2 * mle_sigma2) + np.sum((data - mle_mu)**2) / (2 * mle_sigma2**2)

   # Step 4: Fisher information and standard errors
   # I(mu) = 1/sigma^2, I(sigma^2) = 1/(2*sigma^4) for one observation
   se_mu = np.sqrt(mle_sigma2 / n)
   se_sigma2 = mle_sigma2 * np.sqrt(2.0 / n)

   # Step 5: 95% CI
   z = 1.96
   ci_mu = (mle_mu - z * se_mu, mle_mu + z * se_mu)
   ci_sigma2 = (mle_sigma2 - z * se_sigma2, mle_sigma2 + z * se_sigma2)

   print("=== Bolt Diameter Quality Control ===")
   print(f"Sample size: {n}")
   print(f"True:    mu = {true_mu}, sigma^2 = {true_sigma**2}")
   print(f"MLE:     mu = {mle_mu:.4f}, sigma^2 = {mle_sigma2:.6f}")
   print(f"scipy:   mu = {sp_mu:.4f}, sigma^2 = {sp_sigma**2:.6f}")
   print(f"Diff:    mu = {abs(mle_mu - sp_mu):.2e}, "
         f"sigma^2 = {abs(mle_sigma2 - sp_sigma**2):.2e}")
   print(f"\nScore at MLE (should be ~0):")
   print(f"  d ell/d mu      = {score_mu:.10f}")
   print(f"  d ell/d sigma^2 = {score_sigma2:.10f}")
   print(f"\nSE(mu) = {se_mu:.4f}, SE(sigma^2) = {se_sigma2:.6f}")
   print(f"95% CI for mu:      ({ci_mu[0]:.4f}, {ci_mu[1]:.4f})"
         f"  covers true? {'Yes' if ci_mu[0] <= true_mu <= ci_mu[1] else 'No'}")
   print(f"95% CI for sigma^2: ({ci_sigma2[0]:.6f}, {ci_sigma2[1]:.6f})"
         f"  covers true? {'Yes' if ci_sigma2[0] <= true_sigma**2 <= ci_sigma2[1] else 'No'}")

So what? The factory can now answer both questions: the estimated mean
diameter is :math:`\hat\mu \pm \text{SE}`, and if the 95% CI for
:math:`\mu` contains 10.0 mm, the machine is centred correctly.

10.1.4 Second-Derivative Verification
---------------------------------------

The Hessian matrix of :math:`\ell` is

.. math::

   H = \begin{pmatrix}
   \dfrac{\partial^2\ell}{\partial\mu^2}
   & \dfrac{\partial^2\ell}{\partial\mu\,\partial\tau} \\[6pt]
   \dfrac{\partial^2\ell}{\partial\tau\,\partial\mu}
   & \dfrac{\partial^2\ell}{\partial\tau^2}
   \end{pmatrix}
   =
   \begin{pmatrix}
   -n/\sigma^2 & 0 \\
   0 & -n/(2\sigma^4) + (\cdots)
   \end{pmatrix}.

Evaluating at the MLEs, both diagonal entries are negative and the
off-diagonal is zero at the MLE (since the cross-derivative vanishes
when evaluated at :math:`\hat\mu`), confirming the Hessian is negative
definite---a maximum.

.. code-block:: python

   # Verify the Hessian is negative definite at the MLE
   import numpy as np

   # Continuing from above
   n = 60
   mle_mu = 10.0  # placeholder; in practice use computed value
   mle_sigma2 = 0.15**2

   # For demonstration, recompute from the formulas
   H_11 = -n / mle_sigma2
   H_22 = -n / (2 * mle_sigma2**2)
   # (The full H_22 includes a data-dependent term that equals H_22 at the MLE)

   print(f"Hessian diagonal at MLE:")
   print(f"  d^2 ell / d mu^2 =      {H_11:.2f}  (negative)")
   print(f"  d^2 ell / d sigma^4 =   {H_22:.2f}  (negative)")
   print(f"  => Negative definite => confirmed maximum")

10.1.5 Bias of the Variance MLE
---------------------------------

The MLE :math:`\hat\sigma^2` divides by :math:`n`, not :math:`n-1`.
To see the resulting bias, note that

.. math::

   \sum_{i=1}^n (X_i - \bar X)^2
   = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar X - \mu)^2.

Taking expectations:

.. math::

   E\!\left[\sum_{i=1}^n(X_i - \bar X)^2\right]
   = n\sigma^2 - n\cdot\frac{\sigma^2}{n}
   = (n-1)\sigma^2.

Therefore:

.. math::

   E[\hat\sigma^2]
   = \frac{n-1}{n}\,\sigma^2.

The MLE is **biased downward** by a factor of :math:`(n-1)/n`. The
unbiased estimator is :math:`S^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2`.
As discussed in :ref:`ch9_mle_theory`, this :math:`O(1/n)` bias is
typical of MLEs and vanishes as :math:`n \to \infty`.


10.2 Exponential Distribution --- :math:`\text{Exp}(\lambda)`
===============================================================

**Motivating problem.** A cloud computing company needs to estimate how
long its servers run before failing. They have failure-time data for 80
servers. Each server's lifetime is modelled as exponentially distributed
with rate parameter :math:`\lambda` (so the mean lifetime is
:math:`1/\lambda`). Accurately estimating :math:`\lambda` determines
warranty terms, replacement schedules, and maintenance budgets.

The exponential distribution has density

.. math::

   f(x \mid \lambda) = \lambda\,e^{-\lambda x}, \qquad x \geq 0.

If events occur at a constant average rate :math:`\lambda` per unit
time, the time between consecutive events follows an exponential
distribution.

10.2.1 Log-Likelihood
-----------------------

.. math::

   \ell(\lambda)
   = \sum_{i=1}^n \bigl[\log\lambda - \lambda x_i\bigr]
   = n\log\lambda - \lambda\sum_{i=1}^n x_i.

The log-likelihood has a clean structure: one term that increases with
:math:`\lambda` (the :math:`\log\lambda` part) and one that decreases
(the :math:`-\lambda\sum x_i` part). The MLE balances these two forces.

10.2.2 Score and MLE
----------------------

Differentiate with respect to :math:`\lambda`:

.. math::

   \frac{d\ell}{d\lambda}
   = \frac{n}{\lambda} - \sum_{i=1}^n x_i.

Set to zero:

.. math::

   \frac{n}{\lambda} = \sum_{i=1}^n x_i
   \quad\Longrightarrow\quad
   \boxed{\hat\lambda = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar x}.}

The MLE of the rate is the reciprocal of the sample mean. This has a
satisfying interpretation: if the average server lifetime in your data is
:math:`\bar{x}` months, then the estimated failure rate is
:math:`1/\bar{x}` failures per month.

.. code-block:: python

   # Exponential MLE: server lifetime analysis
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   true_lambda = 0.04  # failures per month (mean lifetime = 25 months)
   n = 80
   data = np.random.exponential(scale=1/true_lambda, size=n)

   # Step 1: Analytical MLE
   x_bar = np.mean(data)
   mle_lambda = 1.0 / x_bar

   # Step 2: Compare to scipy
   loc, scale = stats.expon.fit(data, floc=0)
   sp_lambda = 1.0 / scale

   # Step 3: Verify score = 0 at MLE
   score_at_mle = n / mle_lambda - np.sum(data)

   # Step 4: Fisher information and SE
   # I(lambda) = 1/lambda^2 for one observation
   fisher_info = 1.0 / mle_lambda**2
   se_lambda = np.sqrt(1.0 / (n * fisher_info))  # = mle_lambda / sqrt(n)

   # Step 5: 95% CI
   z = 1.96
   ci = (mle_lambda - z * se_lambda, mle_lambda + z * se_lambda)

   print("=== Server Lifetime Analysis ===")
   print(f"Sample size: {n} servers")
   print(f"Mean lifetime: {x_bar:.2f} months")
   print(f"\nTrue lambda:     {true_lambda}")
   print(f"MLE lambda:      {mle_lambda:.4f}")
   print(f"scipy lambda:    {sp_lambda:.4f}")
   print(f"Score at MLE:    {score_at_mle:.10f}  (should be ~0)")
   print(f"\nFisher info:     {fisher_info:.4f}")
   print(f"SE(lambda):      {se_lambda:.4f}")
   print(f"MLE +/- SE:      {mle_lambda:.4f} +/- {se_lambda:.4f}")
   print(f"95% CI:          ({ci[0]:.4f}, {ci[1]:.4f})")
   print(f"Covers true?     {'Yes' if ci[0] <= true_lambda <= ci[1] else 'No'}")

10.2.3 Second-Derivative Check
--------------------------------

.. math::

   \frac{d^2\ell}{d\lambda^2} = -\frac{n}{\lambda^2} < 0

for all :math:`\lambda > 0`, confirming a global maximum.

.. code-block:: python

   # Verify second derivative is negative at the MLE
   d2ell = -n / mle_lambda**2
   print(f"d^2 ell / d lambda^2 = {d2ell:.4f}  (negative => maximum)")

10.2.4 Bias
-------------

As noted in :ref:`ch9_mle_theory`, :math:`E[1/\bar X] = n\lambda/(n-1)`
for exponential data. The MLE is biased upward; the unbiased estimator
is :math:`(n-1)/(n\bar X)`.

.. code-block:: python

   # Verify bias of exponential MLE via simulation
   import numpy as np

   np.random.seed(42)
   true_lambda = 0.04
   n = 80
   n_sims = 50000

   mle_vals = [1.0 / np.mean(np.random.exponential(1/true_lambda, size=n))
               for _ in range(n_sims)]

   print(f"True lambda:         {true_lambda}")
   print(f"E[MLE]:              {np.mean(mle_vals):.6f}")
   print(f"Theoretical E[MLE]:  {n * true_lambda / (n-1):.6f}")
   print(f"Bias:                {np.mean(mle_vals) - true_lambda:.6f}")
   print(f"Theoretical bias:    {true_lambda / (n-1):.6f}")

So what? The cloud company can now estimate that servers fail on average
every :math:`1/\hat\lambda` months, with a quantified uncertainty band.
If the lower CI bound for mean lifetime falls below the warranty period,
they need to reconsider their warranty terms.


10.3 Poisson Distribution --- :math:`\text{Pois}(\lambda)`
============================================================

**Motivating problem.** A software team tracks bugs per 1000 lines of
code across 120 modules. They want to estimate the underlying bug rate
:math:`\lambda` to decide whether their quality targets are being met.
The Poisson distribution is the default model for count data: number of
events in a fixed window.

The Poisson mass function is

.. math::

   f(x \mid \lambda) = \frac{\lambda^x\,e^{-\lambda}}{x!},
   \qquad x = 0, 1, 2, \ldots

10.3.1 Log-Likelihood
-----------------------

.. math::

   \ell(\lambda)
   = \sum_{i=1}^n \bigl[x_i\log\lambda - \lambda - \log(x_i!)\bigr]
   = \log\lambda\sum_{i=1}^n x_i - n\lambda
     - \sum_{i=1}^n \log(x_i!).

The last term is a constant with respect to :math:`\lambda` and can be
ignored during optimisation.

10.3.2 Score and MLE
----------------------

.. math::

   \frac{d\ell}{d\lambda}
   = \frac{1}{\lambda}\sum_{i=1}^n x_i - n.

Setting to zero:

.. math::

   \frac{1}{\lambda}\sum_{i=1}^n x_i = n
   \quad\Longrightarrow\quad
   \boxed{\hat\lambda = \bar x = \frac{1}{n}\sum_{i=1}^n x_i.}

The MLE of the Poisson rate is the sample mean. For the Poisson, the
mean and variance are both :math:`\lambda`, and the sample mean captures
everything the data have to say about this single parameter.

.. code-block:: python

   # Poisson MLE: bug rate per 1000 lines of code
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   true_lambda = 4.2   # bugs per 1000 LOC
   n = 120             # software modules
   data = np.random.poisson(true_lambda, size=n)

   # Step 1: Analytical MLE
   mle_lambda = np.mean(data)

   # Step 2: Compare to numerical optimisation
   def neg_loglik(lam):
       if lam <= 0:
           return 1e10
       return -(np.sum(data) * np.log(lam) - n * lam)

   result = minimize_scalar(neg_loglik, bounds=(0.01, 20), method='bounded')
   num_lambda = result.x

   # Step 3: Verify score = 0
   score_at_mle = np.sum(data) / mle_lambda - n

   # Step 4: Fisher information and SE
   # I(lambda) = 1/lambda for one observation
   fisher_info = 1.0 / mle_lambda
   se_lambda = np.sqrt(mle_lambda / n)

   # Step 5: 95% CI
   z = 1.96
   ci = (mle_lambda - z * se_lambda, mle_lambda + z * se_lambda)

   print("=== Software Bug Rate Estimation ===")
   print(f"Modules analysed:  {n}")
   print(f"Total bugs found:  {np.sum(data)}")
   print(f"\nTrue lambda:       {true_lambda}")
   print(f"MLE lambda:        {mle_lambda:.4f}")
   print(f"Numerical MLE:     {num_lambda:.4f}")
   print(f"Score at MLE:      {score_at_mle:.10f}  (should be ~0)")
   print(f"\nFisher info:       {fisher_info:.4f}")
   print(f"SE(lambda):        {se_lambda:.4f}")
   print(f"MLE +/- SE:        {mle_lambda:.4f} +/- {se_lambda:.4f}")
   print(f"95% CI:            ({ci[0]:.4f}, {ci[1]:.4f})")
   print(f"Covers true?       {'Yes' if ci[0] <= true_lambda <= ci[1] else 'No'}")
   print(f"\nSample variance:   {np.var(data, ddof=1):.4f} "
         f"(should be close to lambda = {mle_lambda:.4f})")

10.3.3 Second-Derivative Check
--------------------------------

.. math::

   \frac{d^2\ell}{d\lambda^2}
   = -\frac{1}{\lambda^2}\sum_{i=1}^n x_i < 0

(assuming at least one observation is positive), confirming a maximum.

10.3.4 Note on Unbiasedness
-----------------------------

Unlike the normal variance, the Poisson MLE :math:`\hat\lambda = \bar X`
is unbiased: :math:`E[\bar X] = \lambda`. This is because the Poisson is
a one-parameter exponential family and :math:`\bar X` is the natural
sufficient statistic.

So what? If the 95% CI for :math:`\lambda` lies entirely above the
team's quality threshold (say, 3 bugs per 1000 LOC), management knows
the codebase needs more review effort---and can quantify exactly how
confident they are in that conclusion.


10.4 Binomial Distribution --- :math:`\text{Bin}(m, p)`
=========================================================

**Motivating problem.** An email provider wants to estimate what
fraction of incoming emails are spam. Each day they sample :math:`m`
emails and classify each as spam or not-spam. Over :math:`n` days they
have :math:`n` independent binomial observations. What is the MLE of the
spam rate :math:`p`?

Here we assume the number of trials :math:`m` is known and we estimate
the success probability :math:`p \in (0,1)`.

**Setup.** Let :math:`x_1, \ldots, x_n` be i.i.d.
:math:`\text{Bin}(m, p)` with mass function

.. math::

   f(x \mid p) = \binom{m}{x}\,p^x\,(1-p)^{m-x},
   \qquad x = 0, 1, \ldots, m.

10.4.1 Log-Likelihood
-----------------------

.. math::

   \ell(p)
   = \sum_{i=1}^n \left[
       \log\binom{m}{x_i} + x_i\log p + (m - x_i)\log(1-p)
     \right].

Dropping the constant:

.. math::

   \ell(p) = \log p \sum_{i=1}^n x_i
             + \log(1-p)\sum_{i=1}^n(m - x_i) + C.

The log-likelihood is a weighted combination of :math:`\log p` and
:math:`\log(1-p)`. The weights---total successes and total
failures---are exactly the sufficient statistics.

10.4.2 Score and MLE
----------------------

.. math::

   \frac{d\ell}{dp}
   = \frac{\sum x_i}{p} - \frac{\sum(m - x_i)}{1 - p}
   = \frac{\sum x_i}{p} - \frac{nm - \sum x_i}{1 - p}.

Setting to zero and solving:

.. math::

   \frac{\sum x_i}{p} = \frac{nm - \sum x_i}{1 - p}.

Cross-multiplying:

.. math::

   (1-p)\sum x_i = p\,(nm - \sum x_i)
   \;\Longrightarrow\;
   \sum x_i = p\,nm.

Therefore:

.. math::

   \boxed{\hat p = \frac{\sum_{i=1}^n x_i}{nm} = \frac{\bar x}{m}.}

The MLE is the total number of successes divided by the total number of
trials---the observed proportion.

.. code-block:: python

   # Binomial MLE: spam rate estimation
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   true_p = 0.42    # true spam rate
   m = 200          # emails sampled per day
   n = 30           # number of days
   data = np.random.binomial(m, true_p, size=n)

   # Step 1: Analytical MLE
   total_spam = np.sum(data)
   total_emails = n * m
   mle_p = total_spam / total_emails

   # Step 2: Compare to numerical optimisation
   def neg_loglik(p):
       if p <= 0 or p >= 1:
           return 1e10
       return -(total_spam * np.log(p) + (total_emails - total_spam) * np.log(1 - p))

   result = minimize_scalar(neg_loglik, bounds=(0.001, 0.999), method='bounded')
   num_p = result.x

   # Step 3: Verify score = 0
   score_at_mle = total_spam / mle_p - (total_emails - total_spam) / (1 - mle_p)

   # Step 4: Fisher information and SE
   # For n Bin(m, p) observations: total info = nm / [p(1-p)]
   # SE of mle_p = sqrt(p(1-p) / (nm))
   fisher_info_one = m / (mle_p * (1 - mle_p))  # per observation
   se_p = np.sqrt(mle_p * (1 - mle_p) / total_emails)

   # Step 5: 95% CI
   z = 1.96
   ci = (mle_p - z * se_p, mle_p + z * se_p)

   print("=== Email Spam Rate Estimation ===")
   print(f"Days sampled:       {n}")
   print(f"Emails per day:     {m}")
   print(f"Total spam found:   {total_spam} / {total_emails}")
   print(f"\nTrue p:             {true_p}")
   print(f"MLE p:              {mle_p:.4f}")
   print(f"Numerical MLE:      {num_p:.4f}")
   print(f"Score at MLE:       {score_at_mle:.10f}  (should be ~0)")
   print(f"\nFisher info (per obs): {fisher_info_one:.4f}")
   print(f"SE(p):              {se_p:.4f}")
   print(f"MLE +/- SE:         {mle_p:.4f} +/- {se_p:.4f}")
   print(f"95% CI:             ({ci[0]:.4f}, {ci[1]:.4f})")
   print(f"Covers true?        {'Yes' if ci[0] <= true_p <= ci[1] else 'No'}")

10.4.3 Second-Derivative Check
--------------------------------

.. math::

   \frac{d^2\ell}{dp^2}
   = -\frac{\sum x_i}{p^2} - \frac{nm - \sum x_i}{(1-p)^2} < 0,

confirming a maximum (as long as the data are not all zeros or all
:math:`m`).

So what? The email provider now has a precise estimate: about
:math:`\hat{p}` of all incoming email is spam, with a confidence band
that tells them how much this estimate might fluctuate. This drives
resource allocation for spam-filtering infrastructure.


10.5 Gamma Distribution --- :math:`\text{Gamma}(\alpha, \beta)`
=================================================================

The Gamma distribution has density

.. math::

   f(x \mid \alpha, \beta)
   = \frac{\beta^\alpha}{\Gamma(\alpha)}\,x^{\alpha-1}\,e^{-\beta x},
   \qquad x > 0,

where :math:`\alpha > 0` is the shape and :math:`\beta > 0` is the
rate.

The Gamma is a flexible family that includes the exponential
(:math:`\alpha = 1`) as a special case. You will encounter it in
survival analysis, Bayesian statistics (as a conjugate prior for the
Poisson), and anywhere you need to model positive continuous data with a
right skew.

10.5.1 Log-Likelihood
-----------------------

.. math::

   \ell(\alpha, \beta)
   = n\alpha\log\beta - n\log\Gamma(\alpha)
     + (\alpha - 1)\sum_{i=1}^n \log x_i
     - \beta\sum_{i=1}^n x_i.

10.5.2 MLE for :math:`\beta` (Conditional on :math:`\alpha`)
--------------------------------------------------------------

Differentiating with respect to :math:`\beta`:

.. math::

   \frac{\partial\ell}{\partial\beta}
   = \frac{n\alpha}{\beta} - \sum_{i=1}^n x_i.

Setting to zero:

.. math::

   \hat\beta = \frac{n\alpha}{\sum_{i=1}^n x_i}
   = \frac{\alpha}{\bar x}.

For **fixed** :math:`\alpha`, the MLE of :math:`\beta` has a clean
closed form. This makes the Gamma a "half-analytical" case: one
parameter can be solved in closed form, but the other requires numerical
methods.

10.5.3 MLE for :math:`\alpha` --- Why Numerical Methods Are Required
----------------------------------------------------------------------

Differentiating with respect to :math:`\alpha`:

.. math::

   \frac{\partial\ell}{\partial\alpha}
   = n\log\beta - n\,\psi(\alpha) + \sum_{i=1}^n \log x_i,

where :math:`\psi(\alpha) = \Gamma'(\alpha)/\Gamma(\alpha)` is the
**digamma function**.

Substituting the conditional MLE :math:`\hat\beta = \alpha/\bar x`:

.. math::

   n\log\frac{\alpha}{\bar x} - n\,\psi(\alpha)
   + \sum_{i=1}^n \log x_i = 0.

Rearranging:

.. math::

   \log\alpha - \psi(\alpha)
   = \log\bar x - \frac{1}{n}\sum_{i=1}^n \log x_i.

The right-hand side is a known constant computed from the data (it
equals :math:`\log\bar x - \overline{\log x}`, which is always positive
by Jensen's inequality). The left-hand side,
:math:`\log\alpha - \psi(\alpha)`, is a monotonically decreasing
function of :math:`\alpha` that maps :math:`(0, \infty) \to (0, \infty)`.

**There is no closed-form inverse.** The digamma function is
transcendental, so :math:`\alpha` must be found numerically---for
example, by Newton--Raphson iteration or a fixed-point scheme. A common
starting value is the method-of-moments estimator
:math:`\tilde\alpha = \bar x^2 / s^2`.

.. code-block:: python

   # Gamma MLE: numerical solution for alpha, then beta = alpha / x_bar
   import numpy as np
   from scipy import stats
   from scipy.special import digamma, polygamma
   from scipy.optimize import brentq

   np.random.seed(42)
   true_alpha, true_beta = 3.0, 2.0
   n = 200
   data = np.random.gamma(true_alpha, scale=1/true_beta, size=n)

   x_bar = np.mean(data)
   log_x_bar = np.log(x_bar)
   mean_log_x = np.mean(np.log(data))
   s = log_x_bar - mean_log_x  # always positive by Jensen

   # Solve: log(alpha) - digamma(alpha) = s
   def score_alpha(a):
       return np.log(a) - digamma(a) - s

   mle_alpha = brentq(score_alpha, 0.01, 1000)
   mle_beta = mle_alpha / x_bar

   # Verify score is zero at MLE for both parameters
   score_a = n * np.log(mle_beta) - n * digamma(mle_alpha) + np.sum(np.log(data))
   score_b = n * mle_alpha / mle_beta - np.sum(data)

   # Compare to scipy
   sp_alpha, sp_loc, sp_scale = stats.gamma.fit(data, floc=0)
   sp_beta = 1.0 / sp_scale

   # Fisher information (diagonal elements of 2x2 matrix)
   fi_alpha = polygamma(1, mle_alpha)  # trigamma
   fi_beta = mle_alpha / mle_beta**2
   se_alpha = 1.0 / np.sqrt(n * fi_alpha)
   se_beta = 1.0 / np.sqrt(n * fi_beta)

   z = 1.96
   ci_alpha = (mle_alpha - z * se_alpha, mle_alpha + z * se_alpha)
   ci_beta = (mle_beta - z * se_beta, mle_beta + z * se_beta)

   print("=== Gamma MLE ===")
   print(f"True:    alpha = {true_alpha}, beta = {true_beta}")
   print(f"MLE:     alpha = {mle_alpha:.4f}, beta = {mle_beta:.4f}")
   print(f"scipy:   alpha = {sp_alpha:.4f}, beta = {sp_beta:.4f}")
   print(f"\nScore at MLE (should be ~0):")
   print(f"  d ell/d alpha = {score_a:.6f}")
   print(f"  d ell/d beta  = {score_b:.6f}")
   print(f"\nSE(alpha) = {se_alpha:.4f}, SE(beta) = {se_beta:.4f}")
   print(f"95% CI alpha: ({ci_alpha[0]:.4f}, {ci_alpha[1]:.4f})"
         f"  covers true? {'Yes' if ci_alpha[0] <= true_alpha <= ci_alpha[1] else 'No'}")
   print(f"95% CI beta:  ({ci_beta[0]:.4f}, {ci_beta[1]:.4f})"
         f"  covers true? {'Yes' if ci_beta[0] <= true_beta <= ci_beta[1] else 'No'}")

10.5.4 Second-Derivative Check
--------------------------------

The Hessian with respect to :math:`(\alpha, \beta)` has diagonal
entries:

.. math::

   \frac{\partial^2\ell}{\partial\alpha^2}
   = -n\,\psi'(\alpha), \qquad
   \frac{\partial^2\ell}{\partial\beta^2}
   = -\frac{n\alpha}{\beta^2}.

Since the trigamma function :math:`\psi'(\alpha) > 0` for all
:math:`\alpha > 0`, both diagonal entries are negative, and the
log-likelihood is jointly concave, confirming a unique maximum.


10.6 Beta Distribution --- :math:`\text{Beta}(\alpha, \beta)`
===============================================================

The Beta distribution has density

.. math::

   f(x \mid \alpha, \beta)
   = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\,\Gamma(\beta)}\,
     x^{\alpha-1}(1-x)^{\beta-1},
   \qquad x \in (0,1).

The Beta is the natural distribution for modelling proportions and
probabilities. It appears as the conjugate prior for the Binomial in
Bayesian statistics and is widely used in A/B testing, sports analytics,
and any context where you need a flexible distribution on :math:`(0,1)`.

10.6.1 Log-Likelihood
-----------------------

.. math::

   \ell(\alpha, \beta)
   &= n\bigl[\log\Gamma(\alpha+\beta) - \log\Gamma(\alpha)
      - \log\Gamma(\beta)\bigr] \\
   &\quad + (\alpha-1)\sum_{i=1}^n \log x_i
     + (\beta-1)\sum_{i=1}^n \log(1-x_i).

10.6.2 Score Equations
-----------------------

.. math::

   \frac{\partial\ell}{\partial\alpha}
   &= n\bigl[\psi(\alpha+\beta) - \psi(\alpha)\bigr]
      + \sum_{i=1}^n \log x_i = 0, \\[6pt]
   \frac{\partial\ell}{\partial\beta}
   &= n\bigl[\psi(\alpha+\beta) - \psi(\beta)\bigr]
      + \sum_{i=1}^n \log(1-x_i) = 0.

These are two coupled equations involving the digamma function. Unlike
the Gamma case, neither parameter can be eliminated to yield a
one-dimensional equation. Both equations are **transcendental** and
must be solved simultaneously by numerical optimisation.

10.6.3 Numerical Approaches
-----------------------------

- **Newton--Raphson.** Compute the :math:`2 \times 2` Hessian using the
  trigamma function :math:`\psi'` and iterate:

  .. math::

     \begin{pmatrix}\alpha\\\beta\end{pmatrix}^{(t+1)}
     =
     \begin{pmatrix}\alpha\\\beta\end{pmatrix}^{(t)}
     - H^{-1}\,\nabla\ell.

  The Hessian is

  .. math::

     H = n\begin{pmatrix}
     \psi'(\alpha+\beta)-\psi'(\alpha)
       & \psi'(\alpha+\beta) \\
     \psi'(\alpha+\beta)
       & \psi'(\alpha+\beta)-\psi'(\beta)
     \end{pmatrix}.

- **Fixed-point iteration.** Rewrite the score equations as

  .. math::

     \psi(\alpha) &= \psi(\alpha+\beta)
       + \frac{1}{n}\sum \log x_i, \\
     \psi(\beta) &= \psi(\alpha+\beta)
       + \frac{1}{n}\sum \log(1-x_i),

  and apply the inverse digamma function :math:`\psi^{-1}` iteratively.
  This converges more slowly than Newton--Raphson but is simpler to
  implement.

- **Method-of-moments initialisation.** A good starting point is

  .. math::

     \tilde\alpha = \bar x\!\left(
       \frac{\bar x(1-\bar x)}{s^2} - 1
     \right), \qquad
     \tilde\beta = (1-\bar x)\!\left(
       \frac{\bar x(1-\bar x)}{s^2} - 1
     \right).

.. code-block:: python

   # Beta MLE: numerical optimisation with full verification
   import numpy as np
   from scipy import stats
   from scipy.optimize import minimize
   from scipy.special import gammaln, digamma, polygamma

   np.random.seed(42)
   true_alpha, true_beta = 2.0, 5.0
   n = 200
   data = np.random.beta(true_alpha, true_beta, size=n)

   sum_log_x = np.sum(np.log(data))
   sum_log_1mx = np.sum(np.log(1 - data))

   # Negative log-likelihood
   def neg_log_lik(params):
       a, b = params
       if a <= 0 or b <= 0:
           return 1e10
       return -(n * (gammaln(a + b) - gammaln(a) - gammaln(b))
                + (a - 1) * sum_log_x + (b - 1) * sum_log_1mx)

   # Method-of-moments starting values
   xbar = np.mean(data)
   s2 = np.var(data, ddof=1)
   common = xbar * (1 - xbar) / s2 - 1
   a0, b0 = xbar * common, (1 - xbar) * common

   result = minimize(neg_log_lik, [a0, b0], method='Nelder-Mead')
   mle_a, mle_b = result.x

   # Verify score is zero at MLE
   score_a = n * (digamma(mle_a + mle_b) - digamma(mle_a)) + sum_log_x
   score_b = n * (digamma(mle_a + mle_b) - digamma(mle_b)) + sum_log_1mx

   # Compare to scipy
   sp_a, sp_b, sp_loc, sp_scale = stats.beta.fit(data, floc=0, fscale=1)

   # Fisher information (2x2 matrix)
   psi1_ab = polygamma(1, mle_a + mle_b)
   fi_aa = n * (polygamma(1, mle_a) - psi1_ab)
   fi_bb = n * (polygamma(1, mle_b) - psi1_ab)
   fi_ab = -n * psi1_ab
   fi_matrix = np.array([[fi_aa, fi_ab], [fi_ab, fi_bb]])
   cov_matrix = np.linalg.inv(fi_matrix)
   se_a = np.sqrt(cov_matrix[0, 0])
   se_b = np.sqrt(cov_matrix[1, 1])

   z = 1.96
   ci_a = (mle_a - z * se_a, mle_a + z * se_a)
   ci_b = (mle_b - z * se_b, mle_b + z * se_b)

   print("=== Beta MLE ===")
   print(f"True:    alpha = {true_alpha}, beta = {true_beta}")
   print(f"MLE:     alpha = {mle_a:.4f}, beta = {mle_b:.4f}")
   print(f"scipy:   alpha = {sp_a:.4f}, beta = {sp_b:.4f}")
   print(f"\nScore at MLE (should be ~0):")
   print(f"  d ell/d alpha = {score_a:.6f}")
   print(f"  d ell/d beta  = {score_b:.6f}")
   print(f"\nSE(alpha) = {se_a:.4f}, SE(beta) = {se_b:.4f}")
   print(f"95% CI alpha: ({ci_a[0]:.4f}, {ci_a[1]:.4f})"
         f"  covers true? {'Yes' if ci_a[0] <= true_alpha <= ci_a[1] else 'No'}")
   print(f"95% CI beta:  ({ci_b[0]:.4f}, {ci_b[1]:.4f})"
         f"  covers true? {'Yes' if ci_b[0] <= true_beta <= ci_b[1] else 'No'}")


10.7 Uniform Distribution --- :math:`\text{Unif}(0, \theta)`
==============================================================

The Uniform distribution provides an instructive example where
regularity conditions fail and the MLE behaves very differently from the
standard theory in :ref:`ch9_mle_theory`.

**Setup.** Let :math:`x_1, \ldots, x_n \sim \text{Unif}(0, \theta)`
with density

.. math::

   f(x \mid \theta) = \frac{1}{\theta}\,\mathbf{1}(0 \leq x \leq \theta).

.. admonition:: Common Pitfall

   Students often try to find the MLE by differentiating and setting
   equal to zero, as with the other distributions. That approach fails
   here because the support of the distribution depends on
   :math:`\theta`. The log-likelihood has no interior critical point---the
   MLE sits at a boundary. Always check whether the regularity conditions
   hold before blindly applying the score equation.

10.7.1 Likelihood
-------------------

The likelihood is nonzero only if all observations fall in
:math:`[0, \theta]`:

.. math::

   L(\theta) = \prod_{i=1}^n \frac{1}{\theta}\,
   \mathbf{1}(x_i \leq \theta)
   = \frac{1}{\theta^n}\,\mathbf{1}(\theta \geq x_{(n)}),

where :math:`x_{(n)} = \max(x_1, \ldots, x_n)` is the largest
observation.

10.7.2 Log-Likelihood
-----------------------

For :math:`\theta \geq x_{(n)}`:

.. math::

   \ell(\theta) = -n\log\theta.

This is a strictly **decreasing** function of :math:`\theta`. Therefore,
the log-likelihood has **no interior critical point**---the derivative
:math:`d\ell/d\theta = -n/\theta < 0` never equals zero.

10.7.3 The MLE via Order Statistics
--------------------------------------

Since :math:`\ell(\theta)` is decreasing for
:math:`\theta \geq x_{(n)}` and :math:`L(\theta) = 0` for
:math:`\theta < x_{(n)}`, the likelihood is maximised at the smallest
permissible value:

.. math::

   \boxed{\hat\theta = x_{(n)} = \max(x_1, \ldots, x_n).}

The MLE is the maximum order statistic. There is no score equation to
solve---the maximum occurs at the **boundary** of the feasible region.

.. code-block:: python

   # Uniform MLE: maximum order statistic
   import numpy as np

   np.random.seed(42)
   true_theta = 10.0
   n = 50
   data = np.random.uniform(0, true_theta, size=n)

   mle_theta = np.max(data)
   unbiased_theta = (n + 1) / n * mle_theta

   print(f"True theta:       {true_theta}")
   print(f"MLE (max):        {mle_theta:.4f}")
   print(f"Unbiased version: {unbiased_theta:.4f}")
   print(f"MLE bias:         {mle_theta - true_theta:.4f} (underestimates)")

10.7.4 Properties
-------------------

The standard asymptotic theory does not apply here because the support
depends on :math:`\theta` (violating regularity condition 2 in
:ref:`ch9_mle_theory`). The MLE converges at rate :math:`n` rather than
:math:`\sqrt{n}`:

.. math::

   n(\theta - \hat\theta) \;\xrightarrow{d}\; \text{Exp}(1/\theta).

Furthermore, the MLE is biased:

.. math::

   E[X_{(n)}] = \frac{n}{n+1}\,\theta,

so :math:`\hat\theta` underestimates :math:`\theta`. The unbiased
correction is :math:`\frac{n+1}{n}\,x_{(n)}`.

This faster convergence rate is actually a bonus---the Uniform MLE is
*super-consistent*, converging faster than any regular MLE. The price we
pay is that the standard confidence interval formulas no longer apply.

.. code-block:: python

   # Verify super-consistency: Uniform MLE converges at rate n, not sqrt(n)
   import numpy as np

   np.random.seed(42)
   true_theta = 10.0
   n_sims = 50000

   print(f"Rate of convergence for Uniform MLE:")
   print(f"{'n':>6s}  {'sqrt(n)*|bias|':>15s}  {'n*|bias|':>15s}")
   print("-" * 42)
   for n in [20, 50, 100, 500, 1000]:
       errors = [true_theta - np.max(np.random.uniform(0, true_theta, size=n))
                 for _ in range(n_sims)]
       mean_error = np.mean(errors)
       print(f"{n:6d}  {np.sqrt(n) * mean_error:15.4f}  "
             f"{n * mean_error:15.4f}")

   print("\n(n * |bias| should stabilise => convergence rate is 1/n, not 1/sqrt(n))")


10.8 Multinomial Distribution
===============================

The Multinomial distribution generalises the Binomial to :math:`k`
categories. It is fundamental in categorical data analysis, natural
language processing, and genetics.

**Setup.** A single multinomial trial produces a vector
:math:`(x_1, \ldots, x_k)` with :math:`\sum x_j = m` and mass function

.. math::

   f(x_1, \ldots, x_k \mid p_1, \ldots, p_k)
   = \frac{m!}{x_1!\cdots x_k!}\;\prod_{j=1}^k p_j^{x_j},

subject to the constraint :math:`\sum_{j=1}^k p_j = 1`. With :math:`n`
independent multinomial observations (or equivalently, pooling counts
into :math:`n_j = \sum_{i=1}^n x_{ij}`):

10.8.1 Log-Likelihood
-----------------------

Dropping constants:

.. math::

   \ell(p_1, \ldots, p_k)
   = \sum_{j=1}^k n_j \log p_j,

where :math:`n_j` is the total count in category :math:`j` and
:math:`N = \sum_j n_j` is the grand total.

10.8.2 Constrained Optimisation via Lagrange Multipliers
----------------------------------------------------------

We must maximise :math:`\ell` subject to :math:`\sum_j p_j = 1`. Form
the Lagrangian:

.. math::

   \mathcal{L}(p_1, \ldots, p_k, \lambda)
   = \sum_{j=1}^k n_j\log p_j
     + \lambda\!\left(1 - \sum_{j=1}^k p_j\right).

Differentiate with respect to :math:`p_j`:

.. math::

   \frac{\partial\mathcal{L}}{\partial p_j}
   = \frac{n_j}{p_j} - \lambda = 0
   \quad\Longrightarrow\quad
   p_j = \frac{n_j}{\lambda}.

Now enforce the constraint:

.. math::

   \sum_{j=1}^k p_j = 1
   \;\Longrightarrow\;
   \sum_{j=1}^k \frac{n_j}{\lambda} = 1
   \;\Longrightarrow\;
   \lambda = N.

Therefore:

.. math::

   \boxed{\hat p_j = \frac{n_j}{N}, \qquad j = 1, \ldots, k.}

The MLE of each category probability is the **observed relative
frequency**---an intuitive and satisfying result. Each :math:`\hat p_j`
is unbiased for :math:`p_j`.

.. code-block:: python

   # Multinomial MLE: observed relative frequencies
   import numpy as np

   np.random.seed(42)
   true_probs = np.array([0.1, 0.3, 0.15, 0.25, 0.2])
   total_count = 500
   counts = np.random.multinomial(total_count, true_probs)

   mle_probs = counts / counts.sum()

   print("Category  True p   Count   MLE p")
   print("-" * 40)
   for j in range(len(true_probs)):
       print(f"   {j+1}      {true_probs[j]:.2f}     {counts[j]:4d}    {mle_probs[j]:.4f}")
   print(f"\nTotal count: {counts.sum()}")
   print(f"MLE sums to: {mle_probs.sum():.4f}")

10.8.3 Second-Derivative Check
--------------------------------

The bordered Hessian for the constrained problem can be verified by
noting that the log-likelihood is concave (since :math:`\log` is
concave and :math:`n_j \geq 0`). Alternatively, the unconstrained
parameterisation using :math:`k-1` free probabilities yields a negative
definite Hessian on the probability simplex.


10.9 Summary of Analytical MLEs
=================================

The following table collects the results derived in this chapter.

.. list-table:: Analytical MLE Solutions
   :header-rows: 1
   :widths: 25 30 20

   * - Distribution
     - Parameter(s)
     - MLE
   * - Normal
     - :math:`\mu,\;\sigma^2`
     - :math:`\bar x,\;\frac{1}{n}\sum(x_i-\bar x)^2`
   * - Exponential
     - :math:`\lambda`
     - :math:`1/\bar x`
   * - Poisson
     - :math:`\lambda`
     - :math:`\bar x`
   * - Binomial
     - :math:`p`
     - :math:`\bar x / m`
   * - Gamma
     - :math:`\alpha,\;\beta`
     - :math:`\beta = \alpha/\bar x`; :math:`\alpha` numerical
   * - Beta
     - :math:`\alpha,\;\beta`
     - Both numerical (digamma equations)
   * - Uniform :math:`(0,\theta)`
     - :math:`\theta`
     - :math:`x_{(n)} = \max_i x_i`
   * - Multinomial
     - :math:`p_1, \ldots, p_k`
     - :math:`n_j / N`

Let's verify every entry in this table with a single comprehensive code
block:

.. code-block:: python

   # Comprehensive verification: all MLEs in one place
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   results = []

   # Normal
   d = np.random.normal(10.0, 0.15, size=100)
   results.append(("Normal mu", 10.0, np.mean(d)))
   results.append(("Normal sigma^2", 0.15**2, np.mean((d - np.mean(d))**2)))

   # Exponential
   d = np.random.exponential(1/0.04, size=100)
   results.append(("Exponential lambda", 0.04, 1.0 / np.mean(d)))

   # Poisson
   d = np.random.poisson(4.2, size=100)
   results.append(("Poisson lambda", 4.2, np.mean(d)))

   # Binomial
   m_binom = 200
   d = np.random.binomial(m_binom, 0.42, size=30)
   results.append(("Binomial p", 0.42, np.sum(d) / (30 * m_binom)))

   # Uniform
   d = np.random.uniform(0, 10.0, size=50)
   results.append(("Uniform theta", 10.0, np.max(d)))

   print(f"{'Distribution':<22s}  {'True':>8s}  {'MLE':>8s}  {'Error':>10s}")
   print("-" * 55)
   for name, true, mle in results:
       print(f"{name:<22s}  {true:8.4f}  {mle:8.4f}  {abs(mle - true):10.6f}")

Several patterns emerge:

- For **exponential-family** models (Normal, Poisson, Binomial,
  Exponential, Multinomial), the MLE is a simple function of sufficient
  statistics and often has a closed form.

- For models with **transcendental** normalising constants (Gamma shape,
  Beta), the score equation involves the digamma function and must be
  solved numerically.

- For models with **parameter-dependent support** (Uniform), the MLE
  occurs at a boundary, regularity conditions fail, and the standard
  asymptotics do not apply.

.. admonition:: Intuition

   There is a unifying theme across all of these derivations: the MLE
   always "matches" some feature of the data to the model. For the
   Poisson, it matches the sample mean to the theoretical mean. For the
   Binomial, it matches the observed proportion to the theoretical
   probability. For the Uniform, it matches the range of the data to
   the support of the distribution. This "matching" is not a coincidence
   ---it reflects the deeper connection between MLEs and sufficient
   statistics in exponential families.

The next chapter shows how to use these MLEs to build confidence
intervals and hypothesis tests.
