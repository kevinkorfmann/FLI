.. _ch17_bayesian:

==========================================
Chapter 17 -- Bayesian Connections
==========================================

You have spent Parts I--IV mastering the likelihood function: finding the MLE,
quantifying uncertainty with Fisher information, and testing hypotheses with
likelihood ratios.  But here is a situation that MLE alone cannot handle well.

**The motivating problem.**
A pharmaceutical company is running a clinical trial for a new drug.  They
enroll 40 patients and observe 23 responders.  The MLE of the response rate is
:math:`23/40 = 0.575`.  But this is not the first study of this drug---a pilot
study with 20 patients had already suggested a response rate around 60%.
Throwing away that pilot data feels wasteful.  And the regulatory agency wants
to know: "What is the *probability* that the response rate exceeds 50%?"  The
MLE is a point estimate; it cannot answer probability questions about parameters.

Bayesian inference solves both problems.  It combines the likelihood (what the
current data say) with a *prior distribution* (what we already knew) to produce
a *posterior distribution*---a complete probabilistic summary of the parameter
after seeing the data.

.. code-block:: python

   # The clinical trial scenario we will develop throughout this chapter
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Pilot study: 12 responders out of 20 patients (~60% response rate)
   # This becomes our prior: Beta(12, 8)
   alpha_prior, beta_prior = 12, 8

   # New trial: 23 responders out of 40 patients
   n_trial, s_trial = 40, 23

   # The MLE ignores the pilot study entirely
   mle = s_trial / n_trial
   print(f"MLE (new trial only):  {mle:.4f}")
   print(f"Pilot estimate:        {12/20:.4f}")
   print(f"Question: Can we combine these? -> Chapter 17 shows how.")


17.1 Bayes' Theorem for Parameters
------------------------------------

**Why we need this.**
Maximum likelihood gives us a single "best" parameter value, but it cannot
directly answer "What is the probability that :math:`\theta > 0.5`?"  To make
probability statements about parameters, we need a framework that treats the
parameter as a random quantity.  Bayes' theorem provides exactly that.

Recall the elementary form of Bayes' theorem for events :math:`A` and
:math:`B`:

.. math::

   P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}.

Now replace event :math:`A` with a parameter vector :math:`\theta` and event
:math:`B` with the observed data :math:`\mathbf{x}`:

.. math::

   p(\theta \mid \mathbf{x})
   = \frac{p(\mathbf{x} \mid \theta)\, p(\theta)}{p(\mathbf{x})}.

Each piece has a name:

* :math:`p(\theta \mid \mathbf{x})` --- the **posterior distribution**, our
  updated belief about :math:`\theta` after seeing data.
* :math:`p(\mathbf{x} \mid \theta)` --- the **likelihood**, the same function
  :math:`L(\theta)` we have studied throughout this guide.
* :math:`p(\theta)` --- the **prior distribution**, encoding what we believe
  about :math:`\theta` before seeing the data.
* :math:`p(\mathbf{x}) = \int p(\mathbf{x} \mid \theta)\, p(\theta)\, d\theta`
  --- the **marginal likelihood** (or *evidence*), a normalizing constant
  ensuring the posterior integrates to one.

Because the marginal likelihood does not depend on :math:`\theta`, we often
write the proportionality:

.. math::

   p(\theta \mid \mathbf{x}) \;\propto\; L(\theta)\, p(\theta).

In words: *the posterior is proportional to the likelihood times the prior*.
This single line is the bridge between the likelihood world and the Bayesian
world.

Let us verify this computationally.  We will compute the posterior on a grid
of :math:`\theta` values and compare with the exact conjugate answer.

.. code-block:: python

   # Verify Bayes' theorem: posterior = likelihood * prior (up to normalization)
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Clinical trial data: 23 responders in 40 patients
   n, s = 40, 23

   # Prior from pilot study: Beta(12, 8)
   alpha_prior, beta_prior = 12, 8

   # Grid of theta values
   theta = np.linspace(0.001, 0.999, 2000)

   # Each ingredient of Bayes' theorem
   likelihood = stats.binom.pmf(s, n, theta)
   prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
   unnorm_posterior = likelihood * prior

   # Normalize by trapezoidal integration
   posterior = unnorm_posterior / np.trapz(unnorm_posterior, theta)

   # The exact answer: Beta(12+23, 8+17) = Beta(35, 25)
   exact_posterior = stats.beta.pdf(theta, alpha_prior + s, beta_prior + n - s)

   print("Verifying posterior = likelihood * prior:")
   print(f"  Grid posterior mean:  {np.trapz(theta * posterior, theta):.4f}")
   print(f"  Exact posterior mean: {(alpha_prior + s) / (alpha_prior + beta_prior + n):.4f}")
   print(f"  MLE (ignores prior):  {s / n:.4f}")
   print(f"  Prior mean:           {alpha_prior / (alpha_prior + beta_prior):.4f}")
   print(f"  Max grid error:       {np.max(np.abs(posterior - exact_posterior)):.6f}")

.. admonition:: Intuition

   Imagine you are tuning a radio.  The likelihood is the signal strength at
   each dial position---it tells you where the broadcast probably is.  The prior
   is your friend telling you, "I think it's somewhere around 99 FM."  The
   posterior combines both: you listen to the signal *and* trust your friend,
   weighting each by how informative they are.


17.2 Prior Specification
--------------------------

**Why priors matter.**
The prior :math:`p(\theta)` encodes external knowledge (or ignorance) about
the parameter before the data arrive.  Choosing a prior is both the most
powerful and most controversial aspect of Bayesian inference.  Let's walk
through the main strategies and see how they affect the clinical trial.

.. code-block:: python

   # How different priors affect the posterior for our clinical trial
   import numpy as np
   from scipy import stats

   n, s = 40, 23  # trial data
   theta = np.linspace(0.001, 0.999, 1000)

   priors = {
       "Informative (pilot study)":   (12, 8),    # strong prior from pilot
       "Weakly informative":          (2, 2),     # mild belief in fairness
       "Non-informative (uniform)":   (1, 1),     # flat prior
       "Jeffreys":                    (0.5, 0.5), # parameterization-invariant
   }

   print(f"{'Prior':<32} {'Prior mean':>10} {'Post mean':>10} {'Post 95% CI':>20}")
   print("-" * 76)
   for name, (a, b) in priors.items():
       post = stats.beta(a + s, b + n - s)
       ci = post.ppf([0.025, 0.975])
       print(f"{name:<32} {a/(a+b):>10.4f} {post.mean():>10.4f} "
             f"{'[{:.3f}, {:.3f}]'.format(ci[0], ci[1]):>20}")

   print(f"\n{'MLE (no prior)':<32} {'---':>10} {s/n:>10.4f} {'---':>20}")
   print("\nNote: with n=40 data points, even strong priors have modest effect.")

Informative priors
^^^^^^^^^^^^^^^^^^

When genuine domain knowledge exists --- for example, a physicist knows a mass
must be positive, or a clinician has data from a pilot study --- we encode it
directly.  Our pilot study of 20 patients with 12 responders translates to a
:math:`\text{Beta}(12, 8)` prior, which concentrates around 0.60.

.. admonition:: Why does this matter?

   Informative priors let you formally incorporate decades of scientific
   knowledge into your analysis.  Without them, you are pretending each new
   study starts from a blank slate---which is rarely true and often wasteful.

Non-informative (vague) priors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we wish to "let the data speak," we use priors that are as flat as
possible.  A common choice for a location parameter is:

.. math::

   p(\theta) \propto 1, \qquad \theta \in (-\infty, \infty).

This is an *improper* prior --- it does not integrate to a finite value ---
but the posterior can still be proper provided the likelihood is integrable.

Jeffreys prior
^^^^^^^^^^^^^^

Harold Jeffreys proposed a prior that is invariant under reparameterization.
The idea: a "non-informative" prior should give the same inferences whether we
parameterize by :math:`\theta` or by :math:`\phi = g(\theta)`.

**Derivation.**
The Fisher information (see :ref:`Part II <part2>`) for a scalar parameter is:

.. math::

   I(\theta) = -E\!\left[\frac{\partial^2}{\partial \theta^2}
               \log p(\mathbf{x} \mid \theta)\right].

Under the transformation :math:`\phi = g(\theta)` the information transforms as

.. math::

   I_\phi(\phi) = I_\theta(\theta)\, \left(\frac{d\theta}{d\phi}\right)^2.

Here is the key step: if we set the prior proportional to the square root of
the Fisher information, the result is invariant to how we parameterize the
model.

.. math::

   p_J(\theta) \;\propto\; \sqrt{I(\theta)},

then under :math:`\phi = g(\theta)`:

.. math::

   p_J(\phi)
   = p_J(\theta)\, \left|\frac{d\theta}{d\phi}\right|
   \propto \sqrt{I_\theta(\theta)}\, \left|\frac{d\theta}{d\phi}\right|
   = \sqrt{I_\phi(\phi)},

which has the same functional form.  The prior is therefore
*parameterization-invariant*.

**Example: Bernoulli model.**
For :math:`X \sim \text{Bernoulli}(\theta)` the Fisher information is
:math:`I(\theta) = 1/[\theta(1-\theta)]`, so

.. math::

   p_J(\theta) \propto \theta^{-1/2}(1-\theta)^{-1/2},

which is a :math:`\text{Beta}(1/2, 1/2)` distribution --- the *arcsine*
distribution on :math:`[0,1]`.

.. code-block:: python

   # Compute the Jeffreys prior from the Fisher information directly
   import numpy as np
   from scipy import stats

   theta = np.linspace(0.01, 0.99, 500)

   # Fisher information for Bernoulli: I(theta) = 1 / [theta * (1 - theta)]
   fisher_info = 1.0 / (theta * (1 - theta))

   # Jeffreys prior: sqrt(I(theta))
   jeffreys_unnorm = np.sqrt(fisher_info)
   jeffreys_density = jeffreys_unnorm / np.trapz(jeffreys_unnorm, theta)

   # Compare with the exact Beta(0.5, 0.5) density
   exact_jeffreys = stats.beta.pdf(theta, 0.5, 0.5)

   print("Jeffreys prior for Bernoulli = Beta(1/2, 1/2):")
   print(f"  At theta=0.1: computed={jeffreys_density[50]:.3f}, "
         f"exact={stats.beta.pdf(0.1, 0.5, 0.5):.3f}")
   print(f"  At theta=0.5: computed={jeffreys_density[250]:.3f}, "
         f"exact={stats.beta.pdf(0.5, 0.5, 0.5):.3f}")
   print(f"  At theta=0.9: computed={jeffreys_density[450]:.3f}, "
         f"exact={stats.beta.pdf(0.9, 0.5, 0.5):.3f}")
   print("  Note: more mass near 0 and 1 than a uniform prior.")

Reference priors
^^^^^^^^^^^^^^^^

Reference priors (Bernardo, 1979) extend Jeffreys' idea to multi-parameter
settings by maximizing the expected Kullback--Leibler divergence between prior
and posterior, ensuring the data have maximal influence.  They often agree with
Jeffreys' prior for single parameters but handle nuisance parameters more
carefully.


17.3 Conjugate Priors
-----------------------

**Why conjugacy is useful.**
We want the posterior in closed form --- no integrals, no MCMC, no numerical
headaches.  Conjugacy delivers exactly this.  If the prior and the posterior
belong to the same distributional family, the posterior parameters can be read
off directly.

Formally, a family :math:`\mathcal{F}` of distributions is conjugate to a
likelihood :math:`p(\mathbf{x} \mid \theta)` if

.. math::

   p(\theta) \in \mathcal{F}
   \;\;\Longrightarrow\;\;
   p(\theta \mid \mathbf{x}) \in \mathcal{F}.

Table of common conjugate families
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Likelihood
     - Parameter
     - Conjugate prior
     - Posterior
   * - Bernoulli / Binomial
     - :math:`\theta` (probability)
     - :math:`\text{Beta}(\alpha, \beta)`
     - :math:`\text{Beta}(\alpha + s,\; \beta + n - s)`
   * - Poisson
     - :math:`\lambda` (rate)
     - :math:`\text{Gamma}(\alpha, \beta)`
     - :math:`\text{Gamma}(\alpha + \sum x_i,\; \beta + n)`
   * - Normal (known :math:`\sigma^2`)
     - :math:`\mu` (mean)
     - :math:`\text{Normal}(\mu_0, \sigma_0^2)`
     - :math:`\text{Normal}(\mu_n, \sigma_n^2)`
   * - Normal (known :math:`\mu`)
     - :math:`\sigma^2` (variance)
     - :math:`\text{Inv-Gamma}(\alpha, \beta)`
     - :math:`\text{Inv-Gamma}(\alpha + n/2,\; \beta + \tfrac{1}{2}\sum(x_i-\mu)^2)`
   * - Exponential
     - :math:`\lambda` (rate)
     - :math:`\text{Gamma}(\alpha, \beta)`
     - :math:`\text{Gamma}(\alpha + n,\; \beta + \sum x_i)`
   * - Multinomial
     - :math:`\boldsymbol{\theta}` (probabilities)
     - :math:`\text{Dirichlet}(\boldsymbol{\alpha})`
     - :math:`\text{Dirichlet}(\boldsymbol{\alpha} + \mathbf{n})`

Let us verify the entire table computationally.

.. code-block:: python

   # Verify conjugate updates for three families
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   print("=== Beta-Binomial ===")
   alpha, beta_ = 3, 7  # prior
   data_binom = np.random.binomial(1, 0.4, size=30)
   s = data_binom.sum()
   n = len(data_binom)
   post = stats.beta(alpha + s, beta_ + n - s)
   print(f"  Prior: Beta({alpha}, {beta_}), mean={alpha/(alpha+beta_):.3f}")
   print(f"  Data: {s} successes in {n} trials")
   print(f"  Posterior: Beta({alpha+s}, {beta_+n-s}), mean={post.mean():.3f}")

   print("\n=== Gamma-Poisson ===")
   a_gam, b_gam = 2, 1  # prior: Gamma(shape=2, rate=1)
   data_pois = np.random.poisson(3.0, size=20)
   a_post = a_gam + data_pois.sum()
   b_post = b_gam + len(data_pois)
   print(f"  Prior: Gamma({a_gam}, {b_gam}), mean={a_gam/b_gam:.3f}")
   print(f"  Data: {len(data_pois)} obs, sum={data_pois.sum()}")
   print(f"  Posterior: Gamma({a_post}, {b_post}), mean={a_post/b_post:.3f}")
   print(f"  MLE: {data_pois.mean():.3f}")

   print("\n=== Normal-Normal (known sigma) ===")
   mu_0, sigma_0 = 0, 5  # prior
   sigma = 2  # known data sd
   data_norm = np.random.normal(3.0, sigma, size=15)
   x_bar = data_norm.mean()
   prec_prior = 1 / sigma_0**2
   prec_data = len(data_norm) / sigma**2
   mu_n = (prec_data * x_bar + prec_prior * mu_0) / (prec_prior + prec_data)
   sigma_n = np.sqrt(1 / (prec_prior + prec_data))
   print(f"  Prior: N({mu_0}, {sigma_0}^2)")
   print(f"  Data: n={len(data_norm)}, x_bar={x_bar:.3f}")
   print(f"  Posterior: N({mu_n:.3f}, {sigma_n:.3f}^2)")
   print(f"  Weight on data: {prec_data/(prec_prior+prec_data):.1%}")

Derivation: Beta--Binomial conjugacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose we observe :math:`s` successes in :math:`n` Bernoulli trials and place
a :math:`\text{Beta}(\alpha, \beta)` prior on the success probability
:math:`\theta`.

The likelihood is:

.. math::

   L(\theta) = \binom{n}{s} \theta^s (1-\theta)^{n-s}.

The prior density is:

.. math::

   p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},

where :math:`B(\alpha,\beta)` is the Beta function.

Multiplying:

.. math::

   p(\theta \mid s)
   &\propto \theta^s (1-\theta)^{n-s} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} \\
   &= \theta^{(\alpha + s) - 1} (1-\theta)^{(\beta + n - s) - 1}.

This is the kernel of a :math:`\text{Beta}(\alpha + s,\; \beta + n - s)`
distribution.  The posterior is therefore:

.. math::

   \theta \mid s \;\sim\; \text{Beta}(\alpha + s,\; \beta + n - s).

Each success adds one to :math:`\alpha`; each failure adds one to
:math:`\beta`.  The prior "pseudo-counts" :math:`\alpha` and :math:`\beta`
play the role of imaginary previous observations.

Now we return to our clinical trial.  The pilot study gives us
:math:`\text{Beta}(12, 8)`, and the new trial contributes 23 successes and 17
failures.

.. code-block:: python

   # Clinical trial: watch the posterior evolve patient by patient
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Prior from pilot study
   alpha, beta_ = 12, 8

   # Simulate individual patient outcomes (23 responders out of 40)
   outcomes = np.array([1]*23 + [0]*17)
   np.random.shuffle(outcomes)

   print(f"{'Patient':>8} {'Outcome':>8} {'alpha':>7} {'beta':>7} "
         f"{'Post mean':>10} {'95% CI':>22}")
   print("-" * 66)
   print(f"{'Prior':>8} {'---':>8} {alpha:>7} {beta_:>7} "
         f"{alpha/(alpha+beta_):>10.4f} "
         f"{'[{:.3f}, {:.3f}]'.format(*stats.beta.ppf([0.025, 0.975], alpha, beta_)):>22}")

   for i, y in enumerate(outcomes, 1):
       alpha += y
       beta_ += (1 - y)
       post = stats.beta(alpha, beta_)
       ci = post.ppf([0.025, 0.975])
       if i <= 5 or i >= 38 or i % 10 == 0:
           print(f"{i:>8} {y:>8} {alpha:>7} {beta_:>7} "
                 f"{post.mean():>10.4f} "
                 f"{'[{:.3f}, {:.3f}]'.format(ci[0], ci[1]):>22}")

   print(f"\nFinal posterior: Beta({alpha}, {beta_})")
   print(f"P(theta > 0.5) = {1 - stats.beta.cdf(0.5, alpha, beta_):.4f}")

**The shrinkage effect.**  The posterior mean is pulled between the prior mean
and the MLE, with the pull depending on sample size.  Let us demonstrate
this directly: as :math:`n` grows, the posterior converges to the MLE.

.. code-block:: python

   # Shrinkage: posterior moves from prior toward MLE as n grows
   import numpy as np
   from scipy import stats

   true_theta = 0.55
   alpha_prior, beta_prior = 12, 8  # prior mean = 0.60

   print(f"{'n':>6} {'MLE':>8} {'Post mean':>10} {'Prior weight':>13}")
   print("-" * 40)
   for n in [5, 10, 20, 40, 100, 500, 2000]:
       s = int(n * true_theta)
       alpha_post = alpha_prior + s
       beta_post = beta_prior + n - s
       post_mean = alpha_post / (alpha_post + beta_post)
       mle = s / n
       # Weight on prior = prior_n / (prior_n + data_n)
       prior_n = alpha_prior + beta_prior
       prior_weight = prior_n / (prior_n + n)
       print(f"{n:>6} {mle:>8.4f} {post_mean:>10.4f} {prior_weight:>12.1%}")

   print("\nAs n grows, prior weight -> 0 and posterior mean -> MLE.")

Derivation: Normal--Normal conjugacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose :math:`x_1, \ldots, x_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)`
with :math:`\sigma^2` known, and we place a :math:`N(\mu_0, \sigma_0^2)` prior
on :math:`\mu`.

The log-likelihood is:

.. math::

   \ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2)
               - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2.

Keeping only terms involving :math:`\mu`:

.. math::

   \ell(\mu) \propto -\frac{n}{2\sigma^2}(\bar{x} - \mu)^2,

where :math:`\bar{x} = n^{-1}\sum x_i`.

The log-prior is:

.. math::

   \log p(\mu) \propto -\frac{1}{2\sigma_0^2}(\mu - \mu_0)^2.

Adding and completing the square in :math:`\mu`:

.. math::

   \log p(\mu \mid \mathbf{x})
   &\propto -\frac{1}{2}\left[\frac{n}{\sigma^2}(\mu - \bar{x})^2
            + \frac{1}{\sigma_0^2}(\mu - \mu_0)^2\right].

Define the posterior precision (reciprocal of variance):

.. math::

   \frac{1}{\sigma_n^2} = \frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}.

Expanding both squared terms, collecting the coefficient of :math:`\mu` and
:math:`\mu^2`, we find the posterior mean:

.. math::

   \mu_n = \sigma_n^2 \left(\frac{n \bar{x}}{\sigma^2}
           + \frac{\mu_0}{\sigma_0^2}\right).

Therefore:

.. math::

   \mu \mid \mathbf{x} \;\sim\; N(\mu_n, \sigma_n^2).

The posterior mean is a precision-weighted average of the data mean
:math:`\bar{x}` and the prior mean :math:`\mu_0`.  As :math:`n \to \infty`
the posterior concentrates around :math:`\bar{x}` --- the prior is overwhelmed
by the data.

.. code-block:: python

   # Normal-Normal conjugate update: variable names match the math exactly
   import numpy as np

   np.random.seed(42)

   # Prior belief: mu ~ N(mu_0, sigma_0^2)
   mu_0 = 5.0
   sigma_0 = 2.0

   # Known data variance, observed data
   sigma = 1.0
   data = np.random.normal(loc=7.0, scale=sigma, size=15)
   n = len(data)
   x_bar = data.mean()

   # Posterior parameters --- these formulas match the math above exactly
   precision_prior = 1 / sigma_0**2          # 1/sigma_0^2
   precision_data  = n / sigma**2            # n/sigma^2
   precision_post  = precision_prior + precision_data  # 1/sigma_n^2

   sigma_n = np.sqrt(1 / precision_post)
   mu_n = (precision_data * x_bar + precision_prior * mu_0) / precision_post

   print(f"Prior:     N(mu_0={mu_0:.2f}, sigma_0={sigma_0:.2f})")
   print(f"Data:      n={n}, x_bar={x_bar:.4f}, sigma={sigma:.2f}")
   print(f"Posterior: N(mu_n={mu_n:.4f}, sigma_n={sigma_n:.4f})")
   print(f"\nPrecision decomposition:")
   print(f"  Prior precision:  1/sigma_0^2 = {precision_prior:.4f}")
   print(f"  Data precision:   n/sigma^2   = {precision_data:.4f}")
   print(f"  Post precision:   sum         = {precision_post:.4f}")
   print(f"  Weight on data:   {precision_data/precision_post:.1%}")
   print(f"  Weight on prior:  {precision_prior/precision_post:.1%}")

Derivation: Gamma--Poisson conjugacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose :math:`x_1, \ldots, x_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)`
and :math:`\lambda \sim \text{Gamma}(\alpha, \beta)` with density

.. math::

   p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)}\,
                \lambda^{\alpha-1} e^{-\beta\lambda}.

The likelihood is:

.. math::

   L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
              = \frac{\lambda^{\sum x_i} e^{-n\lambda}}{\prod x_i!}.

Multiplying prior and likelihood:

.. math::

   p(\lambda \mid \mathbf{x})
   &\propto \lambda^{\sum x_i} e^{-n\lambda} \cdot
            \lambda^{\alpha-1} e^{-\beta\lambda} \\
   &= \lambda^{(\alpha + \sum x_i) - 1}\,
      e^{-(\beta + n)\lambda}.

This is the kernel of a :math:`\text{Gamma}(\alpha + \sum x_i,\; \beta + n)`
distribution.  Each observation adds its count to the shape and increments the
rate by one.


17.4 MAP Estimation
---------------------

**Why MAP?**
You have done the full Bayesian update and have a posterior distribution.  But
your collaborator says: "Just give me a single number."  The *maximum a
posteriori* (MAP) estimate is the mode of the posterior---the single most
probable value:

.. math::

   \hat{\theta}_{\text{MAP}}
   = \arg\max_\theta \; p(\theta \mid \mathbf{x})
   = \arg\max_\theta \; \bigl[\log L(\theta) + \log p(\theta)\bigr].

The MAP criterion is identical to *penalized likelihood* where the penalty is
:math:`-\log p(\theta)`.

.. admonition:: Intuition

   MAP estimation is just MLE with a "nudge" from the prior.  If you have ever
   used ridge or lasso regression, you have already been doing MAP estimation
   without calling it that.

Connection to penalized likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Ridge regression** corresponds to a Normal prior on regression coefficients:
  :math:`p(\beta_j) = N(0, \tau^2)`.  Then :math:`-\log p(\beta_j) \propto
  \beta_j^2 / (2\tau^2)`, giving the :math:`L_2` penalty.
* **Lasso** corresponds to a Laplace (double-exponential) prior:
  :math:`p(\beta_j) \propto \exp(-|\beta_j|/b)`, giving the :math:`L_1`
  penalty.

Let us compute MLE, MAP, and the posterior mean side by side for our
clinical trial, and then see how each behaves as the prior gets stronger.

.. code-block:: python

   # MLE vs MAP vs posterior mean: the three Bayesian point estimates
   import numpy as np
   from scipy import stats

   n, s = 40, 23  # clinical trial data

   print(f"{'Prior':>28} {'MLE':>8} {'MAP':>8} {'Post mean':>10} {'Prior mean':>11}")
   print("-" * 70)

   for label, (a, b) in [("Flat: Beta(1,1)",        (1, 1)),
                          ("Mild: Beta(2,2)",        (2, 2)),
                          ("Pilot: Beta(12,8)",      (12, 8)),
                          ("Strong: Beta(60,40)",    (60, 40))]:
       mle = s / n
       # MAP for Beta-Binomial: (a+s-1) / (a+b+n-2)
       map_est = (a + s - 1) / (a + b + n - 2)
       # Posterior mean: (a+s) / (a+b+n)
       post_mean = (a + s) / (a + b + n)
       prior_mean = a / (a + b)
       print(f"{label:>28} {mle:>8.4f} {map_est:>8.4f} "
             f"{post_mean:>10.4f} {prior_mean:>11.4f}")

   print("\nNote: with a flat prior, MAP = MLE exactly.")

When MAP equals MLE
^^^^^^^^^^^^^^^^^^^^

If the prior is flat (uniform over the parameter space), then
:math:`\log p(\theta) = \text{const}` and the MAP estimate reduces to the MLE.
More generally, as :math:`n \to \infty` the likelihood dominates and
:math:`\hat{\theta}_{\text{MAP}} \to \hat{\theta}_{\text{MLE}}`.

**Bayesian A/B test: answering the question MLE cannot.**

An e-commerce company runs an A/B test.  Version A (control) had 120
conversions in 1000 visits; version B (new design) had 135 in 1000.
The MLE says B is better (13.5% vs 12.0%), but is that just noise?  The
Bayesian approach answers directly: "What is the probability that B beats A?"

.. code-block:: python

   # Bayesian A/B test: P(B > A) via Monte Carlo
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Data
   visitors_A, conversions_A = 1000, 120
   visitors_B, conversions_B = 1000, 135

   # Posteriors with uniform prior Beta(1,1)
   post_A = stats.beta(1 + conversions_A, 1 + visitors_A - conversions_A)
   post_B = stats.beta(1 + conversions_B, 1 + visitors_B - conversions_B)

   # Monte Carlo: draw S samples and compare
   S = 100_000
   samples_A = post_A.rvs(S)
   samples_B = post_B.rvs(S)

   prob_B_wins = np.mean(samples_B > samples_A)
   lift_samples = (samples_B - samples_A) / samples_A
   expected_lift = np.mean(lift_samples)

   print("A/B Test Results:")
   print(f"  MLE for A: {conversions_A/visitors_A:.4f}")
   print(f"  MLE for B: {conversions_B/visitors_B:.4f}")
   print(f"  P(B > A):  {prob_B_wins:.4f}")
   print(f"  Expected lift if B is deployed: {expected_lift:.2%}")
   print(f"  95% CI on lift: [{np.percentile(lift_samples, 2.5):.2%}, "
         f"{np.percentile(lift_samples, 97.5):.2%}]")
   print(f"\n  Interpretation: There is a {prob_B_wins:.0%} probability that B is better.")
   if prob_B_wins > 0.95:
       print("  -> Strong evidence to deploy B.")
   else:
       print("  -> Evidence not conclusive. Consider running the test longer.")


17.5 The Laplace Approximation
---------------------------------

**Why approximate?**
In most problems the posterior cannot be computed in closed form.  The Laplace
approximation replaces the posterior with a Gaussian, using only the MAP
estimate and the curvature of the log-posterior at the MAP.  It is one of the
oldest and most practical tools in the Bayesian toolkit.

**Derivation.**
Let :math:`h(\theta) = \log p(\theta \mid \mathbf{x})`.  Taylor-expand
:math:`h` around the MAP :math:`\hat{\theta}`:

.. math::

   h(\theta)
   \approx h(\hat{\theta})
   + \underbrace{h'(\hat{\theta})}_{=\,0}(\theta - \hat{\theta})
   + \frac{1}{2}\, h''(\hat{\theta})\,(\theta - \hat{\theta})^2.

The first-derivative term vanishes because :math:`\hat{\theta}` is a maximum.
Exponentiating:

.. math::

   p(\theta \mid \mathbf{x})
   \approx p(\hat{\theta} \mid \mathbf{x})\,
   \exp\!\left[\frac{1}{2}\, h''(\hat{\theta})\,
   (\theta - \hat{\theta})^2\right].

This is the kernel of a Gaussian with mean :math:`\hat{\theta}` and variance
:math:`\sigma^2 = -1/h''(\hat{\theta})`.

In the multivariate case with parameter vector
:math:`\boldsymbol{\theta} \in \mathbb{R}^p`:

.. math::

   p(\boldsymbol{\theta} \mid \mathbf{x})
   \approx N\!\left(\hat{\boldsymbol{\theta}},\;
   \left[-\mathbf{H}(\hat{\boldsymbol{\theta}})\right]^{-1}\right),

where :math:`\mathbf{H}` is the Hessian matrix of :math:`h` evaluated at the
MAP.

Let us compute the Laplace approximation for our clinical trial posterior and
overlay it with the exact Beta posterior.

.. code-block:: python

   # Laplace approximation to the clinical trial posterior
   import numpy as np
   from scipy import stats, optimize

   # Data: 23 successes in 40 trials; Prior: Beta(12, 8)
   n, s = 40, 23
   alpha_prior, beta_prior = 12, 8

   # -- Step 1: find the MAP (theta_hat) --
   def neg_log_post(theta):
       """Negative log-posterior (to minimize)."""
       if theta <= 0 or theta >= 1:
           return 1e10
       return -((alpha_prior + s - 1) * np.log(theta)
                + (beta_prior + n - s - 1) * np.log(1 - theta))

   result = optimize.minimize_scalar(neg_log_post, bounds=(0.01, 0.99),
                                     method='bounded')
   theta_hat = result.x

   # -- Step 2: compute h''(theta_hat) via numerical second derivative --
   h = lambda t: -neg_log_post(t)  # log-posterior
   eps = 1e-5
   h_double_prime = (h(theta_hat + eps) - 2*h(theta_hat)
                     + h(theta_hat - eps)) / eps**2

   # -- Step 3: Laplace variance = -1 / h''(theta_hat) --
   laplace_var = -1.0 / h_double_prime
   laplace_std = np.sqrt(laplace_var)

   # -- Compare with exact posterior: Beta(35, 25) --
   exact = stats.beta(alpha_prior + s, beta_prior + n - s)

   print("Laplace approximation vs exact posterior:")
   print(f"  MAP (theta_hat):    {theta_hat:.4f}")
   print(f"  Exact mode:         "
         f"{(alpha_prior+s-1)/(alpha_prior+beta_prior+n-2):.4f}")
   print(f"  Laplace std:        {laplace_std:.4f}")
   print(f"  Exact std:          {exact.std():.4f}")
   print(f"  Laplace 95% CI:     [{theta_hat - 1.96*laplace_std:.4f}, "
         f"{theta_hat + 1.96*laplace_std:.4f}]")
   print(f"  Exact 95% CI:       [{exact.ppf(0.025):.4f}, "
         f"{exact.ppf(0.975):.4f}]")

   # -- Pointwise comparison --
   theta_grid = np.linspace(0.3, 0.85, 7)
   print(f"\n  {'theta':>8} {'Exact':>10} {'Laplace':>10} {'Error':>10}")
   for t in theta_grid:
       exact_val = exact.pdf(t)
       laplace_val = stats.norm.pdf(t, theta_hat, laplace_std)
       print(f"  {t:>8.2f} {exact_val:>10.4f} {laplace_val:>10.4f} "
             f"{abs(exact_val - laplace_val):>10.4f}")

**Connection to Fisher information.**
When the prior is flat, the Hessian of the log-posterior equals the Hessian of
the log-likelihood, which is the *observed* Fisher information matrix
:math:`\mathcal{J}(\hat{\theta})`.  The Laplace approximation then gives

.. math::

   p(\boldsymbol{\theta} \mid \mathbf{x})
   \approx N\!\left(\hat{\boldsymbol{\theta}}_{\text{MLE}},\;
   \mathcal{J}(\hat{\boldsymbol{\theta}})^{-1}\right),

recovering the large-sample Normal approximation from classical theory
(see :ref:`Part II <part2>`).

**Accuracy.**
The Laplace approximation is exact when the posterior is Gaussian (e.g.,
Normal--Normal conjugate model).  It is a good approximation when the posterior
is unimodal and roughly symmetric.  It degrades for skewed, multimodal, or
heavy-tailed posteriors.  Let us see this by comparing it on both a symmetric
and a skewed posterior.

.. code-block:: python

   # When Laplace works well vs when it fails
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   theta = np.linspace(0.001, 0.999, 1000)

   cases = [
       ("n=100, s=50 (symmetric)", 100, 50, 1, 1),
       ("n=10,  s=1  (skewed)",     10,  1, 1, 1),
   ]
   for label, n, s, a, b in cases:
       exact = stats.beta(a + s, b + n - s)
       mode = (a + s - 1) / (a + b + n - 2)
       # Laplace std from Beta distribution second derivative
       h2 = -(a+s-1)/mode**2 - (b+n-s-1)/(1-mode)**2
       lap_std = np.sqrt(-1.0/h2)

       exact_pdf = exact.pdf(theta)
       laplace_pdf = stats.norm.pdf(theta, mode, lap_std)
       # Normalize Laplace to [0,1]
       laplace_pdf /= np.trapz(laplace_pdf, theta)

       max_err = np.max(np.abs(exact_pdf - laplace_pdf))
       print(f"{label}:")
       print(f"  Mode={mode:.3f}, Laplace std={lap_std:.3f}, "
             f"max pointwise error={max_err:.3f}")


17.6 Credible Intervals vs Confidence Intervals
--------------------------------------------------

Both Bayesian credible intervals and frequentist confidence intervals aim to
quantify uncertainty, but they answer different questions.  Understanding the
distinction is crucial for interpreting results correctly.

**Confidence interval (frequentist).**
A :math:`100(1-\alpha)\%` confidence interval is a random interval
:math:`[L(\mathbf{X}), U(\mathbf{X})]` constructed so that, if we repeated the
experiment many times, the interval would contain the true :math:`\theta` in
:math:`100(1-\alpha)\%` of repetitions:

.. math::

   P_\theta\!\bigl(\theta \in [L(\mathbf{X}), U(\mathbf{X})]\bigr)
   = 1 - \alpha \quad \text{for all } \theta.

The interval is random; the parameter is fixed.

**Credible interval (Bayesian).**
A :math:`100(1-\alpha)\%` credible interval is an interval :math:`[a, b]` such
that the posterior probability of :math:`\theta` lying in the interval is
:math:`1-\alpha`:

.. math::

   P(\theta \in [a, b] \mid \mathbf{x}) = 1 - \alpha.

The parameter is random (given the posterior); the interval endpoints are fixed
given the data.

A common choice is the *highest posterior density* (HPD) interval, which is the
shortest interval containing :math:`1-\alpha` posterior probability.

.. code-block:: python

   # Compare credible intervals and confidence intervals in our clinical trial
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   n, s = 40, 23

   # --- Bayesian: 95% credible interval from Beta posterior ---
   alpha_prior, beta_prior = 12, 8
   post = stats.beta(alpha_prior + s, beta_prior + n - s)
   cred_lo, cred_hi = post.ppf(0.025), post.ppf(0.975)

   # --- Frequentist: Wald 95% confidence interval from MLE ---
   p_hat = s / n
   se = np.sqrt(p_hat * (1 - p_hat) / n)
   conf_lo, conf_hi = p_hat - 1.96 * se, p_hat + 1.96 * se

   print("95% intervals for the response rate:")
   print(f"  Bayesian credible:  [{cred_lo:.4f}, {cred_hi:.4f}]  (width={cred_hi-cred_lo:.4f})")
   print(f"  Frequentist Wald:   [{conf_lo:.4f}, {conf_hi:.4f}]  (width={conf_hi-conf_lo:.4f})")
   print(f"\nInterpretation difference:")
   print(f"  Credible:   'P(theta in [{cred_lo:.3f}, {cred_hi:.3f}] | data) = 0.95'")
   print(f"  Confidence: 'If we repeat this trial, 95% of CIs will contain the true theta'")

   # --- Simulation: verify the confidence interval has 95% coverage ---
   true_theta = 0.575
   n_sims = 10_000
   covers = 0
   for _ in range(n_sims):
       s_sim = np.random.binomial(n, true_theta)
       p_sim = s_sim / n
       se_sim = np.sqrt(p_sim * (1 - p_sim) / n)
       if p_sim - 1.96*se_sim <= true_theta <= p_sim + 1.96*se_sim:
           covers += 1
   print(f"\n  Wald CI coverage ({n_sims} simulations): {covers/n_sims:.3f}")

.. admonition:: Common Pitfall

   A 95% confidence interval does *not* mean "there is a 95% probability that
   the parameter is in this interval."  That is the Bayesian credible interval
   interpretation.  Confusing the two is one of the most widespread statistical
   misunderstandings.

**When do they coincide?**
Under a flat prior and with large samples, the Laplace approximation shows that
the posterior is approximately :math:`N(\hat{\theta}_{\text{MLE}},\,
\mathcal{J}^{-1})`, and the Bayesian credible interval numerically coincides
with the Wald confidence interval from classical theory.


17.7 Bayesian Model Comparison
---------------------------------

**Why compare models?**
Given two competing models :math:`M_1` and :math:`M_2` for the same data, we
want to know which is better supported.  The Bayesian framework offers an
elegant solution through the marginal likelihood.

Marginal likelihood
^^^^^^^^^^^^^^^^^^^^

The marginal likelihood (evidence) for model :math:`M_k` is:

.. math::

   p(\mathbf{x} \mid M_k)
   = \int p(\mathbf{x} \mid \theta_k, M_k)\, p(\theta_k \mid M_k)\, d\theta_k.

This integrates the likelihood over the entire prior, automatically penalizing
complex models that spread their prior mass over large parameter spaces (a
built-in Occam's razor).

.. code-block:: python

   # Compute marginal likelihoods for two models analytically
   import numpy as np
   from scipy.special import betaln

   # Data: 23 successes in 40 trials
   n, s = 40, 23

   # Model 1: theta ~ Beta(12, 8) -- informative prior from pilot study
   # Model 2: theta ~ Beta(1, 1)  -- no prior information (uniform)
   # Marginal likelihood for Beta-Binomial:
   #   p(s | n, alpha, beta) = C(n,s) * B(alpha+s, beta+n-s) / B(alpha, beta)
   # where B is the Beta function.  Taking logs:

   from scipy.special import comb
   log_comb = np.log(comb(n, s, exact=True))

   models = [("Informative Beta(12,8)", 12, 8),
             ("Uniform Beta(1,1)",       1, 1)]

   log_marginals = []
   for name, a, b in models:
       log_ml = log_comb + betaln(a + s, b + n - s) - betaln(a, b)
       log_marginals.append(log_ml)
       print(f"  {name}: log p(data|M) = {log_ml:.4f}")

   log_bf = log_marginals[0] - log_marginals[1]
   print(f"\n  Log Bayes factor (M1 vs M2): {log_bf:.4f}")
   print(f"  Bayes factor:                {np.exp(log_bf):.2f}")
   if log_bf > 0:
       print("  -> Data favor the informative prior model.")
   else:
       print("  -> Data favor the uniform prior model.")

Bayes factor
^^^^^^^^^^^^^

The Bayes factor comparing :math:`M_1` to :math:`M_2` is:

.. math::

   \text{BF}_{12}
   = \frac{p(\mathbf{x} \mid M_1)}{p(\mathbf{x} \mid M_2)}.

Using Bayes' theorem on models:

.. math::

   \frac{p(M_1 \mid \mathbf{x})}{p(M_2 \mid \mathbf{x})}
   = \text{BF}_{12} \;\times\;
     \frac{p(M_1)}{p(M_2)}.

The posterior odds equal the Bayes factor times the prior odds.

**Interpretation (Kass and Raftery, 1995).**

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - :math:`\log_{10}(\text{BF}_{12})`
     - Evidence for :math:`M_1`
   * - 0 to 0.5
     - Barely worth mentioning
   * - 0.5 to 1
     - Substantial
   * - 1 to 2
     - Strong
   * - > 2
     - Decisive

**Computing Bayes factors.**
The marginal likelihood is often a high-dimensional integral with no closed
form.  Methods include:

1. **Conjugate models** --- closed-form marginal likelihood.
2. **Laplace approximation** --- using the result from Section 17.5 gives
   :math:`p(\mathbf{x} \mid M) \approx (2\pi)^{p/2}\, |\mathbf{H}|^{-1/2}\,
   L(\hat{\theta})\, p(\hat{\theta})`.
3. **Harmonic mean estimator** --- simple but notoriously unstable.
4. **Thermodynamic integration / path sampling** --- more stable MCMC-based
   methods (see :ref:`Chapter 18 <ch18_computational>`).
5. **Bridge sampling** --- optimal Monte Carlo estimator using samples from
   prior and posterior.

**Sensitivity to priors.**
Unlike posterior inference, which is often robust to the prior for large
samples, the Bayes factor can be very sensitive to the prior, especially for
parameters not well-constrained by the data.  Improper priors generally cannot
be used for Bayes factors because the arbitrary normalizing constants do not
cancel.

.. code-block:: python

   # Sensitivity of Bayes factors to prior choice
   import numpy as np
   from scipy.special import betaln, comb

   n, s = 40, 23
   log_comb_val = np.log(comb(n, s, exact=True))

   # Model 2 is always uniform Beta(1,1)
   log_ml_uniform = log_comb_val + betaln(1+s, 1+n-s) - betaln(1, 1)

   print("Sensitivity of Bayes factor to prior strength:")
   print(f"{'Prior':>20} {'BF vs Uniform':>15} {'Verdict':>15}")
   print("-" * 52)
   for label, a, b in [("Beta(2,2)",   2,  2),
                        ("Beta(6,4)",   6,  4),
                        ("Beta(12,8)", 12,  8),
                        ("Beta(30,20)",30, 20),
                        ("Beta(60,40)",60, 40)]:
       log_ml = log_comb_val + betaln(a+s, b+n-s) - betaln(a, b)
       bf = np.exp(log_ml - log_ml_uniform)
       print(f"{label:>20} {bf:>15.2f} "
             f"{'favors prior' if bf > 1 else 'favors uniform':>15}")


17.8 Summary
--------------

* The posterior is proportional to the likelihood times the prior.
* Jeffreys' prior is derived from the Fisher information and is invariant
  to reparameterization.
* Conjugate priors yield closed-form posteriors; the Beta--Binomial,
  Normal--Normal, and Gamma--Poisson families are the workhorses.
* MAP estimation is penalized likelihood; it reduces to MLE with a flat prior.
* The Laplace approximation replaces the posterior by a Gaussian at the MAP,
  with covariance given by the inverse Hessian.
* Credible intervals make probability statements about parameters; confidence
  intervals make probability statements about the procedure.
* Bayes factors compare models via the ratio of marginal likelihoods, with a
  built-in penalty for complexity.

.. code-block:: python

   # Chapter 17 summary: all three point estimates for the clinical trial
   import numpy as np
   from scipy import stats

   n, s = 40, 23
   alpha_prior, beta_prior = 12, 8

   mle = s / n
   map_est = (alpha_prior + s - 1) / (alpha_prior + beta_prior + n - 2)
   post = stats.beta(alpha_prior + s, beta_prior + n - s)

   print("Clinical trial summary (n=40, s=23, prior=Beta(12,8)):")
   print(f"  MLE:             {mle:.4f}")
   print(f"  MAP:             {map_est:.4f}")
   print(f"  Posterior mean:  {post.mean():.4f}")
   print(f"  Posterior std:   {post.std():.4f}")
   print(f"  95% credible:    [{post.ppf(0.025):.4f}, {post.ppf(0.975):.4f}]")
   print(f"  P(theta > 0.5): {1 - post.cdf(0.5):.4f}")
