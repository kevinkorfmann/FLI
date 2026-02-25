.. _ch18_computational:

==========================================
Chapter 18 -- Computational Methods
==========================================

**The motivating problem.**
In Chapter 17 we enjoyed the luxury of conjugate priors: the posterior had a
closed form.  But real-world models are rarely so cooperative.  Consider a
hierarchical model for clinical trials across multiple hospitals.  Hospital
:math:`j` has response rate :math:`\theta_j`, and we believe these rates come
from a common population: :math:`\theta_j \sim \text{Beta}(\mu\kappa,\,
(1-\mu)\kappa)`, where :math:`\mu` and :math:`\kappa` are unknown
hyperparameters.

The posterior is:

.. math::

   p(\mu, \kappa, \boldsymbol{\theta} \mid \mathbf{y})
   \;\propto\;
   \prod_{j=1}^J \binom{n_j}{y_j} \theta_j^{y_j}(1-\theta_j)^{n_j - y_j}
   \;\cdot\;
   \prod_{j=1}^J \text{Beta}(\theta_j \mid \mu\kappa, (1-\mu)\kappa)
   \;\cdot\; p(\mu)\, p(\kappa).

We can evaluate this pointwise, but we cannot integrate it analytically.  What
now?  This chapter builds the computational highway system that gets you to
*any* posterior, no matter how complicated.

.. code-block:: python

   # The hierarchical model we will tackle throughout this chapter
   import numpy as np
   from scipy.special import gammaln

   np.random.seed(42)

   # Data: 8 hospitals, each with n_j patients and y_j responders
   n_hospitals = 8
   n_j = np.array([20, 30, 25, 40, 15, 35, 28, 22])
   y_j = np.array([11, 16, 10, 25,  7, 19, 14,  9])

   print("Hospital data:")
   print(f"  {'Hospital':>10} {'n_j':>5} {'y_j':>5} {'MLE':>8}")
   print("  " + "-" * 30)
   for j in range(n_hospitals):
       print(f"  {j+1:>10} {n_j[j]:>5} {y_j[j]:>5} {y_j[j]/n_j[j]:>8.3f}")
   print(f"\n  Hospital MLEs range from {min(y_j/n_j):.3f} to {max(y_j/n_j):.3f}")
   print("  Can we borrow strength across hospitals?  -> Chapter 18 shows how.")

   def log_posterior(theta_vec, mu, kappa):
       """Unnormalized log-posterior for the hierarchical model."""
       a = mu * kappa
       b = (1 - mu) * kappa
       # Likelihood
       ll = np.sum(y_j * np.log(theta_vec) + (n_j - y_j) * np.log(1 - theta_vec))
       # Prior on theta_j: Beta(a, b)
       lp_theta = np.sum((a - 1)*np.log(theta_vec) + (b - 1)*np.log(1 - theta_vec)
                         - gammaln(a) - gammaln(b) + gammaln(a + b))
       # Hyperpriors: mu ~ Beta(2,2), kappa ~ Gamma(2, 0.1)
       lp_mu = (2-1)*np.log(mu) + (2-1)*np.log(1-mu) if 0 < mu < 1 else -1e10
       lp_kappa = (2-1)*np.log(kappa) - 0.1*kappa if kappa > 0 else -1e10
       return ll + lp_theta + lp_mu + lp_kappa


18.1 Monte Carlo Integration
-------------------------------

**Why we need this.**
Many quantities of interest --- posterior means, marginal likelihoods, predictive
distributions --- are integrals of the form

.. math::

   I = \int g(\theta)\, p(\theta \mid \mathbf{x})\, d\theta.

When the integral has no closed form we resort to *Monte Carlo* methods:
generate samples :math:`\theta^{(1)}, \ldots, \theta^{(S)}` from
:math:`p(\theta \mid \mathbf{x})` and approximate

.. math::

   \hat{I} = \frac{1}{S}\sum_{s=1}^{S} g(\theta^{(s)}).

By the law of large numbers :math:`\hat{I} \to I` as :math:`S \to \infty`, and
the central limit theorem gives the Monte Carlo standard error:

.. math::

   \text{SE}(\hat{I}) = \frac{\text{sd}(g(\theta^{(s)}))}{\sqrt{S}}.

The error decreases as :math:`1/\sqrt{S}` regardless of the dimension of
:math:`\theta` --- a crucial advantage over deterministic quadrature in high
dimensions.

.. code-block:: python

   # Monte Carlo integration: watch the estimate converge
   import numpy as np

   np.random.seed(42)

   # Goal: estimate E[theta] where theta ~ Beta(35, 25)
   # (our clinical trial posterior from Chapter 17)
   from scipy import stats
   true_mean = 35 / (35 + 25)

   samples = stats.beta.rvs(35, 25, size=10_000)

   print(f"{'S':>7} {'MC estimate':>12} {'SE':>10} {'|Error|':>10}")
   print("-" * 42)
   for S in [10, 50, 100, 500, 1000, 5000, 10000]:
       mc_est = samples[:S].mean()
       mc_se = samples[:S].std() / np.sqrt(S)
       print(f"{S:>7} {mc_est:>12.6f} {mc_se:>10.6f} "
             f"{abs(mc_est - true_mean):>10.6f}")
   print(f"{'True':>7} {true_mean:>12.6f}")
   print("\nSE decreases as 1/sqrt(S) regardless of dimension.")

.. admonition:: Intuition

   Monte Carlo integration is essentially "estimation by random polling."  Just
   as a political poll of 1000 random voters can approximate the opinion of
   millions, 10,000 random draws from a posterior can approximate intractable
   integrals to several decimal places.

Importance sampling
^^^^^^^^^^^^^^^^^^^^

Often we cannot sample directly from the target :math:`p(\theta \mid
\mathbf{x})` but we can sample from a *proposal* distribution :math:`q(\theta)`.
Write

.. math::

   I = \int g(\theta)\, \frac{p(\theta \mid \mathbf{x})}{q(\theta)}\,
       q(\theta)\, d\theta
     = E_q\!\left[g(\theta)\, w(\theta)\right],

where the *importance weight* is

.. math::

   w(\theta) = \frac{p(\theta \mid \mathbf{x})}{q(\theta)}.

In plain English: the weight measures how much more (or less) probable
:math:`\theta` is under the target distribution compared to the proposal.
Samples from regions where the proposal underrepresents the target get
upweighted; samples from overrepresented regions get downweighted.

Drawing :math:`\theta^{(s)} \sim q` and computing

.. math::

   \hat{I} = \frac{\sum_{s=1}^S w(\theta^{(s)})\, g(\theta^{(s)})}
                  {\sum_{s=1}^S w(\theta^{(s)})}

gives a consistent (self-normalized) importance sampling estimator.  The
denominator normalizes the weights so they sum to one, which is why this is
called "self-normalized" --- we do not need to know the normalizing constants
of :math:`p` or :math:`q`.  The
quality depends on how well :math:`q` covers the tails of :math:`p`; if :math:`q`
has lighter tails the variance can be infinite.

.. code-block:: python

   # Importance sampling: good vs bad proposal
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Target: Beta(35, 25), want E[theta]
   true_mean = 35 / 60
   S = 10_000

   # Good proposal: N(0.58, 0.07) -- covers the target well
   # Bad proposal:  N(0.3, 0.05) -- centered far from the target
   for label, mu_q, sigma_q in [("Good proposal N(0.58, 0.07)", 0.58, 0.07),
                                  ("Bad proposal  N(0.30, 0.05)", 0.30, 0.05)]:
       z = np.random.normal(mu_q, sigma_q, S)
       z = z[(z > 0.01) & (z < 0.99)]  # keep in valid range

       log_w = (stats.beta.logpdf(z, 35, 25)
                - stats.norm.logpdf(z, mu_q, sigma_q))
       log_w -= log_w.max()  # for numerical stability
       w = np.exp(log_w)

       estimate = np.sum(w * z) / np.sum(w)
       ess = np.sum(w)**2 / np.sum(w**2)

       print(f"{label}:")
       print(f"  Estimate: {estimate:.6f} (true: {true_mean:.6f})")
       print(f"  ESS:      {ess:.0f} / {len(z)} samples")
       print(f"  ESS ratio: {ess/len(z):.1%}\n")

.. admonition:: Common Pitfall

   If your proposal distribution :math:`q` has thinner tails than the target
   :math:`p`, a few extreme importance weights will dominate the estimate,
   making it highly unstable.  Always check the effective sample size of your
   importance weights: :math:`\text{ESS} = (\sum w_s)^2 / \sum w_s^2`.


18.2 The Metropolis--Hastings Algorithm
-----------------------------------------

**Why Markov chains?**
Importance sampling works well in low dimensions, but in high dimensions it is
very hard to find a good proposal :math:`q`.  Instead, we construct a *Markov
chain* whose stationary distribution is the target posterior.  After a burn-in
period the chain produces (correlated) samples from the posterior.

We will build a Metropolis--Hastings sampler for our hierarchical model.  But
first, let us derive the algorithm from first principles.

Detailed balance
^^^^^^^^^^^^^^^^^^

A Markov chain with transition kernel :math:`T(\theta' \mid \theta)` has
stationary distribution :math:`\pi(\theta) = p(\theta \mid \mathbf{x})` if
*detailed balance* holds:

.. math::

   \pi(\theta)\, T(\theta' \mid \theta)
   = \pi(\theta')\, T(\theta \mid \theta').

This says the probability of being at :math:`\theta` and moving to
:math:`\theta'` equals the probability of the reverse.

Derivation of the acceptance probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We choose a proposal distribution :math:`q(\theta' \mid \theta)` and set the
transition as: propose :math:`\theta'` from :math:`q`, then accept with
probability :math:`\alpha(\theta, \theta')`.  Thus

.. math::

   T(\theta' \mid \theta) = q(\theta' \mid \theta)\, \alpha(\theta, \theta').

Substituting into the detailed balance condition:

.. math::

   \pi(\theta)\, q(\theta' \mid \theta)\, \alpha(\theta, \theta')
   = \pi(\theta')\, q(\theta \mid \theta')\, \alpha(\theta', \theta).

Rearranging:

.. math::

   \frac{\alpha(\theta, \theta')}{\alpha(\theta', \theta)}
   = \frac{\pi(\theta')\, q(\theta \mid \theta')}
          {\pi(\theta)\, q(\theta' \mid \theta)}.

We need acceptance probabilities :math:`\alpha` that satisfy this ratio and lie
in :math:`[0, 1]`.  The Metropolis--Hastings choice picks the largest such
probabilities:

.. math::

   \alpha(\theta, \theta')
   = \min\!\left(1,\;
     \frac{\pi(\theta')\, q(\theta \mid \theta')}
          {\pi(\theta)\, q(\theta' \mid \theta)}\right).

Reading this formula: the numerator is the posterior density at the proposed
point times the probability of proposing a move *back* to where we are; the
denominator is the same thing in the opposite direction.  If the proposed
point is "better" (higher posterior density, easy to get back from), the ratio
exceeds 1 and we always accept.  If it is "worse," we accept with a
probability equal to the ratio, which gives worse points a fighting chance
--- this is essential for exploring the full posterior.

One can verify this satisfies the ratio above.  Because :math:`\pi` is the
posterior, only the ratio :math:`\pi(\theta')/\pi(\theta)` is needed ---
the normalizing constant cancels.

.. admonition:: Why does this matter?

   The cancellation of the normalizing constant is what makes MCMC so powerful.
   You never need to compute the marginal likelihood :math:`p(\mathbf{x})`.
   You only need to evaluate the *unnormalized* posterior --- which is just the
   likelihood times the prior.  This is the engine that makes Bayesian
   inference practical for complex models.

Algorithm
^^^^^^^^^^

1. Initialize :math:`\theta^{(0)}`.
2. For :math:`t = 1, 2, \ldots`:

   a. Draw :math:`\theta^* \sim q(\cdot \mid \theta^{(t-1)})`.
   b. Compute

      .. math::

         \alpha = \min\!\left(1,\;
           \frac{p(\theta^* \mid \mathbf{x})\, q(\theta^{(t-1)} \mid \theta^*)}
                {p(\theta^{(t-1)} \mid \mathbf{x})\, q(\theta^* \mid \theta^{(t-1)})}\right).

   c. With probability :math:`\alpha` set :math:`\theta^{(t)} = \theta^*`;
      otherwise set :math:`\theta^{(t)} = \theta^{(t-1)}`.

**Special case: random-walk Metropolis.**
If :math:`q(\theta' \mid \theta) = q(\theta \mid \theta')` (symmetric
proposal, e.g., :math:`\theta' = \theta + \epsilon` with
:math:`\epsilon \sim N(0, \sigma^2)`) the proposal ratio cancels and

.. math::

   \alpha = \min\!\left(1,\;
     \frac{p(\theta^* \mid \mathbf{x})}
          {p(\theta^{(t-1)} \mid \mathbf{x})}\right).

Now let us apply this to Bayesian logistic regression, printing the first 20
iterations so you can see the sampler in action.

.. code-block:: python

   # Metropolis-Hastings for logistic regression: iteration by iteration
   import numpy as np
   from scipy.special import expit

   np.random.seed(42)

   # Simulate: 50 patients, dosage x predicts recovery y
   n = 50
   x = np.random.normal(0, 1, n)
   true_beta = np.array([0.5, 1.5])
   prob = expit(true_beta[0] + true_beta[1] * x)
   y = np.random.binomial(1, prob)

   def log_posterior(beta):
       eta = beta[0] + beta[1] * x
       log_lik = np.sum(y * eta - np.log(1 + np.exp(eta)))
       log_prior = -0.5 * np.sum(beta**2) / 100  # N(0, 10^2)
       return log_lik + log_prior

   # MH sampler
   T = 20_000
   samples = np.zeros((T, 2))
   samples[0] = [0.0, 0.0]
   sigma_prop = 0.3
   accepted = 0

   # Print header for first 20 iterations
   print(f"{'t':>4} {'beta0*':>8} {'beta1*':>8} {'log_alpha':>10} "
         f"{'Accept?':>8} {'beta0':>8} {'beta1':>8}")
   print("-" * 60)

   for t in range(1, T):
       proposal = samples[t-1] + np.random.normal(0, sigma_prop, 2)
       log_alpha = log_posterior(proposal) - log_posterior(samples[t-1])
       u = np.random.uniform()
       accept = np.log(u) < log_alpha

       if accept:
           samples[t] = proposal
           accepted += 1
       else:
           samples[t] = samples[t-1]

       if t <= 20:
           print(f"{t:>4} {proposal[0]:>8.3f} {proposal[1]:>8.3f} "
                 f"{log_alpha:>10.3f} {'YES' if accept else 'no':>8} "
                 f"{samples[t, 0]:>8.3f} {samples[t, 1]:>8.3f}")

   burn_in = 5000
   post_samples = samples[burn_in:]
   print(f"\n--- After {T} iterations (burn-in={burn_in}) ---")
   print(f"Acceptance rate:     {accepted / T:.3f}")
   print(f"Posterior mean beta: [{post_samples[:,0].mean():.3f}, "
         f"{post_samples[:,1].mean():.3f}]")
   print(f"Posterior std beta:  [{post_samples[:,0].std():.3f}, "
         f"{post_samples[:,1].std():.3f}]")
   print(f"True beta:           [{true_beta[0]:.3f}, {true_beta[1]:.3f}]")

Proposal tuning
^^^^^^^^^^^^^^^^

The step size :math:`\sigma` controls the trade-off between exploring
widely (large steps, many rejections) and staying local (small steps,
high acceptance but slow exploration).  The optimal acceptance rate for a
:math:`d`-dimensional Gaussian target is approximately 23.4\% (Roberts,
Gelman, and Gilks, 1997).

.. code-block:: python

   # Effect of proposal step size on acceptance rate and mixing
   import numpy as np

   np.random.seed(42)

   # Target: standard bivariate normal
   def log_target(theta):
       return -0.5 * np.sum(theta**2)

   def run_mh(sigma_prop, T=10_000, start=np.array([5.0, 5.0])):
       samples = np.zeros((T, 2))
       samples[0] = start
       acc = 0
       for t in range(1, T):
           proposal = samples[t-1] + np.random.normal(0, sigma_prop, 2)
           log_alpha = log_target(proposal) - log_target(samples[t-1])
           if np.log(np.random.uniform()) < log_alpha:
               samples[t] = proposal
               acc += 1
           else:
               samples[t] = samples[t-1]
       return samples, acc / T

   print(f"{'sigma_prop':>12} {'Accept rate':>13} {'Post mean':>16} "
         f"{'Post std':>12}")
   print("-" * 56)
   for sigma in [0.05, 0.3, 1.0, 2.4, 5.0, 20.0]:
       samp, rate = run_mh(sigma)
       post = samp[2000:]
       print(f"{sigma:>12.2f} {rate:>13.3f} "
             f"[{post[:,0].mean():>5.2f}, {post[:,1].mean():>5.2f}] "
             f"[{post[:,0].std():>4.2f}, {post[:,1].std():>4.2f}]")

   print("\nOptimal acceptance rate for d=2: ~23.4%")
   print("sigma_prop ~ 2.4 hits the sweet spot.")


18.3 Gibbs Sampling
---------------------

**Why Gibbs?**
In many models the *full conditional* distribution of each parameter block
--- the distribution of one parameter given all others and the data --- has
a known form (especially with conjugate priors).  Instead of proposing moves
in all dimensions at once, we cycle through one dimension at a time, always
drawing from the exact conditional distribution.  Every draw is accepted.

Full conditional distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a parameter vector :math:`\boldsymbol{\theta} = (\theta_1, \ldots,
\theta_p)`, the full conditional of :math:`\theta_j` is

.. math::

   p(\theta_j \mid \theta_{-j}, \mathbf{x})
   \;\propto\; p(\mathbf{x} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta}),

treating all :math:`\theta_k` for :math:`k \neq j` as fixed.  In conjugate
models these are standard distributions.

Algorithm
^^^^^^^^^^

1. Initialize :math:`\boldsymbol{\theta}^{(0)}`.
2. For :math:`t = 1, 2, \ldots`:

   a. Draw :math:`\theta_1^{(t)} \sim p(\theta_1 \mid \theta_2^{(t-1)},
      \ldots, \theta_p^{(t-1)}, \mathbf{x})`.
   b. Draw :math:`\theta_2^{(t)} \sim p(\theta_2 \mid \theta_1^{(t)},
      \theta_3^{(t-1)}, \ldots, \theta_p^{(t-1)}, \mathbf{x})`.
   c. Continue cycling through all components.

Gibbs as a special case of Metropolis--Hastings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Derivation.**
In MH, set the proposal for updating :math:`\theta_j` to be the full
conditional itself: :math:`q(\theta_j' \mid \theta_j, \theta_{-j}) =
p(\theta_j' \mid \theta_{-j}, \mathbf{x})`.  Then the acceptance ratio is:

.. math::

   \alpha
   &= \frac{p(\theta_j', \theta_{-j} \mid \mathbf{x})\;
            p(\theta_j \mid \theta_{-j}, \mathbf{x})}
           {p(\theta_j, \theta_{-j} \mid \mathbf{x})\;
            p(\theta_j' \mid \theta_{-j}, \mathbf{x})} \\
   &= \frac{p(\theta_j' \mid \theta_{-j}, \mathbf{x})\,
            p(\theta_{-j} \mid \mathbf{x})\,
            p(\theta_j \mid \theta_{-j}, \mathbf{x})}
           {p(\theta_j \mid \theta_{-j}, \mathbf{x})\,
            p(\theta_{-j} \mid \mathbf{x})\,
            p(\theta_j' \mid \theta_{-j}, \mathbf{x})} \\
   &= 1.

Every proposal is accepted!  Gibbs sampling is MH with an optimal
component-wise proposal.

**Example: Normal model with unknown mean and variance.**
With :math:`x_i \sim N(\mu, \sigma^2)`,
:math:`\mu \sim N(\mu_0, \sigma_0^2)`, and
:math:`\sigma^2 \sim \text{Inv-Gamma}(a, b)`, the full conditionals are:

* :math:`\mu \mid \sigma^2, \mathbf{x} \sim N(\mu_n, \sigma_n^2)` (Normal--Normal
  conjugacy from :ref:`Chapter 17 <ch17_bayesian>`).
* :math:`\sigma^2 \mid \mu, \mathbf{x} \sim \text{Inv-Gamma}(a + n/2,\;
  b + \frac{1}{2}\sum(x_i - \mu)^2)`.

Let us implement this, printing every iteration for the first 15 to see the
alternating updates converge.

.. code-block:: python

   # Gibbs sampler for Normal model: mu and sigma^2 with alternating updates
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Generate data from N(mu=5, sigma^2=4)
   true_mu, true_sigma2 = 5.0, 4.0
   data = np.random.normal(true_mu, np.sqrt(true_sigma2), size=30)
   n = len(data)
   x_bar = data.mean()
   S_data = np.sum((data - x_bar)**2)

   # Hyperpriors
   mu_0, sigma_0_sq = 0.0, 100.0     # mu ~ N(0, 100)
   a_prior, b_prior = 0.01, 0.01     # sigma^2 ~ InvGamma(0.01, 0.01)

   # Gibbs sampler
   T = 5_000
   mu_samples = np.zeros(T)
   sigma2_samples = np.zeros(T)
   mu_samples[0] = 0.0       # start far from truth
   sigma2_samples[0] = 20.0  # start far from truth

   print(f"{'t':>4} {'mu':>10} {'sigma2':>10}   Full conditional used")
   print("-" * 55)
   print(f"{'0':>4} {mu_samples[0]:>10.4f} {sigma2_samples[0]:>10.4f}   (initial)")

   for t in range(1, T):
       # -- Update mu | sigma^2, data --
       prec_prior = 1.0 / sigma_0_sq
       prec_data = n / sigma2_samples[t-1]
       prec_post = prec_prior + prec_data
       mu_post = (prec_prior * mu_0 + prec_data * x_bar) / prec_post
       sigma_post = np.sqrt(1.0 / prec_post)
       mu_samples[t] = np.random.normal(mu_post, sigma_post)

       # -- Update sigma^2 | mu, data --
       a_post = a_prior + n / 2.0
       b_post = b_prior + 0.5 * np.sum((data - mu_samples[t])**2)
       sigma2_samples[t] = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

       if t <= 15 or t == T-1:
           label = f"  N({mu_post:.2f}, {sigma_post:.2f}^2) -> " \
                   f"IG({a_post:.1f}, {b_post:.1f})"
           print(f"{t:>4} {mu_samples[t]:>10.4f} {sigma2_samples[t]:>10.4f} {label}")

   burn_in = 500
   print(f"\n--- After {T} iterations (burn-in={burn_in}) ---")
   print(f"Posterior E[mu]:      {mu_samples[burn_in:].mean():.4f}  (true={true_mu})")
   print(f"Posterior E[sigma^2]: {sigma2_samples[burn_in:].mean():.4f}  (true={true_sigma2})")
   print(f"Posterior sd(mu):     {mu_samples[burn_in:].std():.4f}")
   print(f"Data mean:            {x_bar:.4f}")

Now let us tackle the full hierarchical model from the chapter opening using
a Gibbs sampler.  Each :math:`\theta_j` has a conjugate full conditional
(Beta), while :math:`\mu` and :math:`\kappa` require Metropolis steps within
the Gibbs.

.. code-block:: python

   # Gibbs sampler for the hierarchical Beta-Binomial model
   import numpy as np
   from scipy.special import gammaln

   np.random.seed(42)

   # Data from chapter opening
   n_hospitals = 8
   n_j = np.array([20, 30, 25, 40, 15, 35, 28, 22])
   y_j = np.array([11, 16, 10, 25,  7, 19, 14,  9])

   T = 10_000
   theta_samples = np.zeros((T, n_hospitals))
   mu_samples = np.zeros(T)
   kappa_samples = np.zeros(T)

   # Initialize
   theta_samples[0] = y_j / n_j
   mu_samples[0] = 0.5
   kappa_samples[0] = 10.0

   def log_beta_binomial_hyperprior(mu, kappa, thetas):
       """Log-density of the hyperprior part: Beta priors on theta_j + hyperpriors."""
       if not (0.01 < mu < 0.99 and kappa > 0.5):
           return -1e10
       a = mu * kappa
       b = (1 - mu) * kappa
       lp = np.sum((a-1)*np.log(thetas) + (b-1)*np.log(1-thetas)
                    - gammaln(a) - gammaln(b) + gammaln(a+b))
       lp += (2-1)*np.log(mu) + (2-1)*np.log(1-mu)    # mu ~ Beta(2,2)
       lp += (2-1)*np.log(kappa) - 0.1*kappa           # kappa ~ Gamma(2,0.1)
       return lp

   accepted_mu, accepted_kappa = 0, 0

   for t in range(1, T):
       mu_cur = mu_samples[t-1]
       kappa_cur = kappa_samples[t-1]

       # -- Gibbs step: update each theta_j from its Beta full conditional --
       a = mu_cur * kappa_cur + y_j
       b = (1 - mu_cur) * kappa_cur + n_j - y_j
       theta_samples[t] = np.random.beta(a, b)

       # -- MH step: update mu --
       mu_prop = mu_cur + np.random.normal(0, 0.05)
       if 0.01 < mu_prop < 0.99:
           log_r = (log_beta_binomial_hyperprior(mu_prop, kappa_cur, theta_samples[t])
                    - log_beta_binomial_hyperprior(mu_cur, kappa_cur, theta_samples[t]))
           if np.log(np.random.uniform()) < log_r:
               mu_cur = mu_prop
               accepted_mu += 1
       mu_samples[t] = mu_cur

       # -- MH step: update kappa --
       kappa_prop = kappa_cur * np.exp(np.random.normal(0, 0.2))  # log-scale proposal
       log_r = (log_beta_binomial_hyperprior(mu_cur, kappa_prop, theta_samples[t])
                - log_beta_binomial_hyperprior(mu_cur, kappa_cur, theta_samples[t])
                + np.log(kappa_prop) - np.log(kappa_cur))  # Jacobian
       if np.log(np.random.uniform()) < log_r:
           kappa_cur = kappa_prop
           accepted_kappa += 1
       kappa_samples[t] = kappa_cur

   burn_in = 3000
   print("Hierarchical Beta-Binomial: Gibbs + MH results")
   print(f"  mu acceptance rate:    {accepted_mu/T:.3f}")
   print(f"  kappa acceptance rate: {accepted_kappa/T:.3f}")
   print(f"\n  E[mu]:     {mu_samples[burn_in:].mean():.4f}")
   print(f"  E[kappa]:  {kappa_samples[burn_in:].mean():.2f}")
   print(f"\n  {'Hospital':>10} {'MLE':>8} {'Post mean':>10} {'Shrinkage':>10}")
   print("  " + "-" * 40)
   for j in range(n_hospitals):
       mle = y_j[j] / n_j[j]
       pm = theta_samples[burn_in:, j].mean()
       shrink = (mle - pm) / (mle - mu_samples[burn_in:].mean())
       print(f"  {j+1:>10} {mle:>8.3f} {pm:>10.3f} {shrink:>10.1%}")
   print("\n  Note: smaller hospitals are shrunk more toward the overall mean.")


18.4 Hamiltonian Monte Carlo
-------------------------------

**Why HMC?**
Random-walk Metropolis explores the parameter space by diffusion, which
becomes extremely slow in high dimensions.  HMC uses the *gradient* of the
log-posterior to make large, directed moves while maintaining a high acceptance
rate.  Think of it as giving the sampler a "GPS" rather than letting it wander
blindly.

Hamiltonian mechanics analogy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imagine a frictionless puck sliding on a surface whose height is
:math:`U(\theta) = -\log p(\theta \mid \mathbf{x})`.  The puck has position
:math:`\theta` and momentum :math:`\mathbf{r}`.  The total energy
(Hamiltonian) is:

.. math::

   H(\theta, \mathbf{r}) = U(\theta) + K(\mathbf{r}),

where :math:`K(\mathbf{r}) = \frac{1}{2}\mathbf{r}^T M^{-1} \mathbf{r}` is
the kinetic energy and :math:`M` is a "mass matrix" (usually set to the
identity or an estimate of the posterior covariance).

Hamilton's equations give the dynamics:

.. math::

   \frac{d\theta}{dt} &= \frac{\partial H}{\partial \mathbf{r}}
                       = M^{-1}\mathbf{r}, \\
   \frac{d\mathbf{r}}{dt} &= -\frac{\partial H}{\partial \theta}
                            = -\nabla U(\theta)
                            = \nabla \log p(\theta \mid \mathbf{x}).

In plain English: the first equation says the parameter moves in the direction
of the momentum (the puck slides).  The second equation says the momentum
changes according to the gradient of the log-posterior --- the puck accelerates
downhill toward regions of high posterior density.  Together, these create
smooth, curving trajectories that efficiently explore the posterior landscape.

These equations are *energy-preserving* and *volume-preserving* (symplectic),
which is exactly what we need for a valid MCMC proposal.

The leapfrog integrator
^^^^^^^^^^^^^^^^^^^^^^^^^

We cannot solve Hamilton's equations analytically, so we use the *leapfrog*
integrator with step size :math:`\varepsilon` and :math:`L` steps:

1. :math:`\mathbf{r} \leftarrow \mathbf{r} + \frac{\varepsilon}{2}\,
   \nabla \log p(\theta \mid \mathbf{x})`  (half-step for momentum).
2. For :math:`l = 1, \ldots, L`:

   a. :math:`\theta \leftarrow \theta + \varepsilon\, M^{-1}\mathbf{r}`
      (full step for position).
   b. If :math:`l < L`:
      :math:`\mathbf{r} \leftarrow \mathbf{r} + \varepsilon\,
      \nabla \log p(\theta \mid \mathbf{x})`  (full step for momentum).

3. :math:`\mathbf{r} \leftarrow \mathbf{r} + \frac{\varepsilon}{2}\,
   \nabla \log p(\theta \mid \mathbf{x})`  (final half-step for momentum).

The leapfrog integrator is symplectic (volume-preserving) and time-reversible,
so the MH acceptance probability is:

.. math::

   \alpha = \min\!\bigl(1,\; \exp\!\bigl[-H(\theta^*, \mathbf{r}^*)
            + H(\theta, \mathbf{r})\bigr]\bigr),

where :math:`(\theta^*, \mathbf{r}^*)` is the endpoint.  If the integrator
were exact, :math:`H` would be conserved and :math:`\alpha = 1`.  In practice,
the discretization error is small and acceptance rates are typically above 80\%.

.. code-block:: python

   # HMC sampler: sampling a 2D correlated Gaussian
   import numpy as np

   np.random.seed(42)

   # Target: bivariate normal with correlation 0.95
   rho = 0.95
   Sigma = np.array([[1, rho], [rho, 1]])
   Sigma_inv = np.linalg.inv(Sigma)

   def log_prob(q):
       return -0.5 * q @ Sigma_inv @ q

   def grad_log_prob(q):
       return -Sigma_inv @ q

   def hmc_step(q, epsilon, L):
       """One HMC step with leapfrog integration."""
       p = np.random.normal(size=2)  # sample momentum
       current_H = -log_prob(q) + 0.5 * p @ p

       q_new = q.copy()
       p_new = p.copy()

       # Leapfrog
       p_new += 0.5 * epsilon * grad_log_prob(q_new)
       for l in range(L):
           q_new += epsilon * p_new
           if l < L - 1:
               p_new += epsilon * grad_log_prob(q_new)
       p_new += 0.5 * epsilon * grad_log_prob(q_new)

       proposed_H = -log_prob(q_new) + 0.5 * p_new @ p_new
       if np.log(np.random.uniform()) < current_H - proposed_H:
           return q_new, True
       return q, False

   # Run HMC and compare with random-walk MH
   T = 5_000
   hmc_samples = np.zeros((T, 2))
   hmc_samples[0] = [0.0, 0.0]
   hmc_acc = 0

   for t in range(1, T):
       hmc_samples[t], accepted = hmc_step(hmc_samples[t-1], epsilon=0.15, L=20)
       hmc_acc += accepted

   # Random-walk MH for comparison
   mh_samples = np.zeros((T, 2))
   mh_samples[0] = [0.0, 0.0]
   mh_acc = 0
   for t in range(1, T):
       prop = mh_samples[t-1] + np.random.normal(0, 0.3, 2)
       if np.log(np.random.uniform()) < log_prob(prop) - log_prob(mh_samples[t-1]):
           mh_samples[t] = prop
           mh_acc += 1
       else:
           mh_samples[t] = mh_samples[t-1]

   def ess_1d(chain):
       n = len(chain)
       rho1 = np.corrcoef(chain[:-1], chain[1:])[0, 1]
       return max(1, n * (1 - rho1) / (1 + rho1))

   burn = 500
   print(f"{'':>14} {'HMC':>12} {'Random-walk MH':>16}")
   print("-" * 44)
   print(f"{'Accept rate':>14} {hmc_acc/T:>12.3f} {mh_acc/T:>16.3f}")
   print(f"{'Mean[0]':>14} {hmc_samples[burn:,0].mean():>12.3f} "
         f"{mh_samples[burn:,0].mean():>16.3f}")
   print(f"{'Std[0]':>14} {hmc_samples[burn:,0].std():>12.3f} "
         f"{mh_samples[burn:,0].std():>16.3f}")
   print(f"{'Corr':>14} "
         f"{np.corrcoef(hmc_samples[burn:].T)[0,1]:>12.3f} "
         f"{np.corrcoef(mh_samples[burn:].T)[0,1]:>16.3f}")
   print(f"{'ESS[0]':>14} {ess_1d(hmc_samples[burn:,0]):>12.0f} "
         f"{ess_1d(mh_samples[burn:,0]):>16.0f}")
   print(f"\nHMC achieves much higher ESS for correlated targets.")

The No-U-Turn Sampler (NUTS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NUTS (Hoffman and Gelman, 2014) eliminates the need to hand-tune :math:`L` by
building a binary tree of leapfrog steps and stopping when the trajectory
begins to "double back" (the U-turn criterion).  Combined with dual averaging
to adapt :math:`\varepsilon`, NUTS is the default sampler in Stan and PyMC.


18.5 Convergence Diagnostics
-------------------------------

Markov chains are guaranteed to converge only in the limit.  In practice we
need diagnostics to assess whether our finite chains have mixed well.  We want
concrete numbers, not just "it looks okay."

R-hat (Gelman--Rubin statistic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run :math:`m \ge 2` chains from dispersed starting values.  Let :math:`B` be
the between-chain variance of the chain means and :math:`W` the average
within-chain variance.  The potential scale reduction factor is:

.. math::

   \hat{R} = \sqrt{\frac{(n-1)/n \cdot W + B/n}{W}},

where :math:`n` is the chain length.

Reading this formula: the numerator estimates the true variance of the
posterior using a mixture of within-chain and between-chain information.
If the chains have converged to the same distribution, the between-chain
variance :math:`B` will be small relative to :math:`W`, and the ratio
will be close to 1.  If the chains are still exploring different regions,
:math:`B` will be large, inflating :math:`\hat{R}`.  Values close to 1
(typically :math:`\hat{R} < 1.01`) indicate convergence.

Effective sample size (ESS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because MCMC samples are autocorrelated, :math:`S` samples carry less
information than :math:`S` independent samples.  The ESS is:

.. math::

   \text{ESS} = \frac{S}{1 + 2\sum_{k=1}^\infty \rho_k},

where :math:`\rho_k` is the autocorrelation at lag :math:`k`.

In plain English: if consecutive MCMC samples are highly correlated (the
chain moves slowly), the denominator grows large, and the ESS shrinks well
below the nominal number of samples :math:`S`.  For example, if every sample
is nearly identical to the previous one (:math:`\rho_k \approx 1` for many
lags), you might have 10,000 samples but an ESS of only 50 --- meaning your
10,000 correlated draws carry as much information as 50 independent ones.
A low ESS means we need more iterations or a better sampler.

Let us compute R-hat and ESS from scratch, running multiple chains.

.. code-block:: python

   # Convergence diagnostics from scratch: R-hat and ESS
   import numpy as np

   np.random.seed(42)

   # Target: N(3, 2^2).  Run 4 chains from dispersed starts.
   def mh_chain(T, sigma_prop, start, target_mu=3.0, target_sigma=2.0):
       chain = np.zeros(T)
       chain[0] = start
       for t in range(1, T):
           proposal = chain[t-1] + np.random.normal(0, sigma_prop)
           log_alpha = (-0.5*((proposal - target_mu)/target_sigma)**2
                        + 0.5*((chain[t-1] - target_mu)/target_sigma)**2)
           if np.log(np.random.uniform()) < log_alpha:
               chain[t] = proposal
           else:
               chain[t] = chain[t-1]
       return chain

   # 4 chains from dispersed starting values
   T = 3000
   starts = [-10, 0, 10, 20]
   chains = [mh_chain(T, sigma_prop=2.5, start=s) for s in starts]

   # Discard first half as burn-in
   burn = T // 2
   chains_post = [c[burn:] for c in chains]
   m = len(chains_post)
   n = len(chains_post[0])

   # -- R-hat --
   chain_means = np.array([c.mean() for c in chains_post])
   chain_vars = np.array([c.var(ddof=1) for c in chains_post])
   W = chain_vars.mean()
   B = n * chain_means.var(ddof=1)
   var_hat = (n - 1) / n * W + B / n
   R_hat = np.sqrt(var_hat / W)

   # -- ESS (using initial positive sequence estimator) --
   def compute_ess(chain):
       n = len(chain)
       mean = chain.mean()
       var = chain.var()
       if var == 0:
           return n
       acf_sum = 0
       for k in range(1, n):
           rho_k = np.corrcoef(chain[:-k], chain[k:])[0, 1]
           if rho_k < 0.05:
               break
           acf_sum += rho_k
       return n / (1 + 2 * acf_sum)

   ess_values = [compute_ess(c) for c in chains_post]

   print("Convergence Diagnostics (target: N(3, 4))")
   print(f"  Chains: {m}, length: {T}, burn-in: {burn}")
   print(f"\n  {'Chain':>6} {'Start':>7} {'Mean':>8} {'Std':>8} {'ESS':>8}")
   print("  " + "-" * 42)
   for i, (c, s) in enumerate(zip(chains_post, starts)):
       print(f"  {i+1:>6} {s:>7.1f} {c.mean():>8.3f} {c.std():>8.3f} "
             f"{ess_values[i]:>8.0f}")

   print(f"\n  R-hat:         {R_hat:.4f}  {'(converged)' if R_hat < 1.01 else '(NOT converged)'}")
   print(f"  Total ESS:     {sum(ess_values):.0f}")
   print(f"  W (within):    {W:.4f}")
   print(f"  B/n (between): {B/n:.4f}")

Trace plots
^^^^^^^^^^^^

A simple visual diagnostic: plot each parameter against the iteration number.
The chain should look like a "fuzzy caterpillar" --- stationary and well-mixed
--- rather than showing trends or long excursions.

.. code-block:: python

   # Contrast good vs bad mixing by printing trajectory summaries
   import numpy as np

   np.random.seed(42)

   # Well-tuned chain (sigma=2.5) vs poorly-tuned chain (sigma=0.05)
   T = 5000
   good = np.zeros(T)
   bad = np.zeros(T)
   good[0] = bad[0] = 10.0  # start away from target mean=3

   for t in range(1, T):
       # Good chain
       prop = good[t-1] + np.random.normal(0, 2.5)
       if np.log(np.random.uniform()) < -0.5*((prop-3)/2)**2 + 0.5*((good[t-1]-3)/2)**2:
           good[t] = prop
       else:
           good[t] = good[t-1]
       # Bad chain
       prop = bad[t-1] + np.random.normal(0, 0.05)
       if np.log(np.random.uniform()) < -0.5*((prop-3)/2)**2 + 0.5*((bad[t-1]-3)/2)**2:
           bad[t] = prop
       else:
           bad[t] = bad[t-1]

   # Print snapshots of trajectory
   print(f"{'Iteration':>10} {'Good chain':>12} {'Bad chain':>12}")
   print("-" * 36)
   for t in [0, 10, 50, 100, 500, 1000, 2000, 4999]:
       print(f"{t:>10} {good[t]:>12.3f} {bad[t]:>12.3f}")

   print(f"\n{'Diagnostic':>18} {'Good':>10} {'Bad':>10}")
   print("-" * 40)
   rho_good = np.corrcoef(good[:-1], good[1:])[0, 1]
   rho_bad = np.corrcoef(bad[:-1], bad[1:])[0, 1]
   ess_good = T * (1 - rho_good) / (1 + rho_good)
   ess_bad = T * (1 - rho_bad) / (1 + rho_bad)
   print(f"{'Mean (last 3000)':>18} {good[2000:].mean():>10.3f} {bad[2000:].mean():>10.3f}")
   print(f"{'Std (last 3000)':>18} {good[2000:].std():>10.3f} {bad[2000:].std():>10.3f}")
   print(f"{'Lag-1 autocorr':>18} {rho_good:>10.3f} {rho_bad:>10.3f}")
   print(f"{'ESS':>18} {ess_good:>10.0f} {ess_bad:>10.0f}")
   print(f"{'True mean':>18} {'3.000':>10} {'3.000':>10}")


18.6 Variational Inference
-----------------------------

**Why variational methods?**
MCMC is asymptotically exact but can be slow --- our hierarchical model takes
thousands of iterations per second.  What if we had a million observations?
Variational inference (VI) turns the inference problem into an *optimization*
problem, trading some accuracy for speed.

The ELBO
^^^^^^^^^

We want to approximate the posterior :math:`p(\theta \mid \mathbf{x})` by a
simpler distribution :math:`q(\theta)` from some family :math:`\mathcal{Q}`.
The KL divergence from :math:`q` to the posterior is:

.. math::

   \text{KL}(q \| p)
   = \int q(\theta)\, \log \frac{q(\theta)}{p(\theta \mid \mathbf{x})}\, d\theta.

This is non-negative and zero only when :math:`q = p`.  We want to minimize it,
but it involves the intractable :math:`p(\theta \mid \mathbf{x})`.

**Derivation of the ELBO.**
Write :math:`p(\theta \mid \mathbf{x}) = p(\mathbf{x}, \theta) / p(\mathbf{x})`:

.. math::

   \text{KL}(q \| p)
   &= \int q(\theta)\, \log \frac{q(\theta)\, p(\mathbf{x})}{p(\mathbf{x}, \theta)}\, d\theta \\
   &= \log p(\mathbf{x})
      + \int q(\theta)\, \log \frac{q(\theta)}{p(\mathbf{x}, \theta)}\, d\theta.

Rearranging:

.. math::

   \log p(\mathbf{x})
   = \text{KL}(q \| p)
     - \int q(\theta)\, \log \frac{q(\theta)}{p(\mathbf{x}, \theta)}\, d\theta.

Because :math:`\text{KL} \ge 0`, we have:

.. math::

   \log p(\mathbf{x})
   \;\ge\;
   \underbrace{\int q(\theta)\, \log \frac{p(\mathbf{x}, \theta)}{q(\theta)}\, d\theta}
   _{\text{ELBO}(\,q\,)}.

The **Evidence Lower BOund** (ELBO) is:

.. math::

   \text{ELBO}(q)
   = E_q[\log p(\mathbf{x}, \theta)] - E_q[\log q(\theta)]
   = E_q[\log p(\mathbf{x} \mid \theta)] - \text{KL}(q \| p(\theta)).

Reading the second form: the ELBO has two competing terms.  The first,
:math:`E_q[\log p(\mathbf{x} \mid \theta)]`, rewards the approximation
:math:`q` for placing mass on parameter values that explain the data well
(good fit).  The second, :math:`\text{KL}(q \| p(\theta))`, penalizes
:math:`q` for straying too far from the prior (regularization).  Maximizing
the ELBO finds the best trade-off --- the approximate posterior that fits the
data as well as possible while remaining close to prior expectations.

Maximizing the ELBO is equivalent to minimizing :math:`\text{KL}(q \| p)`.

Let us implement variational inference for a Beta posterior, optimizing the
ELBO with gradient ascent and printing the ELBO at each iteration.

.. code-block:: python

   # Variational inference: fit a Gaussian to a Beta(35,25) posterior
   # Watch the ELBO increase iteration by iteration
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # True posterior: Beta(35, 25) from clinical trial
   a_true, b_true = 35, 25
   true_mean = a_true / (a_true + b_true)
   true_std = np.sqrt(a_true * b_true / ((a_true+b_true)**2 * (a_true+b_true+1)))

   # Variational family: q(theta) = N(mu_q, sigma_q^2) (on logit scale)
   # Parameterize on logit scale for unconstrained optimization
   mu_q = 0.0       # logit scale
   log_sigma_q = -1.0  # log(sigma_q)

   lr = 0.05
   S = 5_000  # MC samples for ELBO gradient

   print(f"{'Iter':>5} {'ELBO':>10} {'mu_q (logit)':>13} {'sigma_q':>9} "
         f"{'mean(theta)':>12}")
   print("-" * 52)

   for iteration in range(80):
       sigma_q = np.exp(log_sigma_q)

       # Reparameterization trick: z = mu_q + sigma_q * eps
       eps = np.random.normal(size=S)
       z_logit = mu_q + sigma_q * eps
       z = 1.0 / (1.0 + np.exp(-z_logit))  # inverse logit -> theta in (0,1)
       z = np.clip(z, 1e-10, 1 - 1e-10)

       # log p(x, theta) using Beta kernel
       log_joint = (a_true - 1) * np.log(z) + (b_true - 1) * np.log(1 - z)
       # log q(logit(theta)) with Jacobian
       log_q = stats.norm.logpdf(z_logit, mu_q, sigma_q)

       elbo = np.mean(log_joint - log_q)

       # Gradients (score function estimator)
       score_mu = (z_logit - mu_q) / sigma_q**2
       score_log_sigma = ((z_logit - mu_q)**2 / sigma_q**2 - 1)
       advantage = log_joint - log_q - elbo

       grad_mu = np.mean(advantage * score_mu)
       grad_log_sigma = np.mean(advantage * score_log_sigma)

       mu_q += lr * grad_mu
       log_sigma_q += lr * 0.1 * grad_log_sigma

       if iteration < 10 or iteration % 10 == 0:
           theta_mean = np.mean(z)
           print(f"{iteration:>5} {elbo:>10.3f} {mu_q:>13.4f} "
                 f"{sigma_q:>9.4f} {theta_mean:>12.4f}")

   print(f"\nTrue posterior: mean={true_mean:.4f}, std={true_std:.4f}")
   print(f"VI approx:     mean={np.mean(z):.4f}, std={np.std(z):.4f}")

Mean-field approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest variational family assumes the parameters are independent:

.. math::

   q(\boldsymbol{\theta})
   = \prod_{j=1}^p q_j(\theta_j).

Each factor :math:`q_j` is optimized in turn, holding the others fixed.

**Coordinate ascent VI (CAVI).**
The optimal update for :math:`q_j` is:

.. math::

   \log q_j^*(\theta_j)
   = E_{-j}[\log p(\mathbf{x}, \boldsymbol{\theta})] + \text{const},

where :math:`E_{-j}` denotes expectation with respect to all :math:`q_k` for
:math:`k \neq j`.

In plain English: to update our approximation for one parameter
:math:`\theta_j`, we take the log of the full joint distribution, then average
out all the other parameters using their current approximate distributions.
The result tells us the optimal shape for :math:`q_j`.  This is analogous to
fitting one variable at a time while holding the others fixed, which is why
it is called "coordinate ascent."  This is derived by taking the functional
derivative of the ELBO with respect to :math:`q_j` and setting it to zero.

CAVI iterates over all components until the ELBO converges.

.. code-block:: python

   # Coordinate Ascent VI for a mixture of two Gaussians (mean-field)
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Data from a Gaussian with unknown mean and precision
   true_mu, true_tau = 3.0, 0.5  # tau = 1/sigma^2
   data = np.random.normal(true_mu, 1/np.sqrt(true_tau), size=50)
   n = len(data)
   x_bar = data.mean()
   x_var = data.var()

   # Mean-field: q(mu, tau) = q_mu(mu) * q_tau(tau)
   # q_mu = N(m, s^2), q_tau = Gamma(a, b)
   # Hyperpriors: mu ~ N(0, 100), tau ~ Gamma(0.01, 0.01)

   m, s2 = 0.0, 10.0
   a_q, b_q = 1.0, 1.0

   print(f"{'Iter':>5} {'E[mu]':>8} {'E[tau]':>8} {'E[sigma]':>9} {'ELBO':>10}")
   print("-" * 43)
   for it in range(20):
       # Update q(mu): N(m, s^2)
       E_tau = a_q / b_q
       s2 = 1.0 / (1.0/100.0 + n * E_tau)
       m = s2 * (0.0/100.0 + E_tau * n * x_bar)

       # Update q(tau): Gamma(a_q, b_q)
       E_mu = m
       E_mu2 = s2 + m**2
       a_q = 0.01 + n / 2.0
       b_q = 0.01 + 0.5 * (np.sum(data**2) - 2 * E_mu * np.sum(data) + n * E_mu2)

       # Compute ELBO (simplified)
       elbo = (-n/2 * np.log(2*np.pi) + n/2 * (np.euler_gamma + np.log(2*b_q)
               - 1/(a_q)) - 0.5 * n * (a_q/b_q) * x_var)

       print(f"{it:>5} {m:>8.4f} {a_q/b_q:>8.4f} "
             f"{1/np.sqrt(a_q/b_q):>9.4f} {elbo:>10.2f}")

   print(f"\nTrue: mu={true_mu}, sigma={1/np.sqrt(true_tau):.3f}")

Stochastic variational inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large datasets, computing the ELBO over all data is expensive.  Stochastic
VI (Hoffman et al., 2013) uses mini-batches: subsample the data, compute a
noisy gradient of the ELBO, and update the variational parameters via
stochastic gradient ascent.  This scales VI to millions of observations.


18.7 Integrated Nested Laplace Approximations (INLA)
-------------------------------------------------------

**Why INLA?**
For a large class of models --- *latent Gaussian models* (LGMs) --- INLA
provides fast, deterministic, and accurate posterior approximations without
MCMC.

A latent Gaussian model has three stages:

1. **Observations:** :math:`y_i \mid \eta_i, \boldsymbol{\psi}` from an
   exponential family.
2. **Latent field:** :math:`\boldsymbol{\eta} \mid \boldsymbol{\psi} \sim
   N(\boldsymbol{\mu}_\psi, \boldsymbol{Q}_\psi^{-1})` (Gaussian with a
   sparse precision matrix :math:`\boldsymbol{Q}`).
3. **Hyperparameters:** :math:`\boldsymbol{\psi}` with prior
   :math:`p(\boldsymbol{\psi})`.

The key insight: with a Gaussian latent field, the Laplace approximation
(Section 17.5 in :ref:`Chapter 17 <ch17_bayesian>`) is very accurate for the
conditional posterior :math:`p(\boldsymbol{\eta} \mid \boldsymbol{\psi},
\mathbf{y})`.  INLA then numerically integrates over the (typically
low-dimensional) :math:`\boldsymbol{\psi}`.

**When to use INLA.**
INLA is the method of choice when the model fits the LGM framework: generalized
linear models, spatial models, time series, survival models, etc.  It is
implemented in the R package ``R-INLA``.


18.8 Comparison of Methods
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Accuracy
     - Speed
     - Scalability
     - When to use
   * - Laplace approx.
     - Moderate
     - Very fast
     - High
     - Unimodal, symmetric posteriors; quick screening
   * - MCMC (MH/Gibbs)
     - Exact (asymptotic)
     - Slow
     - Low--moderate
     - Gold standard; moderate dimensions
   * - HMC / NUTS
     - Exact (asymptotic)
     - Moderate
     - Moderate--high
     - Continuous parameters; up to thousands of dimensions
   * - Variational inference
     - Approximate
     - Fast
     - Very high
     - Large datasets; exploratory analysis; deep generative models
   * - INLA
     - Very good
     - Fast
     - High
     - Latent Gaussian models

Let us run a horse race: apply three methods to the same problem and compare.

.. code-block:: python

   # Head-to-head comparison: Laplace vs MH vs Gibbs on Normal-Normal
   import numpy as np
   from scipy import stats, optimize
   import time

   np.random.seed(42)

   # Data: N(mu, 1), mu ~ N(0, 10^2), observe 50 points
   true_mu = 3.0
   data = np.random.normal(true_mu, 1.0, size=50)
   n = len(data)
   x_bar = data.mean()

   # --- Exact posterior (conjugate) ---
   prec_post = 1/100 + n
   mu_post_exact = (0/100 + n*x_bar) / prec_post
   sd_post_exact = np.sqrt(1/prec_post)

   # --- Method 1: Laplace approximation ---
   t0 = time.time()
   neg_log_post = lambda mu: 0.5*n*(mu - x_bar)**2 + 0.5*mu**2/100
   res = optimize.minimize_scalar(neg_log_post)
   mu_lap = res.x
   h2 = n + 1/100
   sd_lap = np.sqrt(1/h2)
   t_lap = time.time() - t0

   # --- Method 2: Metropolis-Hastings ---
   t0 = time.time()
   T_mh = 10_000
   mh = np.zeros(T_mh)
   mh[0] = 0.0
   acc_mh = 0
   for t in range(1, T_mh):
       prop = mh[t-1] + np.random.normal(0, 0.3)
       la = (-0.5*n*(prop-x_bar)**2 - 0.5*prop**2/100
             + 0.5*n*(mh[t-1]-x_bar)**2 + 0.5*mh[t-1]**2/100)
       if np.log(np.random.uniform()) < la:
           mh[t] = prop
           acc_mh += 1
       else:
           mh[t] = mh[t-1]
   t_mh = time.time() - t0
   mh_post = mh[2000:]

   # --- Method 3: Direct sampling (exact conjugate) ---
   t0 = time.time()
   exact_samples = np.random.normal(mu_post_exact, sd_post_exact, 10_000)
   t_exact = time.time() - t0

   print(f"{'Method':>18} {'Mean':>8} {'Std':>8} {'Time (ms)':>10}")
   print("-" * 48)
   print(f"{'Exact':>18} {mu_post_exact:>8.4f} {sd_post_exact:>8.4f} {t_exact*1000:>10.2f}")
   print(f"{'Laplace':>18} {mu_lap:>8.4f} {sd_lap:>8.4f} {t_lap*1000:>10.2f}")
   print(f"{'MH (10k iter)':>18} {mh_post.mean():>8.4f} {mh_post.std():>8.4f} "
         f"{t_mh*1000:>10.2f}")

**Rules of thumb.**

* If the model is a latent Gaussian model, try INLA first.
* For moderate-dimensional continuous models, use HMC/NUTS (Stan, PyMC).
* For massive datasets or deep generative models, use variational inference.
* Use Gibbs sampling when full conditionals are available and the dimension
  is not too high.
* Use random-walk Metropolis as a baseline or when gradients are unavailable.


18.9 Summary
--------------

* Monte Carlo integration approximates integrals by sampling; importance
  sampling reweights samples from a proposal.
* Metropolis--Hastings constructs a Markov chain satisfying detailed balance;
  the acceptance probability is derived from the stationarity condition.
* Gibbs sampling is MH with full conditional proposals and acceptance rate 1.
* HMC uses Hamiltonian dynamics and the leapfrog integrator to make large,
  gradient-informed moves; NUTS automates the trajectory length.
* Convergence is assessed with :math:`\hat{R}`, ESS, and trace plots.
* Variational inference turns inference into optimization via the ELBO.
* INLA combines Laplace approximations with numerical integration for latent
  Gaussian models.

.. code-block:: python

   # Summary: which method for which problem?
   methods = [
       ("Conjugate (Beta-Binomial)",  "Exact",       "< 1 ms",   "Low-dim conjugate"),
       ("Laplace approximation",      "Good",        "< 10 ms",  "Unimodal posteriors"),
       ("Metropolis-Hastings",        "Exact (lim)", "seconds",   "General, low-dim"),
       ("Gibbs sampling",             "Exact (lim)", "seconds",   "Conjugate conditionals"),
       ("HMC / NUTS",                 "Exact (lim)", "seconds",   "Continuous, high-dim"),
       ("Variational inference",      "Approximate", "< 1 sec",  "Large data, deep models"),
   ]
   print(f"{'Method':<30} {'Accuracy':<14} {'Speed':<12} {'Best for'}")
   print("-" * 80)
   for name, acc, speed, use in methods:
       print(f"{name:<30} {acc:<14} {speed:<12} {use}")
