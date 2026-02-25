.. _ch15_em:

========================================
Chapter 15: The EM Algorithm
========================================

Look at this histogram of customer purchase amounts at an online retailer:

.. code-block:: python

   # The motivating picture: a bimodal histogram
   import numpy as np

   np.random.seed(42)
   # Two types of customers: budget shoppers and premium buyers
   budget  = np.random.normal(loc=25, scale=8, size=300)
   premium = np.random.normal(loc=75, scale=12, size=200)
   purchases = np.concatenate([budget, premium])
   np.random.shuffle(purchases)

   # Simple text histogram
   bins = np.linspace(0, 120, 13)
   counts, _ = np.histogram(purchases, bins=bins)
   print("Purchase Amount Distribution")
   print("-" * 40)
   for i in range(len(counts)):
       bar = "#" * (counts[i] // 2)
       print(f"${bins[i]:5.0f}-${bins[i+1]:5.0f}: {bar} ({counts[i]})")

Two bumps. You suspect there are two types of customers---budget shoppers
centered around $25 and premium buyers centered around $75---but you do not
know which customer belongs to which group. The group label is *hidden*. How do
you fit a model when part of the data is missing?

This is exactly the problem the EM algorithm solves. It alternates between
guessing the hidden labels (the E-step) and updating the model parameters given
those guesses (the M-step). Each iteration is guaranteed to improve the
likelihood, and the whole procedure converges to a (local) maximum.

Many statistical models involve quantities that are not directly observed:
latent class memberships, missing measurements, censored survival times,
random effects. In these settings the *observed-data* log-likelihood is
complicated --- often a sum-of-logs that resists closed-form maximization ---
while the *complete-data* log-likelihood (the one we would write if we could
see everything) is simple. The Expectation--Maximization (EM) algorithm
(Dempster, Laird, and Rubin, 1977) exploits this structure.

.. admonition:: Why EM?

   Imagine you are trying to find the best-fitting model, but some of the data
   are hidden.  You cannot optimize directly because the objective function
   involves a nasty log-of-a-sum.  EM sidesteps this by saying: "Pretend we
   know the hidden data (using our current best guess), optimize as if the data
   were complete, then update our guess."  Repeat.  This simple two-step dance
   always improves the likelihood, and it turns many intractable problems into
   sequences of easy ones.


15.1 The Incomplete-Data Problem
==================================

Observed vs. Complete Data
---------------------------

Let :math:`\mathbf{X}` denote the *observed* (incomplete) data and
:math:`\mathbf{Z}` the *latent* (unobserved) variables. Together,
:math:`(\mathbf{X}, \mathbf{Z})` form the *complete data*. We write

- :math:`p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})` for the
  complete-data density,
- :math:`p(\mathbf{X} \mid \boldsymbol{\theta})` for the observed-data
  (marginal) density,
- :math:`p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})` for the
  conditional density of the latent variables given the observed data.

The relationship is

.. math::
   :label: marginal

   p(\mathbf{X} \mid \boldsymbol{\theta})
   = \int p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})\,d\mathbf{Z}
   = \int p(\mathbf{X} \mid \mathbf{Z}, \boldsymbol{\theta})\,
     p(\mathbf{Z} \mid \boldsymbol{\theta})\,d\mathbf{Z}.

(Replace the integral with a sum when :math:`\mathbf{Z}` is discrete.)

The observed-data log-likelihood is

.. math::

   \ell(\boldsymbol{\theta})
   = \log p(\mathbf{X} \mid \boldsymbol{\theta})
   = \log \int p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})\,d\mathbf{Z}.

The log-of-an-integral is hard to optimize. Let us see *why* it is hard, with
our purchase-amount data.

.. code-block:: python

   # Why direct maximization is hard: the log-of-a-sum
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   x = np.concatenate([np.random.normal(25, 8, 300),
                        np.random.normal(75, 12, 200)])

   # The observed-data log-likelihood for a 2-component Gaussian mixture
   def obs_log_lik(pi1, mu1, sigma1, mu2, sigma2):
       pi2 = 1 - pi1
       ll = 0.0
       for xi in x:
           # This is log(pi1 * N(xi|mu1,sigma1) + pi2 * N(xi|mu2,sigma2))
           # -- a LOG OF A SUM, not a sum of logs!
           ll += np.log(pi1 * norm.pdf(xi, mu1, sigma1)
                      + pi2 * norm.pdf(xi, mu2, sigma2))
       return ll

   # Trying to differentiate this w.r.t. mu1 gives a messy expression
   # because d/d(mu1) log(A + B) = (dA/d(mu1)) / (A + B)
   # -- the denominator couples ALL parameters. No closed form.
   ll = obs_log_lik(0.5, 30, 10, 70, 15)
   print(f"Log-likelihood at initial guess: {ll:.2f}")
   print("Direct maximization requires iterating over all 5 parameters")
   print("with coupled gradients -- no closed-form solution exists.")

By contrast, the complete-data log-likelihood

.. math::

   \ell_c(\boldsymbol{\theta})
   = \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})

is often much simpler --- typically a member of an exponential family. The EM
algorithm turns the hard observed-data problem into a sequence of easy
complete-data problems.


15.2 Derivation via Jensen's Inequality
==========================================

We now derive the EM algorithm from first principles, using only Jensen's
inequality.

Introducing an Auxiliary Distribution
--------------------------------------

Let :math:`q(\mathbf{Z})` be *any* distribution over the latent variables
:math:`\mathbf{Z}`. Starting from the definition

.. math::

   \ell(\boldsymbol{\theta})
   = \log p(\mathbf{X} \mid \boldsymbol{\theta})
   = \log \int p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})\,d\mathbf{Z},

we multiply and divide by :math:`q(\mathbf{Z})` inside the integral:

.. math::

   \ell(\boldsymbol{\theta})
   = \log \int q(\mathbf{Z})\,
     \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
     \,d\mathbf{Z}.

This is a clever trick: we have rewritten the log-likelihood as the log of an
expectation (under :math:`q`).  Jensen's inequality will now let us push the
log inside the expectation, giving us a tractable lower bound.

Applying Jensen's Inequality
------------------------------

Since :math:`\log` is concave, Jensen's inequality gives

.. math::
   :label: elbo_ineq

   \ell(\boldsymbol{\theta})
   \;\geq\;
   \int q(\mathbf{Z})\,\log
     \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
     \,d\mathbf{Z}
   \;=:\;
   \mathcal{L}(q, \boldsymbol{\theta}).

The quantity :math:`\mathcal{L}(q, \boldsymbol{\theta})` is called the
**evidence lower bound (ELBO)**. It is a lower bound on the log-likelihood
for *any* choice of :math:`q`.

This bound is the foundation of the entire EM algorithm.  By choosing :math:`q`
wisely, we can make the bound tight; by maximizing the bound over
:math:`\boldsymbol{\theta}`, we push the log-likelihood upward.

Decomposing the ELBO
----------------------

We can rewrite the ELBO as

.. math::

   \mathcal{L}(q, \boldsymbol{\theta})
   = \int q(\mathbf{Z})\,\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})
     \,d\mathbf{Z}
   - \int q(\mathbf{Z})\,\log q(\mathbf{Z})\,d\mathbf{Z}.

The first term is the expected complete-data log-likelihood under :math:`q`.
The second is the entropy of :math:`q`.

Alternatively, the gap between the log-likelihood and the ELBO is

.. math::
   :label: gap

   \ell(\boldsymbol{\theta}) - \mathcal{L}(q, \boldsymbol{\theta})
   = \operatorname{KL}\!\bigl(q(\mathbf{Z})
     \;\|\; p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})\bigr)
   \;\geq\; 0,

where :math:`\operatorname{KL}` denotes the Kullback--Leibler divergence:

.. math::

   \operatorname{KL}(q \| p)
   = \int q(\mathbf{Z})\,\log \frac{q(\mathbf{Z})}
     {p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}\,d\mathbf{Z}.

To verify :eq:`gap`, note:

.. math::

   \ell(\boldsymbol{\theta})
   &= \log p(\mathbf{X} \mid \boldsymbol{\theta})
   = \int q(\mathbf{Z})\,\log p(\mathbf{X} \mid \boldsymbol{\theta})\,d\mathbf{Z} \\
   &= \int q(\mathbf{Z})\,\log
     \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}
     \,d\mathbf{Z} \\
   &= \int q(\mathbf{Z})\,\log
     \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
     \,d\mathbf{Z}
   + \int q(\mathbf{Z})\,\log
     \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}
     \,d\mathbf{Z} \\
   &= \mathcal{L}(q, \boldsymbol{\theta})
   + \operatorname{KL}(q \| p).

Since :math:`\operatorname{KL} \geq 0`, the ELBO is indeed a lower bound, and
equality holds if and only if
:math:`q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})`.

The E-Step and M-Step
-----------------------

The EM algorithm alternately maximizes the ELBO with respect to :math:`q` and
:math:`\boldsymbol{\theta}`:

**E-step (iteration** :math:`t` **):** Fix :math:`\boldsymbol{\theta}^{(t)}` and
choose :math:`q` to make the bound tight:

.. math::

   q^{(t+1)}(\mathbf{Z})
   = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}).

This sets the KL divergence in :eq:`gap` to zero, so
:math:`\mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t)}) = \ell(\boldsymbol{\theta}^{(t)})`.

**M-step:** Fix :math:`q = q^{(t+1)}` and maximize over
:math:`\boldsymbol{\theta}`:

.. math::

   \boldsymbol{\theta}^{(t+1)}
   = \arg\max_{\boldsymbol{\theta}}\;
     \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}).

Since the entropy of :math:`q^{(t+1)}` does not depend on
:math:`\boldsymbol{\theta}`, this reduces to

.. math::
   :label: mstep

   \boldsymbol{\theta}^{(t+1)}
   = \arg\max_{\boldsymbol{\theta}}\;
     Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}),

where the **Q-function** is

.. math::
   :label: qfunc

   Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})
   = \mathbb{E}_{\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)}}
     \!\left[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})\right]
   = \int p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})\,
     \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})\,d\mathbf{Z}.

In words: the E-step computes the expected complete-data log-likelihood,
averaging over the current conditional distribution of the latent variables;
the M-step maximizes this expected log-likelihood.


15.3 Monotone Ascent: EM Never Decreases the Likelihood
==========================================================

**Theorem.** The sequence of observed-data log-likelihoods produced by EM is
non-decreasing:

.. math::

   \ell(\boldsymbol{\theta}^{(t+1)}) \;\geq\; \ell(\boldsymbol{\theta}^{(t)})
   \quad\text{for all } t.

This is one of the most reassuring properties in optimization: no matter how
you initialize, no matter how complex the model, EM will never make things
worse. Each iteration either improves the likelihood or holds it steady.

**Proof.** After the E-step, the bound is tight:

.. math::

   \ell(\boldsymbol{\theta}^{(t)})
   = \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t)}).

After the M-step,
:math:`\boldsymbol{\theta}^{(t+1)}` maximizes :math:`\mathcal{L}(q^{(t+1)}, \cdot)`,
so

.. math::

   \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t+1)})
   \;\geq\;
   \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t)})
   = \ell(\boldsymbol{\theta}^{(t)}).

Since the ELBO is always a lower bound on the log-likelihood:

.. math::

   \ell(\boldsymbol{\theta}^{(t+1)})
   \;\geq\;
   \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}^{(t+1)})
   \;\geq\;
   \ell(\boldsymbol{\theta}^{(t)}).

:math:`\square`

**Convergence to a stationary point.** If the log-likelihood is bounded above,
the monotone sequence :math:`\ell(\boldsymbol{\theta}^{(t)})` converges. Under
regularity conditions (Wu, 1983), the limit point is a stationary point of the
log-likelihood. It is *not* guaranteed to be the global maximum; EM, like
Newton's method, can converge to a local maximum or a saddle point.


15.4 Example: Gaussian Mixture Model
=======================================

The Gaussian mixture model (GMM) is the canonical example for EM, and we will
use it for our customer segmentation problem: "Are there two types of
customers?"

Model Specification
--------------------

We observe :math:`x_1, \dots, x_n \in \mathbb{R}^d` and model them as drawn
from a mixture of :math:`K` Gaussian components:

.. math::

   p(x_i \mid \boldsymbol{\theta})
   = \sum_{k=1}^K \pi_k \,
     \mathcal{N}(x_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),

where :math:`\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K`,
the mixing weights satisfy :math:`\pi_k \geq 0` and
:math:`\sum_k \pi_k = 1`, and

.. math::

   \mathcal{N}(x \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})
   = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}
     \exp\!\left(-\tfrac{1}{2}(x - \boldsymbol{\mu})^{\!\top}
     \boldsymbol{\Sigma}^{-1}(x - \boldsymbol{\mu})\right).

Latent Variables
-----------------

Introduce latent indicator variables :math:`z_i \in \{1, \dots, K\}` such that
:math:`z_i = k` means observation :math:`i` came from component :math:`k`.
Then

.. math::

   p(z_i = k \mid \boldsymbol{\theta}) &= \pi_k, \\
   p(x_i \mid z_i = k, \boldsymbol{\theta})
     &= \mathcal{N}(x_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).

Complete-Data Log-Likelihood
-----------------------------

.. math::

   \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})
   = \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}[z_i = k]
     \bigl[\log \pi_k
     + \log \mathcal{N}(x_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\bigr].

This is a simple sum of Gaussian log-densities, weighted by indicator
functions --- much easier to handle than the log-sum in the observed-data
likelihood.

E-Step: Computing Responsibilities
------------------------------------

The E-step requires the posterior probability that observation :math:`i` belongs
to component :math:`k`, given the current parameter values
:math:`\boldsymbol{\theta}^{(t)}`. By Bayes' theorem:

.. math::
   :label: responsibilities

   \gamma_{ik}^{(t)}
   := p(z_i = k \mid x_i, \boldsymbol{\theta}^{(t)})
   = \frac{\pi_k^{(t)} \,
     \mathcal{N}(x_i \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}
     {\displaystyle\sum_{j=1}^K \pi_j^{(t)} \,
     \mathcal{N}(x_i \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}.

These are called **responsibilities** --- the "responsibility" of component
:math:`k` for observation :math:`i`.

Let us compute responsibilities for a few data points and see what they look
like.

.. code-block:: python

   # E-step in action: computing responsibilities for our purchase data
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   budget  = np.random.normal(25, 8, 300)
   premium = np.random.normal(75, 12, 200)
   x = np.concatenate([budget, premium])
   np.random.shuffle(x)
   n = len(x)

   # Current parameter guesses (deliberately imperfect)
   pi = np.array([0.5, 0.5])
   mu = np.array([20.0, 60.0])       # off from true 25, 75
   sigma = np.array([10.0, 15.0])

   # E-step: gamma[i, k] = P(customer i is type k | purchase amount x_i)
   gamma = np.zeros((n, 2))
   for k in range(2):
       gamma[:, k] = pi[k] * norm.pdf(x, mu[k], sigma[k])
   gamma /= gamma.sum(axis=1, keepdims=True)

   # Show responsibilities for a few informative data points
   print("E-step: Who belongs to which group?")
   print(f"{'Customer':>8s}  {'Purchase':>8s}  {'P(budget)':>10s}  {'P(premium)':>10s}  {'Assignment'}")
   print("-" * 58)
   # Pick points from different parts of the distribution
   for i in [0, 50, 100, 200, 300, 400, 499]:
       label = "budget" if gamma[i, 0] > 0.5 else "premium"
       print(f"{i:8d}  ${x[i]:7.2f}  {gamma[i, 0]:10.4f}  {gamma[i, 1]:10.4f}  {label}")

   print(f"\nEffective cluster sizes: N_1={gamma[:, 0].sum():.1f}, N_2={gamma[:, 1].sum():.1f}")

Notice how customers with low purchase amounts have high responsibility for the
budget component, and vice versa. Points near the overlap region have split
responsibilities.

The Q-function is then

.. math::

   Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})
   = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik}^{(t)}
     \bigl[\log \pi_k
     + \log \mathcal{N}(x_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\bigr].

M-Step: Updating Parameters
-----------------------------

We maximize :math:`Q` with respect to each parameter in turn.

**Mixing weights.** Maximize :math:`\sum_k \sum_i \gamma_{ik} \log \pi_k`
subject to :math:`\sum_k \pi_k = 1`. Using a Lagrange multiplier
(see :ref:`ch16_constrained`):

.. math::

   \mathcal{L} = \sum_{k=1}^K N_k \log \pi_k + \lambda\Bigl(1 - \sum_{k=1}^K \pi_k\Bigr),

where :math:`N_k = \sum_{i=1}^n \gamma_{ik}^{(t)}` is the effective number of
points assigned to component :math:`k`. Setting :math:`\partial\mathcal{L}/\partial\pi_k = 0`:

.. math::

   \frac{N_k}{\pi_k} - \lambda = 0
   \quad\Longrightarrow\quad
   \pi_k = \frac{N_k}{\lambda}.

Summing over :math:`k`: :math:`1 = \sum_k N_k / \lambda`, so :math:`\lambda = n` and

.. math::
   :label: pi_update

   \pi_k^{(t+1)} = \frac{N_k}{n}.

**Means.** For each component :math:`k`, maximize

.. math::

   \sum_{i=1}^n \gamma_{ik}^{(t)}
   \log \mathcal{N}(x_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
   \propto -\frac{1}{2}\sum_{i=1}^n \gamma_{ik}^{(t)}
   (x_i - \boldsymbol{\mu}_k)^{\!\top}\boldsymbol{\Sigma}_k^{-1}
   (x_i - \boldsymbol{\mu}_k).

Differentiating with respect to :math:`\boldsymbol{\mu}_k` and setting to zero:

.. math::

   \sum_{i=1}^n \gamma_{ik}^{(t)}\,\boldsymbol{\Sigma}_k^{-1}
   (x_i - \boldsymbol{\mu}_k) = \mathbf{0}
   \quad\Longrightarrow\quad
   \boldsymbol{\mu}_k^{(t+1)}
   = \frac{\sum_{i=1}^n \gamma_{ik}^{(t)}\,x_i}{N_k}.

.. math::
   :label: mu_update

   \boldsymbol{\mu}_k^{(t+1)}
   = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}^{(t)}\,x_i.

This is a weighted average of the data, weighted by the responsibilities.

**Covariances.** Differentiating
:math:`Q` with respect to :math:`\boldsymbol{\Sigma}_k^{-1}` (or equivalently
using the standard matrix-calculus result for the MLE of a Gaussian covariance):

.. math::
   :label: sigma_update

   \boldsymbol{\Sigma}_k^{(t+1)}
   = \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}^{(t)}\,
     (x_i - \boldsymbol{\mu}_k^{(t+1)})(x_i - \boldsymbol{\mu}_k^{(t+1)})^{\!\top}.

This is the responsibility-weighted sample covariance.

Let us trace one M-step so we can see the old and new parameters side by side.

.. code-block:: python

   # M-step in action: updating parameters from responsibilities
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   budget  = np.random.normal(25, 8, 300)
   premium = np.random.normal(75, 12, 200)
   x = np.concatenate([budget, premium])
   np.random.shuffle(x)
   n = len(x)

   # Old parameters
   pi_old = np.array([0.5, 0.5])
   mu_old = np.array([20.0, 60.0])
   sigma_old = np.array([10.0, 15.0])

   # E-step
   gamma = np.zeros((n, 2))
   for k in range(2):
       gamma[:, k] = pi_old[k] * norm.pdf(x, mu_old[k], sigma_old[k])
   gamma /= gamma.sum(axis=1, keepdims=True)

   # M-step
   N_k = gamma.sum(axis=0)
   pi_new = N_k / n
   mu_new = (gamma.T @ x) / N_k
   sigma_new = np.sqrt(np.array([
       np.sum(gamma[:, k] * (x - mu_new[k])**2) / N_k[k] for k in range(2)
   ]))

   # Show old -> new
   print("M-step: Parameter updates")
   print(f"{'Parameter':>12s}  {'Old':>10s}  {'New':>10s}  {'True':>10s}")
   print("-" * 48)
   for k in range(2):
       label = "budget" if k == 0 else "premium"
       true_pi = 0.6 if k == 0 else 0.4
       true_mu = 25.0 if k == 0 else 75.0
       true_sig = 8.0 if k == 0 else 12.0
       print(f"  pi_{label:>7s}  {pi_old[k]:10.4f}  {pi_new[k]:10.4f}  {true_pi:10.4f}")
       print(f"  mu_{label:>7s}  {mu_old[k]:10.4f}  {mu_new[k]:10.4f}  {true_mu:10.4f}")
       print(f" sig_{label:>7s}  {sigma_old[k]:10.4f}  {sigma_new[k]:10.4f}  {true_sig:10.4f}")

After just one E-step and M-step, the parameters are already moving toward the
true values.

Full EM: Watching Convergence
------------------------------

Now let us run the full EM algorithm for 20 iterations and print a convergence
table showing the parameter estimates and log-likelihood at every step.

.. code-block:: python

   # Full EM for 2-component Gaussian mixture (customer segmentation)
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   budget  = np.random.normal(25, 8, 300)
   premium = np.random.normal(75, 12, 200)
   x = np.concatenate([budget, premium])
   np.random.shuffle(x)
   n = len(x)
   K = 2

   # Initialize (deliberately off)
   pi = np.array([0.5, 0.5])
   mu = np.array([15.0, 55.0])
   sigma = np.array([12.0, 18.0])

   print(f"{'Iter':>4s}  {'pi1':>6s}  {'mu1':>8s}  {'sig1':>6s}  "
         f"{'pi2':>6s}  {'mu2':>8s}  {'sig2':>6s}  "
         f"{'Log-lik':>12s}  {'Change':>10s}")
   print("-" * 85)

   log_liks = []
   for t in range(20):
       # E-step: compute responsibilities
       gamma = np.zeros((n, K))
       for k in range(K):
           gamma[:, k] = pi[k] * norm.pdf(x, mu[k], sigma[k])
       gamma /= gamma.sum(axis=1, keepdims=True)

       # M-step: update parameters
       N_k = gamma.sum(axis=0)
       pi = N_k / n
       mu = (gamma.T @ x) / N_k
       sigma = np.sqrt(np.array([
           np.sum(gamma[:, k] * (x - mu[k])**2) / N_k[k] for k in range(K)
       ]))

       # Log-likelihood (observed data)
       ll = np.sum(np.log(sum(pi[k] * norm.pdf(x, mu[k], sigma[k])
                               for k in range(K))))
       change = ll - log_liks[-1] if log_liks else 0.0
       log_liks.append(ll)

       print(f"{t:4d}  {pi[0]:6.3f}  {mu[0]:8.2f}  {sigma[0]:6.2f}  "
             f"{pi[1]:6.3f}  {mu[1]:8.2f}  {sigma[1]:6.2f}  "
             f"{ll:12.2f}  {change:+10.4f}")

   print(f"\nTrue values: pi=(0.60, 0.40), mu=(25.0, 75.0), sigma=(8.0, 12.0)")

Two critical things to notice in the table:

1. **The log-likelihood is monotonically increasing** --- it never goes down,
   confirming the theorem we proved in Section 15.3.
2. **The parameters converge** to values close to the truth within about 10
   iterations.

Extending to 2D: Customer Segmentation with Two Features
----------------------------------------------------------

For a more realistic scenario, let us fit a 2D Gaussian mixture with 3
components (budget, regular, premium customers) using both purchase amount and
visit frequency.

.. code-block:: python

   # 2D GMM-EM: customer segmentation with two features
   import numpy as np
   from scipy.stats import multivariate_normal

   np.random.seed(42)

   # Generate: (avg_order_value, visits_per_month)
   X = np.vstack([
       np.random.randn(100, 2) @ [[8, 0], [0, 1.5]] + [25, 2],     # budget
       np.random.randn(80, 2)  @ [[6, 0], [0, 2.0]] + [50, 5],     # regular
       np.random.randn(70, 2)  @ [[12, 0], [0, 1.0]] + [80, 8],    # premium
   ])
   n, d = X.shape
   K = 3

   # Initialize
   pi = np.ones(K) / K
   idx = np.random.choice(n, K, replace=False)
   mu = X[idx].copy()
   Sigma = np.array([np.eye(d) * 50] * K)

   print(f"{'Iter':>4s}  ", end="")
   for k in range(K):
       print(f"{'mu_'+str(k+1):>14s}  ", end="")
   print(f"{'Log-lik':>12s}  {'Change':>10s}")
   print("-" * 80)

   log_liks = []
   for t in range(30):
       # E-step
       gamma = np.zeros((n, K))
       for k in range(K):
           gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k])
       gamma /= gamma.sum(axis=1, keepdims=True)

       # M-step
       N_k = gamma.sum(axis=0)
       for k in range(K):
           pi[k] = N_k[k] / n
           mu[k] = (gamma[:, k] @ X) / N_k[k]
           diff = X - mu[k]
           Sigma[k] = (gamma[:, k][:, None] * diff).T @ diff / N_k[k]

       # Log-likelihood
       ll = sum(np.log(sum(pi[k] * multivariate_normal.pdf(X, mu[k], Sigma[k])
                           for k in range(K))))
       change = ll - log_liks[-1] if log_liks else 0.0
       log_liks.append(ll)

       if t < 10 or t % 5 == 0 or abs(change) < 0.01:
           print(f"{t:4d}  ", end="")
           for k in range(K):
               print(f"({mu[k][0]:5.1f},{mu[k][1]:4.1f})  ", end="")
           print(f"{ll:12.2f}  {change:+10.4f}")

       if t > 0 and abs(change) < 1e-6:
           print(f"Converged at iteration {t}.")
           break

   print(f"\nFinal cluster proportions: {np.round(pi, 3)}")
   for k in range(K):
       print(f"  Cluster {k+1}: mu={np.round(mu[k], 1)}, "
             f"N_eff={N_k[k]:.0f}")


15.5 Example: Simple Missing Data
====================================

Setup
-----

Suppose we observe :math:`n` i.i.d. bivariate normal vectors
:math:`(X_{i1}, X_{i2}) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})`
but for some observations :math:`X_{i2}` is missing. Let
:math:`\mathcal{O}` be the set of complete cases and :math:`\mathcal{M}` the
set of cases missing :math:`X_{i2}`.

This is extremely common. Think of a customer survey that asks for both income
and spending, but 30% of respondents skip the income question. We want to
estimate the joint distribution using *all* observations, not just the complete
cases.

Complete-Data Log-Likelihood
-----------------------------

.. math::

   \ell_c(\boldsymbol{\mu}, \boldsymbol{\Sigma})
   = -\frac{n}{2}\log|\boldsymbol{\Sigma}|
   - \frac{1}{2}\sum_{i=1}^n
     (x_i - \boldsymbol{\mu})^{\!\top}\boldsymbol{\Sigma}^{-1}
     (x_i - \boldsymbol{\mu}).

E-Step
------

For complete cases :math:`i \in \mathcal{O}`, nothing changes. For missing
cases :math:`i \in \mathcal{M}`, we need the conditional expectation of
:math:`X_{i2}` and of :math:`X_{i2}^2` given :math:`X_{i1}` and the current
parameters :math:`\boldsymbol{\theta}^{(t)}`.

From the conditional-normal formula:

.. math::

   \mathbb{E}[X_{i2} \mid X_{i1}, \boldsymbol{\theta}^{(t)}]
   &= \mu_2^{(t)} + \frac{\sigma_{12}^{(t)}}{\sigma_{11}^{(t)}}
     (X_{i1} - \mu_1^{(t)}), \\
   \operatorname{Var}(X_{i2} \mid X_{i1}, \boldsymbol{\theta}^{(t)})
   &= \sigma_{22}^{(t)} - \frac{(\sigma_{12}^{(t)})^2}{\sigma_{11}^{(t)}}.

We replace each missing :math:`X_{i2}` with its conditional expectation, and
each missing :math:`X_{i2}^2` with its conditional second moment
(:math:`\operatorname{Var} + (\text{mean})^2`). This gives us the sufficient
statistics needed for the M-step.

M-Step
------

With the imputed sufficient statistics, the M-step simply computes the usual
sample mean and covariance --- exactly as if the data were complete, but using
the imputed values.

This "fill-in-and-maximize" pattern is characteristic of EM for exponential
families: the E-step computes expected sufficient statistics, and the M-step
uses them in the standard formulas.

Let us implement this and watch the parameters converge, comparing with the
"complete-case" estimate (which discards incomplete observations) and the oracle
(which knows the true missing values).

.. code-block:: python

   # EM for bivariate normal with missing data (survey example)
   import numpy as np

   np.random.seed(42)
   n = 500
   mu_true = np.array([50000.0, 3000.0])  # income, monthly spending
   Sigma_true = np.array([[1e8, 4e6], [4e6, 5e5]])

   # Generate complete data, then mask 30% of income (X_2)
   L = np.linalg.cholesky(Sigma_true)
   data_complete = np.random.randn(n, 2) @ L.T + mu_true
   missing = np.random.rand(n) < 0.3
   n_missing = missing.sum()
   print(f"Total observations: {n}, Missing income: {n_missing} ({n_missing/n:.0%})")

   # Oracle: use all complete data
   mu_oracle = data_complete.mean(axis=0)
   Sigma_oracle = np.cov(data_complete.T, bias=True)

   # Complete-case analysis: discard incomplete rows
   complete_cases = data_complete[~missing]
   mu_cc = complete_cases.mean(axis=0)
   Sigma_cc = np.cov(complete_cases.T, bias=True)

   # EM: use ALL data
   data = data_complete.copy()
   data[missing, 1] = np.nan  # mark as missing

   # Initialize from complete cases
   mu = mu_cc.copy()
   Sigma = Sigma_cc.copy()

   print(f"\n{'Iter':>4s}  {'mu1':>10s}  {'mu2':>10s}  "
         f"{'Sig11':>12s}  {'Sig12':>12s}  {'Sig22':>12s}")
   print("-" * 68)

   for iteration in range(25):
       # E-step: impute missing X_2
       data_filled = data_complete.copy()  # start from observed
       x2_sq_adj = np.zeros(n)
       for i in range(n):
           if missing[i]:
               cond_mean = mu[1] + Sigma[0, 1] / Sigma[0, 0] * (data_complete[i, 0] - mu[0])
               cond_var = Sigma[1, 1] - Sigma[0, 1]**2 / Sigma[0, 0]
               data_filled[i, 1] = cond_mean
               x2_sq_adj[i] = cond_var

       # M-step
       mu_new = data_filled.mean(axis=0)
       diff = data_filled - mu_new
       Sigma_new = (diff.T @ diff) / n
       Sigma_new[1, 1] += x2_sq_adj.sum() / n
       mu, Sigma = mu_new, Sigma_new

       if iteration < 5 or iteration % 5 == 0:
           print(f"{iteration:4d}  {mu[0]:10.0f}  {mu[1]:10.0f}  "
                 f"{Sigma[0,0]:12.0f}  {Sigma[0,1]:12.0f}  {Sigma[1,1]:12.0f}")

   # Final comparison
   print(f"\n{'Method':>15s}  {'mu1':>10s}  {'mu2':>10s}  {'Sig12':>12s}")
   print("-" * 52)
   print(f"{'Truth':>15s}  {mu_true[0]:10.0f}  {mu_true[1]:10.0f}  {Sigma_true[0,1]:12.0f}")
   print(f"{'Complete-case':>15s}  {mu_cc[0]:10.0f}  {mu_cc[1]:10.0f}  {Sigma_cc[0,1]:12.0f}")
   print(f"{'EM':>15s}  {mu[0]:10.0f}  {mu[1]:10.0f}  {Sigma[0,1]:12.0f}")
   print(f"{'Oracle':>15s}  {mu_oracle[0]:10.0f}  {mu_oracle[1]:10.0f}  {Sigma_oracle[0,1]:12.0f}")

EM uses all the data and typically produces estimates closer to the oracle than
the complete-case analysis. The improvement is especially visible in the
off-diagonal covariance :math:`\sigma_{12}`, which governs the income--spending
relationship.


15.6 ECM: Expectation Conditional Maximization
=================================================

When the M-step does not have a closed-form solution for all parameters
simultaneously, one can replace it with a sequence of **conditional
maximization** steps.

**ECM** (Meng and Rubin, 1993) replaces the M-step with :math:`S`
conditional maximizations: on each pass, maximize :math:`Q` with respect to
one block of parameters while holding the others fixed:

.. math::

   \boldsymbol{\theta}^{(t+1)}
   = \text{CM}_S \circ \cdots \circ \text{CM}_2 \circ \text{CM}_1
     (\boldsymbol{\theta}^{(t)}).

Each conditional maximization increases :math:`Q`, so the monotone-ascent
property is preserved. ECM converges at the same rate as EM asymptotically,
but each iteration is simpler.

**Example:** In a mixed-effects model, one might update the fixed effects
:math:`\boldsymbol{\beta}` in one CM step (a weighted regression) and the
variance components :math:`\sigma^2, \tau^2` in a second CM step.


15.7 MCEM: Monte Carlo EM
============================

Sometimes the E-step integral :eq:`qfunc` is intractable --- the conditional
distribution :math:`p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})`
does not have a convenient closed form. **Monte Carlo EM** (Wei and Tanner,
1990) replaces the exact expectation with a Monte Carlo average.

At iteration :math:`t`:

1. **E-step:** Draw :math:`M` samples
   :math:`\mathbf{Z}^{(1)}, \dots, \mathbf{Z}^{(M)}`
   from :math:`p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{(t)})` (e.g.,
   using MCMC).

2. **M-step:** Maximize the Monte Carlo approximation

   .. math::

      \hat{Q}(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})
      = \frac{1}{M}\sum_{m=1}^M
        \log p(\mathbf{X}, \mathbf{Z}^{(m)} \mid \boldsymbol{\theta}).

The monotone-ascent property is lost (the Monte Carlo error introduces noise),
but under regularity conditions (with increasing :math:`M`), MCEM converges to
a stationary point. In practice, one starts with a small :math:`M` and
increases it as the algorithm approaches convergence.

.. code-block:: python

   # MCEM for a simple censored-data example
   # Observe x_i if x_i < C, otherwise observe "x_i >= C" (right-censored)
   import numpy as np
   from scipy.stats import truncnorm

   np.random.seed(42)
   n = 300
   mu_true, sigma_true = 5.0, 2.0
   C = 6.0  # censoring threshold

   x_full = np.random.normal(mu_true, sigma_true, n)
   observed = x_full < C
   x_obs = x_full[observed]
   n_obs = observed.sum()
   n_cens = n - n_obs
   print(f"Observed: {n_obs}, Censored: {n_cens}")

   # MCEM: use Monte Carlo samples for censored values
   mu, sigma = 4.0, 1.5  # initial guess
   M = 200  # Monte Carlo samples per censored observation

   print(f"\n{'Iter':>4s}  {'mu':>8s}  {'sigma':>8s}  {'M':>5s}")
   print("-" * 30)
   for t in range(20):
       # E-step: sample censored values from truncated normal [C, inf)
       a = (C - mu) / sigma  # lower bound in standard normal units
       z_samples = truncnorm.rvs(a, np.inf, loc=mu, scale=sigma,
                                  size=(n_cens, M))

       # M-step: MLE using observed + sampled values
       # Sufficient statistics
       sum_x = x_obs.sum() + z_samples.mean(axis=1).sum()
       sum_x2 = (x_obs**2).sum() + (z_samples**2).mean(axis=1).sum()
       mu = sum_x / n
       sigma = np.sqrt(sum_x2 / n - mu**2)

       if t < 5 or t % 5 == 0:
           print(f"{t:4d}  {mu:8.4f}  {sigma:8.4f}  {M:5d}")

   print(f"\nTrue: mu={mu_true}, sigma={sigma_true}")
   print(f"MCEM: mu={mu:.4f}, sigma={sigma:.4f}")

.. admonition:: Common Pitfall

   EM can converge to a local maximum.  For mixture models, the log-likelihood
   surface is typically multimodal, with one local maximum for each permutation
   of the component labels plus potentially other spurious modes.  Always run
   EM from multiple random initializations and keep the solution with the
   highest log-likelihood.  For GMMs with :math:`K` components, 10--50 random
   restarts is common practice.

.. code-block:: python

   # Multiple random restarts to avoid local maxima
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   x = np.concatenate([np.random.normal(25, 8, 300),
                        np.random.normal(75, 12, 200)])
   n = len(x)

   def run_em(x, mu_init, n_iters=50):
       """Run 2-component 1D GMM-EM and return final log-likelihood + params."""
       pi = np.array([0.5, 0.5])
       mu = mu_init.copy()
       sigma = np.array([15.0, 15.0])
       for _ in range(n_iters):
           # E-step
           gamma = np.column_stack([pi[k] * norm.pdf(x, mu[k], sigma[k])
                                     for k in range(2)])
           gamma /= gamma.sum(axis=1, keepdims=True)
           # M-step
           N_k = gamma.sum(axis=0)
           pi = N_k / len(x)
           mu = (gamma.T @ x) / N_k
           sigma = np.sqrt(np.array([
               np.sum(gamma[:, k] * (x - mu[k])**2) / N_k[k]
               for k in range(2)]))
       ll = np.sum(np.log(sum(pi[k] * norm.pdf(x, mu[k], sigma[k])
                               for k in range(2))))
       return ll, pi, mu, sigma

   print(f"{'Restart':>7s}  {'Init mu':>16s}  {'Final mu':>16s}  {'Log-lik':>10s}")
   print("-" * 55)
   best_ll = -np.inf
   for restart in range(10):
       mu_init = np.sort(np.random.choice(x, 2, replace=False))
       ll, pi, mu, sigma = run_em(x, mu_init)
       print(f"{restart:7d}  ({mu_init[0]:5.1f}, {mu_init[1]:5.1f})  "
             f"({mu[0]:5.1f}, {mu[1]:5.1f})  {ll:10.2f}")
       if ll > best_ll:
           best_ll, best_params = ll, (pi, mu, sigma)

   pi, mu, sigma = best_params
   print(f"\nBest solution: mu=({mu[0]:.1f}, {mu[1]:.1f}), LL={best_ll:.2f}")


15.8 Variational EM
======================

Connection to Variational Inference
--------------------------------------

The EM derivation in Section 15.2 can be seen as a special case of
**variational inference**. The ELBO is

.. math::

   \mathcal{L}(q, \boldsymbol{\theta})
   = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})]
   - \mathbb{E}_q[\log q(\mathbf{Z})].

In standard EM, we set :math:`q` to the exact posterior
:math:`p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})`. In **variational
EM**, we restrict :math:`q` to a tractable family :math:`\mathcal{Q}` (e.g.,
factorized distributions) and optimize within that family:

**Variational E-step:**

.. math::

   q^{(t+1)} = \arg\max_{q \in \mathcal{Q}}\;
   \mathcal{L}(q, \boldsymbol{\theta}^{(t)}).

**M-step (unchanged):**

.. math::

   \boldsymbol{\theta}^{(t+1)}
   = \arg\max_{\boldsymbol{\theta}}\;
   \mathcal{L}(q^{(t+1)}, \boldsymbol{\theta}).

Mean-Field Approximation
--------------------------

The most common choice is the **mean-field** family, where :math:`q` factorizes
over groups of latent variables:

.. math::

   q(\mathbf{Z}) = \prod_{j} q_j(Z_j).

The optimal :math:`q_j` (holding others fixed) satisfies

.. math::

   \log q_j^*(Z_j)
   = \mathbb{E}_{q_{-j}}[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})]
   + \text{const},

where :math:`\mathbb{E}_{q_{-j}}` denotes expectation over all latent variables
except :math:`Z_j`. This yields a coordinate-ascent algorithm on the ELBO.

Because the variational E-step does not make the bound perfectly tight, the
monotone-ascent guarantee applies to the ELBO, not to the log-likelihood
itself. Nevertheless, the ELBO increases at every step, providing a useful
convergence diagnostic.


15.9 Convergence Rate of EM
==============================

EM converges, but how fast? The answer involves the *fraction of missing
information*.

Linear Convergence
--------------------

Near a stationary point :math:`\boldsymbol{\theta}^*`, the EM iterates satisfy

.. math::

   \boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^*
   \;\approx\;
   \mathbf{M}\,(\boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*),

where :math:`\mathbf{M}` is the **EM rate matrix**. The convergence is
**linear** (geometric), with rate equal to the spectral radius of
:math:`\mathbf{M}`.

The Fraction of Missing Information
-------------------------------------

Dempster, Laird, and Rubin (1977) showed that

.. math::
   :label: rate_matrix

   \mathbf{M}
   = \mathbf{I}
   - \mathcal{I}_{\text{obs}}(\boldsymbol{\theta}^*)^{-1}\,
     \mathcal{I}_{\text{com}}(\boldsymbol{\theta}^*),

where

- :math:`\mathcal{I}_{\text{com}}` is the (expected) complete-data information,
- :math:`\mathcal{I}_{\text{obs}}` is the observed-data information.

The **missing information** is
:math:`\mathcal{I}_{\text{mis}} = \mathcal{I}_{\text{com}} - \mathcal{I}_{\text{obs}}`,
so

.. math::

   \mathbf{M}
   = \mathcal{I}_{\text{obs}}^{-1}\,\mathcal{I}_{\text{mis}}.

The fraction of missing information is (roughly) the largest eigenvalue of
:math:`\mathbf{M}`. When most of the information is missing (e.g., heavy
censoring, many latent variables), :math:`\mathbf{M}` is near the identity and
EM converges very slowly. When little information is missing, :math:`\mathbf{M}`
is near zero and EM converges quickly.

Let us demonstrate this by varying the fraction of missing data and measuring
how many EM iterations are needed.

.. code-block:: python

   # EM convergence speed vs. fraction of missing data
   import numpy as np
   from scipy.stats import norm

   np.random.seed(42)
   n = 1000

   def em_iters_to_converge(frac_missing, tol=1e-6):
       """Run 1D normal EM with missing data, return iterations to converge."""
       mu_true, sigma_true = 5.0, 2.0
       x = np.random.normal(mu_true, sigma_true, n)
       missing = np.random.rand(n) < frac_missing
       x_obs = x[~missing]

       mu, sigma = 3.0, 3.0  # init
       for t in range(500):
           # E-step: E[x_i | missing] = mu, E[x_i^2 | missing] = sigma^2 + mu^2
           sum_x = x_obs.sum() + missing.sum() * mu
           sum_x2 = (x_obs**2).sum() + missing.sum() * (sigma**2 + mu**2)
           # M-step
           mu_new = sum_x / n
           sigma_new = np.sqrt(sum_x2 / n - mu_new**2)
           if abs(mu_new - mu) + abs(sigma_new - sigma) < tol:
               return t + 1, mu_new, sigma_new
           mu, sigma = mu_new, sigma_new
       return 500, mu, sigma

   print(f"{'% Missing':>10s}  {'Iterations':>10s}  {'mu_hat':>8s}  {'sigma_hat':>10s}")
   print("-" * 45)
   for frac in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90]:
       iters, mu_hat, sig_hat = em_iters_to_converge(frac)
       print(f"{frac:10.0%}  {iters:10d}  {mu_hat:8.4f}  {sig_hat:10.4f}")

   print(f"\nTrue: mu=5.0, sigma=2.0")
   print("More missing data -> slower convergence (more iterations)")

.. admonition:: What's the Intuition?

   EM's convergence speed is governed by how much you are "guessing" versus how
   much you actually "know."  If 90% of the information comes from the observed
   data and only 10% has to be filled in from the latent variables, EM
   converges fast --- the guesses barely matter.  But if 90% of the information
   is missing, EM is mostly chasing its own tail: each E-step makes a guess
   based on noisy estimates, and the M-step can only improve things marginally.

Practical Implications
-----------------------

- EM is **globally convergent** (monotone ascent from any start) but
  **locally linearly convergent** --- slower than Newton's quadratic rate.
- For problems with a large fraction of missing information, EM can be
  painfully slow. Remedies include:

  - **Aitken acceleration:** Extrapolate the linear convergence pattern.
  - **Louis' method:** Compute the observed information from EM output to
    get standard errors without running Newton.
  - **Hybrid methods:** Run EM for a few iterations to get near the optimum,
    then switch to Newton for fast convergence.

- The EM standard errors require additional computation. The observed
  information matrix can be obtained via **Louis' formula** (1982):

  .. math::

     \mathcal{I}_{\text{obs}}(\boldsymbol{\theta})
     = \mathcal{I}_{\text{com}}(\boldsymbol{\theta})
     - \mathcal{I}_{\text{mis}}(\boldsymbol{\theta}),

  where both terms are expectations under the posterior
  :math:`p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})`.


15.10 Generalizations and Connections
=======================================

The EM framework is remarkably general:

- **GEM (Generalized EM):** The M-step need not find the global maximum of
  :math:`Q`; any :math:`\boldsymbol{\theta}^{(t+1)}` with
  :math:`Q(\boldsymbol{\theta}^{(t+1)} \mid \boldsymbol{\theta}^{(t)}) \geq
  Q(\boldsymbol{\theta}^{(t)} \mid \boldsymbol{\theta}^{(t)})` suffices for
  the monotone-ascent property. This is useful when the M-step is itself
  solved iteratively (e.g., one gradient step).

- **Connection to majorization-minimization (MM):** EM is a special case of
  the MM principle, where the Q-function serves as a minorizer of the
  log-likelihood. Any algorithm that iteratively maximizes a tight lower
  bound enjoys the same convergence guarantees.

- **Connection to coordinate ascent on the ELBO:** As shown in Section 15.8,
  EM alternately optimizes the ELBO over :math:`q` and
  :math:`\boldsymbol{\theta}`. This perspective unifies EM with variational
  Bayes, wake-sleep algorithms, and expectation propagation.


15.11 Summary
==============

The EM algorithm transforms a difficult incomplete-data optimization problem
into a sequence of simpler complete-data problems. Its derivation via Jensen's
inequality leads naturally to the ELBO and the Q-function. The monotone-ascent
property guarantees that the likelihood never decreases, and convergence to a
stationary point follows under mild regularity conditions. The convergence rate
is linear, governed by the fraction of missing information --- fast when
little is missing, slow otherwise.

For Gaussian mixture models, EM yields the elegant responsibilities-based
algorithm that is the standard fitting method. When the E-step or M-step is
intractable, extensions such as MCEM, ECM, and variational EM provide
practical alternatives. The EM framework connects naturally to variational
inference, majorization-minimization, and the general theory of optimization
on lower bounds.

Combining EM with the gradient and Newton methods from :ref:`ch12_gradient`
and :ref:`ch13_newton` --- for instance, using EM to navigate the global
landscape and Newton to polish the final answer --- is a powerful practical
strategy. When constraints are present, the methods of :ref:`ch16_constrained`
can be incorporated into the M-step.
