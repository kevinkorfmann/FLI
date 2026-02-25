.. _ch8_specialized:

==========================================
Chapter 8 --- Specialized Likelihoods
==========================================

The "standard" likelihood --- the product of i.i.d. densities --- is a powerful
tool, but many practical problems require modifications. This chapter surveys
the most important variants, each built around a concrete problem where the
standard approach breaks down.

**Three running scenarios thread through this chapter:**

1. **Censored data** --- a clinical trial where patients drop out before the
   study ends, so some survival times are only partially observed.
2. **High-dimensional regression** --- gene expression data with 20,000 genes
   and only 100 patients, where the standard MLE does not exist.
3. **Spatial dependence** --- ecological surveys where nearby plots are
   correlated, making the full joint likelihood intractable.

Each method trades something --- efficiency, full distributional assumptions,
exact probability interpretation --- in exchange for tractability or robustness.
Knowing which tool to reach for is a core skill in applied statistics.

.. contents:: Topics in this chapter
   :local:
   :depth: 1
   :class: this-will-duplicate-information-and-it-is-still-useful-here


.. _sec_profile:

8.1 Profile Likelihood
========================

The problem with nuisance parameters
--------------------------------------

You are estimating the mean :math:`\mu` of a Normal distribution, but the
variance :math:`\sigma^2` is also unknown.  The log-likelihood
:math:`\ell(\mu, \sigma^2)` depends on both, yet you only care about
:math:`\mu`.  Reporting a confidence interval for :math:`\mu` that accounts
for uncertainty in :math:`\sigma^2` requires eliminating the nuisance
parameter.  The **profile likelihood** does this by, for each candidate
:math:`\mu`, plugging in the best-fitting :math:`\sigma^2`.

Definition
----------

The profile log-likelihood for :math:`\theta` is

.. math::

   \ell_P(\theta)
     = \max_{\lambda}\;\ell(\theta, \lambda)
     = \ell\!\left(\theta,\; \hat{\lambda}(\theta)\right),

where :math:`\hat{\lambda}(\theta)` is the constrained MLE of the nuisance
parameter :math:`\lambda` holding :math:`\theta` fixed.

.. code-block:: python

   # Profile likelihood: the idea in code
   # For each candidate mu, find the best sigma^2, then record the log-lik
   import numpy as np

   np.random.seed(42)
   mu_true, sigma_true = 5.0, 2.0
   n = 50
   data = np.random.normal(mu_true, sigma_true, size=n)

   def full_loglik(mu, sigma2):
       """Normal log-likelihood for given mu and sigma^2."""
       return -n/2 * np.log(2 * np.pi * sigma2) - np.sum((data - mu)**2) / (2 * sigma2)

   # For each mu, the optimal sigma^2 is the sample variance around mu
   def sigma2_hat(mu):
       return np.mean((data - mu)**2)

   mu_grid = np.linspace(3.5, 6.5, 200)
   profile_ll = np.array([full_loglik(mu, sigma2_hat(mu)) for mu in mu_grid])

   mu_hat = data.mean()
   print(f"MLE mu_hat = {mu_hat:.4f} (true = {mu_true})")
   print(f"Profile log-lik at MLE = {full_loglik(mu_hat, sigma2_hat(mu_hat)):.4f}")

Confidence intervals via Wilks' theorem
-----------------------------------------

Wilks' theorem states that, under regularity conditions,

.. math::

   2\bigl[\ell_P(\hat{\theta}) - \ell_P(\theta_0)\bigr]
     \;\xrightarrow{d}\; \chi^2_1,

so a :math:`(1-\alpha)` confidence region is
:math:`\{\theta : 2[\ell_P(\hat\theta) - \ell_P(\theta)] \le \chi^2_{1,\alpha}\}`.

.. code-block:: python

   # Profile CI via Wilks' theorem
   from scipy.stats import chi2

   ll_max = full_loglik(mu_hat, sigma2_hat(mu_hat))
   cutoff = chi2.ppf(0.95, df=1) / 2

   # Find the CI by checking which mu values are within the cutoff
   in_ci = mu_grid[profile_ll >= ll_max - cutoff]
   ci_lo, ci_hi = in_ci[0], in_ci[-1]

   # Compare to naive Wald CI
   se_wald = np.sqrt(sigma2_hat(mu_hat) / n)

   print(f"Profile 95% CI:  [{ci_lo:.4f}, {ci_hi:.4f}]")
   print(f"Wald 95% CI:     [{mu_hat - 1.96*se_wald:.4f}, {mu_hat + 1.96*se_wald:.4f}]")
   print(f"True mu = {mu_true}")

.. admonition:: When to use profile likelihood

   Use profile likelihood when you have nuisance parameters that can be
   maximised out, and you need confidence intervals for a subset of
   parameters.  It is especially valuable when the Wald approximation is
   poor (small samples, parameters near boundaries).


.. _sec_partial:

8.2 Partial Likelihood
========================

The problem with unknown baselines
------------------------------------

You are running a clinical trial with 200 cancer patients. You want to know
whether a new drug reduces mortality, adjusting for age and tumour stage.
The survival times have a complex baseline hazard :math:`h_0(t)` that you
cannot specify parametrically. The **partial likelihood** extracts the
information about covariate effects while leaving the baseline completely
unspecified.

Cox proportional hazards model
-------------------------------

The hazard for individual :math:`i` with covariate vector
:math:`\mathbf{z}_i` is

.. math::

   h(t \mid \mathbf{z}_i) = h_0(t)\,\exp(\boldsymbol{\beta}^\top \mathbf{z}_i),

where :math:`h_0(t)` is an unspecified baseline hazard and
:math:`\boldsymbol{\beta}` is the parameter of interest.

Construction of the partial likelihood
----------------------------------------

At each event time :math:`t_{(j)}`, the probability that the specific
individual :math:`d_j` fails (given exactly one failure occurs) is

.. math::

   \frac{\exp(\boldsymbol{\beta}^\top \mathbf{z}_{d_j})}
        {\sum_{i \in \mathcal{R}_j}\exp(\boldsymbol{\beta}^\top \mathbf{z}_i)},

where the risk set :math:`\mathcal{R}_j` contains everyone still at risk.
The baseline :math:`h_0(t_{(j)})` cancels. The partial likelihood is

.. math::

   L_{\text{partial}}(\boldsymbol{\beta})
     = \prod_{j=1}^{D}
       \frac{\exp(\boldsymbol{\beta}^\top \mathbf{z}_{d_j})}
            {\sum_{i \in \mathcal{R}_j}\exp(\boldsymbol{\beta}^\top \mathbf{z}_i)}.

.. code-block:: python

   # Partial likelihood: Cox model from scratch
   import numpy as np
   from scipy.optimize import minimize

   np.random.seed(42)
   n = 200

   # Simulate: drug (0/1) and age (standardized)
   drug = np.random.binomial(1, 0.5, size=n)
   age = np.random.randn(n)
   beta_true = np.array([-0.5, 0.3])  # drug reduces risk, age increases it

   # Simulate survival times from Cox model with Weibull baseline
   linear_pred = beta_true[0] * drug + beta_true[1] * age
   baseline_times = np.random.weibull(1.5, size=n)
   true_times = baseline_times * np.exp(-linear_pred)

   # Administrative censoring at 5 years + random dropout
   censor_times = np.minimum(5.0, np.random.exponential(8.0, size=n))
   time = np.minimum(true_times, censor_times)
   event = (true_times <= censor_times).astype(int)

   print(f"Events: {event.sum()}, Censored: {n - event.sum()}")

   # Build covariate matrix
   Z = np.column_stack([drug, age])

   # Partial log-likelihood
   def partial_loglik(beta):
       eta = Z @ beta
       ll = 0.0
       for j in range(n):
           if event[j] == 1:
               risk_set = time >= time[j]
               ll += eta[j] - np.log(np.sum(np.exp(eta[risk_set])))
       return ll

   # Maximize
   result = minimize(lambda b: -partial_loglik(b), x0=[0.0, 0.0], method='Nelder-Mead')
   beta_hat = result.x

   print(f"\nTrue beta:     drug = {beta_true[0]:.3f}, age = {beta_true[1]:.3f}")
   print(f"Partial MLE:   drug = {beta_hat[0]:.3f}, age = {beta_hat[1]:.3f}")
   print(f"Hazard ratio (drug): {np.exp(beta_hat[0]):.3f}  "
         f"(<1 means drug is protective)")

.. code-block:: python

   # Partial likelihood: numerical Hessian for SEs
   eps = 1e-5
   H = np.zeros((2, 2))
   for i in range(2):
       for j in range(2):
           pp = beta_hat.copy(); pp[i] += eps; pp[j] += eps
           pm = beta_hat.copy(); pm[i] += eps; pm[j] -= eps
           mp = beta_hat.copy(); mp[i] -= eps; mp[j] += eps
           mm = beta_hat.copy(); mm[i] -= eps; mm[j] -= eps
           H[i, j] = (partial_loglik(pp) - partial_loglik(pm)
                       - partial_loglik(mp) + partial_loglik(mm)) / (4 * eps**2)

   obs_info = -H
   se = np.sqrt(np.diag(np.linalg.inv(obs_info)))

   print(f"{'Covariate':<8} {'beta_hat':>10} {'SE':>8} {'95% CI':>22} {'HR':>8}")
   print("-" * 56)
   for k, name in enumerate(["drug", "age"]):
       ci_lo = beta_hat[k] - 1.96 * se[k]
       ci_hi = beta_hat[k] + 1.96 * se[k]
       print(f"{name:<8} {beta_hat[k]:10.4f} {se[k]:8.4f} "
             f"[{ci_lo:8.4f}, {ci_hi:8.4f}] {np.exp(beta_hat[k]):8.3f}")

.. admonition:: Why the partial likelihood works

   The baseline hazard :math:`h_0(t)` is an infinite-dimensional nuisance
   parameter. The partial likelihood sidesteps it entirely by conditioning on
   the event times, making the baseline cancel. You can learn about what
   *increases or decreases* risk without specifying the overall level of risk.


.. _sec_marginal:

8.3 Marginal Likelihood
=========================

The problem with too many parameters
--------------------------------------

A psychometrician fits a random-effects model to test scores from students
nested within schools. Each school has its own intercept (a nuisance
parameter), but she cares about the between-school variance. With 500
schools, profiling out 500 intercepts is awkward. The **marginal likelihood**
integrates out all school effects simultaneously using a prior/mixing
distribution.

Definition
----------

If :math:`\theta` is the parameter of interest and :math:`\lambda` is a
nuisance parameter with distribution :math:`\pi(\lambda)`:

.. math::

   L_M(\theta)
     = \int L(\theta, \lambda)\,\pi(\lambda)\,d\lambda.

.. code-block:: python

   # Marginal likelihood: one-way random effects model
   import numpy as np
   from scipy import integrate
   from scipy.stats import norm
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   n_schools = 30
   n_per_school = 20
   mu_true = 50.0          # grand mean
   sigma_b_true = 5.0      # between-school SD
   sigma_w_true = 10.0     # within-school SD

   # Simulate data: y_{ij} = mu + b_i + e_{ij}
   school_effects = np.random.normal(0, sigma_b_true, size=n_schools)
   data = []
   school_ids = []
   for i in range(n_schools):
       y = mu_true + school_effects[i] + np.random.normal(0, sigma_w_true, size=n_per_school)
       data.append(y)
       school_ids.extend([i] * n_per_school)
   data = np.concatenate(data)
   school_ids = np.array(school_ids)

   # For fixed sigma_b, the marginal likelihood integrates out school effects
   # For school i with data y_i, the marginal is Normal:
   #   y_i | mu, sigma_b, sigma_w ~ N(mu * 1, sigma_b^2 * J + sigma_w^2 * I)
   def marginal_loglik(sigma_b, mu=None, sigma_w=None):
       if mu is None:
           mu = data.mean()
       if sigma_w is None:
           sigma_w = sigma_w_true  # known for simplicity
       ll = 0.0
       for i in range(n_schools):
           y_i = data[school_ids == i]
           n_i = len(y_i)
           # Marginal variance of y_bar_i
           var_i = sigma_b**2 + sigma_w**2 / n_i
           y_bar_i = y_i.mean()
           ll += norm.logpdf(y_bar_i, loc=mu, scale=np.sqrt(var_i))
       return ll

   # Maximize over sigma_b
   result = minimize_scalar(lambda sb: -marginal_loglik(max(sb, 0.01)),
                            bounds=(0.01, 20), method='bounded')
   sigma_b_hat = result.x

   print(f"True sigma_b = {sigma_b_true}")
   print(f"Marginal MLE sigma_b = {sigma_b_hat:.4f}")
   print(f"Grand mean = {data.mean():.2f} (true = {mu_true})")

Connection to Bayesian model evidence
---------------------------------------

The fully marginal likelihood :math:`m(\mathbf{x}) = \int\!\int
f(\mathbf{x} \mid \theta, \lambda)\,\pi(\theta)\,\pi(\lambda)\,
d\theta\,d\lambda` is the **model evidence**, used for Bayes factors:

.. math::

   \text{BF}_{12} = \frac{m_1(\mathbf{x})}{m_2(\mathbf{x})}.

.. code-block:: python

   # Bayes factor: comparing two models for the school data
   # Model 1: sigma_b > 0 (random effects)
   # Model 2: sigma_b = 0 (no school effects)
   import numpy as np
   from scipy.stats import norm

   # Log marginal likelihood at the MLEs
   ll_model1 = marginal_loglik(sigma_b_hat)
   ll_model2 = marginal_loglik(0.001)  # effectively sigma_b = 0

   log_bf = ll_model1 - ll_model2
   print(f"Log Bayes factor (random effects vs. fixed): {log_bf:.2f}")
   print(f"Bayes factor: {np.exp(log_bf):.2f}")
   if log_bf > 3:
       print("=> Strong evidence for between-school variation")


.. _sec_conditional:

8.4 Conditional Likelihood
============================

The problem with incidental parameters
----------------------------------------

A matched case-control study has :math:`K = 200` matched pairs. Each pair
has its own intercept :math:`\alpha_k` (representing the shared environment
of the pair), but you care about the effect :math:`\beta` of an exposure.
With 200 nuisance parameters and only 400 observations, the unconditional
MLE for :math:`\beta` is inconsistent (the Neyman-Scott problem). The
**conditional likelihood** eliminates all :math:`\alpha_k` by conditioning on
sufficient statistics.

Definition
----------

.. math::

   L_C(\theta) = f(\mathbf{x} \mid T = t;\; \theta),

where :math:`T` is a sufficient statistic for the nuisance parameters.

Example: matched pairs in logistic regression
------------------------------------------------

The logistic model is
:math:`\text{logit}\,P(Y_{kj}=1) = \alpha_k + \beta\,x_{kj}`.
Conditioning on the pair totals :math:`Y_{k1} + Y_{k0} = 1`:

.. math::

   L_C(\beta)
     = \prod_{k=1}^{K}
       \frac{1}{1 + \exp(-\beta\,(x_{k1} - x_{k0}))}.

The :math:`\alpha_k` cancel entirely.

.. code-block:: python

   # Conditional likelihood: matched case-control study
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   K = 200  # matched pairs
   beta_true = 0.8

   # Simulate exposure for cases and controls
   x_case = np.random.normal(1.0, 1.0, size=K)     # cases tend to be more exposed
   x_control = np.random.normal(0.0, 1.0, size=K)

   # Pair-specific intercepts (nuisance) --- these cancel out
   alpha = np.random.normal(0, 2.0, size=K)

   # Generate outcomes (one case and one control per pair, total = 1)
   # The conditional likelihood depends only on d_k = x_case - x_control
   d = x_case - x_control

   # Conditional log-likelihood: product of logistic terms
   def cond_loglik(beta):
       return np.sum(-np.log(1 + np.exp(-beta * d)))

   # Maximize
   result = minimize_scalar(lambda b: -cond_loglik(b), bounds=(-5, 5), method='bounded')
   beta_hat = result.x

   # SE from observed information
   eps = 1e-5
   H = (cond_loglik(beta_hat + eps) - 2 * cond_loglik(beta_hat)
        + cond_loglik(beta_hat - eps)) / eps**2
   se = 1.0 / np.sqrt(-H)

   print(f"True beta = {beta_true}")
   print(f"Conditional MLE = {beta_hat:.4f}")
   print(f"SE = {se:.4f}")
   print(f"95% CI = [{beta_hat - 1.96*se:.4f}, {beta_hat + 1.96*se:.4f}]")
   print(f"Odds ratio = {np.exp(beta_hat):.3f}")


.. _sec_composite:

8.5 Composite Likelihood
==========================

The problem with intractable joint distributions
--------------------------------------------------

**Running scenario: spatial ecology.** An ecologist surveys 200 plots across
a landscape, recording presence/absence of a tree species. Nearby plots are
correlated (if a species thrives in one spot, it likely thrives 100m away).
The full joint likelihood for 200 correlated binary variables requires a
200-dimensional integral --- intractable. The **composite likelihood** breaks
this into manageable pieces: pairs of observations whose bivariate
distributions are tractable.

Definition
----------

Let :math:`\{A_1, \dots, A_K\}` be subsets of the data. The composite
likelihood is

.. math::

   L_{\text{CL}}(\theta)
     = \prod_{k=1}^{K} f_{A_k}(\mathbf{x}_{A_k} \mid \theta).

**Pairwise likelihood** (the most common variant):

.. math::

   L_{\text{pair}}(\theta) = \prod_{i < j} f(x_i, x_j \mid \theta).

.. code-block:: python

   # Composite likelihood: spatial ecology scenario
   import numpy as np
   from scipy.optimize import minimize_scalar
   from scipy.stats import norm

   np.random.seed(42)
   n_plots = 200

   # Spatial coordinates on a 2D landscape
   coords = np.column_stack([
       np.random.uniform(0, 10, size=n_plots),
       np.random.uniform(0, 10, size=n_plots)
   ])

   # True spatial correlation: exponential with range parameter theta
   theta_true = 2.0
   dists = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(axis=2))
   Sigma = np.exp(-dists / theta_true)
   Sigma += 0.01 * np.eye(n_plots)  # numerical stability

   # Simulate correlated binary data via latent Gaussian
   L_chol = np.linalg.cholesky(Sigma)
   z = L_chol @ np.random.randn(n_plots)
   presence = (z > 0).astype(int)

   print(f"Species present at {presence.sum()}/{n_plots} plots")

Here is the key issue: the full joint likelihood requires computing the
determinant and inverse of a 200x200 correlation matrix for binary data ---
and the probit link makes this a 200-dimensional integral over a truncated
multivariate normal.  Instead, we use **pairwise composite likelihood**,
considering only pairs of nearby plots.

.. code-block:: python

   # Pairwise composite log-likelihood
   from scipy.stats import mvn

   def pairwise_cll(theta):
       """Pairwise composite log-likelihood for spatial binary data."""
       rho_mat = np.exp(-dists / theta)
       cll = 0.0
       n_pairs = 0

       for i in range(n_plots):
           for j in range(i + 1, n_plots):
               if dists[i, j] > 4.0:  # only nearby pairs contribute
                   continue
               rho = rho_mat[i, j]

               # Bivariate probit probability
               # P(Y_i=y_i, Y_j=y_j) via bivariate normal CDF
               lo_i = 0.0 if presence[i] == 1 else -np.inf
               hi_i = np.inf if presence[i] == 1 else 0.0
               lo_j = 0.0 if presence[j] == 1 else -np.inf
               hi_j = np.inf if presence[j] == 1 else 0.0

               cov = np.array([[1.0, rho], [rho, 1.0]])
               lower = np.array([lo_i, lo_j])
               upper = np.array([hi_i, hi_j])

               # Use scipy.stats.mvn for bivariate normal rectangle prob
               p, _ = mvn.mvnun(lower, upper, np.array([0.0, 0.0]), cov)
               cll += np.log(max(p, 1e-15))
               n_pairs += 1

       return cll, n_pairs

   # Optimize over theta
   def neg_cll(theta):
       val, _ = pairwise_cll(max(theta, 0.1))
       return -val

   result = minimize_scalar(neg_cll, bounds=(0.3, 8.0), method='bounded')
   theta_hat = result.x
   _, n_pairs_used = pairwise_cll(theta_hat)

   print(f"\nTrue theta = {theta_true}")
   print(f"Composite MLE theta = {theta_hat:.4f}")
   print(f"Pairs used: {n_pairs_used}")

Properties
----------

* The composite MLE is consistent but not fully efficient.
* Standard errors require the **Godambe sandwich correction**:

  .. math::

     \text{Var}(\hat\theta_{\text{CL}})
       \approx H^{-1}\,J\,H^{-1},

  where :math:`H = -E[\nabla^2 \ell_{\text{CL}}]` and
  :math:`J = \text{Var}(\nabla\ell_{\text{CL}})`.

.. code-block:: python

   # Sandwich SE for composite likelihood (numerical)
   eps = 1e-4
   cll_plus, _ = pairwise_cll(theta_hat + eps)
   cll_center, _ = pairwise_cll(theta_hat)
   cll_minus, _ = pairwise_cll(theta_hat - eps)

   # H = -second derivative (sensitivity)
   H_num = -(cll_plus - 2 * cll_center + cll_minus) / eps**2

   # For the sandwich, we would ideally compute J from per-pair scores.
   # As a rough approximation, use the inverse Hessian SE:
   se_naive = 1.0 / np.sqrt(abs(H_num))

   print(f"Composite MLE = {theta_hat:.4f}")
   print(f"Naive SE = {se_naive:.4f}")
   print(f"95% CI = [{theta_hat - 1.96*se_naive:.4f}, {theta_hat + 1.96*se_naive:.4f}]")
   print(f"True theta = {theta_true}")


.. _sec_quasi:

8.6 Quasi-Likelihood
======================

The problem with unknown distributions
----------------------------------------

An entomologist counts insect larvae on crop plants. The counts show
variance about 3 times the mean (overdispersion). She does not want to
commit to a specific distribution (Poisson? Negative Binomial?), but she
does know that the variance is proportional to the mean.
**Quasi-likelihood** requires only a mean--variance relationship and produces
valid inference even without a fully specified distribution.

Definition
----------

Suppose :math:`E[Y_i] = \mu_i(\boldsymbol{\beta})` and
:math:`\text{Var}(Y_i) = \phi\,V(\mu_i)`. The **quasi-score** is

.. math::

   q_i(\boldsymbol{\beta})
     = \frac{y_i - \mu_i}{V(\mu_i)}\,\frac{\partial\mu_i}{\partial\boldsymbol{\beta}}.

.. code-block:: python

   # Quasi-likelihood: overdispersed Poisson regression
   import numpy as np
   from scipy.optimize import minimize

   np.random.seed(42)
   n = 200

   # Covariates: fertiliser amount (standardized)
   x = np.random.randn(n)
   beta_true = np.array([1.5, 0.8])  # intercept and slope (log link)

   # True mean: mu = exp(beta0 + beta1 * x)
   eta_true = beta_true[0] + beta_true[1] * x
   mu_true = np.exp(eta_true)

   # Simulate overdispersed counts (NegBin as stand-in for unknown dist)
   phi_true = 3.0  # overdispersion factor
   # NegBin with mean mu and var = phi * mu
   r_nb = mu_true / (phi_true - 1)  # NegBin r parameter
   p_nb = r_nb / (r_nb + mu_true)   # NegBin p parameter
   y = np.array([np.random.negative_binomial(max(ri, 0.1), pi)
                 for ri, pi in zip(r_nb, p_nb)])

   print(f"Sample mean = {y.mean():.2f}, Sample var = {y.var():.2f}")
   print(f"Var/Mean ratio = {y.var()/y.mean():.2f} (true phi ~ {phi_true})")

   # Quasi-Poisson: V(mu) = mu, estimate beta by solving quasi-score = 0
   # This is equivalent to Poisson GLM (ignoring overdispersion for point estimates)
   X = np.column_stack([np.ones(n), x])

   def quasi_score_norm(beta):
       eta = X @ beta
       mu = np.exp(eta)
       # Quasi-score: sum of [(y - mu) / V(mu)] * d_mu/d_beta
       # With log link: d_mu/d_beta = mu * X
       residuals = (y - mu) / mu  # (y-mu)/V(mu) with V(mu) = mu
       score = X.T @ (residuals * mu)  # = X.T @ (y - mu)
       return -np.sum(score**2)  # minimize negative squared norm

   # Solve via IRLS (equivalent to Poisson GLM fit)
   result = minimize(lambda b: np.sum((y - np.exp(X @ b))**2 / np.exp(X @ b)),
                     x0=[0.0, 0.0], method='Nelder-Mead')
   beta_hat = result.x

   # Estimate dispersion from Pearson residuals
   mu_hat = np.exp(X @ beta_hat)
   pearson_resid = (y - mu_hat) / np.sqrt(mu_hat)
   phi_hat = np.sum(pearson_resid**2) / (n - 2)

   print(f"\nTrue beta:  intercept = {beta_true[0]:.3f}, slope = {beta_true[1]:.3f}")
   print(f"Quasi MLE:  intercept = {beta_hat[0]:.3f}, slope = {beta_hat[1]:.3f}")
   print(f"Estimated dispersion phi = {phi_hat:.2f} (true ~ {phi_true})")

   # Sandwich SE (accounts for overdispersion)
   # Naive SE from Poisson * sqrt(phi) = quasi SE
   H = np.zeros((2, 2))
   for i in range(n):
       xi = X[i:i+1].T
       H += mu_hat[i] * (xi @ xi.T)
   naive_cov = np.linalg.inv(H)
   quasi_cov = phi_hat * naive_cov
   quasi_se = np.sqrt(np.diag(quasi_cov))

   print(f"\nQuasi-SEs: intercept = {quasi_se[0]:.4f}, slope = {quasi_se[1]:.4f}")
   print(f"95% CI for slope: [{beta_hat[1]-1.96*quasi_se[1]:.4f}, "
         f"{beta_hat[1]+1.96*quasi_se[1]:.4f}]")


.. _sec_pseudo:

8.7 Pseudo-Likelihood
=======================

The problem with intractable normalizing constants
-----------------------------------------------------

A neuroscientist models the joint firing patterns of 50 neurons as an Ising
model on a graph. The joint probability involves a normalizing constant that
sums over :math:`2^{50}` configurations --- utterly intractable. The
**pseudo-likelihood** replaces the joint density with the product of full
conditional densities, each of which is a simple logistic function.

Definition
----------

.. math::

   L_{\text{PL}}(\theta)
     = \prod_{i=1}^{n} f(x_i \mid x_{-i};\; \theta).

For a Markov random field, each conditional depends only on the neighbours:

.. math::

   P(x_i = 1 \mid x_{\partial i})
     = \frac{\exp\bigl(\beta\sum_{j \in \partial i} x_j\bigr)}
            {1 + \exp\bigl(\beta\sum_{j \in \partial i} x_j\bigr)}.

.. code-block:: python

   # Pseudo-likelihood: Ising model on a grid
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   grid_size = 20
   n = grid_size * grid_size  # 400 nodes
   beta_true = 0.4

   # Simulate Ising model via Gibbs sampling
   x = np.random.choice([0, 1], size=(grid_size, grid_size))
   for sweep in range(500):  # burn-in
       for i in range(grid_size):
           for j in range(grid_size):
               neighbours = []
               if i > 0: neighbours.append(x[i-1, j])
               if i < grid_size-1: neighbours.append(x[i+1, j])
               if j > 0: neighbours.append(x[i, j-1])
               if j < grid_size-1: neighbours.append(x[i, j+1])
               s = sum(neighbours)
               prob = 1 / (1 + np.exp(-beta_true * s))
               x[i, j] = int(np.random.rand() < prob)

   print(f"Fraction of 1s: {x.mean():.3f}")

   # Pseudo-log-likelihood
   def pseudo_loglik(beta):
       pll = 0.0
       for i in range(grid_size):
           for j in range(grid_size):
               neighbours = []
               if i > 0: neighbours.append(x[i-1, j])
               if i < grid_size-1: neighbours.append(x[i+1, j])
               if j > 0: neighbours.append(x[i, j-1])
               if j < grid_size-1: neighbours.append(x[i, j+1])
               s = sum(neighbours)
               eta = beta * s
               pll += x[i, j] * eta - np.log(1 + np.exp(eta))
       return pll

   # Maximize
   result = minimize_scalar(lambda b: -pseudo_loglik(b),
                            bounds=(0.0, 2.0), method='bounded')
   beta_hat = result.x

   # SE from curvature
   eps = 1e-5
   H = (pseudo_loglik(beta_hat+eps) - 2*pseudo_loglik(beta_hat)
        + pseudo_loglik(beta_hat-eps)) / eps**2
   se = 1.0 / np.sqrt(-H)

   print(f"True beta = {beta_true}")
   print(f"Pseudo-MLE = {beta_hat:.4f}")
   print(f"SE = {se:.4f}")
   print(f"95% CI = [{beta_hat - 1.96*se:.4f}, {beta_hat + 1.96*se:.4f}]")


.. _sec_empirical:

8.8 Empirical Likelihood
==========================

The problem with distributional assumptions
----------------------------------------------

A labour economist has income data that are heavily right-skewed. She wants a
confidence interval for the population mean, but the data are far from Normal
and the sample is too small for the CLT to give accurate coverage. The
**empirical likelihood** constructs likelihood-ratio-type confidence intervals
without assuming any parametric family, automatically adapting to skewness.

Definition
----------

Given observations :math:`x_1, \dots, x_n`, the empirical likelihood for the
mean :math:`\mu` is

.. math::

   L_{\text{EL}}(\mu) = \max_{w_i}
     \left\{\prod_{i=1}^{n} w_i \;\Bigg|\;
       \sum w_i(x_i - \mu) = 0,\;
       \sum w_i = 1,\; w_i \ge 0
     \right\}.

Owen's theorem: :math:`-2\ln R(\mu_0) \xrightarrow{d} \chi^2_1`.

.. code-block:: python

   # Empirical likelihood: CI for the mean of skewed data
   import numpy as np
   from scipy.optimize import brentq
   from scipy.stats import chi2

   np.random.seed(42)
   # Simulate right-skewed income data (log-normal)
   n = 60
   data = np.random.lognormal(mean=3.5, sigma=0.8, size=n)

   print(f"Sample mean = {data.mean():.2f}")
   print(f"Sample median = {np.median(data):.2f}")
   print(f"Skewness ~ {((data - data.mean())**3).mean() / data.std()**3:.2f}")

   # Compute -2 log R(mu) for a given mu
   def neg2logR(mu):
       """Compute -2 log empirical likelihood ratio at mu."""
       g = data - mu
       # Find Lagrange multiplier lambda via root-finding
       def equation(lam):
           return np.sum(g / (1 + lam * g))

       # Bracket: lam must satisfy 1 + lam * g_i > 0 for all i
       lam_lo = -1 / (max(g) + 1e-10) + 1e-10 if max(g) > 0 else -1e6
       lam_hi = -1 / (min(g) - 1e-10) - 1e-10 if min(g) < 0 else 1e6
       lam_lo = max(lam_lo, -1e6)
       lam_hi = min(lam_hi, 1e6)

       try:
           lam = brentq(equation, lam_lo, lam_hi)
           return 2 * np.sum(np.log(1 + lam * g))
       except:
           return np.inf

   # Find EL confidence interval: {mu : -2 log R(mu) <= chi2_{0.95}}
   cutoff = chi2.ppf(0.95, df=1)
   x_bar = data.mean()

   # Search for CI boundaries
   mu_grid = np.linspace(data.mean() * 0.7, data.mean() * 1.3, 500)
   el_values = np.array([neg2logR(mu) for mu in mu_grid])
   in_ci = mu_grid[el_values <= cutoff]

   if len(in_ci) > 0:
       el_ci_lo, el_ci_hi = in_ci[0], in_ci[-1]
   else:
       el_ci_lo, el_ci_hi = np.nan, np.nan

   # Compare to Wald CI
   se = data.std() / np.sqrt(n)
   wald_ci_lo = x_bar - 1.96 * se
   wald_ci_hi = x_bar + 1.96 * se

   print(f"\nEmpirical likelihood 95% CI: [{el_ci_lo:.2f}, {el_ci_hi:.2f}]")
   print(f"Wald 95% CI:                 [{wald_ci_lo:.2f}, {wald_ci_hi:.2f}]")
   print(f"EL CI is asymmetric: left width = {x_bar - el_ci_lo:.2f}, "
         f"right width = {el_ci_hi - x_bar:.2f}")
   print(f"(Asymmetry correctly reflects the right skew of the data)")


.. _sec_censored:

8.9 Censored and Truncated Likelihoods
========================================

The problem with incomplete observations
------------------------------------------

**Running scenario: clinical trial with dropouts.** A cancer trial follows
200 patients for up to 5 years. Some patients die during the trial (observed
events). Others are lost to follow-up or are still alive when the study ends
(censored). You cannot simply throw away censored observations --- that would
bias your estimates. And you cannot treat censored times as actual death
times --- that would also be wrong. The likelihood must reflect *exactly what
you know* about each patient.

The core principle: for each observation, contribute to the likelihood
exactly the information you have.

* Event observed: contribute the density :math:`f(y_i)`.
* Right censored (only know :math:`T_i > y_i`): contribute :math:`S(y_i)`.
* Left censored (only know :math:`T_i \le y_i`): contribute :math:`F(y_i)`.
* Interval censored (:math:`T_i \in (a_i, b_i]`): contribute
  :math:`F(b_i) - F(a_i)`.

Right-censored likelihood
--------------------------

.. math::

   L(\theta)
     = \prod_{i=1}^{n} f(y_i;\;\theta)^{\delta_i}\,S(y_i;\;\theta)^{1-\delta_i}.

where :math:`\delta_i = 1` if the event was observed, 0 if censored.

.. code-block:: python

   # Censored likelihood: clinical trial with Exponential survival
   import numpy as np

   np.random.seed(42)
   n = 200
   lam_true = 0.3  # true mortality rate

   # Simulate true event times ~ Exp(lambda)
   true_times = np.random.exponential(1 / lam_true, size=n)

   # Simulate censoring: administrative at 5 years + random dropout
   admin_censor = 5.0 * np.ones(n)
   dropout_censor = np.random.exponential(8.0, size=n)
   censor_times = np.minimum(admin_censor, dropout_censor)

   # What we actually observe
   y = np.minimum(true_times, censor_times)
   delta = (true_times <= censor_times).astype(int)

   d = delta.sum()
   print(f"Events: {d}, Censored: {n - d}")
   print(f"Total person-years: {y.sum():.1f}")
   print(f"Median observed time: {np.median(y):.2f} years")

Now the standard Poisson-type MLE :math:`\hat\lambda = n / \sum y_i` would be
*wrong* because it treats censored observations as events. The censored MLE
correctly uses only the number of *events* in the numerator.

.. code-block:: python

   # Censored Exponential log-likelihood
   # ell(lambda) = d * log(lambda) - lambda * sum(y_i)
   lam_grid = np.linspace(0.05, 0.8, 300)
   loglik = d * np.log(lam_grid) - lam_grid * y.sum()

   lam_peak = lam_grid[np.argmax(loglik)]
   lam_hat = d / y.sum()  # exact MLE

   print(f"Grid search MLE: {lam_peak:.4f}")
   print(f"Exact MLE:       {lam_hat:.4f}")
   print(f"Naive (wrong):   {n / y.sum():.4f}")
   print(f"True lambda:     {lam_true}")

.. code-block:: python

   # Censored Exponential: SE and CI
   # Fisher info for censored exponential: approx d / lambda^2
   se = lam_hat / np.sqrt(d)
   ci_lo = lam_hat - 1.96 * se
   ci_hi = lam_hat + 1.96 * se

   # Verify score = 0 at MLE
   score = d / lam_hat - y.sum()

   print(f"MLE = {lam_hat:.4f}, SE = {se:.4f}, "
         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}], true = {lam_true}")
   print(f"Score at MLE = {score:.2e}")
   print(f"Median survival (estimated): {np.log(2)/lam_hat:.2f} years")
   print(f"Median survival (true):      {np.log(2)/lam_true:.2f} years")

.. code-block:: python

   # Why censoring matters: compare correct vs. naive estimates
   import numpy as np

   # Correct: events / total person-time
   lam_correct = d / y.sum()

   # Wrong approach 1: pretend censored = events
   lam_naive1 = n / y.sum()

   # Wrong approach 2: throw away censored observations
   lam_naive2 = d / y[delta == 1].sum()

   print(f"{'Method':<35} {'Estimate':>10} {'Error':>8}")
   print("-" * 56)
   print(f"{'Correct (censored likelihood)':<35} {lam_correct:10.4f} {abs(lam_correct - lam_true):8.4f}")
   print(f"{'Wrong: all obs are events':<35} {lam_naive1:10.4f} {abs(lam_naive1 - lam_true):8.4f}")
   print(f"{'Wrong: drop censored obs':<35} {lam_naive2:10.4f} {abs(lam_naive2 - lam_true):8.4f}")
   print(f"{'True lambda':<35} {lam_true:10.4f}")

Truncation
----------

**Left truncation** (delayed entry): individual :math:`i` is observed only if
:math:`T_i > t_{\text{entry},i}`.  The contribution is
:math:`f(y_i) / S(t_{\text{entry},i})`.

.. code-block:: python

   # Left truncation: delayed entry in a survival study
   import numpy as np
   from scipy.optimize import minimize_scalar

   np.random.seed(42)
   n_potential = 500
   lam_true = 0.2

   # Simulate event times
   true_times = np.random.exponential(1 / lam_true, size=n_potential)

   # Left truncation: only observe individuals who survive past entry time
   entry_times = np.random.uniform(0, 2.0, size=n_potential)
   observed = true_times > entry_times
   y = true_times[observed]
   t_entry = entry_times[observed]
   n_obs = observed.sum()

   print(f"Potential subjects: {n_potential}")
   print(f"Observed (survived past entry): {n_obs}")

   # Truncated log-likelihood: sum of [log f(y_i) - log S(t_entry_i)]
   # For Exp(lambda): log f(y) = log(lambda) - lambda*y
   #                  log S(t) = -lambda * t
   def trunc_loglik(lam):
       return np.sum(np.log(lam) - lam * y - (-lam * t_entry))

   result = minimize_scalar(lambda l: -trunc_loglik(l),
                            bounds=(0.01, 2.0), method='bounded')
   lam_hat = result.x

   # Naive (ignoring truncation)
   lam_naive = n_obs / y.sum()

   print(f"\nTrue lambda = {lam_true}")
   print(f"Truncation-adjusted MLE = {lam_hat:.4f}")
   print(f"Naive MLE (ignoring truncation) = {lam_naive:.4f}")


.. _sec_penalized:

8.10 Penalized Likelihood
===========================

The problem with too many predictors
--------------------------------------

**Running scenario: gene expression.** A geneticist measures the expression
of 20,000 genes in 100 cancer patients and wants to predict tumour growth
rate.  With :math:`p = 20{,}000` predictors and :math:`n = 100`
observations, the ordinary least-squares estimator does not exist ---
:math:`\mathbf{X}^\top\mathbf{X}` is singular.  Even if it did exist, it
would overfit catastrophically.  **Penalized likelihood** adds a penalty that
shrinks coefficients toward zero, trading a little bias for a dramatic
reduction in variance.

General form
-------------

.. math::

   \ell_{\text{pen}}(\theta) = \ell(\theta) - \lambda\,P(\theta),

where :math:`P(\theta) \ge 0` is a penalty and :math:`\lambda \ge 0` controls
the strength of regularisation.

.. code-block:: python

   # The problem: OLS fails when p >> n
   import numpy as np

   np.random.seed(42)
   n_patients = 100
   n_genes = 500  # using 500 for speed; real genomics has 20000+
   n_causal = 10

   X = np.random.randn(n_patients, n_genes)
   true_beta = np.zeros(n_genes)
   causal_idx = np.random.choice(n_genes, n_causal, replace=False)
   true_beta[causal_idx] = np.random.randn(n_causal) * 2
   y = X @ true_beta + np.random.randn(n_patients) * 0.5

   # OLS: X'X is singular
   XtX = X.T @ X
   print(f"Dimensions: n = {n_patients}, p = {n_genes}")
   print(f"Rank of X'X = {np.linalg.matrix_rank(XtX)} (need {n_genes} for OLS)")
   print(f"=> OLS is impossible. We need penalization.")

Ridge regression (L2 penalty)
------------------------------

.. math::

   \hat{\boldsymbol{\beta}}_{\text{ridge}}
     = (\mathbf{X}^\top\mathbf{X} + 2\lambda\sigma^2\mathbf{I})^{-1}
       \mathbf{X}^\top\mathbf{y}.

Ridge always exists, even when :math:`\mathbf{X}^\top\mathbf{X}` is singular.

.. code-block:: python

   # Ridge regression: closed-form solution
   import numpy as np

   lam_ridge = 1.0
   I = np.eye(n_genes)
   beta_ridge = np.linalg.solve(X.T @ X + 2 * lam_ridge * I, X.T @ y)

   # How well does it recover the true signal?
   mse_ridge = np.mean((beta_ridge - true_beta)**2)
   nonzero_ridge = np.sum(np.abs(beta_ridge) > 0.01)

   print(f"Ridge MSE(beta): {mse_ridge:.6f}")
   print(f"Ridge nonzero coefficients: {nonzero_ridge} (true: {n_causal})")
   print(f"Ridge does NOT produce sparsity --- all {n_genes} coefficients are nonzero")

LASSO (L1 penalty)
--------------------

.. math::

   P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|.

The L1 penalty sets some coefficients to **exactly zero**, performing
simultaneous estimation and variable selection.

.. code-block:: python

   # LASSO: sparse solution via coordinate descent (scikit-learn)
   import numpy as np
   from sklearn.linear_model import LassoCV

   np.random.seed(42)
   lasso = LassoCV(cv=5, max_iter=10000)
   lasso.fit(X, y)

   selected = np.where(lasso.coef_ != 0)[0]
   recovered = set(selected) & set(causal_idx)
   false_positives = set(selected) - set(causal_idx)

   print(f"Best lambda: {lasso.alpha_:.4f}")
   print(f"True causal genes: {n_causal}")
   print(f"Selected by LASSO: {len(selected)}")
   print(f"Correctly identified: {len(recovered)}")
   print(f"False positives: {len(false_positives)}")
   print(f"R^2 (training): {lasso.score(X, y):.4f}")

.. code-block:: python

   # Compare Ridge vs LASSO: which coefficients survive?
   import numpy as np

   print(f"{'Gene':>6} {'True':>8} {'Ridge':>8} {'LASSO':>8} {'Causal?':>8}")
   print("-" * 44)

   # Show causal genes
   for idx in sorted(causal_idx):
       marker = "***"
       print(f"{idx:6d} {true_beta[idx]:8.3f} {beta_ridge[idx]:8.3f} "
             f"{lasso.coef_[idx]:8.3f} {marker:>8}")

   # Show a few non-causal genes for comparison
   non_causal = [i for i in range(min(5, n_genes)) if i not in causal_idx]
   for idx in non_causal[:3]:
       print(f"{idx:6d} {true_beta[idx]:8.3f} {beta_ridge[idx]:8.3f} "
             f"{lasso.coef_[idx]:8.3f} {'':>8}")

Elastic net
-----------

.. math::

   P(\boldsymbol{\beta})
     = \alpha\|\boldsymbol{\beta}\|_1
       + \frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2.

The elastic net combines L1 sparsity with L2 grouping: when covariates are
correlated, it tends to select them together.

Choosing the tuning parameter
-------------------------------

:math:`\lambda` is typically chosen by **cross-validation**: divide the data
into :math:`K` folds, fit on :math:`K-1`, evaluate on the held-out fold,
and select the :math:`\lambda` that minimises prediction error.

.. code-block:: python

   # Cross-validation path: how many genes are selected at each lambda?
   import numpy as np
   from sklearn.linear_model import lasso_path

   alphas, coefs, _ = lasso_path(X, y, n_alphas=50)

   print(f"{'lambda':>10} {'Nonzero':>10} {'MSE':>10}")
   print("-" * 34)
   for i in range(0, len(alphas), 10):
       n_nonzero = np.sum(coefs[:, i] != 0)
       pred = X @ coefs[:, i]
       mse = np.mean((y - pred)**2)
       print(f"{alphas[i]:10.4f} {n_nonzero:10d} {mse:10.4f}")

Connection to Bayesian priors
-------------------------------

.. list-table:: Penalties and their Bayesian priors
   :header-rows: 1
   :widths: 25 25 30

   * - Penalty
     - Prior
     - Effect
   * - L2 (Ridge)
     - Gaussian
     - Shrinkage, no sparsity
   * - L1 (LASSO)
     - Laplace
     - Shrinkage + sparsity
   * - Elastic net
     - Gaussian + Laplace
     - Shrinkage + grouped sparsity


.. _sec_specialized_summary:

8.11 Summary
=============

.. list-table:: When to use each specialised likelihood
   :header-rows: 1
   :widths: 25 50

   * - Method
     - Use when...
   * - Profile likelihood
     - Nuisance parameters can be maximised out; you need confidence intervals
       for a subset of parameters.
   * - Partial likelihood
     - The baseline hazard is unspecified (Cox model); you care about
       regression coefficients, not the baseline.
   * - Marginal likelihood
     - You want to integrate out nuisance parameters; Bayesian model
       comparison via Bayes factors.
   * - Conditional likelihood
     - A sufficient statistic for the nuisance exists; matched designs.
   * - Composite likelihood
     - The full joint density is intractable; spatial or high-dimensional
       data where marginals/pairs are manageable.
   * - Quasi-likelihood
     - You specify only a mean--variance relationship, not a full distribution;
       overdispersed GLM data.
   * - Pseudo-likelihood
     - Markov random fields or network models with intractable normalising
       constants.
   * - Empirical likelihood
     - You want nonparametric, likelihood-ratio-based inference without
       specifying a distributional family.
   * - Censored / truncated
     - Observations are incompletely observed (survival data, detection
       limits).
   * - Penalized likelihood
     - Over-parametrised models; you need regularisation for prediction or
       variable selection.

.. code-block:: python

   # Decision tree: which likelihood variant do you need?
   import numpy as np

   def diagnose_likelihood(n_obs, n_params, has_censoring, has_spatial_corr,
                           has_nuisance, dist_known):
       """Print which specialized likelihood to consider."""
       print("Likelihood diagnosis:")
       print(f"  n = {n_obs}, p = {n_params}")
       recommendations = []

       if has_censoring:
           recommendations.append("Censored/truncated likelihood (Sec 8.9)")
       if has_spatial_corr:
           recommendations.append("Composite or pseudo-likelihood (Sec 8.5, 8.7)")
       if has_nuisance:
           recommendations.append("Profile, marginal, or conditional likelihood (Sec 8.1, 8.3, 8.4)")
       if n_params > n_obs:
           recommendations.append("Penalized likelihood (Sec 8.10)")
       if not dist_known:
           recommendations.append("Quasi-likelihood or empirical likelihood (Sec 8.6, 8.8)")

       if not recommendations:
           recommendations.append("Standard likelihood should work fine!")

       for r in recommendations:
           print(f"  => {r}")

   # Example: clinical trial
   print("--- Clinical trial ---")
   diagnose_likelihood(200, 3, has_censoring=True, has_spatial_corr=False,
                       has_nuisance=False, dist_known=True)

   print("\n--- Gene expression ---")
   diagnose_likelihood(100, 20000, has_censoring=False, has_spatial_corr=False,
                       has_nuisance=False, dist_known=True)

   print("\n--- Spatial ecology ---")
   diagnose_likelihood(500, 3, has_censoring=False, has_spatial_corr=True,
                       has_nuisance=False, dist_known=True)

.. admonition:: Choosing the right likelihood variant

   When facing a new problem, ask yourself these questions in order:

   1. **Are all observations fully observed?** If not, use censored/truncated
      likelihoods (Section 8.9).
   2. **Is the joint distribution tractable?** If not, consider composite,
      pseudo-, or partial likelihood (Sections 8.2, 8.5, 8.7).
   3. **Are there nuisance parameters?** If so, choose between profile
      (maximise out), marginal (integrate out), or conditional (condition out)
      likelihoods (Sections 8.1, 8.3, 8.4).
   4. **Is the model over-parametrised?** If so, add a penalty (Section 8.10).
   5. **Do you want to avoid distributional assumptions?** Consider empirical
      or quasi-likelihood (Sections 8.6, 8.8).
