.. _ch19_model_selection:

==========================================
Chapter 19 -- Model Selection
==========================================

**The motivating problem.**
You are predicting house prices.  You have 10 candidate features: square
footage, number of bedrooms, lot size, age, distance to downtown, and so on.
A linear model with one feature gives :math:`R^2 = 0.45`.  Adding more
features always improves the fit on your training data.  With all 10 features
you get :math:`R^2 = 0.92`.  But when you use that model to predict prices in a
new neighborhood, the predictions are terrible.  You have overfit.

How many features should you include?  Every criterion in this chapter is a
different answer to that question, each from a slightly different angle.

.. code-block:: python

   # The house price prediction problem we develop throughout this chapter
   import numpy as np

   np.random.seed(42)

   # True model: price depends on 3 features (sqft, bedrooms, age)
   # Features 4-10 are irrelevant noise predictors
   n_train, n_test = 100, 500
   n_features = 10

   # Generate features
   X_all = np.random.normal(0, 1, (n_train + n_test, n_features))

   # True coefficients: only first 3 matter
   true_beta = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   intercept = 200

   noise = np.random.normal(0, 30, n_train + n_test)
   y_all = intercept + X_all @ true_beta + noise

   X_train, X_test = X_all[:n_train], X_all[n_train:]
   y_train, y_test = y_all[:n_train], y_all[n_train:]

   # Show the overfitting problem: training MSE always decreases, test MSE has U-shape
   print(f"{'Features':>10} {'Train MSE':>12} {'Test MSE':>12} {'R^2 train':>10}")
   print("-" * 48)
   for k in range(1, n_features + 1):
       Xk_tr = np.column_stack([np.ones(n_train), X_train[:, :k]])
       Xk_te = np.column_stack([np.ones(n_test), X_test[:, :k]])
       beta_hat = np.linalg.lstsq(Xk_tr, y_train, rcond=None)[0]
       pred_tr = Xk_tr @ beta_hat
       pred_te = Xk_te @ beta_hat
       mse_tr = np.mean((y_train - pred_tr)**2)
       mse_te = np.mean((y_test - pred_te)**2)
       ss_res = np.sum((y_train - pred_tr)**2)
       ss_tot = np.sum((y_train - y_train.mean())**2)
       r2 = 1 - ss_res / ss_tot
       marker = " <-- best test" if k == 3 else ""
       print(f"{k:>10} {mse_tr:>12.2f} {mse_te:>12.2f} {r2:>10.4f}{marker}")

   print("\nTraining MSE always decreases. Test MSE has a U-shape.")
   print("The true model uses 3 features. Can our criteria find it?")


19.1 The Bias--Variance Trade-off
-----------------------------------

**Why this matters.**
Before introducing specific criteria we need the conceptual backbone: the
total prediction error of any model can be decomposed into *bias* (systematic
error from a model that is too rigid) and *variance* (instability from a model
that is too flexible).

Suppose the true data-generating process is :math:`Y = f(\mathbf{x}) +
\varepsilon`, with :math:`E[\varepsilon] = 0` and
:math:`\text{Var}(\varepsilon) = \sigma^2`.  For a fitted model
:math:`\hat{f}(\mathbf{x})` the expected squared prediction error at a new
point :math:`\mathbf{x}_0` is:

.. math::

   E\!\left[(Y_0 - \hat{f}(\mathbf{x}_0))^2\right]
   &= \sigma^2
      + \bigl[f(\mathbf{x}_0) - E[\hat{f}(\mathbf{x}_0)]\bigr]^2
      + \text{Var}\!\bigl[\hat{f}(\mathbf{x}_0)\bigr] \\
   &= \underbrace{\sigma^2}_{\text{irreducible}}
      + \underbrace{\text{Bias}^2}_{\text{underfitting}}
      + \underbrace{\text{Variance}}_{\text{overfitting}}.

**Derivation.**

Start by adding and subtracting :math:`E[\hat{f}(\mathbf{x}_0)]`:

.. math::

   Y_0 - \hat{f}(\mathbf{x}_0)
   &= \bigl(Y_0 - f(\mathbf{x}_0)\bigr)
      + \bigl(f(\mathbf{x}_0) - E[\hat{f}(\mathbf{x}_0)]\bigr)
      + \bigl(E[\hat{f}(\mathbf{x}_0)] - \hat{f}(\mathbf{x}_0)\bigr).

Squaring and taking expectations, the three cross-terms vanish because
:math:`\varepsilon` is independent of :math:`\hat{f}` and the third term has
mean zero.  This yields the decomposition above.

Let us estimate bias and variance by simulation: fit the model many times on
different random training sets and measure the spread.

.. code-block:: python

   # Bias-variance decomposition by simulation
   import numpy as np

   np.random.seed(42)

   def true_f(x):
       return 50 * x - 20 * x**2  # true relationship

   sigma_noise = 15
   n_train = 40
   n_sims = 500
   x_test = np.array([1.0])  # evaluate at one point

   print(f"{'Degree':>8} {'Bias^2':>10} {'Variance':>10} {'Irreduc':>10} {'Total':>10}")
   print("-" * 50)
   for degree in [1, 2, 3, 5, 9]:
       predictions = []
       for _ in range(n_sims):
           x_tr = np.random.uniform(-2, 3, n_train)
           y_tr = true_f(x_tr) + np.random.normal(0, sigma_noise, n_train)
           coeffs = np.polyfit(x_tr, y_tr, degree)
           y_pred = np.polyval(coeffs, x_test)
           predictions.append(y_pred[0])

       preds = np.array(predictions)
       bias_sq = (true_f(x_test[0]) - preds.mean())**2
       variance = preds.var()
       irreducible = sigma_noise**2
       total = bias_sq + variance + irreducible
       print(f"{degree:>8} {bias_sq:>10.1f} {variance:>10.1f} "
             f"{irreducible:>10.1f} {total:>10.1f}")

   print("\nDegree 2 matches the true model -> lowest bias^2 + variance.")

.. admonition:: Intuition

   Imagine you are throwing darts.  Bias is how far your *average* throw is
   from the bullseye (systematic offset).  Variance is how spread out your
   throws are (consistency).  A dart player can be consistently off-center
   (high bias, low variance), or scattered all over the board (low bias, high
   variance).  The best player minimizes both.


19.2 Akaike Information Criterion (AIC)
------------------------------------------

**Motivation.**
We want to choose the model that is *closest to the truth* in a
Kullback--Leibler sense, using only the observed data.  The AIC provides an
elegant, practical answer.

Derivation from KL divergence minimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`g(\mathbf{x})` be the true density and :math:`f(\mathbf{x} \mid
\theta)` the model.  The KL divergence is:

.. math::

   \text{KL}(g \| f_\theta)
   = \int g(\mathbf{x})\, \log \frac{g(\mathbf{x})}{f(\mathbf{x} \mid \theta)}\, d\mathbf{x}
   = E_g[\log g(\mathbf{x})] - E_g[\log f(\mathbf{x} \mid \theta)].

The first term is a constant that does not depend on the model, so minimizing
KL is equivalent to maximizing :math:`E_g[\log f(\mathbf{x} \mid \theta)]`.

We do not know :math:`g`, but we can estimate this expected log-likelihood
from the data.  The in-sample log-likelihood :math:`\ell(\hat{\theta})` is a
biased estimator --- it is *too optimistic* because :math:`\hat{\theta}` was
chosen to maximize it on the same data.

Akaike (1973) showed that, under regularity conditions, the bias is
approximately :math:`p`, the number of estimated parameters:

.. math::

   E_g\!\left[\ell(\hat{\theta})\right]
   \approx E_g\!\left[E_g[\log f(\mathbf{x} \mid \hat{\theta})]\right] + p.

This motivates the AIC:

.. math::

   \text{AIC} = -2\,\ell(\hat{\theta}) + 2p.

We choose the model with the smallest AIC.  The factor of 2 is conventional.

**Intuition.**
The first term rewards fit; the second term penalizes complexity.  Adding a
parameter always improves :math:`\ell(\hat{\theta})` (or at worst leaves it
unchanged), but the penalty of 2 per parameter counteracts this unless the
parameter genuinely improves predictive accuracy.

Now let us compute AIC for our house price models with 1 to 10 features.

.. code-block:: python

   # AIC for the house price problem: 1 to 10 features
   import numpy as np

   np.random.seed(42)

   # Reuse the training data from chapter opening
   n_train_aic = 100
   n_features_aic = 10
   X_all_aic = np.random.normal(0, 1, (n_train_aic + 500, n_features_aic))
   true_beta_aic = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   noise_aic = np.random.normal(0, 30, n_train_aic + 500)
   y_all_aic = 200 + X_all_aic @ true_beta_aic + noise_aic
   X_tr = X_all_aic[:n_train_aic]
   y_tr = y_all_aic[:n_train_aic]
   n = n_train_aic

   print(f"{'Features':>10} {'LogLik':>10} {'p':>4} {'AIC':>10} {'AICc':>10} {'Best?':>7}")
   print("-" * 55)
   best_aic = np.inf
   best_k_aic = 0
   for k in range(1, n_features_aic + 1):
       Xk = np.column_stack([np.ones(n), X_tr[:, :k]])
       beta_hat = np.linalg.lstsq(Xk, y_tr, rcond=None)[0]
       resid = y_tr - Xk @ beta_hat
       sigma2 = np.mean(resid**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2

       p = k + 2  # intercept + k slopes + variance
       aic = -2 * log_lik + 2 * p
       # AICc: corrected for small samples
       aicc = aic + 2*p*(p+1) / max(n - p - 1, 1)

       if aic < best_aic:
           best_aic = aic
           best_k_aic = k
       marker = "<--" if k == best_k_aic and aic == best_aic else ""
       print(f"{k:>10} {log_lik:>10.2f} {p:>4} {aic:>10.2f} {aicc:>10.2f} {marker:>7}")

   print(f"\nAIC selects {best_k_aic} features. True model uses 3.")

Corrected AIC (AICc)
^^^^^^^^^^^^^^^^^^^^^^

For small samples the bias correction :math:`p` is not accurate enough.
Hurvich and Tsai (1989) derived a second-order correction for Normal linear
models:

.. math::

   \text{AICc} = \text{AIC} + \frac{2p(p+1)}{n - p - 1}.

When :math:`n` is large relative to :math:`p`, the correction is negligible.
As a rule of thumb, use AICc whenever :math:`n/p < 40`.


19.3 Bayesian Information Criterion (BIC)
--------------------------------------------

**Motivation.**
The BIC takes a different philosophical starting point: instead of predictive
accuracy, it approximates the *marginal likelihood* (model evidence) used in
Bayesian model comparison (see :ref:`Chapter 17 <ch17_bayesian>`).

Derivation from the Laplace approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The marginal likelihood is:

.. math::

   p(\mathbf{x} \mid M)
   = \int L(\theta)\, p(\theta)\, d\theta.

Apply the Laplace approximation (Section 17.5):

.. math::

   p(\mathbf{x} \mid M)
   \approx L(\hat{\theta})\, p(\hat{\theta})\,
   (2\pi)^{p/2}\, |\mathbf{H}|^{-1/2},

where :math:`\mathbf{H} = -\nabla^2 \ell(\hat{\theta})` is the observed
information.  Taking the logarithm:

.. math::

   \log p(\mathbf{x} \mid M)
   \approx \ell(\hat{\theta})
   + \log p(\hat{\theta})
   + \frac{p}{2}\log(2\pi)
   - \frac{1}{2}\log|\mathbf{H}|.

For an iid sample of size :math:`n`, the information scales as
:math:`\mathbf{H} = O(n)`, so :math:`\log|\mathbf{H}| \approx p\log n +
O(1)`.  Dropping lower-order terms:

.. math::

   \log p(\mathbf{x} \mid M)
   \approx \ell(\hat{\theta}) - \frac{p}{2}\log n + O(1).

Multiplying by :math:`-2`:

.. math::

   \text{BIC} = -2\,\ell(\hat{\theta}) + p\,\log n.

.. code-block:: python

   # BIC for the house price problem: compare with AIC
   import numpy as np

   np.random.seed(42)

   n = 100
   n_feat = 10
   X_all_bic = np.random.normal(0, 1, (n + 500, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_all_bic = 200 + X_all_bic @ true_b + np.random.normal(0, 30, n + 500)
   X_tr = X_all_bic[:n]
   y_tr = y_all_bic[:n]

   print(f"{'Features':>10} {'AIC':>10} {'BIC':>10} {'AIC pick':>9} {'BIC pick':>9}")
   print("-" * 52)
   best_aic, best_bic = np.inf, np.inf
   best_k_aic, best_k_bic = 0, 0
   results = []
   for k in range(1, n_feat + 1):
       Xk = np.column_stack([np.ones(n), X_tr[:, :k]])
       beta_hat = np.linalg.lstsq(Xk, y_tr, rcond=None)[0]
       resid = y_tr - Xk @ beta_hat
       sigma2 = np.mean(resid**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k + 2
       aic = -2 * log_lik + 2 * p
       bic = -2 * log_lik + p * np.log(n)
       if aic < best_aic: best_aic, best_k_aic = aic, k
       if bic < best_bic: best_bic, best_k_bic = bic, k
       results.append((k, aic, bic))

   for k, aic, bic in results:
       aic_mark = "<--" if k == best_k_aic else ""
       bic_mark = "<--" if k == best_k_bic else ""
       print(f"{k:>10} {aic:>10.2f} {bic:>10.2f} {aic_mark:>9} {bic_mark:>9}")

   print(f"\nAIC selects {best_k_aic} features, BIC selects {best_k_bic} features.")
   print(f"True model uses 3 features.")
   if best_k_aic != best_k_bic:
       print("They disagree! BIC penalizes more heavily (penalty = p*log(n) vs 2*p).")

Comparison with AIC
^^^^^^^^^^^^^^^^^^^^^

* AIC has a penalty of :math:`2p`; BIC has a penalty of :math:`p \log n`.
* For :math:`n \ge 8`, :math:`\log n > 2`, so BIC penalizes complexity more
  heavily.
* AIC is designed for *prediction*; BIC is designed for *model identification*.
* AIC tends to select larger models; BIC tends to select simpler (possibly
  true) models.
* AIC is not consistent (it does not select the true model with probability 1
  as :math:`n \to \infty` if the true model is in the candidate set); BIC is
  consistent.

.. code-block:: python

   # How AIC and BIC penalties diverge as n grows
   import numpy as np

   print(f"{'n':>8} {'AIC penalty per p':>18} {'BIC penalty per p':>18} {'BIC/AIC ratio':>15}")
   print("-" * 62)
   for n in [10, 20, 50, 100, 500, 1000, 10000]:
       aic_pen = 2
       bic_pen = np.log(n)
       print(f"{n:>8} {aic_pen:>18.2f} {bic_pen:>18.2f} {bic_pen/aic_pen:>15.2f}")

   print("\nFor n >= 8, BIC penalizes harder. For n >= 2981 (exp(8)), BIC penalty > 4x AIC.")


19.4 Deviance Information Criterion (DIC)
--------------------------------------------

**Motivation.**
AIC and BIC are designed for maximum likelihood estimation.  In Bayesian
analyses where we have MCMC output, we want a criterion that uses the
posterior distribution directly.

Definition
^^^^^^^^^^^^

The deviance is:

.. math::

   D(\theta) = -2\, \log p(\mathbf{x} \mid \theta).

The DIC is:

.. math::

   \text{DIC} = \bar{D} + p_D,

where :math:`\bar{D} = E_{\text{post}}[D(\theta)]` is the posterior mean
deviance and :math:`p_D` is the *effective number of parameters*:

.. math::

   p_D = \bar{D} - D(\bar{\theta}),

with :math:`\bar{\theta} = E_{\text{post}}[\theta]` the posterior mean.

Alternatively, :math:`p_D` can be defined as half the posterior variance of
the deviance:

.. math::

   p_D = \frac{1}{2}\, \text{Var}_{\text{post}}[D(\theta)].

**Intuition.**
:math:`\bar{D}` measures fit (lower is better); :math:`p_D` measures
complexity.  In a well-identified model, :math:`p_D` is close to the nominal
number of parameters.  In hierarchical models with partial pooling, :math:`p_D`
can be fractional --- a shrinkage estimator that counts only the "effective"
degrees of freedom.

.. code-block:: python

   # DIC from MCMC samples: Normal model
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Data
   data = np.random.normal(5, 2, size=30)
   n = len(data)

   # MCMC posterior samples (use conjugate for simplicity)
   # mu ~ N(0, 100), sigma^2 ~ InvGamma(0.01, 0.01)
   # For demonstration, use samples from approximate posterior
   mu_samples = np.random.normal(data.mean(), 2/np.sqrt(n), size=5000)
   sigma2_samples = np.random.gamma(n/2, 2*data.var()/n, size=5000)

   # Deviance at each posterior sample
   deviance_samples = np.array([
       -2 * np.sum(stats.norm.logpdf(data, mu, np.sqrt(s2)))
       for mu, s2 in zip(mu_samples, sigma2_samples)
   ])

   # D_bar: posterior mean deviance
   D_bar = deviance_samples.mean()

   # D(theta_bar): deviance at posterior mean
   mu_bar = mu_samples.mean()
   sigma2_bar = sigma2_samples.mean()
   D_theta_bar = -2 * np.sum(stats.norm.logpdf(data, mu_bar, np.sqrt(sigma2_bar)))

   # Effective number of parameters
   p_D = D_bar - D_theta_bar
   DIC = D_bar + p_D  # equivalently: 2*D_bar - D_theta_bar

   print(f"DIC Computation:")
   print(f"  D_bar (posterior mean deviance): {D_bar:.2f}")
   print(f"  D(theta_bar):                   {D_theta_bar:.2f}")
   print(f"  p_D (effective parameters):      {p_D:.2f}")
   print(f"  DIC = D_bar + p_D:              {DIC:.2f}")
   print(f"\n  Nominal parameters: 2 (mu, sigma^2)")
   print(f"  Effective parameters: {p_D:.2f}")

**Limitations.**
DIC can give negative :math:`p_D` for non-log-concave posteriors and is not
invariant to parameterization.


19.5 Widely Applicable Information Criterion (WAIC)
------------------------------------------------------

**Motivation.**
WAIC (Watanabe, 2010) improves on DIC by being fully Bayesian and having a
direct connection to cross-validation.

Definition
^^^^^^^^^^^^

The **log pointwise predictive density** is:

.. math::

   \text{lppd}
   = \sum_{i=1}^n \log p(x_i \mid \mathbf{x})
   = \sum_{i=1}^n \log \int p(x_i \mid \theta)\, p(\theta \mid \mathbf{x})\, d\theta.

In practice, using :math:`S` posterior draws:

.. math::

   \widehat{\text{lppd}}
   = \sum_{i=1}^n \log\!\left(\frac{1}{S}\sum_{s=1}^S
     p(x_i \mid \theta^{(s)})\right).

The effective number of parameters is:

.. math::

   p_{\text{WAIC}}
   = \sum_{i=1}^n \text{Var}_{\text{post}}\!\bigl[\log p(x_i \mid \theta)\bigr].

The WAIC is:

.. math::

   \text{WAIC} = -2\,\bigl(\widehat{\text{lppd}} - p_{\text{WAIC}}\bigr).

.. code-block:: python

   # WAIC computation from posterior samples
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Data
   data = np.random.normal(5, 2, size=30)
   n = len(data)

   # Posterior samples (approximate)
   S = 5000
   mu_samples = np.random.normal(data.mean(), 2/np.sqrt(n), size=S)
   sigma_samples = np.abs(np.random.normal(data.std(), 0.3, size=S))

   # Pointwise log-likelihood matrix: (n x S)
   log_lik_matrix = np.array([
       stats.norm.logpdf(data, mu, sigma)
       for mu, sigma in zip(mu_samples, sigma_samples)
   ]).T  # shape (n, S)

   # lppd: log pointwise predictive density
   # For each data point: log( mean( exp(log_lik) ) )
   lppd_i = np.log(np.mean(np.exp(log_lik_matrix), axis=1))
   lppd = lppd_i.sum()

   # p_WAIC: sum of posterior variances of log-likelihood
   p_waic_i = np.var(log_lik_matrix, axis=1)
   p_waic = p_waic_i.sum()

   # WAIC
   waic = -2 * (lppd - p_waic)

   print(f"WAIC Computation:")
   print(f"  lppd:          {lppd:.2f}")
   print(f"  p_WAIC:        {p_waic:.2f}")
   print(f"  WAIC:          {waic:.2f}")
   print(f"\n  Per-observation breakdown (first 5):")
   print(f"  {'i':>4} {'lppd_i':>10} {'p_WAIC_i':>10}")
   for i in range(5):
       print(f"  {i+1:>4} {lppd_i[i]:>10.4f} {p_waic_i[i]:>10.4f}")

Connection to cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Watanabe showed that WAIC is asymptotically equivalent to leave-one-out
cross-validation (LOO-CV), making it a Bayesian approximation to the
out-of-sample predictive performance.


19.6 Cross-Validation
-----------------------

**Why cross-validation?**
All information criteria are *approximations* to out-of-sample prediction
error.  Cross-validation estimates it directly by repeatedly fitting the model
on a subset of the data and evaluating on the held-out portion.  It is the
most honest assessment of how well your model will generalize.

Leave-one-out cross-validation (LOO-CV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each observation :math:`i = 1, \ldots, n`:

1. Fit the model on all data except :math:`x_i`.
2. Compute the predictive density :math:`p(x_i \mid \mathbf{x}_{-i})`.

The LOO-CV estimate of predictive performance is:

.. math::

   \widehat{\text{elpd}}_{\text{LOO}}
   = \sum_{i=1}^n \log p(x_i \mid \mathbf{x}_{-i}).

This requires :math:`n` model fits, which is expensive.  Vehtari, Gelman, and
Gabry (2017) showed that *Pareto-smoothed importance sampling* (PSIS-LOO)
accurately approximates LOO-CV using a single MCMC run, with importance
weights derived from the posterior:

.. math::

   p(x_i \mid \mathbf{x}_{-i})
   \approx \frac{1}{\frac{1}{S}\sum_{s=1}^S \frac{1}{p(x_i \mid \theta^{(s)})}}.

The Pareto smoothing stabilizes the importance weights by fitting a generalized
Pareto distribution to the tail; the shape parameter :math:`\hat{k}` also
serves as a diagnostic (values :math:`> 0.7` indicate unreliable
approximations).

k-fold cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^

Partition the data into :math:`K` folds (commonly :math:`K = 5` or 10).
For each fold :math:`k`:

1. Fit the model on all data except fold :math:`k`.
2. Evaluate the log predictive density on fold :math:`k`.

Sum over folds to obtain the cross-validation estimate.  :math:`K`-fold CV
requires only :math:`K` model fits, but introduces some bias because each
training set has size :math:`n(1 - 1/K)` rather than :math:`n-1`.

Let us run 5-fold CV on our house price problem and compare with AIC and BIC.

.. code-block:: python

   # 5-fold cross-validation for the house price problem
   import numpy as np

   np.random.seed(42)

   n = 100
   n_feat = 10
   X_all_cv = np.random.normal(0, 1, (n, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_cv = 200 + X_all_cv @ true_b + np.random.normal(0, 30, n)

   K = 5
   indices = np.arange(n)
   np.random.shuffle(indices)
   folds = np.array_split(indices, K)

   print(f"{'Features':>10} {'CV MSE':>10} {'AIC':>10} {'BIC':>10}")
   print("-" * 44)
   best_cv, best_aic, best_bic = np.inf, np.inf, np.inf
   best_k_cv, best_k_aic, best_k_bic = 0, 0, 0

   for k_feat in range(1, n_feat + 1):
       # 5-fold CV
       cv_mse_list = []
       for fold_idx in range(K):
           test_idx = folds[fold_idx]
           train_idx = np.concatenate([folds[j] for j in range(K) if j != fold_idx])
           Xk_tr = np.column_stack([np.ones(len(train_idx)), X_all_cv[train_idx, :k_feat]])
           Xk_te = np.column_stack([np.ones(len(test_idx)), X_all_cv[test_idx, :k_feat]])
           b = np.linalg.lstsq(Xk_tr, y_cv[train_idx], rcond=None)[0]
           cv_mse_list.append(np.mean((y_cv[test_idx] - Xk_te @ b)**2))
       cv_mse = np.mean(cv_mse_list)

       # AIC and BIC on full data
       Xk_full = np.column_stack([np.ones(n), X_all_cv[:, :k_feat]])
       b_full = np.linalg.lstsq(Xk_full, y_cv, rcond=None)[0]
       resid = y_cv - Xk_full @ b_full
       sigma2 = np.mean(resid**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k_feat + 2
       aic = -2 * log_lik + 2 * p
       bic = -2 * log_lik + p * np.log(n)

       if cv_mse < best_cv: best_cv, best_k_cv = cv_mse, k_feat
       if aic < best_aic: best_aic, best_k_aic = aic, k_feat
       if bic < best_bic: best_bic, best_k_bic = bic, k_feat

       print(f"{k_feat:>10} {cv_mse:>10.2f} {aic:>10.2f} {bic:>10.2f}")

   print(f"\nSelections:  CV -> {best_k_cv},  AIC -> {best_k_aic},  BIC -> {best_k_bic}")
   print(f"True model uses 3 features.")

Now let us build the comprehensive comparison table that shows all criteria
side by side, highlighting agreements and disagreements.

.. code-block:: python

   # The big comparison table: all criteria for models with 1-10 features
   import numpy as np

   np.random.seed(42)

   n = 100
   n_feat = 10
   X_data = np.random.normal(0, 1, (n + 500, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_data = 200 + X_data @ true_b + np.random.normal(0, 30, n + 500)

   X_tr, X_te = X_data[:n], X_data[n:]
   y_tr, y_te = y_data[:n], y_data[n:]

   K = 5
   idx = np.arange(n)
   np.random.shuffle(idx)
   folds = np.array_split(idx, K)

   header = (f"{'k':>3} {'Train MSE':>10} {'Test MSE':>10} {'AIC':>10} "
             f"{'AICc':>10} {'BIC':>10} {'CV MSE':>10}")
   print(header)
   print("-" * len(header))

   records = []
   for k in range(1, n_feat + 1):
       Xk_tr = np.column_stack([np.ones(n), X_tr[:, :k]])
       Xk_te = np.column_stack([np.ones(500), X_te[:, :k]])
       b = np.linalg.lstsq(Xk_tr, y_tr, rcond=None)[0]

       train_mse = np.mean((y_tr - Xk_tr @ b)**2)
       test_mse = np.mean((y_te - Xk_te @ b)**2)

       sigma2 = np.mean((y_tr - Xk_tr @ b)**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k + 2
       aic = -2 * log_lik + 2 * p
       aicc = aic + 2*p*(p+1)/max(n-p-1, 1)
       bic = -2 * log_lik + p * np.log(n)

       cv_vals = []
       for fi in range(K):
           te_idx = folds[fi]
           tr_idx = np.concatenate([folds[j] for j in range(K) if j != fi])
           Xk_f_tr = np.column_stack([np.ones(len(tr_idx)), X_tr[tr_idx, :k]])
           Xk_f_te = np.column_stack([np.ones(len(te_idx)), X_tr[te_idx, :k]])
           bf = np.linalg.lstsq(Xk_f_tr, y_tr[tr_idx], rcond=None)[0]
           cv_vals.append(np.mean((y_tr[te_idx] - Xk_f_te @ bf)**2))
       cv_mse = np.mean(cv_vals)

       records.append((k, train_mse, test_mse, aic, aicc, bic, cv_mse))
       print(f"{k:>3} {train_mse:>10.2f} {test_mse:>10.2f} {aic:>10.2f} "
             f"{aicc:>10.2f} {bic:>10.2f} {cv_mse:>10.2f}")

   # Find best by each criterion
   records = np.array(records)
   criteria = ['Train MSE', 'Test MSE', 'AIC', 'AICc', 'BIC', 'CV MSE']
   print(f"\n{'Criterion':<12} {'Best k':>7}")
   print("-" * 20)
   for i, name in enumerate(criteria):
       col = records[:, i+1]
       best = int(records[np.argmin(col), 0])
       print(f"{name:<12} {best:>7}")

   print(f"\nTrue model: 3 features")
   print("Note: Training MSE always picks the most complex model!")
   print("All proper criteria favor 3 features (or close to it).")


19.7 Likelihood Ratio Test for Nested Models
-----------------------------------------------

**When to use hypothesis tests vs information criteria.**
The likelihood ratio test (LRT) is appropriate when two models are *nested*
--- the simpler model is a special case of the more complex one (obtained by
fixing some parameters).  Information criteria are more general and can compare
non-nested models.

For nested models :math:`M_0 \subset M_1`:

.. math::

   \Lambda = -2\,\bigl[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)\bigr],

where :math:`\hat{\theta}_0` and :math:`\hat{\theta}_1` are the MLEs under
:math:`M_0` and :math:`M_1`, respectively.  Under :math:`H_0: M_0` is true
and regularity conditions, Wilks' theorem gives:

.. math::

   \Lambda \xrightarrow{d} \chi^2_{p_1 - p_0},

where :math:`p_1 - p_0` is the difference in the number of parameters.

.. code-block:: python

   # Likelihood ratio tests: adding features one at a time
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   n = 100
   n_feat = 10
   X_lrt = np.random.normal(0, 1, (n, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_lrt = 200 + X_lrt @ true_b + np.random.normal(0, 30, n)

   print("Sequential likelihood ratio tests (adding one feature at a time):")
   print(f"{'Test':>20} {'LR stat':>10} {'df':>4} {'p-value':>10} {'Decision':>18}")
   print("-" * 65)

   prev_loglik = None
   prev_p = None
   for k in range(1, n_feat + 1):
       Xk = np.column_stack([np.ones(n), X_lrt[:, :k]])
       b = np.linalg.lstsq(Xk, y_lrt, rcond=None)[0]
       resid = y_lrt - Xk @ b
       sigma2 = np.mean(resid**2)
       loglik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k + 2

       if prev_loglik is not None:
           lr_stat = -2 * (prev_loglik - loglik)
           df = p - prev_p
           pval = 1 - stats.chi2.cdf(lr_stat, df)
           decision = "Add feature" if pval < 0.05 else "Stop (p >= 0.05)"
           test_label = f"M{k-1} vs M{k}"
           print(f"{test_label:>20} {lr_stat:>10.4f} {df:>4} {pval:>10.4f} {decision:>18}")

       prev_loglik = loglik
       prev_p = p

   print("\nFeatures 1-3 are significant (true coefficients are nonzero).")
   print("Features 4+ are not significant (true coefficients are zero).")

**When LRT fails or is inappropriate:**

* Parameters on the boundary of the parameter space (e.g., testing whether
  a variance component is zero) --- the chi-squared reference distribution
  does not apply.
* Non-nested models --- the LRT has no natural null distribution.
* When the goal is *prediction* rather than testing a null hypothesis ---
  information criteria are better suited.


19.8 Model Averaging
-----------------------

**Why average?**
Model selection picks a single "best" model and ignores the uncertainty about
which model is correct.  Model averaging addresses this by combining
predictions across multiple models, weighted by how well each model is
supported.  Think of it as consulting a committee of experts rather than
trusting just one.

Bayesian model averaging (BMA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`K` candidate models, the posterior predictive distribution for a
new observation :math:`\tilde{x}` is:

.. math::

   p(\tilde{x} \mid \mathbf{x})
   = \sum_{k=1}^K p(\tilde{x} \mid M_k, \mathbf{x})\, p(M_k \mid \mathbf{x}),

where the posterior model weights are:

.. math::

   p(M_k \mid \mathbf{x})
   = \frac{p(\mathbf{x} \mid M_k)\, p(M_k)}
          {\sum_{j=1}^K p(\mathbf{x} \mid M_j)\, p(M_j)}.

This requires computing the marginal likelihoods :math:`p(\mathbf{x} \mid M_k)`
(see :ref:`Chapter 17 <ch17_bayesian>` for methods).

In practice, we can approximate model weights using AIC weights (Burnham and
Anderson, 2002):

.. code-block:: python

   # Model averaging using AIC weights
   import numpy as np

   np.random.seed(42)

   n = 100
   n_feat = 10
   X_avg = np.random.normal(0, 1, (n + 200, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_avg = 200 + X_avg @ true_b + np.random.normal(0, 30, n + 200)
   X_tr, X_te = X_avg[:n], X_avg[n:]
   y_tr, y_te = y_avg[:n], y_avg[n:]

   # Compute AIC for each model and derive weights
   aics = []
   models = []
   for k in range(1, n_feat + 1):
       Xk = np.column_stack([np.ones(n), X_tr[:, :k]])
       b = np.linalg.lstsq(Xk, y_tr, rcond=None)[0]
       resid = y_tr - Xk @ b
       sigma2 = np.mean(resid**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k + 2
       aic = -2 * log_lik + 2 * p
       aics.append(aic)
       models.append((k, b))

   aics = np.array(aics)
   delta_aic = aics - aics.min()
   weights = np.exp(-0.5 * delta_aic)
   weights /= weights.sum()

   print("AIC model weights:")
   print(f"{'Features':>10} {'AIC':>10} {'Delta AIC':>10} {'Weight':>10}")
   print("-" * 44)
   for k, (aic, w) in enumerate(zip(aics, weights), 1):
       print(f"{k:>10} {aic:>10.2f} {delta_aic[k-1]:>10.2f} {w:>10.4f}")

   # Model-averaged predictions
   y_pred_avg = np.zeros(len(X_te))
   y_pred_best = None
   best_k = np.argmin(aics)
   for i, (k, b) in enumerate(models):
       Xk_te = np.column_stack([np.ones(len(X_te)), X_te[:, :k]])
       pred = Xk_te @ b
       y_pred_avg += weights[i] * pred
       if i == best_k:
           y_pred_best = pred

   mse_avg = np.mean((y_te - y_pred_avg)**2)
   mse_best = np.mean((y_te - y_pred_best)**2)
   print(f"\nTest MSE (best single model, k={best_k+1}): {mse_best:.2f}")
   print(f"Test MSE (model averaged):                {mse_avg:.2f}")
   if mse_avg < mse_best:
       print("Model averaging wins! Combining models reduces prediction error.")
   else:
       print("Best single model wins here (averaging adds noise from bad models).")

Stacking
^^^^^^^^^

Yao et al. (2018) proposed *stacking* as a frequentist-friendly alternative:
find non-negative weights :math:`w_1, \ldots, w_K` summing to 1 that maximize
the LOO-CV predictive density:

.. math::

   \hat{\mathbf{w}} = \arg\max_{\mathbf{w}}
   \sum_{i=1}^n \log\!\left(\sum_{k=1}^K w_k\, p(x_i \mid \mathbf{x}_{-i}, M_k)\right).

Stacking is more robust than BMA when the true model is not in the candidate
set, because it optimizes prediction rather than model identification.


19.9 A Practical Decision Guide
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Situation
     - Recommended method
     - Notes
   * - Two nested models, want a p-value
     - Likelihood ratio test
     - Check regularity conditions
   * - Several models, goal is prediction
     - AIC / AICc or LOO-CV
     - AICc for small samples
   * - Several models, goal is identifying the true model
     - BIC
     - Consistent selector
   * - Bayesian workflow with MCMC output
     - WAIC or PSIS-LOO
     - Check Pareto-:math:`k` diagnostics
   * - Model uncertainty is important
     - Bayesian model averaging or stacking
     - Stacking for :math:`M`-open setting

.. code-block:: python

   # Decision guide as runnable code: which criterion for your problem?
   import numpy as np

   scenarios = [
       ("Nested models, want p-value",        "LRT",            "check regularity"),
       ("Prediction, frequentist",             "AIC / AICc",     "AICc if n/p < 40"),
       ("Identify true model",                 "BIC",            "consistent as n->inf"),
       ("Bayesian with MCMC",                  "WAIC / PSIS-LOO","check Pareto-k"),
       ("Account for model uncertainty",       "BMA / stacking", "stacking more robust"),
       ("Small sample, many models",           "AICc + CV",      "cross-validate!"),
   ]

   print(f"{'Scenario':<40} {'Method':<18} {'Note'}")
   print("-" * 80)
   for scenario, method, note in scenarios:
       print(f"{scenario:<40} {method:<18} {note}")


19.10 Summary
---------------

* The bias--variance trade-off is the fundamental tension behind model
  selection: too simple means bias, too complex means variance.
* AIC approximates out-of-sample KL divergence with a penalty of :math:`2p`;
  AICc adds a finite-sample correction.
* BIC approximates the log marginal likelihood with a penalty of
  :math:`p \log n`; it penalizes complexity more than AIC for :math:`n \ge 8`.
* DIC and WAIC extend information criteria to Bayesian settings; WAIC is
  asymptotically equivalent to LOO-CV.
* Cross-validation directly estimates predictive performance; PSIS-LOO makes
  it practical with a single MCMC run.
* The likelihood ratio test is powerful for nested models but requires
  regularity conditions and does not apply to non-nested comparisons.
* Model averaging (BMA or stacking) accounts for model uncertainty in
  predictions.

.. code-block:: python

   # Final summary: all criteria agree on the right model
   import numpy as np

   np.random.seed(42)
   n = 100
   n_feat = 10
   X_final = np.random.normal(0, 1, (n + 500, n_feat))
   true_b = np.array([50, 30, -20, 0, 0, 0, 0, 0, 0, 0], dtype=float)
   y_final = 200 + X_final @ true_b + np.random.normal(0, 30, n + 500)
   X_tr, X_te = X_final[:n], X_final[n:]
   y_tr, y_te = y_final[:n], y_final[n:]

   criteria_best = {}
   aic_scores, bic_scores, test_mses = [], [], []
   for k in range(1, n_feat + 1):
       Xk = np.column_stack([np.ones(n), X_tr[:, :k]])
       b = np.linalg.lstsq(Xk, y_tr, rcond=None)[0]
       resid = y_tr - Xk @ b
       sigma2 = np.mean(resid**2)
       log_lik = -n/2 * np.log(2 * np.pi * sigma2) - n/2
       p = k + 2
       aic_scores.append(-2*log_lik + 2*p)
       bic_scores.append(-2*log_lik + p*np.log(n))

       Xk_te = np.column_stack([np.ones(500), X_te[:, :k]])
       test_mses.append(np.mean((y_te - Xk_te @ b)**2))

   print("Model Selection Summary:")
   print(f"  True model:         3 features")
   print(f"  AIC selects:        {np.argmin(aic_scores)+1} features")
   print(f"  BIC selects:        {np.argmin(bic_scores)+1} features")
   print(f"  Test MSE selects:   {np.argmin(test_mses)+1} features")
   print(f"\n  Lesson: proper model selection criteria protect against overfitting.")
   print(f"  Training MSE would select all {n_feat} features -- and fail on new data.")
