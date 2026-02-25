.. _ch7_multivariate:

==========================================
Chapter 7 --- Multivariate Likelihoods
==========================================

Moving from scalar to vector- and matrix-valued random variables introduces
new mathematical machinery---matrix calculus, determinants, and traces---but
the core ideas of likelihood, score, information, and maximum-likelihood
estimation carry over intact. This chapter treats the most important
multivariate distributions and the calculus tools required for their
derivations.

If you have made it through the univariate distributions in
:ref:`ch5_discrete` and :ref:`ch6_continuous`, the good news is that the
*logic* is identical here. The only difference is notation: scalars become
vectors, division becomes matrix inversion, and squares become quadratic forms.
Let's build up the tools and then put them to work.

.. contents:: Topics in this chapter
   :local:
   :depth: 1
   :class: this-will-duplicate-information-and-it-is-still-useful-here


.. _sec_matrix_calculus:

7.1 Matrix Calculus Essentials
===============================

Before tackling multivariate likelihoods we collect the key matrix-calculus
identities that appear repeatedly in the derivations below. The reader who is
already comfortable with these results can skip ahead to
:ref:`sec_mvn`.

Think of this section as your reference card. You will find yourself coming back
to these identities every time you differentiate a multivariate log-likelihood.

Notation
--------

* :math:`\mathbf{x}` --- a column vector in :math:`\mathbb{R}^p`.
* :math:`\mathbf{A}` --- a :math:`p \times p` matrix.
* :math:`\text{tr}(\mathbf{A})` --- the trace (sum of diagonal entries).
* :math:`|\mathbf{A}|` --- the determinant.
* :math:`\mathbf{A}^{-1}` --- the matrix inverse (when it exists).
* :math:`\mathbf{A}^\top` --- the transpose.

Key identities
---------------

1. **Derivative of a scalar quadratic form.** For a symmetric, positive
   definite matrix :math:`\boldsymbol{\Sigma}`:

   .. math::

      \frac{\partial}{\partial\boldsymbol{\mu}}
        (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}
        (\mathbf{x} - \boldsymbol{\mu})
      = -2\,\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}).

   This is the multivariate analogue of :math:`d(x - \mu)^2/d\mu = -2(x - \mu)`.

2. **Derivative of log-determinant.** For a positive definite matrix
   :math:`\boldsymbol{\Sigma}`:

   .. math::

      \frac{\partial}{\partial\boldsymbol{\Sigma}}
        \ln|\boldsymbol{\Sigma}|
      = \boldsymbol{\Sigma}^{-1}.

   Here the derivative is taken in the sense that
   :math:`[\partial\ln|\boldsymbol{\Sigma}|/\partial\boldsymbol{\Sigma}]_{jk}`
   is the derivative with respect to :math:`\Sigma_{jk}`.

   .. admonition:: Intuition

      This identity is the multivariate version of :math:`d\ln\sigma^2/d\sigma^2
      = 1/\sigma^2`. The inverse plays the role of the reciprocal.

3. **Derivative of inverse times vector.**

   .. math::

      \frac{\partial}{\partial\boldsymbol{\Sigma}}
        \text{tr}\!\left(\boldsymbol{\Sigma}^{-1}\mathbf{A}\right)
      = -\boldsymbol{\Sigma}^{-1}\mathbf{A}\,\boldsymbol{\Sigma}^{-1}.

4. **Derivative of trace of product.**

   .. math::

      \frac{\partial}{\partial\mathbf{A}}\text{tr}(\mathbf{A}\mathbf{B})
      = \mathbf{B}^\top.

5. **Quadratic form as trace.** For vectors :math:`\mathbf{a}` and
   :math:`\mathbf{b}`:

   .. math::

      \mathbf{a}^\top \mathbf{B} \mathbf{a}
      = \text{tr}(\mathbf{B}\,\mathbf{a}\mathbf{a}^\top).

   This identity is used constantly to convert sums of quadratic forms into a
   single trace expression.

6. **Sum of outer products.** Given observations
   :math:`\mathbf{x}_1, \dots, \mathbf{x}_n` with sample mean
   :math:`\bar{\mathbf{x}}`:

   .. math::

      \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})
        (\mathbf{x}_i - \boldsymbol{\mu})^\top
      = \mathbf{S}_n + n(\bar{\mathbf{x}} - \boldsymbol{\mu})
        (\bar{\mathbf{x}} - \boldsymbol{\mu})^\top,

   where :math:`\mathbf{S}_n = \sum_{i=1}^{n}
   (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top`
   is the scatter matrix.


.. _sec_mvn:

7.2 Multivariate Normal Distribution
======================================

Motivation
----------

The Multivariate Normal (MVN) is the cornerstone of multivariate statistics.
It arises from the multivariate Central Limit Theorem, serves as the default
model for continuous vector data, and underpins linear regression, principal
component analysis, factor analysis, and Gaussian processes.

You can think of the MVN as the natural extension of the univariate Normal to
higher dimensions. The mean becomes a vector, the variance becomes a covariance
matrix, and the bell curve becomes a bell-shaped ellipsoid.

PDF
---

Let :math:`\mathbf{x} \in \mathbb{R}^p` with mean
:math:`\boldsymbol{\mu} \in \mathbb{R}^p` and :math:`p \times p` positive
definite covariance matrix :math:`\boldsymbol{\Sigma}`:

.. math::

   f(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})
     = (2\pi)^{-p/2}\,|\boldsymbol{\Sigma}|^{-1/2}
       \exp\!\left(
         -\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top
         \boldsymbol{\Sigma}^{-1}
         (\mathbf{x} - \boldsymbol{\mu})
       \right).

The exponent :math:`(\mathbf{x} - \boldsymbol{\mu})^\top
\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})` is the squared
**Mahalanobis distance** from :math:`\mathbf{x}` to the mean. It generalises
:math:`(x - \mu)^2/\sigma^2` by accounting for correlations between components.

**The Mahalanobis distance in practice.**
A hospital measures three blood markers on each patient.  Raw Euclidean distance
from the population mean treats all markers equally and ignores correlations.
The Mahalanobis distance rescales by the covariance, so a patient who is 2
standard deviations away on a *correlated* pair of markers is properly flagged
as unusual.  Under a true MVN, the squared Mahalanobis distance follows a
:math:`\chi^2_p` distribution, giving a principled threshold for outlier
detection.

.. code-block:: python

   # Mahalanobis distance: outlier detection under the MVN
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   p = 3
   n = 500
   mu = np.array([5.0, 3.0, 8.0])
   Sigma = np.array([[2.0, 0.8, 0.3],
                      [0.8, 1.5, 0.1],
                      [0.3, 0.1, 1.0]])

   # Generate "healthy" patients from MVN
   X = np.random.multivariate_normal(mu, Sigma, size=n)

   # Plant 5 outlier patients
   outliers = np.array([[12.0, 7.0, 4.0],
                         [1.0, -2.0, 14.0],
                         [10.0, 8.0, 11.0],
                         [0.0, 0.0, 0.0],
                         [9.0, 6.0, 12.0]])
   X_all = np.vstack([X, outliers])

   # Compute Mahalanobis distances
   mu_hat = X.mean(axis=0)
   Sigma_hat = np.cov(X, rowvar=False)
   Sigma_inv = np.linalg.inv(Sigma_hat)

   diff = X_all - mu_hat
   mahal_sq = np.sum(diff @ Sigma_inv * diff, axis=1)  # vectorized quadratic form

   # Under MVN, mahal^2 ~ chi2(p) -> threshold at 99th percentile
   threshold = stats.chi2.ppf(0.99, df=p)

   n_flagged_normal = np.sum(mahal_sq[:n] > threshold)
   n_flagged_outlier = np.sum(mahal_sq[n:] > threshold)

   print(f"Chi-squared threshold (p={p}, 99%): {threshold:.2f}")
   print(f"Normal patients flagged:  {n_flagged_normal}/{n} "
         f"(expect ~{n*0.01:.0f})")
   print(f"Planted outliers flagged: {n_flagged_outlier}/{len(outliers)}")
   print(f"\nMahalanobis^2 of outliers: {np.round(mahal_sq[n:], 1)}")
   print(f"Euclidean distances:       "
         f"{np.round(np.sqrt(np.sum(diff[n:]**2, axis=1)), 1)}")
   print("Note: Mahalanobis accounts for correlations; Euclidean does not.")

Likelihood
----------

For :math:`n` i.i.d. observations
:math:`\mathbf{x}_1, \dots, \mathbf{x}_n`:

.. math::

   L(\boldsymbol{\mu}, \boldsymbol{\Sigma})
     = (2\pi)^{-np/2}\,|\boldsymbol{\Sigma}|^{-n/2}
       \exp\!\left(
         -\tfrac{1}{2}\sum_{i=1}^{n}
         (\mathbf{x}_i - \boldsymbol{\mu})^\top
         \boldsymbol{\Sigma}^{-1}
         (\mathbf{x}_i - \boldsymbol{\mu})
       \right).

Log-likelihood
--------------

Using the trace identity :math:`\mathbf{a}^\top\mathbf{B}\mathbf{a} =
\text{tr}(\mathbf{B}\mathbf{a}\mathbf{a}^\top)` to collect the sum:

.. math::

   \ell(\boldsymbol{\mu}, \boldsymbol{\Sigma})
     = -\frac{np}{2}\ln(2\pi)
       - \frac{n}{2}\ln|\boldsymbol{\Sigma}|
       - \frac{1}{2}\text{tr}\!\left(
           \boldsymbol{\Sigma}^{-1}\sum_{i=1}^{n}
           (\mathbf{x}_i - \boldsymbol{\mu})
           (\mathbf{x}_i - \boldsymbol{\mu})^\top
         \right).

Define the matrix

.. math::

   \mathbf{W}(\boldsymbol{\mu})
     = \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})
       (\mathbf{x}_i - \boldsymbol{\mu})^\top,

so that

.. math::

   \ell = -\frac{np}{2}\ln(2\pi) - \frac{n}{2}\ln|\boldsymbol{\Sigma}|
          - \frac{1}{2}\text{tr}\!\left(\boldsymbol{\Sigma}^{-1}\mathbf{W}\right).

The entire dataset is summarised by two sufficient statistics: the sample mean
:math:`\bar{\mathbf{x}}` and the scatter matrix :math:`\mathbf{W}`. No matter
how many observations you have, these are all you need.

Score for :math:`\boldsymbol{\mu}`
------------------------------------

Using identity 1 from :ref:`sec_matrix_calculus`:

.. math::

   \frac{\partial\ell}{\partial\boldsymbol{\mu}}
     = \boldsymbol{\Sigma}^{-1}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu}).

Setting to zero (and noting :math:`\boldsymbol{\Sigma}^{-1}` is invertible):

.. math::

   \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu}) = \mathbf{0}
   \;\Longrightarrow\;
   \hat{\boldsymbol{\mu}} = \bar{\mathbf{x}}
     = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i.

Just as in the univariate case, the MLE for the mean is the sample mean---now
a vector.

Score for :math:`\boldsymbol{\Sigma}`
---------------------------------------

Using identities 2 and 3:

.. math::

   \frac{\partial\ell}{\partial\boldsymbol{\Sigma}}
     = -\frac{n}{2}\boldsymbol{\Sigma}^{-1}
       + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{W}\,\boldsymbol{\Sigma}^{-1}.

Setting to zero and multiplying on the right by :math:`\boldsymbol{\Sigma}`:

.. math::

   \frac{n}{2}\boldsymbol{\Sigma}^{-1}
     = \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{W}\,\boldsymbol{\Sigma}^{-1}
   \;\Longrightarrow\;
   n\mathbf{I} = \mathbf{W}\,\boldsymbol{\Sigma}^{-1}
   \;\Longrightarrow\;
   \hat{\boldsymbol{\Sigma}} = \frac{1}{n}\mathbf{W}(\hat{\boldsymbol{\mu}}).

Evaluating :math:`\mathbf{W}` at :math:`\hat{\boldsymbol{\mu}} = \bar{\mathbf{x}}`:

.. math::

   \hat{\boldsymbol{\Sigma}}
     = \frac{1}{n}\sum_{i=1}^{n}
       (\mathbf{x}_i - \bar{\mathbf{x}})
       (\mathbf{x}_i - \bar{\mathbf{x}})^\top
     = \frac{1}{n}\mathbf{S}_n.

As in the univariate case the MLE divides by :math:`n`, not :math:`n-1`.

**Computing the MLE and verifying the score equations.**
The score equations tell us that at the MLE, the gradient of the log-likelihood
must be zero.  This is the multivariate analogue of
:math:`\partial\ell/\partial\mu = 0` and :math:`\partial\ell/\partial\sigma = 0`
from the univariate Normal.  Let us generate MVN data, compute the MLE, and
verify that the score is indeed zero at the estimates.

.. code-block:: python

   # MVN MLE: estimate mean and covariance, verify score = 0
   import numpy as np

   np.random.seed(42)
   p = 3
   n = 200
   mu_true = np.array([1.0, -2.0, 3.0])
   Sigma_true = np.array([[2.0, 0.5, 0.3],
                           [0.5, 1.5, -0.2],
                           [0.3, -0.2, 1.0]])

   data = np.random.multivariate_normal(mu_true, Sigma_true, size=n)

   # MLE for mean
   mu_hat = data.mean(axis=0)

   # MLE for covariance (divides by n, not n-1)
   centered = data - mu_hat
   Sigma_hat = centered.T @ centered / n

   print("True mean:", mu_true)
   print("MLE mean: ", np.round(mu_hat, 4))
   print("\nTrue Sigma:\n", Sigma_true)
   print("\nMLE Sigma:\n", np.round(Sigma_hat, 4))

   # --- Verify score = 0 at the MLE ---
   # Score for mu: Sigma_inv @ sum(x_i - mu)
   Sigma_inv = np.linalg.inv(Sigma_hat)
   score_mu = Sigma_inv @ np.sum(data - mu_hat, axis=0)
   print(f"\nScore for mu at MLE: {np.round(score_mu, 12)}")
   print(f"  (should be zero vector)")

   # Score for Sigma: -n/2 * Sigma_inv + 1/2 * Sigma_inv @ W @ Sigma_inv
   W = centered.T @ centered  # = n * Sigma_hat
   score_Sigma = -n/2 * Sigma_inv + 0.5 * Sigma_inv @ W @ Sigma_inv
   print(f"\nScore for Sigma at MLE (Frobenius norm): "
         f"{np.linalg.norm(score_Sigma):.2e}")
   print(f"  (should be ~0)")

.. admonition:: Real-World Example

   **Portfolio returns.** A financial analyst models the daily returns of 3
   stocks as a Multivariate Normal to estimate expected returns and the
   covariance structure (which determines portfolio risk).

   .. code-block:: text

      Scenario: 252 trading days of daily returns for 3 stocks
      Data: n x 3 matrix of log-returns
      Goal: estimate mean return vector and covariance matrix
      Approach: MVN(mu, Sigma) -> MLE gives sample mean and sample covariance
      Use: portfolio optimisation, Value-at-Risk

   .. code-block:: python

      # Portfolio returns: MVN MLE for mean and covariance
      import numpy as np

      np.random.seed(42)
      n_days = 252  # one trading year
      p = 3         # three stocks

      # Simulate realistic daily log-returns
      mu_true = np.array([0.0005, 0.0003, 0.0008])  # daily expected returns
      Sigma_true = np.array([
          [0.0004, 0.0001, 0.00015],
          [0.0001, 0.0003, 0.00005],
          [0.00015, 0.00005, 0.0005]
      ])

      returns = np.random.multivariate_normal(mu_true, Sigma_true, size=n_days)

      mu_hat = returns.mean(axis=0)
      Sigma_hat = (returns - mu_hat).T @ (returns - mu_hat) / n_days

      # Annualised estimates
      mu_annual = mu_hat * 252
      vol_annual = np.sqrt(np.diag(Sigma_hat) * 252)
      corr = Sigma_hat / np.sqrt(np.outer(np.diag(Sigma_hat),
                                           np.diag(Sigma_hat)))

      print("Annualised expected returns:", np.round(mu_annual * 100, 2), "%")
      print("Annualised volatilities:    ", np.round(vol_annual * 100, 2), "%")
      print("\nCorrelation matrix:\n", np.round(corr, 3))

Fisher information matrix
--------------------------

The Fisher information for a single observation is a matrix indexed by the
:math:`p` entries of :math:`\boldsymbol{\mu}` and the
:math:`p(p+1)/2` unique entries of :math:`\boldsymbol{\Sigma}`. In block
form:

.. math::

   \mathcal{I}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
     = \begin{pmatrix}
         \boldsymbol{\Sigma}^{-1} & \mathbf{0} \\
         \mathbf{0} & \tfrac{1}{2}
           (\boldsymbol{\Sigma}^{-1} \otimes \boldsymbol{\Sigma}^{-1})
       \end{pmatrix},

where :math:`\otimes` denotes the Kronecker product (with appropriate
duplication-matrix adjustments for the symmetric parametrisation). The key
point is that the mean and covariance blocks are **orthogonal** in the
information sense, meaning :math:`\bar{\mathbf{x}}` and
:math:`\hat{\boldsymbol{\Sigma}}` are asymptotically independent.

.. admonition:: Why does this matter?

   The block-diagonal structure means that your estimate of the mean vector is
   equally good whether or not you know the covariance, and vice versa. This
   orthogonality is a special property of the MVN that does *not* hold for most
   other multivariate distributions.

**Verifying the block-diagonal structure.**
The Fisher information matrix of the MVN is block-diagonal: the cross-information
between :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Sigma}` is zero.  We
can verify this numerically by computing the empirical Fisher information from
simulated score vectors and checking that the off-diagonal block is negligible.

.. code-block:: python

   # MVN Fisher information: verify block-diagonal structure
   import numpy as np

   np.random.seed(42)
   p = 2  # keep small for clear display
   n_sim = 50_000
   mu = np.array([1.0, -1.0])
   Sigma = np.array([[2.0, 0.5],
                      [0.5, 1.0]])
   Sigma_inv = np.linalg.inv(Sigma)

   # Compute score vectors for many single observations
   scores = []
   for _ in range(n_sim):
       x = np.random.multivariate_normal(mu, Sigma)

       # Score for mu: Sigma_inv @ (x - mu), length p
       s_mu = Sigma_inv @ (x - mu)

       # Score for vech(Sigma): use the unique upper-triangle entries
       # d ell / d Sigma = -1/2 Sigma_inv + 1/2 Sigma_inv (x-mu)(x-mu)^T Sigma_inv
       outer = np.outer(x - mu, x - mu)
       dL_dSigma = -0.5 * Sigma_inv + 0.5 * Sigma_inv @ outer @ Sigma_inv

       # Extract upper triangle (unique entries for symmetric matrix)
       idx = np.triu_indices(p)
       s_sigma = dL_dSigma[idx]

       scores.append(np.concatenate([s_mu, s_sigma]))

   scores = np.array(scores)

   # Empirical Fisher = Cov(score) = E[score @ score^T] (since E[score]=0)
   empirical_fisher = scores.T @ scores / n_sim

   print("Empirical Fisher information matrix:")
   print(np.round(empirical_fisher, 3))
   print(f"\nDimensions: {p} (mu) + {p*(p+1)//2} (vech Sigma) "
         f"= {empirical_fisher.shape[0]}")
   print(f"\nOff-diagonal block (mu vs Sigma), should be ~0:")
   print(np.round(empirical_fisher[:p, p:], 4))
   print("\nThis confirms the block-diagonal structure of the Fisher information.")


.. _sec_wishart:

7.3 Wishart Distribution
==========================

Motivation
----------

The Wishart distribution is the multivariate generalisation of the
Chi-squared. If :math:`\mathbf{x}_1, \dots, \mathbf{x}_n` are i.i.d.
:math:`\mathcal{N}_p(\mathbf{0}, \boldsymbol{\Sigma})`, then the scatter
matrix :math:`\mathbf{S} = \sum_{i=1}^{n}\mathbf{x}_i\mathbf{x}_i^\top`
follows a Wishart distribution. It plays a central role in multivariate
hypothesis testing and as the likelihood for covariance matrices.

Think of the Wishart as the distribution of "sample covariance matrices". Just
as the Chi-squared tells you how sample variances behave, the Wishart tells
you how sample covariance matrices behave.

PDF
---

Let :math:`\mathbf{W} \sim \mathcal{W}_p(n, \boldsymbol{\Sigma})` where
:math:`n \ge p` (degrees of freedom) and :math:`\boldsymbol{\Sigma}` is
:math:`p \times p` positive definite:

.. math::

   f(\mathbf{W} \mid n, \boldsymbol{\Sigma})
     = \frac{|\mathbf{W}|^{(n-p-1)/2}}{
         2^{np/2}\,|\boldsymbol{\Sigma}|^{n/2}\,\Gamma_p(n/2)
       }
       \exp\!\left(
         -\tfrac{1}{2}\text{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{W})
       \right),

where :math:`\Gamma_p` is the multivariate gamma function:

.. math::

   \Gamma_p(a) = \pi^{p(p-1)/4}\prod_{j=1}^{p}\Gamma\!\left(a + \frac{1-j}{2}\right).

Log-likelihood
--------------

For a single observation :math:`\mathbf{W}` with known :math:`n`:

.. math::

   \ell(\boldsymbol{\Sigma})
     = \frac{n-p-1}{2}\ln|\mathbf{W}|
       - \frac{n}{2}\ln|\boldsymbol{\Sigma}|
       - \frac{1}{2}\text{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{W})
       + \text{const}.

MLE for :math:`\boldsymbol{\Sigma}`
-------------------------------------

Differentiating with respect to :math:`\boldsymbol{\Sigma}` using the same
identities as for the MVN:

.. math::

   \frac{\partial\ell}{\partial\boldsymbol{\Sigma}}
     = -\frac{n}{2}\boldsymbol{\Sigma}^{-1}
       + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{W}\,\boldsymbol{\Sigma}^{-1}.

Setting to zero:

.. math::

   \hat{\boldsymbol{\Sigma}} = \frac{1}{n}\mathbf{W}.

Connection to MVN MLE
----------------------

When :math:`\mathbf{W} = \sum_{i=1}^{n}(\mathbf{x}_i - \bar{\mathbf{x}})
(\mathbf{x}_i - \bar{\mathbf{x}})^\top` (the scatter matrix from :math:`n`
MVN observations), the Wishart MLE :math:`\mathbf{W}/n` coincides with the
MVN MLE :math:`\hat{\boldsymbol{\Sigma}}`.

**Simulating scatter matrices and verifying the Wishart distribution.**
The Wishart distribution tells us how scatter matrices behave.  If we draw
many samples of size :math:`n` from a MVN, compute the scatter matrix each
time, and look at properties like the expected value and the determinant, the
results should match the Wishart theory.  The expected value of
:math:`\mathbf{W} \sim \mathcal{W}_p(n, \boldsymbol{\Sigma})` is
:math:`n\boldsymbol{\Sigma}`.

.. code-block:: python

   # Wishart: simulate scatter matrices, verify E[W] = n * Sigma
   import numpy as np

   np.random.seed(42)
   p = 3
   n = 20       # degrees of freedom (sample size per scatter matrix)
   n_sim = 5000  # number of scatter matrices to simulate
   Sigma = np.array([[2.0, 0.5, 0.3],
                      [0.5, 1.5, -0.2],
                      [0.3, -0.2, 1.0]])

   # Simulate many scatter matrices
   scatter_matrices = np.zeros((n_sim, p, p))
   log_dets = np.zeros(n_sim)

   for s in range(n_sim):
       X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
       W = X.T @ X  # scatter matrix ~ Wishart(n, Sigma)
       scatter_matrices[s] = W
       log_dets[s] = np.linalg.slogdet(W)[1]

   # Verify E[W] = n * Sigma
   E_W = scatter_matrices.mean(axis=0)
   print("E[W] / n (should approximate Sigma):")
   print(np.round(E_W / n, 3))
   print("\nTrue Sigma:")
   print(Sigma)
   print(f"\nMax absolute error in E[W]/n: {np.max(np.abs(E_W/n - Sigma)):.4f}")

   # Verify log-determinant distribution
   # E[log|W|] = p*log(2) + log|Sigma| + sum(digamma((n+1-j)/2))
   from scipy.special import digamma
   expected_logdet = (p * np.log(2) + np.linalg.slogdet(Sigma)[1]
                      + sum(digamma((n + 1 - j) / 2) for j in range(1, p + 1)))
   print(f"\nE[log|W|]: simulated = {log_dets.mean():.3f}, "
         f"theoretical = {expected_logdet:.3f}")

**So what?** The Wishart distribution is not just a mathematical curiosity.
Every time you report a sample covariance matrix and wonder "how much could
this vary?", the Wishart is the answer.  The simulation above confirms that
the theory correctly predicts the sampling behavior of scatter matrices.


.. _sec_inv_wishart:

7.4 Inverse-Wishart Distribution
==================================

Motivation
----------

The Inverse-Wishart is the distribution of the inverse of a Wishart-distributed
matrix. In Bayesian statistics it serves as the **conjugate prior** for the
covariance matrix of a Multivariate Normal: if
:math:`\boldsymbol{\Sigma} \sim \text{Inv-Wishart}(\nu, \boldsymbol{\Psi})`
and the data are :math:`\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})`,
then the posterior for :math:`\boldsymbol{\Sigma}` is also Inverse-Wishart.

This conjugacy makes the Inverse-Wishart extremely convenient for Bayesian
analysis. The posterior update has the same form as the prior, with the
parameters simply absorbing the data.

PDF
---

Let :math:`\boldsymbol{\Sigma} \sim \text{IW}_p(\nu, \boldsymbol{\Psi})`
where :math:`\nu > p + 1` and :math:`\boldsymbol{\Psi}` is :math:`p \times p`
positive definite:

.. math::

   f(\boldsymbol{\Sigma} \mid \nu, \boldsymbol{\Psi})
     = \frac{|\boldsymbol{\Psi}|^{\nu/2}}{
         2^{\nu p/2}\,\Gamma_p(\nu/2)
       }
       \,|\boldsymbol{\Sigma}|^{-(\nu+p+1)/2}
       \exp\!\left(
         -\tfrac{1}{2}\text{tr}(\boldsymbol{\Psi}\,\boldsymbol{\Sigma}^{-1})
       \right).

Properties
----------

* **Mean:** :math:`E[\boldsymbol{\Sigma}] = \boldsymbol{\Psi}/(\nu - p - 1)`
  for :math:`\nu > p + 1`.
* **Mode:** :math:`\boldsymbol{\Psi}/(\nu + p + 1)`.

Posterior under MVN sampling
-----------------------------

Prior: :math:`\boldsymbol{\Sigma} \sim \text{IW}_p(\nu_0, \boldsymbol{\Psi}_0)`.

Data: :math:`\mathbf{x}_1, \dots, \mathbf{x}_n \mid \boldsymbol{\mu},
\boldsymbol{\Sigma} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})`.

Posterior (with :math:`\boldsymbol{\mu}` known):

.. math::

   \boldsymbol{\Sigma} \mid \text{data}
     \sim \text{IW}_p\!\left(
       \nu_0 + n,\;
       \boldsymbol{\Psi}_0
       + \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})
         (\mathbf{x}_i - \boldsymbol{\mu})^\top
     \right).

This shows how the prior "pseudo-observations" :math:`\nu_0` and
"pseudo-scatter" :math:`\boldsymbol{\Psi}_0` combine with the data scatter
matrix.

**Bayesian conjugate update in action.**
Suppose a lab measures two biomarkers on patients.  Before collecting data, the
researcher has a prior belief about the covariance structure (perhaps from
a pilot study).  After observing new patients, the Inverse-Wishart conjugate
update combines prior and data seamlessly.  We verify that the posterior mean
is a weighted average of the prior mean and the sample covariance.

.. code-block:: python

   # Inverse-Wishart conjugate update for MVN covariance
   import numpy as np

   np.random.seed(42)
   p = 2

   # True covariance (unknown to the analyst)
   Sigma_true = np.array([[1.0, 0.6],
                           [0.6, 2.0]])
   mu_known = np.array([0.0, 0.0])  # assume mean is known

   # Prior: IW(nu_0, Psi_0) with E[Sigma] = Psi_0 / (nu_0 - p - 1)
   nu_0 = 5
   Psi_0 = (nu_0 - p - 1) * np.eye(p)  # prior mean = I
   prior_mean = Psi_0 / (nu_0 - p - 1)
   print("Prior mean for Sigma:")
   print(prior_mean)

   # Observe data
   n = 30
   data = np.random.multivariate_normal(mu_known, Sigma_true, size=n)
   scatter = (data - mu_known).T @ (data - mu_known)

   # Posterior: IW(nu_0 + n, Psi_0 + scatter)
   nu_post = nu_0 + n
   Psi_post = Psi_0 + scatter
   posterior_mean = Psi_post / (nu_post - p - 1)

   # Sample covariance for comparison
   sample_cov = scatter / n

   print(f"\nSample covariance (n={n}):")
   print(np.round(sample_cov, 3))
   print(f"\nPosterior mean for Sigma (nu_post={nu_post}):")
   print(np.round(posterior_mean, 3))
   print(f"\nTrue Sigma:")
   print(Sigma_true)

   # Show it is a weighted combination
   w_prior = (nu_0 - p - 1) / (nu_post - p - 1)
   w_data = n / (nu_post - p - 1)
   weighted = w_prior * prior_mean + w_data * sample_cov
   print(f"\nWeights: prior={w_prior:.3f}, data={w_data:.3f}")
   print(f"Weighted combination:")
   print(np.round(weighted, 3))
   print(f"Match posterior mean: {np.allclose(weighted, posterior_mean)}")

   # Simulate from posterior to get credible intervals
   n_post_samples = 5000
   post_samples_11 = []
   post_samples_12 = []
   for _ in range(n_post_samples):
       # IW(nu, Psi) = inv(Wishart(nu, Psi_inv))
       W = np.random.multivariate_normal(
           np.zeros(p), np.linalg.inv(Psi_post), size=nu_post)
       S = W.T @ W
       Sigma_sample = np.linalg.inv(S)
       post_samples_11.append(Sigma_sample[0, 0])
       post_samples_12.append(Sigma_sample[0, 1])

   post_samples_11 = np.array(post_samples_11)
   post_samples_12 = np.array(post_samples_12)
   print(f"\n95% credible interval for Sigma[1,1]: "
         f"({np.percentile(post_samples_11, 2.5):.3f}, "
         f"{np.percentile(post_samples_11, 97.5):.3f}), "
         f"true={Sigma_true[0,0]}")
   print(f"95% credible interval for Sigma[1,2]: "
         f"({np.percentile(post_samples_12, 2.5):.3f}, "
         f"{np.percentile(post_samples_12, 97.5):.3f}), "
         f"true={Sigma_true[0,1]}")


.. _sec_dirichlet:

7.5 Dirichlet Distribution
============================

Motivation
----------

The Dirichlet distribution is the multivariate generalisation of the Beta. It
is defined on the probability simplex
:math:`\{(p_1, \dots, p_k) : p_j \ge 0,\; \sum p_j = 1\}` and is the
conjugate prior for the Multinomial. It is used in topic modelling (Latent
Dirichlet Allocation), Bayesian nonparametrics, and any setting where one needs
a prior over categorical distributions.

If you need a distribution over *distributions*---a probability distribution
whose samples are themselves probability vectors---the Dirichlet is your
primary tool.

PDF
---

Let :math:`\mathbf{p} = (p_1, \dots, p_k)` lie on the simplex and let
:math:`\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_k)` with each
:math:`\alpha_j > 0`:

.. math::

   f(\mathbf{p} \mid \boldsymbol{\alpha})
     = \frac{\Gamma(\alpha_0)}{\prod_{j=1}^{k}\Gamma(\alpha_j)}
       \prod_{j=1}^{k} p_j^{\alpha_j - 1},

where :math:`\alpha_0 = \sum_{j=1}^{k}\alpha_j` is called the
**concentration**.

.. admonition:: Intuition

   The concentration parameter :math:`\alpha_0` controls how "peaked" the
   distribution is. When :math:`\alpha_0` is large, samples cluster tightly
   around the mean. When :math:`\alpha_0` is small, samples tend toward the
   corners of the simplex (sparse distributions). When all
   :math:`\alpha_j = 1`, you get the Uniform distribution on the simplex.

Likelihood
----------

Given :math:`n` i.i.d. observations
:math:`\mathbf{p}^{(1)}, \dots, \mathbf{p}^{(n)}` on the simplex:

.. math::

   L(\boldsymbol{\alpha})
     = \left[\frac{\Gamma(\alpha_0)}{\prod_j \Gamma(\alpha_j)}\right]^n
       \prod_{i=1}^{n}\prod_{j=1}^{k}
         \bigl(p_j^{(i)}\bigr)^{\alpha_j - 1}.

Log-likelihood
--------------

.. math::

   \ell(\boldsymbol{\alpha})
     = n\!\left[\ln\Gamma(\alpha_0) - \sum_{j=1}^{k}\ln\Gamma(\alpha_j)\right]
       + \sum_{j=1}^{k}(\alpha_j - 1)\sum_{i=1}^{n}\ln p_j^{(i)}.

Score function
--------------

.. math::

   \frac{\partial\ell}{\partial\alpha_j}
     = n\!\left[\psi(\alpha_0) - \psi(\alpha_j)\right]
       + \sum_{i=1}^{n}\ln p_j^{(i)},

where :math:`\psi` is the digamma function. Setting each score to zero:

.. math::

   \psi(\alpha_j) - \psi(\alpha_0)
     = \frac{1}{n}\sum_{i=1}^{n}\ln p_j^{(i)}.

MLE --- Fixed-point iteration
-------------------------------

No closed form exists. The standard method is the **fixed-point iteration**
of Minka (2000). Define :math:`g_j = (1/n)\sum_i \ln p_j^{(i)}`. Then iterate:

.. math::

   \alpha_j^{(\text{new})}
     = \psi^{-1}\!\left(\psi(\alpha_0^{(\text{old})}) + g_j\right),

where :math:`\psi^{-1}` is the inverse digamma function (computed via
Newton's method), and update :math:`\alpha_0 = \sum_j \alpha_j` after each
sweep. This converges reliably for reasonable starting values (e.g., method
of moments).

**Minka's fixed-point iteration with convergence tracking.**
The Dirichlet MLE has no closed form, making it a good case study for iterative
optimization.  We implement the full algorithm, print a convergence table
showing how the estimates improve each iteration, and compare with the true
parameters.

.. code-block:: python

   # Dirichlet MLE: Minka's fixed-point iteration with convergence table
   import numpy as np
   from scipy.special import digamma, polygamma

   np.random.seed(42)
   k = 4
   alpha_true = np.array([2.0, 5.0, 3.0, 1.5])
   n = 300

   # Simulate Dirichlet observations
   data = np.random.dirichlet(alpha_true, size=n)

   # Sufficient statistics
   g = np.log(data).mean(axis=0)

   # Inverse digamma via Newton's method
   def inv_digamma(y, tol=1e-10):
       x = np.exp(y) + 0.5 if y >= -2.22 else -1.0 / (y - digamma(1))
       for _ in range(20):
           x -= (digamma(x) - y) / polygamma(1, x)
       return x

   # Fixed-point iteration with convergence tracking
   alpha_hat = np.ones(k)  # starting values
   print(f"{'Iter':<6} {'alpha_1':<10} {'alpha_2':<10} "
         f"{'alpha_3':<10} {'alpha_4':<10} {'max change':<12}")
   print("-" * 58)

   for iteration in range(50):
       alpha0 = alpha_hat.sum()
       psi_alpha0 = digamma(alpha0)
       alpha_new = np.array([inv_digamma(psi_alpha0 + g[j]) for j in range(k)])
       max_change = np.max(np.abs(alpha_new - alpha_hat))
       alpha_hat = alpha_new

       if iteration < 8 or max_change < 1e-6:
           print(f"{iteration+1:<6} {alpha_hat[0]:<10.4f} {alpha_hat[1]:<10.4f} "
                 f"{alpha_hat[2]:<10.4f} {alpha_hat[3]:<10.4f} {max_change:<12.2e}")

       if max_change < 1e-8:
           print(f"\nConverged in {iteration+1} iterations.")
           break

   print(f"\nTrue alpha:  {alpha_true}")
   print(f"MLE alpha:   {np.round(alpha_hat, 4)}")

   # Verify score ~ 0 at MLE
   alpha0_hat = alpha_hat.sum()
   score = n * (digamma(alpha0_hat) - digamma(alpha_hat)) + n * g
   print(f"\nScore at MLE: {np.round(score, 8)}")
   print(f"  (should be ~0 for each component)")

.. admonition:: Real-World Example

   **Text topic modelling.** In Latent Dirichlet Allocation (LDA), each
   document's topic distribution is modelled as a draw from a Dirichlet.
   Estimating the Dirichlet parameters tells us about the "average" topic
   mixture and how much variation there is across documents.

   .. code-block:: text

      Scenario: 1000 documents, each assigned to a mixture of 5 topics
      Data: for each document, the proportion of words in each topic
      Goal: estimate the Dirichlet concentration parameters
      Approach: Dirichlet(alpha) MLE via Minka's fixed-point iteration

   .. code-block:: python

      # Topic modelling: Dirichlet MLE for document-topic mixtures
      import numpy as np
      from scipy.special import digamma, polygamma

      np.random.seed(42)
      n_docs = 1000
      n_topics = 5
      alpha_true = np.array([0.5, 1.0, 0.3, 0.8, 0.4])  # sparse topics

      topic_mixtures = np.random.dirichlet(alpha_true, size=n_docs)

      # Check sparsity: many documents dominated by 1-2 topics
      max_topic_prob = topic_mixtures.max(axis=1)
      print(f"Mean max topic proportion: {max_topic_prob.mean():.3f}")
      print(f"Proportion with one dominant topic (>0.5): "
            f"{(max_topic_prob > 0.5).mean():.2%}")

      # Sufficient statistics and MLE
      g = np.log(topic_mixtures).mean(axis=0)

      def inv_digamma(y, tol=1e-10):
          x = np.exp(y) + 0.5 if y >= -2.22 else -1.0 / (y - digamma(1))
          for _ in range(20):
              x -= (digamma(x) - y) / polygamma(1, x)
          return x

      alpha_hat = np.ones(n_topics)
      for _ in range(100):
          alpha0 = alpha_hat.sum()
          alpha_hat = np.array([inv_digamma(digamma(alpha0) + g[j])
                                for j in range(n_topics)])

      print(f"\nTrue alpha:  {alpha_true}")
      print(f"MLE alpha:   {np.round(alpha_hat, 4)}")
      print(f"Concentration: {alpha_hat.sum():.2f} "
            f"(true: {alpha_true.sum():.2f})")

Fisher information
------------------

The Fisher information matrix for a single Dirichlet observation is

.. math::

   \mathcal{I}_{jl}
     = \begin{cases}
         n\bigl[\psi'(\alpha_j) - \psi'(\alpha_0)\bigr], & j = l, \\
         -n\,\psi'(\alpha_0), & j \ne l,
       \end{cases}

which can be written compactly as

.. math::

   \mathcal{I}
     = \text{diag}\!\left(\psi'(\alpha_1), \dots, \psi'(\alpha_k)\right)
       - \psi'(\alpha_0)\,\mathbf{1}\mathbf{1}^\top.

The matrix is positive definite because the diagonal elements exceed
:math:`\psi'(\alpha_0)` (a consequence of the strict convexity of
:math:`\ln\Gamma`).


.. _sec_multinomial_mv:

7.6 Multinomial Distribution (Multivariate Treatment)
=======================================================

Motivation
----------

We gave a brief treatment of the Multinomial in :ref:`sec_multinomial` from
the discrete-distribution perspective. Here we revisit it using matrix
notation and the Lagrange-multiplier derivation in full detail, as preparation
for more sophisticated constrained optimisation problems.

This revisit lets you see the same distribution through a different lens.
The result is the same, but the derivation via Lagrange multipliers is a
technique you will use again and again in multivariate settings.

Log-likelihood (restated)
--------------------------

Pooling :math:`n` independent multinomial vectors, each with :math:`m` trials
and :math:`k` categories:

.. math::

   \ell(\mathbf{p}) = \text{const} + \sum_{j=1}^{k} S_j \ln p_j,

where :math:`S_j = \sum_{i=1}^{n} x_j^{(i)}` is the total count in category
:math:`j` and :math:`\sum_j S_j = nm`.

MLE with Lagrange multiplier --- full derivation
---------------------------------------------------

We want to maximise :math:`\ell(\mathbf{p})` subject to
:math:`g(\mathbf{p}) = \sum_{j=1}^{k} p_j - 1 = 0`.

**Step 1.** Form the Lagrangian:

.. math::

   \mathcal{L}(\mathbf{p}, \mu)
     = \sum_{j=1}^{k} S_j \ln p_j - \mu\!\left(\sum_{j=1}^{k} p_j - 1\right).

**Step 2.** Take partial derivatives and set to zero:

.. math::

   \frac{\partial\mathcal{L}}{\partial p_j}
     = \frac{S_j}{p_j} - \mu = 0
   \;\Longrightarrow\;
   p_j = \frac{S_j}{\mu},
   \qquad j = 1, \dots, k.

**Step 3.** Determine :math:`\mu` from the constraint:

.. math::

   \sum_{j=1}^{k} p_j = 1
   \;\Longrightarrow\;
   \sum_{j=1}^{k}\frac{S_j}{\mu} = 1
   \;\Longrightarrow\;
   \mu = \sum_{j=1}^{k} S_j = nm.

**Step 4.** Substitute back:

.. math::

   \hat{p}_j = \frac{S_j}{nm}.

**Step 5.** Verify second-order conditions. The bordered Hessian confirms
this is a maximum because each :math:`\partial^2\mathcal{L}/\partial p_j^2
= -S_j/p_j^2 < 0`.

**Lagrange multiplier MLE with constraint verification.**
The Multinomial MLE is the most natural example of constrained optimization in
statistics: we maximize the log-likelihood subject to the constraint that
probabilities sum to one.  Below we implement the full Lagrange multiplier
approach, compute the MLE, and verify that both the first-order conditions
and the constraint are satisfied.

.. code-block:: python

   # Multinomial MLE via Lagrange multiplier, with constraint verification
   import numpy as np

   np.random.seed(42)
   k = 5           # categories
   m = 10          # trials per observation
   n = 100         # number of observations
   p_true = np.array([0.1, 0.25, 0.35, 0.15, 0.15])

   # Simulate multinomial data
   data = np.random.multinomial(m, p_true, size=n)  # n x k matrix of counts

   # Sufficient statistics: total counts per category
   S = data.sum(axis=0)
   print(f"Total counts S:  {S}")
   print(f"Sum of S (= nm): {S.sum()} (expected: {n*m})")

   # MLE from Lagrange multiplier derivation: p_hat_j = S_j / (nm)
   p_hat = S / (n * m)
   print(f"\nTrue p:     {p_true}")
   print(f"MLE p_hat:  {np.round(p_hat, 4)}")

   # Verify constraint: sum(p_hat) = 1
   print(f"\nConstraint check: sum(p_hat) = {p_hat.sum():.10f}")

   # Verify first-order conditions: S_j / p_hat_j = mu for all j
   # The Lagrange multiplier mu = nm
   mu_lagrange = n * m
   ratios = S / p_hat
   print(f"\nLagrange multiplier mu = nm = {mu_lagrange}")
   print(f"S_j / p_hat_j for each j:     {ratios}")
   print(f"All equal to mu?              {np.allclose(ratios, mu_lagrange)}")

   # Log-likelihood at MLE vs at true p
   def multinomial_loglik(p_vec, counts):
       return np.sum(counts * np.log(p_vec))

   ll_mle = multinomial_loglik(p_hat, S)
   ll_true = multinomial_loglik(p_true, S)
   print(f"\nLog-lik at MLE:    {ll_mle:.4f}")
   print(f"Log-lik at true p: {ll_true:.4f}")
   print(f"Difference:        {ll_mle - ll_true:.4f} (MLE >= true, as expected)")

   # Second-order condition: diagonal of Hessian < 0
   hessian_diag = -S / p_hat**2
   print(f"\nHessian diagonal: {np.round(hessian_diag, 1)}")
   print(f"All negative?     {np.all(hessian_diag < 0)}")

Fisher information matrix (unrestricted parametrisation)
---------------------------------------------------------

Working with the free parameters :math:`p_1, \dots, p_{k-1}` (with
:math:`p_k = 1 - \sum_{j<k} p_j` eliminated), the Fisher information for a
single multinomial observation is

.. math::

   [\mathcal{I}]_{jl}
     = \frac{m\,\delta_{jl}}{p_j} + \frac{m}{p_k},
   \qquad j, l = 1, \dots, k-1.

This can be written as

.. math::

   \mathcal{I}
     = m\!\left[
         \text{diag}(1/p_1, \dots, 1/p_{k-1})
         + \frac{1}{p_k}\,\mathbf{1}\mathbf{1}^\top
       \right].

Derivation:

The log-likelihood for one observation is
:math:`\ell_1 = \sum_{j=1}^{k} x_j \ln p_j`. With :math:`p_k` eliminated:

.. math::

   \frac{\partial\ell_1}{\partial p_j}
     = \frac{x_j}{p_j} - \frac{x_k}{p_k}.

Then

.. math::

   -\frac{\partial^2\ell_1}{\partial p_j\,\partial p_l}
     = \frac{\delta_{jl}\,x_j}{p_j^2} + \frac{x_k}{p_k^2}.

Taking expectations (:math:`E[x_j] = mp_j`, :math:`E[x_k] = mp_k`):

.. math::

   \mathcal{I}_{jl}
     = \frac{m\,\delta_{jl}}{p_j} + \frac{m}{p_k}.


.. _sec_mv_summary:

7.7 Summary
=============

.. list-table:: Multivariate Distributions --- MLEs
   :header-rows: 1
   :widths: 25 40

   * - Distribution
     - MLE
   * - MVN :math:`(\boldsymbol{\mu}, \boldsymbol{\Sigma})`
     - :math:`\hat{\boldsymbol{\mu}} = \bar{\mathbf{x}}`,
       :math:`\hat{\boldsymbol{\Sigma}} = \mathbf{S}_n / n`
   * - Wishart :math:`(\nu, \boldsymbol{\Sigma})`
     - :math:`\hat{\boldsymbol{\Sigma}} = \mathbf{W}/\nu`
   * - Dirichlet :math:`(\boldsymbol{\alpha})`
     - Fixed-point iteration (Minka)
   * - Multinomial :math:`(\mathbf{p})`
     - :math:`\hat{p}_j = S_j / (nm)`

The multivariate Normal is the most commonly encountered multivariate
likelihood. Its MLE has the elegant property that the mean vector and
covariance matrix are estimated independently. The Wishart and Inverse-Wishart
play supporting roles in Bayesian analysis of covariance structures. The
Dirichlet and Multinomial are the workhorses for categorical and compositional
data.

.. admonition:: Common Pitfall

   When :math:`n < p` (more dimensions than observations), the sample
   covariance matrix :math:`\hat{\boldsymbol{\Sigma}}` is singular and cannot
   be inverted. This is the classic "curse of dimensionality" problem. In
   practice, you will need regularisation techniques (shrinkage estimators,
   graphical LASSO) or dimension reduction (PCA) to work with high-dimensional
   covariance estimation.
