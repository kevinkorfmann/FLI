.. _appendix_notation:

====================================
Appendix C: Notation and Glossary
====================================

This appendix provides a centralized reference for all mathematical notation
and terminology used throughout this guide. When a symbol has multiple common
meanings, the one adopted in this text is indicated.


Mathematical Notation
======================

Sets and Spaces
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbb{R}`
     - The set of real numbers
   * - :math:`\mathbb{R}^n`
     - The set of real :math:`n`-dimensional column vectors
   * - :math:`\mathbb{R}^{n \times m}`
     - The set of real :math:`n \times m` matrices
   * - :math:`\mathbb{R}^+`
     - The set of strictly positive real numbers :math:`(0, \infty)`
   * - :math:`\mathbb{Z}`
     - The set of integers
   * - :math:`\mathbb{Z}^+`
     - The set of positive integers :math:`\{1, 2, 3, \ldots\}`
   * - :math:`\mathbb{N}_0`
     - The set of non-negative integers :math:`\{0, 1, 2, \ldots\}`
   * - :math:`\emptyset`
     - The empty set
   * - :math:`\mathcal{X}`
     - The sample space (set of all possible outcomes)
   * - :math:`\Theta`
     - The parameter space
   * - :math:`\mathcal{S}_n^{++}`
     - The cone of :math:`n \times n` symmetric positive definite matrices
   * - :math:`\in`
     - Element of
   * - :math:`\subset, \subseteq`
     - Proper subset, subset (or equal)
   * - :math:`\cup, \cap`
     - Union, intersection
   * - :math:`A^c`
     - Complement of set :math:`A`

Probability Notation
---------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`P(A)`
     - Probability of event :math:`A`
   * - :math:`P(A \mid B)`
     - Conditional probability of :math:`A` given :math:`B`
   * - :math:`P(A \cap B)`
     - Joint probability of :math:`A` and :math:`B`
   * - :math:`f(x)` or :math:`f_X(x)`
     - Probability density function (PDF) of a continuous r.v.
   * - :math:`p(x)` or :math:`p_X(x)`
     - Probability mass function (PMF) of a discrete r.v.
   * - :math:`F(x)` or :math:`F_X(x)`
     - Cumulative distribution function (CDF)
   * - :math:`f(x \mid \theta)`
     - Density of :math:`X` given parameter :math:`\theta`
   * - :math:`f(\mathbf{x} \mid \boldsymbol{\theta})`
     - Joint density of data vector :math:`\mathbf{x}` given parameters
   * - :math:`\sim`
     - "is distributed as" (e.g., :math:`X \sim \mathcal{N}(\mu, \sigma^2)`)
   * - :math:`\stackrel{d}{\to}`
     - Convergence in distribution
   * - :math:`\stackrel{p}{\to}`
     - Convergence in probability
   * - :math:`\stackrel{a.s.}{\to}`
     - Almost sure convergence
   * - :math:`\perp\!\!\!\perp`
     - Statistical independence
   * - :math:`\text{i.i.d.}`
     - Independent and identically distributed

Random Variables and Expectations
----------------------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`X, Y, Z`
     - Random variables (uppercase)
   * - :math:`x, y, z`
     - Observed values / realizations (lowercase)
   * - :math:`\mathbf{X}`
     - Random vector or random matrix
   * - :math:`E[X]` or :math:`\mu`
     - Expected value (mean) of :math:`X`
   * - :math:`E[X \mid Y]`
     - Conditional expectation of :math:`X` given :math:`Y`
   * - :math:`\operatorname{Var}(X)` or :math:`\sigma^2`
     - Variance of :math:`X`
   * - :math:`\operatorname{Cov}(X, Y)`
     - Covariance of :math:`X` and :math:`Y`
   * - :math:`\operatorname{Corr}(X, Y)` or :math:`\rho`
     - Correlation of :math:`X` and :math:`Y`
   * - :math:`\boldsymbol{\Sigma}`
     - Covariance matrix
   * - :math:`M_X(t)`
     - Moment generating function of :math:`X`
   * - :math:`\phi_X(t)`
     - Characteristic function of :math:`X`
   * - :math:`E_\theta[\cdot]`
     - Expectation taken under the distribution indexed by :math:`\theta`

Likelihood and Inference
-------------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`L(\theta)` or :math:`L(\theta ; \mathbf{x})`
     - Likelihood function
   * - :math:`\ell(\theta)` or :math:`\ell(\theta ; \mathbf{x})`
     - Log-likelihood function, :math:`\ell = \log L`
   * - :math:`\hat{\theta}` or :math:`\hat{\theta}_{\text{MLE}}`
     - Maximum likelihood estimator / estimate
   * - :math:`U(\theta)` or :math:`S(\theta)`
     - Score function, :math:`U(\theta) = \partial \ell / \partial \theta`
   * - :math:`\mathcal{I}(\theta)`
     - Fisher information (expected information)
   * - :math:`\mathcal{J}(\theta)` or :math:`J(\hat{\theta})`
     - Observed information, :math:`-\partial^2 \ell / \partial \theta^2`
   * - :math:`\Lambda`
     - Likelihood ratio statistic
   * - :math:`R(\theta)`
     - Profile likelihood or relative likelihood
   * - :math:`\ell_p(\psi)`
     - Profile log-likelihood for parameter of interest :math:`\psi`
   * - :math:`\text{se}(\hat{\theta})`
     - Standard error of estimator :math:`\hat{\theta}`
   * - :math:`\text{AIC}`
     - Akaike Information Criterion, :math:`-2\ell(\hat\theta) + 2p`
   * - :math:`\text{BIC}`
     - Bayesian Information Criterion, :math:`-2\ell(\hat\theta) + p\log n`

Optimization Notation
----------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\nabla f` or :math:`\nabla_{\mathbf{x}} f`
     - Gradient of :math:`f` with respect to :math:`\mathbf{x}`
   * - :math:`\mathbf{H}` or :math:`\nabla^2 f`
     - Hessian matrix (matrix of second partial derivatives)
   * - :math:`\mathbf{J}`
     - Jacobian matrix
   * - :math:`\arg\max_\theta f(\theta)`
     - Value of :math:`\theta` that maximizes :math:`f`
   * - :math:`\arg\min_\theta f(\theta)`
     - Value of :math:`\theta` that minimizes :math:`f`
   * - :math:`\eta`
     - Learning rate / step size
   * - :math:`\theta^{(k)}`
     - Parameter value at iteration :math:`k` of an iterative algorithm
   * - :math:`\epsilon`
     - Convergence tolerance
   * - :math:`O(\cdot)`
     - Big-O notation (asymptotic upper bound)
   * - :math:`o(\cdot)`
     - Little-o notation (asymptotically negligible)
   * - :math:`O_p(\cdot)`
     - Stochastic big-O (bounded in probability)
   * - :math:`o_p(\cdot)`
     - Stochastic little-o (converges to zero in probability)

Matrix Notation
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}`
     - Matrices (bold uppercase)
   * - :math:`\mathbf{x}, \mathbf{y}, \mathbf{z}`
     - Vectors (bold lowercase)
   * - :math:`\mathbf{I}` or :math:`\mathbf{I}_n`
     - Identity matrix (:math:`n \times n`)
   * - :math:`\mathbf{0}`
     - Zero vector or zero matrix
   * - :math:`\mathbf{1}` or :math:`\mathbf{1}_n`
     - Vector of ones
   * - :math:`\mathbf{A}^\top`
     - Transpose of :math:`\mathbf{A}`
   * - :math:`\mathbf{A}^{-1}`
     - Inverse of :math:`\mathbf{A}`
   * - :math:`\mathbf{A}^{-\top}`
     - :math:`(\mathbf{A}^{-1})^\top = (\mathbf{A}^\top)^{-1}`
   * - :math:`\det(\mathbf{A})` or :math:`|\mathbf{A}|`
     - Determinant of :math:`\mathbf{A}`
   * - :math:`\operatorname{tr}(\mathbf{A})`
     - Trace of :math:`\mathbf{A}`
   * - :math:`\operatorname{rank}(\mathbf{A})`
     - Rank of :math:`\mathbf{A}`
   * - :math:`\operatorname{diag}(\mathbf{A})`
     - Vector of diagonal entries of :math:`\mathbf{A}`
   * - :math:`\operatorname{diag}(\mathbf{x})`
     - Diagonal matrix with entries of :math:`\mathbf{x}` on the diagonal
   * - :math:`\lambda_i(\mathbf{A})`
     - :math:`i`-th eigenvalue of :math:`\mathbf{A}`
   * - :math:`\sigma_i(\mathbf{A})`
     - :math:`i`-th singular value of :math:`\mathbf{A}`
   * - :math:`\|\mathbf{x}\|` or :math:`\|\mathbf{x}\|_2`
     - Euclidean (L2) norm, :math:`\sqrt{\mathbf{x}^\top\mathbf{x}}`
   * - :math:`\|\mathbf{A}\|_F`
     - Frobenius norm, :math:`\sqrt{\operatorname{tr}(\mathbf{A}^\top\mathbf{A})}`
   * - :math:`\mathbf{A} \succ 0`
     - :math:`\mathbf{A}` is positive definite
   * - :math:`\mathbf{A} \succeq 0`
     - :math:`\mathbf{A}` is positive semi-definite
   * - :math:`\mathbf{A} \otimes \mathbf{B}`
     - Kronecker product of :math:`\mathbf{A}` and :math:`\mathbf{B}`


Named Distributions
--------------------

The following shorthand is used for standard distributions:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Notation
     - Distribution
   * - :math:`\text{Bernoulli}(p)`
     - Bernoulli with success probability :math:`p`
   * - :math:`\text{Bin}(n, p)`
     - Binomial with :math:`n` trials and success probability :math:`p`
   * - :math:`\text{Poisson}(\lambda)`
     - Poisson with rate :math:`\lambda`
   * - :math:`\text{Geom}(p)`
     - Geometric with success probability :math:`p`
   * - :math:`\text{NegBin}(r, p)`
     - Negative binomial with :math:`r` successes, probability :math:`p`
   * - :math:`\mathcal{U}(a, b)`
     - Uniform on :math:`[a, b]`
   * - :math:`\text{Exp}(\lambda)`
     - Exponential with rate :math:`\lambda`
   * - :math:`\mathcal{N}(\mu, \sigma^2)`
     - Normal with mean :math:`\mu` and variance :math:`\sigma^2`
   * - :math:`\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})`
     - :math:`p`-variate normal with mean :math:`\boldsymbol{\mu}` and covariance :math:`\boldsymbol{\Sigma}`
   * - :math:`\text{Gamma}(\alpha, \beta)`
     - Gamma with shape :math:`\alpha` and rate :math:`\beta`
   * - :math:`\text{Beta}(\alpha, \beta)`
     - Beta with shape parameters :math:`\alpha` and :math:`\beta`
   * - :math:`\chi^2_n`
     - Chi-squared with :math:`n` degrees of freedom
   * - :math:`t_n`
     - Student's :math:`t` with :math:`n` degrees of freedom
   * - :math:`F_{m,n}`
     - :math:`F`-distribution with :math:`m` and :math:`n` degrees of freedom
   * - :math:`\text{Mult}(n, \mathbf{p})`
     - Multinomial with :math:`n` trials and probability vector :math:`\mathbf{p}`
   * - :math:`\text{Dir}(\boldsymbol{\alpha})`
     - Dirichlet with concentration parameter :math:`\boldsymbol{\alpha}`
   * - :math:`\text{Wishart}_p(n, \mathbf{V})`
     - Wishart with :math:`n` degrees of freedom and scale matrix :math:`\mathbf{V}`


Greek Letters and Their Typical Uses
=====================================

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Letter
     - Name
     - Typical Use in Statistics
   * - :math:`\alpha`
     - alpha
     - Significance level; shape parameter; Type I error rate
   * - :math:`\beta`
     - beta
     - Regression coefficient; rate parameter; Type II error rate
   * - :math:`\gamma`
     - gamma
     - Skewness; Euler--Mascheroni constant; threshold parameter
   * - :math:`\delta`
     - delta
     - Effect size; small perturbation; Kronecker delta
   * - :math:`\epsilon, \varepsilon`
     - epsilon
     - Error term; small positive quantity; convergence tolerance
   * - :math:`\zeta`
     - zeta
     - Latent variable; link function parameter
   * - :math:`\eta`
     - eta
     - Natural (canonical) parameter; learning rate
   * - :math:`\theta`
     - theta
     - Generic parameter (the most common choice)
   * - :math:`\iota`
     - iota
     - (Rarely used in statistics)
   * - :math:`\kappa`
     - kappa
     - Cumulant; condition number; concentration parameter
   * - :math:`\lambda`
     - lambda
     - Rate parameter; eigenvalue; Lagrange multiplier; penalty
   * - :math:`\mu`
     - mu
     - Mean; location parameter
   * - :math:`\nu`
     - nu
     - Degrees of freedom
   * - :math:`\xi`
     - xi
     - Latent variable; auxiliary parameter
   * - :math:`\pi`
     - pi
     - Prior probability; the constant 3.14159...
   * - :math:`\rho`
     - rho
     - Correlation coefficient; spectral radius
   * - :math:`\sigma`
     - sigma
     - Standard deviation (:math:`\sigma^2` = variance)
   * - :math:`\tau`
     - tau
     - Precision (:math:`1/\sigma^2`); Kendall's rank correlation
   * - :math:`\upsilon`
     - upsilon
     - (Rarely used in statistics)
   * - :math:`\phi, \varphi`
     - phi
     - Standard normal density; dispersion parameter; basis function
   * - :math:`\chi`
     - chi
     - Chi-squared distribution
   * - :math:`\psi`
     - psi
     - Digamma function; parameter of interest
   * - :math:`\omega`
     - omega
     - Weight; angular frequency

**Uppercase Greek letters** with common statistical uses:

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Letter
     - Name
     - Typical Use
   * - :math:`\Gamma`
     - Gamma
     - Gamma function; Gamma distribution
   * - :math:`\Delta`
     - Delta
     - Change or difference
   * - :math:`\Theta`
     - Theta
     - Parameter space
   * - :math:`\Lambda`
     - Lambda
     - Likelihood ratio; diagonal matrix of eigenvalues
   * - :math:`\Sigma`
     - Sigma
     - Covariance matrix; summation (:math:`\sum`)
   * - :math:`\Phi`
     - Phi
     - Standard normal CDF
   * - :math:`\Psi`
     - Psi
     - Polygamma function
   * - :math:`\Omega`
     - Omega
     - Sample space; precision matrix (:math:`\Sigma^{-1}`)


Glossary of Key Terms
======================

.. glossary::
   :sorted:

   Asymptotic normality
      The property that the distribution of an estimator approaches a normal
      distribution as the sample size grows. Under regularity conditions,
      :math:`\sqrt{n}(\hat\theta - \theta_0) \stackrel{d}{\to} \mathcal{N}(0, \mathcal{I}(\theta_0)^{-1})`.

   Bias
      The difference :math:`E[\hat\theta] - \theta` between the expected value
      of an estimator and the true parameter value.

   Completeness
      A statistic :math:`T` is complete if the only function :math:`g` with
      :math:`E_\theta[g(T)] = 0` for all :math:`\theta` is :math:`g \equiv 0`.

   Confidence interval
      An interval :math:`[L(\mathbf{X}), U(\mathbf{X})]` that contains the
      true parameter with a specified probability (the confidence level).

   Conjugate prior
      A prior distribution that, when combined with the likelihood via Bayes'
      theorem, yields a posterior of the same parametric family.

   Consistency
      An estimator :math:`\hat\theta_n` is consistent if
      :math:`\hat\theta_n \stackrel{p}{\to} \theta_0` as :math:`n \to \infty`.

   Cramér--Rao lower bound
      The minimum variance achievable by any unbiased estimator:
      :math:`\operatorname{Var}(\hat\theta) \geq \mathcal{I}(\theta)^{-1}`.

   Deviance
      Twice the difference between the log-likelihood of the saturated model
      and the fitted model: :math:`D = 2[\ell_{\text{sat}} - \ell(\hat\theta)]`.

   Efficiency
      The ratio of the Cramér--Rao lower bound to the actual variance of an
      estimator. An efficient estimator achieves the bound.

   EM algorithm
      Expectation--Maximization algorithm, an iterative method for finding MLEs
      when the model involves latent variables or missing data.

   Estimator
      A function of the data used to estimate an unknown parameter. The
      distinction between "estimator" (the rule) and "estimate" (the numerical
      value) is maintained in this text.

   Exponential family
      A parametric family whose density can be written as
      :math:`f(x|\theta) = h(x)\exp[\eta(\theta)^\top T(x) - A(\theta)]`.

   Fisher information
      The variance of the score function, or equivalently the negative
      expected Hessian of the log-likelihood:
      :math:`\mathcal{I}(\theta) = E[U(\theta)^2] = -E[\ell''(\theta)]`.

   Gradient descent
      An iterative optimization algorithm:
      :math:`\theta^{(k+1)} = \theta^{(k)} + \eta\,\nabla\ell(\theta^{(k)})`.

   Hessian matrix
      The matrix of second partial derivatives of a function:
      :math:`H_{ij} = \partial^2 f / \partial \theta_i \partial \theta_j`.

   Kullback--Leibler divergence
      A measure of the difference between two distributions:
      :math:`\text{KL}(p \| q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx`.

   Likelihood function
      The joint density or mass function of the data, viewed as a function of
      the parameters: :math:`L(\theta) = f(\mathbf{x} \mid \theta)`.

   Likelihood ratio test
      A hypothesis test based on the statistic
      :math:`\Lambda = 2[\ell(\hat\theta) - \ell(\theta_0)]`, which is
      asymptotically :math:`\chi^2`.

   Log-likelihood
      The natural logarithm of the likelihood function:
      :math:`\ell(\theta) = \log L(\theta)`.

   Maximum likelihood estimator (MLE)
      The parameter value that maximizes the likelihood (or equivalently
      the log-likelihood): :math:`\hat\theta = \arg\max_\theta \ell(\theta)`.

   Method of moments
      An estimation approach that equates sample moments to population
      moments and solves for the parameters.

   Newton--Raphson method
      An iterative root-finding algorithm applied to the score equation:
      :math:`\theta^{(k+1)} = \theta^{(k)} - [\ell''(\theta^{(k)})]^{-1}\,\ell'(\theta^{(k)})`.

   Nuisance parameter
      A parameter that is not of direct interest but must be accounted for
      in the inference procedure.

   Observed information
      The negative Hessian of the log-likelihood evaluated at the MLE:
      :math:`\mathcal{J}(\hat\theta) = -\ell''(\hat\theta)`.

   Power
      The probability of correctly rejecting a false null hypothesis:
      :math:`1 - \beta`, where :math:`\beta` is the Type II error rate.

   Profile likelihood
      The likelihood maximized over nuisance parameters:
      :math:`L_p(\psi) = \max_\lambda L(\psi, \lambda)`.

   p-value
      The probability, under the null hypothesis, of observing a test
      statistic as extreme as or more extreme than the observed value.

   Regularity conditions
      Technical conditions (differentiability, integrability, parameter not
      on boundary) that ensure standard asymptotic results hold for MLEs.

   Score function
      The derivative of the log-likelihood with respect to the parameter:
      :math:`U(\theta) = \partial\ell / \partial\theta`. Under regularity
      conditions, :math:`E[U(\theta_0)] = 0`.

   Sufficient statistic
      A statistic :math:`T(\mathbf{X})` that captures all the information in
      the data about the parameter. By the Fisher--Neyman factorization
      theorem, :math:`T` is sufficient if :math:`f(\mathbf{x}|\theta) = g(T(\mathbf{x}), \theta)\,h(\mathbf{x})`.

   Wald test
      A hypothesis test based on the statistic
      :math:`W = (\hat\theta - \theta_0)^2 / \widehat{\operatorname{Var}}(\hat\theta)`,
      which is asymptotically :math:`\chi^2_1`.


Common Abbreviations
=====================

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Full Name
   * - AIC
     - Akaike Information Criterion
   * - BIC
     - Bayesian Information Criterion
   * - CDF
     - Cumulative distribution function
   * - CLT
     - Central Limit Theorem
   * - CRLB
     - Cramér--Rao lower bound
   * - EM
     - Expectation--Maximization
   * - GLM
     - Generalized linear model
   * - i.i.d.
     - Independent and identically distributed
   * - IRLS
     - Iteratively reweighted least squares
   * - KL
     - Kullback--Leibler
   * - LLN
     - Law of large numbers
   * - LRT
     - Likelihood ratio test
   * - MGF
     - Moment generating function
   * - MLE
     - Maximum likelihood estimator / estimate
   * - MSE
     - Mean squared error
   * - MVN
     - Multivariate normal
   * - NR
     - Newton--Raphson
   * - PDF
     - Probability density function
   * - PMF
     - Probability mass function
   * - r.v.
     - Random variable
   * - SVD
     - Singular value decomposition
   * - UMVUE
     - Uniformly minimum variance unbiased estimator
   * - w.r.t.
     - With respect to
