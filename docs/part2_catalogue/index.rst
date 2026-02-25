.. _part2:

====================================
Part II: The Likelihood Catalogue
====================================

Part II serves as a comprehensive reference catalogue of likelihood functions
encountered across statistics and data science. For each distribution we derive
the likelihood, log-likelihood, score function, Fisher information, and
maximum-likelihood estimator from first principles, showing every algebraic
step so the reader can reproduce each result with pencil and paper.

The catalogue is organised into four chapters:

* **Chapter 5** (:ref:`ch5_discrete`) covers discrete distributions whose
  support is a countable set of integers: Bernoulli, Binomial, Poisson,
  Negative Binomial, Geometric, Hypergeometric, Multinomial, and the
  Zero-Inflated Poisson.

* **Chapter 6** (:ref:`ch6_continuous`) covers continuous distributions
  defined by a probability density function on the real line (or a subset
  thereof): Normal, Exponential, Gamma, Beta, Log-Normal, Weibull, Pareto,
  Student-*t*, Chi-squared, *F*, Uniform, and Cauchy.

* **Chapter 7** (:ref:`ch7_multivariate`) extends the treatment to
  vector-valued and matrix-valued distributions---Multivariate Normal,
  Wishart, Inverse-Wishart, Dirichlet, and Multinomial---introducing the
  matrix-calculus tools required for their derivations.

* **Chapter 8** (:ref:`ch8_specialized`) surveys likelihood variants that
  arise when the standard i.i.d. likelihood is unavailable or inconvenient:
  profile, partial, marginal, conditional, composite, quasi-, pseudo-, and
  empirical likelihoods, together with censored/truncated likelihoods and
  penalised likelihoods.

.. toctree::
   :maxdepth: 2
   :caption: Likelihood Catalogue

   discrete_likelihoods
   continuous_likelihoods
   multivariate_likelihoods
   specialized_likelihoods
