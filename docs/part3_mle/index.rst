.. _part3:

==========================================
Part III: Maximum Likelihood Estimation
==========================================

Part III develops the theory, practice, and inferential machinery of
maximum likelihood estimation (MLE) --- the single most important
estimation principle in modern statistics. Building on the likelihood
functions catalogued in :ref:`part2`, we now ask: *given a likelihood
surface, how do we find its peak, what can we say about that peak, and
how do we use it for inference?*

The treatment is organised into three chapters:

* **Chapter 9** (:ref:`ch9_mle_theory`) establishes the theoretical
  backbone of MLE. We give a formal definition of the maximum likelihood
  estimator, then develop its large-sample properties---consistency,
  asymptotic normality, and efficiency---with full proof sketches. We
  state and explain the regularity conditions that underpin these
  results, prove the invariance (equivariance) property, and discuss
  finite-sample phenomena such as bias.

* **Chapter 10** (:ref:`ch10_analytical_mle`) works through closed-form
  MLE derivations for the most important parametric families: Normal,
  Exponential, Poisson, Binomial, Gamma, Beta, Uniform, and
  Multinomial. Every derivation proceeds step by step---write the
  log-likelihood, differentiate, solve, and verify via the second
  derivative---so that readers can reproduce each result with pencil and
  paper.

* **Chapter 11** (:ref:`ch11_testing`) shows how to convert a fitted
  likelihood into confidence intervals and hypothesis tests. We derive
  the three classical test statistics---likelihood ratio, Wald, and
  score (Rao)---prove Wilks' theorem, construct profile-likelihood
  confidence regions, and discuss multiple-testing corrections.

.. toctree::
   :maxdepth: 2
   :caption: Maximum Likelihood Estimation

   mle_theory
   analytical_mle
   confidence_and_testing
