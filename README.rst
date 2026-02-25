==============================
Likelihood-Based Inference
==============================

*From Foundations to Research*

A comprehensive, novice-accessible guide to likelihood-based statistical
inference. Every result is derived from first principles with plain-English
explanations alongside the mathematics, accompanied by runnable Python code
that demonstrates and verifies each formula.

.. image:: https://img.shields.io/badge/Support_this_project-PayPal-003087?style=for-the-badge&logo=paypal&logoColor=white
   :target: https://www.paypal.com/donate/?hosted_button_id=VTASTXN2KAFJQ
   :alt: Donate via PayPal

Read Online
===========

https://fli.readthedocs.io/en/latest/

Contents
========

**Part I — Foundations** (Chapters 1--4)
  Probability basics, random variables, common distributions, and the
  likelihood function.

**Part II — The Likelihood Catalogue** (Chapters 5--8)
  Discrete, continuous, multivariate, and specialized likelihoods — each
  family with its PDF, log-likelihood, score, Fisher information, and MLE.

**Part III — Maximum Likelihood Estimation** (Chapters 9--11)
  MLE theory and large-sample properties, closed-form derivations for major
  families, confidence intervals, and the likelihood-ratio / Wald / score
  tests.

**Part IV — Optimization for Likelihood** (Chapters 12--16)
  Gradient descent, Newton and Fisher scoring, quasi-Newton (BFGS, L-BFGS),
  the EM algorithm, and constrained optimization.

**Part V — Advanced and Research Topics** (Chapters 17--21)
  Bayesian connections, computational methods (MCMC, variational inference),
  model selection (AIC/BIC/cross-validation), modern research frontiers, and
  numerical considerations.

**Appendices**
  Linear algebra review, calculus review, notation and glossary.

Getting Started
===============

Install dependencies::

    pip install -r requirements.txt

Build the HTML docs locally::

    cd docs
    make html
    # Open _build/html/index.html in your browser

Build the PDF::

    python build_book.py          # incremental
    python build_book.py --clean  # full rebuild

Running Code Examples
=====================

Every chapter includes self-contained Python code blocks with
``np.random.seed(42)`` for reproducibility. Copy any block into a script or
Jupyter notebook and run it directly. The only runtime dependencies are::

    pip install numpy scipy matplotlib

Disclaimer
==========

This project was developed by Kevin Korfmann as an educational resource for
likelihood-based statistical inference. Drafting, code examples, and
explanations were produced with substantial assistance from Claude, an AI
assistant by Anthropic. All content has been reviewed and curated by the
author, but readers should verify implementations against primary literature.
