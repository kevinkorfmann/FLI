==============================
Likelihood-Based Inference
==============================

A comprehensive, novice-accessible guide covering likelihood-based inference
from basic probability through research-grade algorithms.

Read Online
===========

The documentation is hosted on Read the Docs:

https://likelihoodsandoptimization.readthedocs.io/

Getting Started
===============

Install dependencies::

    pip install -r requirements.txt

Build the HTML documentation locally::

    cd docs
    make html
    # Open _build/html/index.html in your browser

Build the PDF book::

    python build_book.py

Structure
=========

- **Part I — Foundations**: Probability, random variables, distributions, the likelihood function
- **Part II — Likelihood Catalogue**: Discrete, continuous, multivariate, and specialized likelihoods
- **Part III — Maximum Likelihood Estimation**: Theory, analytical solutions, testing and confidence intervals
- **Part IV — Optimization**: Gradient methods, Newton methods, quasi-Newton, EM algorithm, constrained optimization
- **Part V — Advanced Topics**: Bayesian connections, computational methods, model selection, modern research, numerical considerations
- **Appendices**: Linear algebra review, calculus review, notation glossary

Running Code Examples
=====================

Every chapter includes runnable Python code blocks that demonstrate and verify
the mathematical formulas. To run them you need NumPy, SciPy, and Matplotlib::

    pip install numpy scipy matplotlib

The code blocks are self-contained — each one includes its own imports and uses
``np.random.seed(42)`` for reproducibility. You can copy any block into a
Python script or Jupyter notebook and run it directly.

Disclaimer
==========

This project was developed by Kevin Korfmann as an educational resource for
likelihood-based statistical inference. Drafting, code examples, and
explanations were produced with substantial assistance from Claude, an AI
assistant by Anthropic. All content has been reviewed and curated by the
author, but readers should verify implementations against primary literature.
