.. _ch2_random_variables:

====================================
Chapter 2: Random Variables
====================================

In :ref:`ch1_probability` we learned how to assign probabilities to *events*
--- subsets of a sample space.  But in practice we almost always care about
*numerical* quantities: the number of heads in 10 flips, the average height of
a sample, the time until a server fails.  **Random variables** bridge the gap
between the abstract world of sample spaces and the numerical world of data.

Once you can attach numbers to outcomes, you can compute averages, measure
spread, and --- most importantly for this book --- write down likelihood
functions.

Two running examples will thread through this chapter.  The first is
**insurance**: an insurer needs to price policies by computing expected claim
costs and quantifying risk.  The second is **stock returns**: a portfolio
analyst needs to understand how two assets co-move and how diversification
reduces risk.  Every formula will be applied to one or both scenarios so you
can see the mathematics earning its keep.

.. code-block:: python

   # ------------------------------------------------------------------
   # Setup: parameters for our two running examples
   # ------------------------------------------------------------------
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # --- Insurance scenario ---
   # A simple auto-insurance model:
   #   With prob 0.85 — no claim ($0)
   #   With prob 0.10 — minor claim (mean $800)
   #   With prob 0.04 — moderate claim (mean $5,000)
   #   With prob 0.01 — major claim (mean $25,000)
   claim_probs = np.array([0.85, 0.10, 0.04, 0.01])
   claim_costs = np.array([0, 800, 5_000, 25_000])
   claim_labels = ["None", "Minor", "Moderate", "Major"]

   # --- Stock returns scenario ---
   # Two stocks: Tech (higher return, higher volatility) and Utility (lower, steadier)
   mu_tech, sigma_tech   = 0.12, 0.25    # annualized mean return 12%, std 25%
   mu_util, sigma_util   = 0.06, 0.12    # annualized mean return  6%, std 12%
   rho_stocks            = 0.30           # correlation between the two

   print("=== Insurance scenario ===")
   for label, prob, cost in zip(claim_labels, claim_probs, claim_costs):
       print(f"  {label:>10s}: P = {prob:.2f}, Cost = ${cost:>7,}")
   print()
   print("=== Stock returns scenario ===")
   print(f"  Tech:    mu = {mu_tech:.0%}, sigma = {sigma_tech:.0%}")
   print(f"  Utility: mu = {mu_util:.0%}, sigma = {sigma_util:.0%}")
   print(f"  Correlation: rho = {rho_stocks}")

.. contents:: Chapter Contents
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here


2.1 What Is a Random Variable?
================================

Your insurance company has 50,000 policyholders.  Each one might or might not
file a claim this year, and if they do, the claim amount could be anything from
a few hundred dollars to tens of thousands.  To set premiums, you need to
attach *numbers* to the random experiment "what happens to policyholder #i this
year."  That mapping from outcome to number is precisely what a random variable
provides.

**Plain-English definition.**  A random variable is a rule that assigns a
number to every outcome of a random experiment.

*Example.*  Roll two dice and let :math:`X` be the sum of the faces.  The
outcome :math:`(3, 5)` maps to :math:`X = 8`.  The outcome :math:`(1, 1)` maps
to :math:`X = 2`.

**Formal definition.**  Given a probability space :math:`(\Omega, \mathcal{F}, P)`,
a **random variable** is a function

.. math::

   X : \Omega \longrightarrow \mathbb{R}

such that for every real number :math:`x`, the set
:math:`\{\omega \in \Omega : X(\omega) \leq x\}` belongs to :math:`\mathcal{F}`.
This measurability condition ensures that we can compute probabilities like
:math:`P(X \leq x)`.

Random variables come in two main flavors:

- **Discrete**: :math:`X` takes on a finite or countably infinite set of
  values (e.g., 0, 1, 2, ...).
- **Continuous**: :math:`X` takes on an uncountably infinite set of values
  (e.g., any real number in an interval).

.. code-block:: python

   # A random variable in action: insurance claim amount
   # The "random experiment" is what happens to one policyholder in a year.
   # The random variable X maps the outcome to a dollar amount.

   # Simulate one year for 10 policyholders
   n_demo = 10
   claim_types = np.random.choice(len(claim_probs), size=n_demo, p=claim_probs)
   X = claim_costs[claim_types]  # the random variable: outcome -> dollar amount

   print("Policyholder  Outcome      X (claim $)")
   print("-" * 42)
   for i in range(n_demo):
       print(f"     {i+1:2d}       {claim_labels[claim_types[i]]:>10s}     ${X[i]:>7,}")

.. admonition:: Intuition

   Think of a random variable as a "measurement device" pointed at a random
   experiment.  The experiment produces an outcome; the random variable reads
   off a number.  Different random variables can measure different things about
   the same experiment --- for instance, the sum of two dice versus whether the
   sum exceeds 7.


2.2 Probability Mass Function (PMF)
=====================================

Suppose you are the insurer's actuary.  You know the probability of each claim
type (none, minor, moderate, major).  The **PMF** is simply the table that
lists every possible value of the random variable alongside its probability ---
exactly the information you need to price a policy.

For a **discrete** random variable :math:`X`, the **probability mass function**
is

.. math::

   p_X(x) = P(X = x).

The PMF must satisfy two properties:

1. :math:`p_X(x) \geq 0` for all :math:`x`.
2. :math:`\displaystyle\sum_{\text{all } x} p_X(x) = 1`.

.. code-block:: python

   # PMF of the insurance claim amount X
   print("x (claim $)    p_X(x)")
   print("-" * 26)
   for cost, prob in zip(claim_costs, claim_probs):
       print(f"  ${cost:>7,}     {prob:.2f}")
   print(f"\n  Sum of p_X(x) = {claim_probs.sum():.2f}  (must be 1.0)")

*Example.*  Let :math:`X` be the result of a single fair die roll.

.. math::

   p_X(x) = \frac{1}{6}, \quad x \in \{1, 2, 3, 4, 5, 6\}.

Let us also build the PMF for the sum of two dice --- a slightly richer
example.

.. code-block:: python

   # PMF of the sum of two fair dice
   counts = np.zeros(13)  # indices 0..12; we use 2..12
   for i in range(1, 7):
       for j in range(1, 7):
           counts[i + j] += 1
   pmf_dice = counts / 36

   print("x   p_X(x) = P(Sum = x)")
   for x in range(2, 13):
       bar = "#" * int(pmf_dice[x] * 120)  # simple text histogram
       print(f"{x:2d}  {pmf_dice[x]:.4f}  {bar}")
   print(f"\nSum of PMF: {pmf_dice.sum():.4f}  (should be 1.0)")


2.3 Probability Density Function (PDF)
========================================

Insurance claims are not always nice round numbers.  A car repair might cost
$1,247.63 or $3,891.02 --- any non-negative real number is possible.  For
such continuous random variables, asking "what is the probability the claim is
exactly $1,247.63?" gives the unhelpful answer zero.  Instead, we describe the
distribution through a *density*: a curve whose area over any interval gives
the probability of landing in that interval.

For a **continuous** random variable :math:`X`, individual points have zero
probability: :math:`P(X = x) = 0`.  Instead we describe the distribution
through a **probability density function** :math:`f_X(x)`, defined by the
property

.. math::

   P(a \leq X \leq b) = \int_a^b f_X(x)\,dx.

The PDF must satisfy:

1. :math:`f_X(x) \geq 0` for all :math:`x`.
2. :math:`\displaystyle\int_{-\infty}^{\infty} f_X(x)\,dx = 1`.

Note: :math:`f_X(x)` is **not** a probability.  It can exceed 1 at some
points.  It is a *density*: probability per unit length.

.. code-block:: python

   # PDF example: Exponential distribution for claim sizes
   # If a minor claim follows Exp(rate = 1/800), the mean claim is $800
   rate = 1 / 800  # lambda
   x_vals = np.linspace(0, 5000, 500)
   f_X = rate * np.exp(-rate * x_vals)  # f_X(x) = lambda * exp(-lambda * x)

   # Verify: integral over [0, inf) should be 1
   dx = x_vals[1] - x_vals[0]
   approx_integral = np.sum(f_X * dx)
   print(f"Exp(rate={rate:.5f}) — approximate integral over [0, 5000]: {approx_integral:.4f}")

   # P(500 <= X <= 1500)
   mask = (x_vals >= 500) & (x_vals <= 1500)
   P_500_1500_numerical = np.sum(f_X[mask] * dx)
   P_500_1500_exact = np.exp(-rate * 500) - np.exp(-rate * 1500)
   print(f"P(500 <= X <= 1500): numerical = {P_500_1500_numerical:.4f}, exact = {P_500_1500_exact:.4f}")

.. admonition:: Common Pitfall

   A common source of confusion: a PDF value of 3.0 at some point does *not*
   mean a 300 % probability.  The density is only meaningful when integrated
   over an interval.  For example, the Uniform(0, 0.1) distribution has a
   constant density of 10, yet it is a perfectly valid probability distribution
   because :math:`10 \times 0.1 = 1`.

.. code-block:: python

   # Demonstrating the pitfall: Uniform(0, 0.1) has density = 10 everywhere
   f_uniform = 1 / 0.1  # = 10
   print(f"Uniform(0, 0.1): f(x) = {f_uniform:.1f} for x in [0, 0.1]")
   print(f"  f(0.05) = {f_uniform}  (NOT a probability!)")
   print(f"  P(0 <= X <= 0.1) = {f_uniform} × 0.1 = {f_uniform * 0.1:.1f}  (this is a probability)")


2.4 Cumulative Distribution Function (CDF)
============================================

The insurer often asks threshold questions: "What is the probability that a
claim is at most $2,000?"  or "What is the probability that annual losses
exceed $1 million?"  The CDF directly answers the first question; the second
follows via the complement rule.

Whether discrete or continuous, every random variable has a **cumulative
distribution function** (CDF):

.. math::

   F_X(x) = P(X \leq x).

The CDF is a unified tool that works for both types of random variables.  It
answers the question: "What is the probability that :math:`X` lands at or
below this threshold?"

**Properties of the CDF** (provable from the axioms):

1. :math:`F_X` is non-decreasing: if :math:`a < b` then :math:`F_X(a) \leq F_X(b)`.
2. :math:`\displaystyle\lim_{x \to -\infty} F_X(x) = 0` and
   :math:`\displaystyle\lim_{x \to +\infty} F_X(x) = 1`.
3. :math:`F_X` is right-continuous: :math:`\displaystyle\lim_{h \downarrow 0} F_X(x+h) = F_X(x)`.

**Relationship to PDF/PMF.**

- Continuous case: :math:`F_X(x) = \int_{-\infty}^{x} f_X(t)\,dt`, hence
  :math:`f_X(x) = F_X'(x)` wherever the derivative exists.
- Discrete case: :math:`F_X(x) = \sum_{t \leq x} p_X(t)`.

**Useful identity.**

.. math::

   P(a < X \leq b) = F_X(b) - F_X(a).

.. code-block:: python

   # CDF of the Exponential(rate = 1/800) claim distribution
   # F_X(x) = 1 - exp(-lambda * x)  for x >= 0
   thresholds = [500, 1000, 2000, 5000, 10000]
   print("Exponential(mean=$800) CDF:")
   print(f"{'Threshold':>12s}  {'F_X(x)':>8s}  {'P(X > x)':>8s}")
   for t in thresholds:
       F_x = 1 - np.exp(-rate * t)
       print(f"  ${t:>8,}  {F_x:8.4f}  {1 - F_x:8.4f}")

   # Useful identity: P(500 < X <= 2000)
   F_500  = 1 - np.exp(-rate * 500)
   F_2000 = 1 - np.exp(-rate * 2000)
   P_interval = F_2000 - F_500
   print(f"\nP(500 < X <= 2000) = F(2000) - F(500) = {F_2000:.4f} - {F_500:.4f} = {P_interval:.4f}")


2.5 Expectation (Expected Value)
==================================

The actuary needs a single number that summarizes "on average, how much will
this policy cost us?"  That number is the **expected value** --- the
probability-weighted average of every possible claim amount.  If the insurer
sells 50,000 identical policies, the total payout will be close to 50,000
times this expected value.

The **expected value** (or **mean**) of a random variable is the
probability-weighted average of its possible values.  It tells you the "center
of mass" of the distribution.

Here is the key idea: if you could repeat the random experiment millions of
times and average the results, you would get something very close to :math:`E[X]`.
The expected value is the theoretical version of that long-run average.

**Discrete case:**

.. math::

   E[X] = \sum_{x} x \, p_X(x).

**Continuous case:**

.. math::

   E[X] = \int_{-\infty}^{\infty} x \, f_X(x) \, dx.

**Insurance example — expected claim cost.**

.. math::

   E[X] = \sum_{x} x \cdot p_X(x)
        = 0(0.85) + 800(0.10) + 5000(0.04) + 25000(0.01)

.. code-block:: python

   # E[X] for the insurance claim random variable
   # E[X] = sum of x * p_X(x)
   E_X = np.sum(claim_costs * claim_probs)

   print("E[X] = sum of x * p_X(x):")
   for cost, prob in zip(claim_costs, claim_probs):
       print(f"  {cost:>7,} × {prob:.2f} = {cost * prob:>8.2f}")
   print(f"  {'':>7s}   {'':>4s}   --------")
   print(f"  {'E[X]':>7s}   {'':>4s}   ${E_X:>7.2f}")
   print(f"\nThe insurer should charge at least ${E_X:.2f} per policy to break even.")

.. code-block:: python

   # Verify E[X] by simulation: 1 million policyholders
   n_sim = 1_000_000
   claim_types_sim = np.random.choice(len(claim_probs), size=n_sim, p=claim_probs)
   X_sim = claim_costs[claim_types_sim]

   print(f"E[X] (formula):     ${E_X:.2f}")
   print(f"E[X] (simulation):  ${X_sim.mean():.2f}  (n = {n_sim:,})")

*Example.*  Fair die: :math:`E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6}
= \frac{1+2+3+4+5+6}{6} = 3.5`.

.. code-block:: python

   # E[X] for a fair die: theory vs. simulation
   E_X_die = sum(x * (1/6) for x in range(1, 7))
   rolls = np.random.randint(1, 7, size=100_000)
   print(f"E[X] (theory)     = {E_X_die:.4f}")
   print(f"E[X] (simulation) = {rolls.mean():.4f}")

**Law of the Unconscious Statistician (LOTUS).**  If :math:`g` is a function,
then

.. math::

   E[g(X)] = \sum_x g(x)\,p_X(x) \quad \text{(discrete)}, \qquad
   E[g(X)] = \int_{-\infty}^{\infty} g(x)\,f_X(x)\,dx \quad \text{(continuous)}.

Why does this matter?  LOTUS lets you compute the expected value of any
transformation of :math:`X` without first deriving the distribution of
:math:`g(X)`.  This saves enormous effort.

.. code-block:: python

   # LOTUS example: the insurer charges a $500 deductible.
   # Actual payout g(X) = max(X - 500, 0).  What is E[g(X)]?
   def g(x):
       """Payout after $500 deductible."""
       return np.maximum(x - 500, 0)

   E_gX = np.sum(g(claim_costs) * claim_probs)

   print("LOTUS: E[g(X)] where g(x) = max(x - 500, 0)")
   for cost, prob in zip(claim_costs, claim_probs):
       print(f"  g({cost:>7,}) = {g(np.array([cost]))[0]:>7,},  × {prob:.2f} = {g(np.array([cost]))[0] * prob:>8.2f}")
   print(f"\n  E[g(X)] = ${E_gX:.2f}  (expected payout after deductible)")
   print(f"  Compare to E[X] = ${E_X:.2f}  (expected claim without deductible)")
   print(f"  The deductible saves the insurer ${E_X - E_gX:.2f} per policy on average.")


2.5.1 Linearity of Expectation
--------------------------------

You insure not just one policyholder but 50,000.  The total expected payout is
simply 50,000 times the expected payout per policy --- no matter how the claims
are correlated across policyholders.  This is linearity of expectation at work.

**Theorem.**  For any random variables :math:`X, Y` and constants :math:`a, b, c`,

.. math::

   E[aX + bY + c] = a\,E[X] + b\,E[Y] + c.

This holds **regardless** of whether :math:`X` and :math:`Y` are independent.

.. code-block:: python

   # Linearity of expectation: total payout for 50,000 policies
   n_policies = 50_000
   E_total = n_policies * E_X
   print(f"E[X per policy]   = ${E_X:.2f}")
   print(f"E[total for {n_policies:,}] = {n_policies} × ${E_X:.2f} = ${E_total:,.2f}")

   # Verify with simulation
   total_sim = sum(
       claim_costs[np.random.choice(len(claim_probs), size=n_policies, p=claim_probs)]
   )
   print(f"Simulated total   = ${total_sim:,.2f}")

.. admonition:: Intuition

   Linearity of expectation is deceptively powerful.  It says that averages
   distribute over sums --- even when the variables are dependent.  This one
   property lets you compute the expected number of fixed points in a random
   permutation, the expected value of a Binomial, and countless other results
   with almost no effort.

**Proof (discrete case for two variables).**  Let :math:`(X, Y)` have joint
PMF :math:`p_{X,Y}(x,y)`.

.. math::

   E[aX + bY]
   &= \sum_x \sum_y (ax + by)\,p_{X,Y}(x,y) \\
   &= a \sum_x \sum_y x\,p_{X,Y}(x,y) + b \sum_x \sum_y y\,p_{X,Y}(x,y) \\
   &= a \sum_x x \underbrace{\sum_y p_{X,Y}(x,y)}_{p_X(x)} \;+\;
      b \sum_y y \underbrace{\sum_x p_{X,Y}(x,y)}_{p_Y(y)} \\
   &= a\,E[X] + b\,E[Y]. \quad \square

Adding a constant :math:`c` contributes :math:`c \sum_x \sum_y p_{X,Y}(x,y) = c`.


2.6 Variance
==============

Two insurance policies might have the same expected cost but very different
risk profiles.  Policy A might pay $530 almost every year; Policy B might
pay $0 most years but occasionally pay $53,000.  The actuary needs to
distinguish these --- and that is exactly what **variance** does.

Expectation tells us the center; **variance** measures the *spread* of a
distribution --- how far values typically fall from the mean.

**Definition.**

.. math::

   \operatorname{Var}(X) = E\!\left[(X - E[X])^2\right].

We also write :math:`\sigma^2 = \operatorname{Var}(X)`.  The **standard
deviation** is :math:`\sigma = \sqrt{\operatorname{Var}(X)}`.


2.6.1 Computational Formula
-----------------------------

The definition involves :math:`E[X]` inside a squared term, which is unwieldy.
A cleaner formula is:

**Theorem.**

.. math::

   \operatorname{Var}(X) = E[X^2] - (E[X])^2.

This is the formula you will reach for most often in practice.  It is
sometimes called the "shortcut formula" because it avoids computing deviations
from the mean one by one.

**Proof.**  Let :math:`\mu = E[X]`.

.. math::

   \operatorname{Var}(X)
   &= E[(X - \mu)^2] \\
   &= E[X^2 - 2\mu X + \mu^2] \\
   &= E[X^2] - 2\mu\,E[X] + \mu^2 \qquad \text{(linearity)} \\
   &= E[X^2] - 2\mu^2 + \mu^2 \\
   &= E[X^2] - \mu^2
    = E[X^2] - (E[X])^2. \quad \square

.. code-block:: python

   # Var(X) for the insurance claim variable, computed both ways
   mu = E_X  # E[X] = 530 from above

   # Method 1: Definition — E[(X - mu)^2]
   Var_X_def = np.sum((claim_costs - mu)**2 * claim_probs)

   # Method 2: Shortcut — E[X^2] - (E[X])^2
   E_X2 = np.sum(claim_costs**2 * claim_probs)
   Var_X_shortcut = E_X2 - mu**2

   sigma = np.sqrt(Var_X_shortcut)

   print("Variance of insurance claim X:")
   print(f"  E[X]              = ${mu:.2f}")
   print(f"  E[X^2]            = {E_X2:,.2f}")
   print(f"  Var(X) (def)      = {Var_X_def:,.2f}")
   print(f"  Var(X) (shortcut) = E[X^2] - (E[X])^2 = {E_X2:,.2f} - {mu:.2f}^2 = {Var_X_shortcut:,.2f}")
   print(f"  Std(X)            = ${sigma:,.2f}")
   print(f"\n  The standard deviation (${sigma:,.0f}) is much larger than the mean (${mu:.0f}),")
   print(f"  reflecting the heavy tail of rare large claims.")

.. code-block:: python

   # Verify Var(X) by simulation
   print(f"Var(X) (formula):    {Var_X_shortcut:>12,.2f}")
   print(f"Var(X) (simulation): {X_sim.var():>12,.2f}")
   print(f"Std(X) (formula):    ${sigma:>10,.2f}")
   print(f"Std(X) (simulation): ${X_sim.std():>10,.2f}")

.. code-block:: python

   # Var(X) for a fair die — a simpler check
   x_die = np.arange(1, 7)
   pmf_die = np.ones(6) / 6

   E_X_die  = np.sum(x_die * pmf_die)
   E_X2_die = np.sum(x_die**2 * pmf_die)
   Var_X_die = E_X2_die - E_X_die**2

   print(f"Fair die:")
   print(f"  E[X]   = {E_X_die:.4f}")
   print(f"  E[X^2] = {E_X2_die:.4f}")
   print(f"  Var(X) = E[X^2] - (E[X])^2 = {E_X2_die:.4f} - {E_X_die:.4f}^2 = {Var_X_die:.4f}")
   print(f"  Std(X) = {np.sqrt(Var_X_die):.4f}")

   # Simulation check
   die_rolls = np.random.randint(1, 7, size=100_000)
   print(f"\n  Simulated: mean = {die_rolls.mean():.4f}, var = {die_rolls.var():.4f}")


2.6.2 Properties of Variance
------------------------------

1. :math:`\operatorname{Var}(X) \geq 0`, with equality if and only if :math:`X`
   is constant almost surely.

2. **Scaling:** :math:`\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)`.

   *Proof.*

   .. math::

      \operatorname{Var}(aX + b)
      &= E[(aX + b - E[aX + b])^2] \\
      &= E[(aX + b - a\mu - b)^2] \\
      &= E[a^2(X - \mu)^2] = a^2 \operatorname{Var}(X). \quad \square

   Notice that adding a constant :math:`b` does not change the variance ---
   it shifts the distribution without stretching it.

3. **Sum of independent variables:**  If :math:`X` and :math:`Y` are
   independent,

   .. math::

      \operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y).

   (We prove this after introducing covariance.)

.. code-block:: python

   # Variance scaling: currency conversion
   # If X is a claim in USD with Var(X), what is the variance in Euros?
   # Let Y = a*X + b where a = exchange rate, b = fixed processing fee
   a = 0.92   # USD to EUR exchange rate
   b = 25     # fixed €25 processing fee (doesn't affect variance)

   Var_Y = a**2 * Var_X_shortcut
   sigma_Y = np.sqrt(Var_Y)

   print(f"Var(X) in USD           = {Var_X_shortcut:>14,.2f}")
   print(f"Var(aX+b) = a^2·Var(X)  = {a}^2 × {Var_X_shortcut:,.2f} = {Var_Y:>14,.2f}")
   print(f"Std in EUR              = €{sigma_Y:>10,.2f}")
   print(f"(The fixed fee b = €{b} does not affect variance.)")


2.7 Moment Generating Functions
=================================

The actuary has computed :math:`E[X]` and :math:`\operatorname{Var}(X)` for the
claim distribution.  But the risk department also wants the skewness (are large
claims much more common than small ones?) and the kurtosis (how heavy is the
tail?).  Computing each moment separately is tedious.  The **moment generating
function** packages *all* moments into a single function, and any individual
moment can be extracted by differentiation.

The **moment generating function** (MGF) of :math:`X` is

.. math::

   M_X(t) = E[e^{tX}],

defined for all :math:`t` in some neighborhood of zero.

**Why is it useful?**

1. **Moments from derivatives.**  The :math:`n`-th moment of :math:`X` can be
   extracted by differentiating the MGF and evaluating at :math:`t = 0`:

   .. math::

      E[X^n] = M_X^{(n)}(0) = \left.\frac{d^n}{dt^n} M_X(t)\right|_{t=0}.

   *Derivation.*  Expand the exponential in a Taylor series:

   .. math::

      M_X(t) = E[e^{tX}] = E\!\left[\sum_{n=0}^{\infty} \frac{(tX)^n}{n!}\right]
      = \sum_{n=0}^{\infty} \frac{t^n}{n!}\,E[X^n].

   The coefficient of :math:`t^n / n!` is :math:`E[X^n]`, so differentiating
   :math:`n` times and setting :math:`t=0` extracts :math:`E[X^n]`.

2. **Uniqueness.**  If two random variables have the same MGF in a neighborhood
   of zero, they have the same distribution.

3. **Sums of independent variables.**  If :math:`X` and :math:`Y` are
   independent, then

   .. math::

      M_{X+Y}(t) = E[e^{t(X+Y)}] = E[e^{tX}]\,E[e^{tY}] = M_X(t)\,M_Y(t).

   The MGF of a sum is the product of the MGFs.

*Example.*  For :math:`X \sim \text{Bernoulli}(p)`:

.. math::

   M_X(t) = E[e^{tX}] = (1-p)\,e^{0} + p\,e^{t} = 1 - p + p\,e^t.

Differentiating: :math:`M_X'(t) = p\,e^t`, so :math:`E[X] = M_X'(0) = p`.
:math:`M_X''(t) = p\,e^t`, so :math:`E[X^2] = p`.  Then
:math:`\operatorname{Var}(X) = p - p^2 = p(1-p)`.

.. code-block:: python

   # MGF of Bernoulli(p): extract moments via numerical differentiation
   p = 0.3

   def mgf_bernoulli(t, p):
       """M_X(t) = (1-p) + p*exp(t)"""
       return (1 - p) + p * np.exp(t)

   # Numerical derivatives at t=0 using central differences
   dt = 1e-7
   # First moment: E[X] = M'(0)
   M_prime_0 = (mgf_bernoulli(dt, p) - mgf_bernoulli(-dt, p)) / (2 * dt)
   # Second moment: E[X^2] = M''(0)
   M_double_prime_0 = (mgf_bernoulli(dt, p) - 2*mgf_bernoulli(0, p)
                        + mgf_bernoulli(-dt, p)) / dt**2

   print(f"Bernoulli(p={p}):")
   print(f"  E[X]   from MGF: {M_prime_0:.6f}  (exact: {p})")
   print(f"  E[X^2] from MGF: {M_double_prime_0:.6f}  (exact: {p})")
   print(f"  Var(X) from MGF: {M_double_prime_0 - M_prime_0**2:.6f}  "
         f"(exact: {p*(1-p):.6f})")

.. code-block:: python

   # MGF of Exponential(rate=lambda): M_X(t) = lambda / (lambda - t) for t < lambda
   # Used for claim-size modeling
   lam = 1 / 800  # rate parameter for minor claim distribution
   def mgf_exponential(t, lam):
       """M_X(t) = lambda / (lambda - t), defined for t < lambda."""
       return lam / (lam - t)

   # Extract moments
   dt = 1e-8
   M1 = (mgf_exponential(dt, lam) - mgf_exponential(-dt, lam)) / (2 * dt)
   M2 = (mgf_exponential(dt, lam) - 2*mgf_exponential(0, lam)
         + mgf_exponential(-dt, lam)) / dt**2

   print(f"\nExponential(rate={lam:.5f}), mean = 1/lambda = ${1/lam:.0f}:")
   print(f"  E[X]   from MGF: ${M1:>10.2f}  (exact: ${1/lam:.2f})")
   print(f"  E[X^2] from MGF: {M2:>12.2f}  (exact: {2/lam**2:.2f})")
   print(f"  Var(X) from MGF: {M2 - M1**2:>12.2f}  (exact: {1/lam**2:.2f})")

.. code-block:: python

   # MGF property: sum of independent variables
   # If X1, X2 ~ iid Exponential(lam), then X1+X2 ~ Gamma(2, lam)
   # M_{X1+X2}(t) = M_X1(t) * M_X2(t) = [lam/(lam-t)]^2
   def mgf_sum_of_two_exp(t, lam):
       return (lam / (lam - t)) ** 2

   # Extract E[X1+X2] — should be 2/lam = 2 * 800 = 1600
   M1_sum = (mgf_sum_of_two_exp(dt, lam) - mgf_sum_of_two_exp(-dt, lam)) / (2 * dt)
   print(f"\nSum of 2 independent Exp(1/800):")
   print(f"  E[X1+X2] from MGF product: ${M1_sum:>10.2f}  (exact: ${2/lam:.2f})")

   # Verify by simulation
   X1_sim = np.random.exponential(1/lam, size=100_000)
   X2_sim = np.random.exponential(1/lam, size=100_000)
   print(f"  E[X1+X2] simulated:        ${(X1_sim + X2_sim).mean():>10.2f}")


2.8 Joint Distributions
=========================

An insurance company does not insure just one risk.  A homeowner's policy
covers *both* fire damage and water damage.  To price the policy correctly, the
actuary must understand how these two claim amounts behave *together* --- does
a year with high fire risk also tend to have high water risk?  This requires
the **joint distribution**.

When we study two (or more) random variables simultaneously, we need their
**joint distribution**.  This tells us not only how each variable behaves on
its own, but how they behave *together*.

**Discrete case.**  The **joint PMF** of :math:`(X, Y)` is

.. math::

   p_{X,Y}(x,y) = P(X = x, Y = y).

**Continuous case.**  The **joint PDF** :math:`f_{X,Y}(x,y)` satisfies

.. math::

   P((X,Y) \in A) = \iint_A f_{X,Y}(x,y)\,dx\,dy.

.. code-block:: python

   # Joint distribution: two correlated stock returns
   # We model (R_tech, R_util) as bivariate normal
   cov_tech_util = rho_stocks * sigma_tech * sigma_util
   mean_vec = [mu_tech, mu_util]
   cov_matrix = [
       [sigma_tech**2,  cov_tech_util],
       [cov_tech_util,  sigma_util**2],
   ]

   # Draw 10,000 joint observations
   returns = np.random.multivariate_normal(mean_vec, cov_matrix, size=10_000)
   R_tech = returns[:, 0]
   R_util = returns[:, 1]

   print("Joint distribution of (R_tech, R_util):")
   print(f"  Sample means:  Tech = {R_tech.mean():.4f}, Utility = {R_util.mean():.4f}")
   print(f"  Sample stds:   Tech = {R_tech.std():.4f}, Utility = {R_util.std():.4f}")
   print(f"  Sample corr:   {np.corrcoef(R_tech, R_util)[0, 1]:.4f}  (target: {rho_stocks})")


2.8.1 Marginal Distributions
------------------------------

The portfolio analyst has the joint distribution of Tech and Utility returns.
But a client who holds only Tech stock cares about the **marginal** distribution
of Tech alone --- the joint information about Utility is irrelevant for them.

The **marginal** distribution of :math:`X` alone is obtained by "summing (or
integrating) out" :math:`Y`:

.. math::

   p_X(x) = \sum_y p_{X,Y}(x,y) \quad \text{(discrete)},

.. math::

   f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy \quad \text{(continuous)}.

.. code-block:: python

   # Marginal distributions: the individual stock returns
   # From the joint simulation above, the marginals should be normal
   # with the parameters we specified
   print("Marginal of R_tech:")
   print(f"  Mean: {R_tech.mean():.4f}  (target: {mu_tech})")
   print(f"  Std:  {R_tech.std():.4f}  (target: {sigma_tech})")
   _, p_val_tech = stats.shapiro(R_tech[:500])
   print(f"  Shapiro-Wilk p-value: {p_val_tech:.4f}  (large p => consistent with normal)")

   print(f"\nMarginal of R_util:")
   print(f"  Mean: {R_util.mean():.4f}  (target: {mu_util})")
   print(f"  Std:  {R_util.std():.4f}  (target: {sigma_util})")
   _, p_val_util = stats.shapiro(R_util[:500])
   print(f"  Shapiro-Wilk p-value: {p_val_util:.4f}")


2.8.2 Independence of Random Variables
----------------------------------------

Are Tech and Utility returns independent?  If they were, knowing Tech had a
great year would tell you nothing about Utility.  In practice, stocks are
rarely independent --- they share exposure to the broader economy.  But
independence is a critical concept: when it holds, joint distributions factor
into products of marginals, and the likelihood function decomposes.

Random variables :math:`X` and :math:`Y` are **independent** if their joint
distribution factors:

.. math::

   p_{X,Y}(x,y) = p_X(x)\,p_Y(y) \quad \text{for all } x, y.

(Similarly for PDFs.)  Independence means knowing :math:`X` tells you nothing
about :math:`Y`, and vice versa.

.. code-block:: python

   # Test for independence: compare joint density to product of marginals
   # We will use a simple binning approach
   from scipy.stats import chi2_contingency

   # Bin the returns into quartiles
   tech_bins = np.digitize(R_tech, np.quantile(R_tech, [0.25, 0.5, 0.75]))
   util_bins = np.digitize(R_util, np.quantile(R_util, [0.25, 0.5, 0.75]))

   # Contingency table
   table = np.zeros((4, 4), dtype=int)
   for t, u in zip(tech_bins, util_bins):
       table[t, u] += 1

   chi2, p_value, dof, expected = chi2_contingency(table)
   print(f"Chi-squared test for independence:")
   print(f"  chi2 = {chi2:.2f}, dof = {dof}, p-value = {p_value:.6f}")
   if p_value < 0.05:
       print(f"  Reject independence (p < 0.05): returns are NOT independent.")
   else:
       print(f"  Cannot reject independence at 5% level.")


2.9 Covariance and Correlation
================================

A portfolio manager is deciding how to split money between Tech and Utility
stocks.  If the two stocks tend to move together (positive covariance),
diversification helps less.  If they move in opposite directions (negative
covariance), holding both dramatically reduces risk.  **Covariance** quantifies
this co-movement.

**Covariance** measures the *linear association* between two random variables.
When :math:`X` tends to be above its mean at the same time :math:`Y` is above
its mean, the covariance is positive.  When they move in opposite directions,
it is negative.

.. math::

   \operatorname{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])].

Expanding:

.. math::

   \operatorname{Cov}(X, Y)
   &= E[XY - X\,E[Y] - E[X]\,Y + E[X]\,E[Y]] \\
   &= E[XY] - E[X]\,E[Y] - E[X]\,E[Y] + E[X]\,E[Y] \\
   &= E[XY] - E[X]\,E[Y].

.. code-block:: python

   # Covariance: compute both ways for Tech and Utility returns
   # Method 1: Definition — E[(X - mu_X)(Y - mu_Y)]
   Cov_def = np.mean((R_tech - R_tech.mean()) * (R_util - R_util.mean()))

   # Method 2: Shortcut — E[XY] - E[X]*E[Y]
   E_XY = np.mean(R_tech * R_util)
   Cov_shortcut = E_XY - R_tech.mean() * R_util.mean()

   # Theoretical value
   Cov_theory = rho_stocks * sigma_tech * sigma_util

   print("Cov(R_tech, R_util):")
   print(f"  Theoretical:        {Cov_theory:.6f}")
   print(f"  Definition method:  {Cov_def:.6f}")
   print(f"  Shortcut method:    {Cov_shortcut:.6f}")
   print(f"  NumPy:              {np.cov(R_tech, R_util, ddof=0)[0, 1]:.6f}")

**Properties:**

- :math:`\operatorname{Cov}(X, X) = \operatorname{Var}(X)`.
- :math:`\operatorname{Cov}(X, Y) = \operatorname{Cov}(Y, X)`.
- :math:`\operatorname{Cov}(aX + b, Y) = a\,\operatorname{Cov}(X, Y)`.
- If :math:`X, Y` are independent, :math:`\operatorname{Cov}(X,Y) = 0`
  (but not conversely).

**Variance of a sum (general):**

.. math::

   \operatorname{Var}(X + Y)
   &= E[(X + Y - E[X+Y])^2] \\
   &= E[((X-\mu_X) + (Y-\mu_Y))^2] \\
   &= E[(X-\mu_X)^2] + 2E[(X-\mu_X)(Y-\mu_Y)] + E[(Y-\mu_Y)^2] \\
   &= \operatorname{Var}(X) + 2\operatorname{Cov}(X,Y) + \operatorname{Var}(Y).

When :math:`X` and :math:`Y` are independent, :math:`\operatorname{Cov}(X,Y) = 0`,
recovering :math:`\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)`.

.. code-block:: python

   # Variance of a portfolio: Var(w*R_tech + (1-w)*R_util)
   # This is the formula that makes portfolio theory work.
   weights = np.linspace(0, 1, 11)  # 0% to 100% in Tech

   print(f"{'w_tech':>6s}  {'E[R_p]':>8s}  {'Std(R_p)':>9s}")
   print("-" * 28)
   for w in weights:
       E_Rp = w * mu_tech + (1 - w) * mu_util
       Var_Rp = (w**2 * sigma_tech**2
                 + (1 - w)**2 * sigma_util**2
                 + 2 * w * (1 - w) * Cov_theory)
       Std_Rp = np.sqrt(Var_Rp)
       print(f"{w:6.1%}  {E_Rp:8.2%}  {Std_Rp:9.2%}")

   print("\nNotice: 100% Utility is NOT the lowest-risk portfolio!")
   print("Diversification (mixing) reduces volatility because rho < 1.")

**Correlation** standardizes covariance to lie in :math:`[-1, 1]`:

.. math::

   \rho(X, Y) = \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)\,\operatorname{Var}(Y)}}.

Values of :math:`\rho = \pm 1` correspond to a perfect linear relationship.

.. code-block:: python

   # Correlation: compute from covariance and standard deviations
   # rho = Cov(X,Y) / (Std(X) * Std(Y))
   rho_computed = Cov_def / (R_tech.std() * R_util.std())

   print(f"Correlation(R_tech, R_util):")
   print(f"  From formula: Cov/(Std_X·Std_Y) = {Cov_def:.6f} / ({R_tech.std():.4f}×{R_util.std():.4f}) = {rho_computed:.4f}")
   print(f"  NumPy:        {np.corrcoef(R_tech, R_util)[0, 1]:.4f}")
   print(f"  Target:       {rho_stocks}")

.. code-block:: python

   # What does correlation LOOK like? Simulate 3 scenarios.
   for rho_val in [-0.8, 0.0, 0.8]:
       cov_val = rho_val * sigma_tech * sigma_util
       cov_mat = [[sigma_tech**2, cov_val], [cov_val, sigma_util**2]]
       sample = np.random.multivariate_normal(mean_vec, cov_mat, size=5000)
       r = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
       print(f"  rho = {rho_val:+.1f}: sample correlation = {r:+.4f}")


2.10 Conditional Distributions and Conditional Expectation
===========================================================

The portfolio analyst learns that Tech stock dropped 10 % this year.
Given this bad news, what should they *expect* for Utility?  This is not the
unconditional mean of Utility (6 %); the correlation between the stocks means
that a bad year for Tech shifts our expectation for Utility.  **Conditional
expectation** formalizes this shift.

**Discrete conditional PMF.**  Given :math:`Y = y` (with :math:`p_Y(y) > 0`),

.. math::

   p_{X|Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}.

**Continuous conditional PDF.**  Given :math:`Y = y` (with :math:`f_Y(y) > 0`),

.. math::

   f_{X|Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}.

**Conditional expectation.**

.. math::

   E[X \mid Y = y] = \sum_x x\,p_{X|Y}(x \mid y) \quad \text{(discrete)},
   \qquad
   E[X \mid Y = y] = \int_{-\infty}^{\infty} x\,f_{X|Y}(x \mid y)\,dx \quad \text{(continuous)}.

.. code-block:: python

   # Conditional expectation for bivariate normal:
   # E[R_util | R_tech = r] = mu_util + rho*(sigma_util/sigma_tech)*(r - mu_tech)
   #
   # If Tech dropped 10%:
   r_tech_observed = -0.10
   E_util_given_tech = (mu_util
                        + rho_stocks * (sigma_util / sigma_tech)
                        * (r_tech_observed - mu_tech))

   print(f"Observed R_tech = {r_tech_observed:.0%}")
   print(f"E[R_util | R_tech = {r_tech_observed:.0%}]")
   print(f"  = mu_util + rho·(sigma_util/sigma_tech)·(r_tech - mu_tech)")
   print(f"  = {mu_util} + {rho_stocks}·({sigma_util}/{sigma_tech})·({r_tech_observed} - {mu_tech})")
   print(f"  = {E_util_given_tech:.4f}  ({E_util_given_tech:.2%})")
   print(f"\nUnconditional E[R_util] = {mu_util:.2%}")
   print(f"The bad Tech year lowers our Utility expectation from {mu_util:.2%} to {E_util_given_tech:.2%}.")

.. code-block:: python

   # Verify by simulation: subset returns where R_tech is near -10%
   mask_near = (R_tech >= -0.12) & (R_tech <= -0.08)
   if mask_near.sum() > 0:
       conditional_mean_sim = R_util[mask_near].mean()
       print(f"Simulated E[R_util | R_tech near -10%] = {conditional_mean_sim:.4f}")
       print(f"Formula:                                  {E_util_given_tech:.4f}")
       print(f"(Based on {mask_near.sum()} observations with R_tech in [-12%, -8%])")

**Law of Total Expectation (iterated expectation).**

.. math::

   E[X] = E\!\big[E[X \mid Y]\big].

This is one of the most useful formulas in probability.  It says you can
compute :math:`E[X]` in two steps: first compute the conditional expectation
given :math:`Y`, then average that over :math:`Y`.

*Proof sketch (discrete).*

.. math::

   E[E[X \mid Y]]
   &= \sum_y E[X \mid Y=y]\,p_Y(y) \\
   &= \sum_y \left(\sum_x x\,p_{X|Y}(x \mid y)\right) p_Y(y) \\
   &= \sum_y \sum_x x\,\frac{p_{X,Y}(x,y)}{p_Y(y)}\,p_Y(y) \\
   &= \sum_x x \sum_y p_{X,Y}(x,y) \\
   &= \sum_x x\,p_X(x) = E[X]. \quad \square

.. code-block:: python

   # Law of total expectation: insurance payout
   # E[X] = E[E[X | claim_type]]
   #       = sum over types: E[X|type] * P(type)
   #
   # This is exactly how we computed E_X earlier — now we see it's an
   # instance of the law of total expectation!
   E_X_via_total_exp = np.sum(claim_costs * claim_probs)
   print(f"E[X] via law of total expectation:")
   for label, cost, prob in zip(claim_labels, claim_costs, claim_probs):
       print(f"  E[X|{label:>10s}]·P({label:>10s}) = ${cost:>7,} × {prob:.2f} = ${cost*prob:>8.2f}")
   print(f"  E[X] = ${E_X_via_total_exp:.2f}")

**Law of Total Variance.**

.. math::

   \operatorname{Var}(X) = E[\operatorname{Var}(X \mid Y)] + \operatorname{Var}(E[X \mid Y]).

.. code-block:: python

   # Law of total variance: insurance example
   # Suppose claim sizes within each type are exponentially distributed.
   # Var(X|type) = (mean_type)^2 for exponential.
   # E[X|type] = mean_type.
   claim_vars = claim_costs**2  # Var of Exp(1/mu) = mu^2; for "None", 0^2=0

   # E[Var(X|type)] = sum P(type) * Var(X|type)
   E_var_given_type = np.sum(claim_probs * claim_vars)

   # Var(E[X|type]) = E[(E[X|type])^2] - (E[E[X|type]])^2
   E_cond_mean_sq = np.sum(claim_probs * claim_costs**2)
   E_cond_mean = np.sum(claim_probs * claim_costs)  # = E[X]
   Var_cond_mean = E_cond_mean_sq - E_cond_mean**2

   Var_total = E_var_given_type + Var_cond_mean

   print("Law of Total Variance:")
   print(f"  E[Var(X|Type)]          = {E_var_given_type:>14,.2f}")
   print(f"  Var(E[X|Type])          = {Var_cond_mean:>14,.2f}")
   print(f"  Total = E[Var] + Var[E] = {Var_total:>14,.2f}")
   print(f"  Direct Var(X)           = {Var_X_shortcut:>14,.2f}")
   print(f"  (They match because Var(X|type=None) = 0 and claim_costs are")
   print(f"   the conditional means, so this reduces to the same calculation.)")


2.11 The Law of Large Numbers
===============================

The insurance company sells 50,000 policies.  Can the CEO sleep well knowing
that the average claim cost will be close to the theoretical expected value ---
so that premium income will cover payouts?  The **Law of Large Numbers**
provides exactly this guarantee: with enough policies, the sample average
converges to the true mean.

The **Law of Large Numbers** (LLN) says that the sample average converges to
the population mean as the sample size grows.  This is the theoretical
guarantee behind the idea that "more data is better."

Let :math:`X_1, X_2, \dots` be i.i.d. random variables with :math:`E[X_i] = \mu`
and :math:`\operatorname{Var}(X_i) = \sigma^2 < \infty`.  Define the sample
mean

.. math::

   \bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i.

**Weak Law of Large Numbers.**  For every :math:`\varepsilon > 0`,

.. math::

   P(|\bar{X}_n - \mu| > \varepsilon) \to 0 \quad \text{as } n \to \infty.

*Intuition.*  The variance of the sample mean is

.. math::

   \operatorname{Var}(\bar{X}_n)
   = \operatorname{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right)
   = \frac{1}{n^2} \sum_{i=1}^n \operatorname{Var}(X_i)
   = \frac{\sigma^2}{n}.

As :math:`n \to \infty`, the variance shrinks to zero, so the sample mean
concentrates around :math:`\mu`.

A quick formal proof uses **Chebyshev's inequality**, which states that for
any random variable :math:`Z` with mean :math:`\mu_Z` and variance
:math:`\sigma_Z^2`,

.. math::

   P(|Z - \mu_Z| > \varepsilon) \leq \frac{\sigma_Z^2}{\varepsilon^2}.

Applying this to :math:`\bar{X}_n`:

.. math::

   P(|\bar{X}_n - \mu| > \varepsilon) \leq \frac{\sigma^2}{n\varepsilon^2} \to 0.
   \quad \square

The **Strong Law of Large Numbers** strengthens this to almost-sure
convergence:

.. math::

   P\!\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1.

The LLN justifies using sample averages as estimates of population quantities
and is the theoretical backbone of frequentist statistics.

.. code-block:: python

   # LLN in action: insurance — average claim converges to E[X]
   # Simulate claims for increasing numbers of policyholders
   mu_claim = E_X  # theoretical mean ($530)
   max_n = 50_000

   # Generate all claims at once, then compute running averages
   all_claim_types = np.random.choice(len(claim_probs), size=max_n, p=claim_probs)
   all_claims = claim_costs[all_claim_types]
   running_mean = np.cumsum(all_claims) / np.arange(1, max_n + 1)

   # Report at selected sample sizes
   checkpoints = [10, 100, 500, 1_000, 5_000, 10_000, 50_000]
   print(f"{'n':>8s}  {'X_bar_n':>10s}  {'|X_bar - mu|':>14s}  {'sigma^2/n':>10s}")
   print("-" * 50)
   for n in checkpoints:
       x_bar = running_mean[n - 1]
       chebyshev_bound = Var_X_shortcut / n  # variance of sample mean
       print(f"{n:>8,}  ${x_bar:>9.2f}  ${abs(x_bar - mu_claim):>13.2f}  {chebyshev_bound:>10.2f}")

   print(f"\nTrue E[X] = ${mu_claim:.2f}")
   print("As n grows, the sample mean converges to the true mean.")

.. code-block:: python

   # Var(X_bar_n) = sigma^2 / n — verify the shrinkage
   n_trials = 5_000
   sample_sizes = [10, 50, 200, 1000, 5000]

   print(f"{'n':>6s}  {'Var(X_bar) theory':>18s}  {'Var(X_bar) simulated':>22s}")
   print("-" * 52)
   for n in sample_sizes:
       # Simulate n_trials sample means, each from n claims
       x_bars = np.array([
           claim_costs[np.random.choice(len(claim_probs), size=n, p=claim_probs)].mean()
           for _ in range(n_trials)
       ])
       var_theory = Var_X_shortcut / n
       var_sim = x_bars.var()
       print(f"{n:>6,}  {var_theory:>18,.2f}  {var_sim:>22,.2f}")


2.12 The Central Limit Theorem
================================

The insurance company has 50,000 policies.  Even though individual claims
follow a highly skewed distribution (most are zero, a few are very large), the
*total* payout across all policies is approximately normal.  This seemingly
magical result is the **Central Limit Theorem** --- arguably the most important
theorem in statistics.  It explains why the normal distribution appears
everywhere and why confidence intervals work.

**Theorem.**  Let :math:`X_1, X_2, \dots` be i.i.d. with mean :math:`\mu` and
variance :math:`\sigma^2 \in (0, \infty)`.  Define the standardized sample
mean

.. math::

   Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}}
       = \frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}}.

Then, as :math:`n \to \infty`,

.. math::

   Z_n \xrightarrow{d} Z \sim N(0, 1).

That is, the distribution of :math:`Z_n` converges to a standard normal.

Equivalently,

.. math::

   \bar{X}_n \;\dot\sim\; N\!\left(\mu, \frac{\sigma^2}{n}\right)
   \quad \text{for large } n.

**Intuition.**  No matter what the original distribution of the :math:`X_i`
looks like (as long as it has finite mean and variance), the sample average
of many observations is approximately normally distributed.  This is why the
normal distribution is so prevalent --- many real-world measurements are sums
or averages of many small, independent contributions.

**Proof sketch using MGFs.**  The standardized variable has MGF

.. math::

   M_{Z_n}(t) = \left[M_{X_1}\!\left(\frac{t}{\sigma\sqrt{n}}\right)\,
   e^{-\mu t / (\sigma\sqrt{n})}\right]^n.

Expanding the MGF of :math:`X_1` in a Taylor series around :math:`t = 0` and
taking the limit as :math:`n \to \infty` yields :math:`e^{t^2/2}`, which is
the MGF of the standard normal.

The CLT is critical for constructing confidence intervals, hypothesis tests,
and understanding the behavior of maximum likelihood estimators
(see :ref:`ch4_likelihood` and later chapters).

.. code-block:: python

   # CLT in action: watch the distribution of Z_n become standard normal
   # Source distribution: our insurance claims (highly skewed!)
   mu_claim = E_X
   sigma_claim = np.sqrt(Var_X_shortcut)

   print("CLT demonstration: insurance claims (highly skewed source)\n")
   print(f"Source: E[X] = ${mu_claim:.2f}, Std(X) = ${sigma_claim:,.2f}\n")
   print(f"{'n':>5s}  {'mean(Z_n)':>10s}  {'std(Z_n)':>10s}  {'theory std':>10s}  {'Shapiro p':>10s}")
   print("-" * 58)

   for n in [1, 5, 10, 30, 50, 100, 500]:
       n_reps = 10_000
       x_bars = np.array([
           claim_costs[np.random.choice(len(claim_probs), size=n, p=claim_probs)].mean()
           for _ in range(n_reps)
       ])
       # Standardize: Z_n = (X_bar - mu) / (sigma / sqrt(n))
       Z_n = (x_bars - mu_claim) / (sigma_claim / np.sqrt(n))

       _, p_val = stats.shapiro(Z_n[:500])
       print(f"{n:>5d}  {Z_n.mean():>10.4f}  {Z_n.std():>10.4f}  {1.0:>10.4f}  {p_val:>10.4f}")

   print("\nAs n increases, mean(Z_n) -> 0, std(Z_n) -> 1, and Shapiro p increases,")
   print("confirming convergence to N(0,1).")

.. code-block:: python

   # CLT with a different source: Exponential(rate=2)
   # This shows the CLT is universal — it works for ANY distribution
   lam_clt = 2.0
   mu_exp = 1 / lam_clt        # 0.5
   sigma_exp = 1 / lam_clt     # 0.5

   print(f"\nCLT with Exponential(rate={lam_clt}): mu={mu_exp}, sigma={sigma_exp}\n")
   print(f"{'n':>5s}  {'mean(X_bar)':>12s}  {'std(X_bar)':>12s}  {'theory std':>12s}  {'Shapiro p':>10s}")
   print("-" * 60)

   for n in [1, 5, 30, 100]:
       sample_means = np.array([
           np.random.exponential(1/lam_clt, size=n).mean()
           for _ in range(10_000)
       ])
       z_scores = (sample_means - mu_exp) / (sigma_exp / np.sqrt(n))
       _, p_val = stats.shapiro(z_scores[:500])
       print(f"{n:>5d}  {sample_means.mean():>12.4f}  {sample_means.std():>12.4f}  "
             f"{sigma_exp/np.sqrt(n):>12.4f}  {p_val:>10.4f}")

.. code-block:: python

   # CLT application: poll sampling — why 1,000 respondents are often enough
   true_p = 0.53
   n_respondents = 1000
   n_polls = 10_000

   # Simulate many polls
   polls = np.random.binomial(n_respondents, true_p, size=n_polls) / n_respondents
   se_theory = np.sqrt(true_p * (1 - true_p) / n_respondents)
   margin = 1.96 * se_theory

   print(f"Poll sampling (CLT in action):")
   print(f"  True proportion:       {true_p}")
   print(f"  n respondents:         {n_respondents}")
   print(f"  Theoretical SE:        {se_theory:.4f}")
   print(f"  Simulated SE:          {polls.std():.4f}")
   print(f"  95% margin of error:   +/- {margin:.4f}")
   print(f"  95% CI:                ({true_p - margin:.4f}, {true_p + margin:.4f})")
   coverage = np.mean(np.abs(polls - true_p) <= margin)
   print(f"  Simulated coverage:    {coverage:.4f}  (should be ~0.95)")


2.13 Summary
==============

- A **random variable** maps outcomes to numbers, enabling us to do arithmetic
  with randomness.
- **PMFs** describe discrete variables; **PDFs** describe continuous ones;
  **CDFs** work for both.
- **Expectation** is the probability-weighted average, and it is **linear**.
- **Variance** measures spread; the formula
  :math:`\operatorname{Var}(X) = E[X^2] - (E[X])^2` is indispensable.
- **MGFs** encode all moments and are a powerful tool for proving distributional
  results.
- **Joint, marginal, and conditional** distributions extend these ideas to
  multiple random variables.
- **Covariance** and **correlation** quantify linear association.
- The **Law of Large Numbers** guarantees that averages converge to means.
- The **Central Limit Theorem** guarantees that averages are approximately
  normal.

Throughout this chapter we computed every formula using two concrete scenarios
--- insurance pricing and stock portfolio analysis --- and verified results
with simulation.  The insurance example showed how :math:`E[X]` sets the break-even
premium, how :math:`\operatorname{Var}(X)` measures risk, and how the CLT
makes the total payout predictable.  The stock example showed how covariance
and correlation drive diversification benefits.

With these tools in hand, we are ready to study the most important
distributions in detail in :ref:`ch3_distributions`.
