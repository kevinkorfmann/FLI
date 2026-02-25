.. _ch3_distributions:

====================================
Chapter 3: Common Distributions
====================================

Imagine you are the operations analyst for a busy hospital emergency department.
Every shift, you face questions that are fundamentally statistical: How many
patients will arrive in the next hour?  How long until the next trauma case?
What fraction of arrivals need intensive care?  Are the vital signs of this
patient abnormally high?

Each of these questions points to a different probability distribution.  This
chapter introduces the distributions you will encounter most often in
likelihood-based inference, and we will use this **hospital ED scenario** as a
running thread that connects them all.  By the end, you will see that the
Poisson, Exponential, Binomial, Normal, and Gamma are not isolated formulas ---
they are different views of the same underlying reality.

For every distribution we provide the probability mass/density function, the
parameters and their meaning, the support, the mean (derived), the variance
(derived), the moment generating function, and code that computes and verifies
each quantity.

When you encounter a new dataset and need to choose a model, you will come back
here.  When you need to derive a likelihood function in :ref:`ch4_likelihood`,
the formulas in this chapter are the starting point.

.. contents:: Chapter Contents
   :local:
   :depth: 2


.. _sec_discrete_distributions:

3.1 Discrete Distributions
============================


3.1.1 Bernoulli Distribution
------------------------------

A single trial with two outcomes: success (1) or failure (0).  This is the
simplest possible random variable --- a single yes/no question --- and it is
the building block for many more complex distributions.

Back at the hospital, each arriving patient either is or is not a trauma case.
If we label trauma as "success" (:math:`X=1`) and non-trauma as "failure"
(:math:`X=0`), a single patient's classification is a Bernoulli trial.

**PMF.**

.. math::

   p(x) = p^x (1-p)^{1-x}, \qquad x \in \{0, 1\}.

**Parameters.** :math:`p \in [0,1]` (probability of success).

**Support.** :math:`\{0, 1\}`.

Let us compute this PMF directly and verify it against SciPy.

.. code-block:: python

   # Bernoulli PMF: compute by hand and verify with SciPy
   import numpy as np
   from scipy import stats

   p = 0.15  # probability a patient is a trauma case

   # PMF from the formula: p(x) = p^x * (1-p)^(1-x)
   for x in [0, 1]:
       pmf_manual = p**x * (1 - p)**(1 - x)
       pmf_scipy  = stats.bernoulli.pmf(x, p)
       print(f"  P(X={x}) = {pmf_manual:.4f}  (scipy: {pmf_scipy:.4f})")

**Mean.**

.. math::

   E[X] = 0 \cdot (1-p) + 1 \cdot p = p.

**Variance.**

.. math::

   E[X^2] = 0^2(1-p) + 1^2 p = p, \qquad
   \operatorname{Var}(X) = p - p^2 = p(1-p).

**MGF.**

.. math::

   M_X(t) = (1-p) + p\,e^t.

.. code-block:: python

   # Bernoulli: verify E[X] and Var(X) by simulation
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   p = 0.15  # trauma probability
   samples = stats.bernoulli.rvs(p, size=100_000)

   E_theory   = p
   Var_theory = p * (1 - p)
   print(f"Bernoulli(p={p}) -- 'Is this patient a trauma case?'")
   print(f"  Theory:     E[X] = {E_theory:.4f},  Var(X) = {Var_theory:.4f}")
   print(f"  Simulation: E[X] = {samples.mean():.4f},  Var(X) = {samples.var():.4f}")

So the Bernoulli answers the simplest hospital question: for one patient, is it
a trauma case or not?  When we observe many patients and count the total number
of trauma cases, we arrive at the Binomial.


3.1.2 Binomial Distribution
------------------------------

Now suppose :math:`n` patients arrive during one shift, and each independently
has probability :math:`p = 0.15` of being a trauma case.  The total number of
trauma cases :math:`X` follows a Binomial distribution:
:math:`X \sim \text{Bin}(n, p)`.

The binomial coefficient :math:`\binom{n}{x}` accounts for all the different
orderings in which :math:`x` successes can occur within :math:`n` trials.

**PMF.**

.. math::

   p(x) = \binom{n}{x} p^x (1-p)^{n-x}, \qquad x \in \{0, 1, \dots, n\}.

**Parameters.** :math:`n \in \{1,2,\dots\}` (number of trials),
:math:`p \in [0,1]` (success probability).

.. code-block:: python

   # Binomial PMF: how many trauma cases in n=30 arrivals?
   import numpy as np
   from scipy import stats
   from scipy.special import comb

   n = 30    # patients arriving during one shift
   p = 0.15  # trauma probability

   print(f"Bin(n={n}, p={p}): PMF for selected x values")
   print(f"{'x':>4s}  {'formula':>10s}  {'scipy':>10s}")
   for x in range(0, 11):
       pmf_manual = comb(n, x, exact=True) * p**x * (1 - p)**(n - x)
       pmf_scipy  = stats.binom.pmf(x, n, p)
       print(f"{x:4d}  {pmf_manual:10.6f}  {pmf_scipy:10.6f}")

**Mean derivation.**  Since :math:`X = \sum_{i=1}^n X_i` where each
:math:`X_i \sim \text{Bernoulli}(p)`, by linearity of expectation:

.. math::

   E[X] = \sum_{i=1}^n E[X_i] = np.

**Variance derivation.**  The :math:`X_i` are independent, so

.. math::

   \operatorname{Var}(X) = \sum_{i=1}^n \operatorname{Var}(X_i) = np(1-p).

**MGF.**  Using independence:

.. math::

   M_X(t) = \prod_{i=1}^n M_{X_i}(t) = \left[(1-p) + pe^t\right]^n.

.. code-block:: python

   # Binomial: verify mean and variance by simulation
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   n, p = 30, 0.15
   samples = stats.binom.rvs(n, p, size=100_000)

   E_theory   = n * p
   Var_theory = n * p * (1 - p)
   print(f"Bin(n={n}, p={p}) -- 'How many trauma cases this shift?'")
   print(f"  Theory:     E[X] = {E_theory:.2f},  Var(X) = {Var_theory:.4f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.4f}")

   # Text histogram: frequency of each count
   print(f"\n  Distribution of trauma counts (out of {n} patients):")
   for x in range(0, 12):
       bar = '#' * int(stats.binom.pmf(x, n, p) * 200)
       print(f"    x={x:2d}: {bar} {stats.binom.pmf(x, n, p):.4f}")

**Relationship to Bernoulli.** The Binomial with :math:`n=1` is the Bernoulli.
We will see in Section 3.3 that when :math:`n` is large and :math:`p` is small,
the Binomial is well-approximated by the Poisson; and when :math:`n` is large
with moderate :math:`p`, by the Normal (CLT).


3.1.3 Poisson Distribution
----------------------------

Here is perhaps the most important discrete distribution for our hospital
scenario.  Patient arrivals to the ED happen at random but at a roughly
constant average rate --- say :math:`\lambda = 4.5` patients per hour.  The
count of arrivals in any given hour follows a Poisson distribution:
:math:`X \sim \text{Pois}(\lambda)`.

What's the intuition?  If events are independent and the rate does not change,
the count in a fixed time interval is Poisson.  This applies to patient
arrivals, phone calls, mutations on a chromosome, and many other settings.

**PMF.**

.. math::

   p(x) = \frac{e^{-\lambda}\,\lambda^x}{x!}, \qquad x \in \{0, 1, 2, \dots\}.

**Parameter.** :math:`\lambda > 0` (rate, average number of events).

.. code-block:: python

   # Poisson PMF: patient arrivals per hour (lambda = 4.5)
   import numpy as np
   from scipy import stats
   from math import factorial, exp

   lam = 4.5  # average patients per hour

   print(f"Poisson(lambda={lam}): PMF for x = 0..12")
   print(f"{'x':>4s}  {'formula':>10s}  {'scipy':>10s}")
   for x in range(13):
       pmf_manual = exp(-lam) * lam**x / factorial(x)
       pmf_scipy  = stats.poisson.pmf(x, lam)
       print(f"{x:4d}  {pmf_manual:10.6f}  {pmf_scipy:10.6f}")

**Mean derivation.**

.. math::

   E[X] = \sum_{x=0}^{\infty} x \frac{e^{-\lambda}\lambda^x}{x!}
        = e^{-\lambda} \sum_{x=1}^{\infty} \frac{\lambda^x}{(x-1)!}
        = e^{-\lambda} \lambda \sum_{k=0}^{\infty} \frac{\lambda^k}{k!}
        = e^{-\lambda} \lambda\, e^{\lambda} = \lambda.

**Variance derivation.**  First compute :math:`E[X(X-1)]`:

.. math::

   E[X(X-1)] = \sum_{x=2}^{\infty} x(x-1)\frac{e^{-\lambda}\lambda^x}{x!}
   = e^{-\lambda}\lambda^2 \sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = \lambda^2.

Then :math:`E[X^2] = E[X(X-1)] + E[X] = \lambda^2 + \lambda`, and

.. math::

   \operatorname{Var}(X) = \lambda^2 + \lambda - \lambda^2 = \lambda.

So the Poisson has the notable property that its **mean equals its variance**.
This is a quick diagnostic: if your count data have a sample mean close to the
sample variance, the Poisson may be a good fit.

.. code-block:: python

   # Poisson: verify mean = variance = lambda by simulation
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5  # patients per hour
   samples = stats.poisson.rvs(lam, size=100_000)

   print(f"Poisson(lambda={lam}) -- 'Patient arrivals per hour'")
   print(f"  Theory:     E[X] = {lam:.2f},  Var(X) = {lam:.2f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.2f}")
   print(f"  Note: mean ~ variance is the hallmark of the Poisson")

   # Practical question: P(more than 8 patients in an hour)?
   P_gt_8 = 1 - stats.poisson.cdf(8, lam)
   sim_gt_8 = np.mean(samples > 8)
   print(f"\n  P(X > 8) = {P_gt_8:.4f}  (simulated: {sim_gt_8:.4f})")
   print(f"  This is the probability of a surge hour for staffing decisions.")

**MGF.**

.. math::

   M_X(t) = \sum_{x=0}^{\infty} e^{tx}\frac{e^{-\lambda}\lambda^x}{x!}
   = e^{-\lambda}\sum_{x=0}^{\infty}\frac{(\lambda e^t)^x}{x!}
   = e^{-\lambda}\,e^{\lambda e^t}
   = \exp\!\left(\lambda(e^t - 1)\right).

**Connection to Exponential.** If patient *counts* per hour are Poisson with
rate :math:`\lambda`, then the *time between* consecutive arrivals is
Exponential with the same rate :math:`\lambda`.  We will verify this connection
in code after introducing the Exponential distribution in Section 3.2.3.


3.1.4 Geometric Distribution
-------------------------------

Back in the ED, suppose each arriving patient independently has probability
:math:`p = 0.15` of being a trauma case.  You want to know: how many patients
do we see before the first trauma case?  That count follows a Geometric
distribution: :math:`X \sim \text{Geom}(p)`.

**PMF.**

.. math::

   p(x) = (1-p)^{x-1}\,p, \qquad x \in \{1, 2, 3, \dots\}.

**Parameter.** :math:`p \in (0,1]`.

.. code-block:: python

   # Geometric PMF: patients until first trauma case
   import numpy as np
   from scipy import stats

   p = 0.15  # trauma probability

   print(f"Geom(p={p}): P(first trauma on patient x)")
   print(f"{'x':>4s}  {'formula':>10s}  {'scipy':>10s}")
   for x in range(1, 16):
       pmf_manual = (1 - p)**(x - 1) * p
       pmf_scipy  = stats.geom.pmf(x, p)
       print(f"{x:4d}  {pmf_manual:10.6f}  {pmf_scipy:10.6f}")

**Mean derivation.**  Let :math:`q = 1-p`.

.. math::

   E[X] = \sum_{x=1}^{\infty} x\,q^{x-1}\,p = p \sum_{x=1}^{\infty} x\,q^{x-1}.

Using the identity :math:`\sum_{x=1}^{\infty} x\,r^{x-1} = \frac{1}{(1-r)^2}`
for :math:`|r| < 1`:

.. math::

   E[X] = p \cdot \frac{1}{(1-q)^2} = p \cdot \frac{1}{p^2} = \frac{1}{p}.

**Variance derivation.**  Similarly,
:math:`E[X(X-1)] = \frac{2q}{p^2}`, so
:math:`E[X^2] = \frac{2q}{p^2} + \frac{1}{p}`, and

.. math::

   \operatorname{Var}(X) = E[X^2] - (E[X])^2
   = \frac{2q}{p^2} + \frac{1}{p} - \frac{1}{p^2}
   = \frac{q}{p^2}
   = \frac{1-p}{p^2}.

**MGF.**

.. math::

   M_X(t) = \frac{pe^t}{1 - (1-p)e^t}, \qquad t < -\ln(1-p).

.. code-block:: python

   # Geometric: verify E[X] = 1/p and Var(X) = (1-p)/p^2
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   p = 0.15
   samples = stats.geom.rvs(p, size=100_000)

   E_theory   = 1 / p
   Var_theory = (1 - p) / p**2
   print(f"Geom(p={p}) -- 'How many patients until first trauma?'")
   print(f"  Theory:     E[X] = {E_theory:.2f},  Var(X) = {Var_theory:.2f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.2f}")
   print(f"  On average, you see {E_theory:.1f} patients before the first trauma case.")


3.1.5 Negative Binomial Distribution
---------------------------------------

The Geometric counts patients until the *first* trauma case.  What if you want
to know how many patients arrive until the *r-th* trauma case?  That is the
Negative Binomial: :math:`X \sim \text{NegBin}(r, p)`.

**PMF.**

.. math::

   p(x) = \binom{x-1}{r-1} p^r (1-p)^{x-r}, \qquad x \in \{r, r+1, r+2, \dots\}.

**Parameters.** :math:`r \in \{1,2,\dots\}` (number of successes needed),
:math:`p \in (0,1]`.

Note: When :math:`r = 1` the Negative Binomial reduces to the Geometric.

**Mean.**  Since :math:`X` is the sum of :math:`r` independent
:math:`\text{Geom}(p)` random variables,

.. math::

   E[X] = \frac{r}{p}.

**Variance.**

.. math::

   \operatorname{Var}(X) = \frac{r(1-p)}{p^2}.

**MGF.**

.. math::

   M_X(t) = \left(\frac{pe^t}{1 - (1-p)e^t}\right)^r, \qquad t < -\ln(1-p).

.. code-block:: python

   # Negative Binomial: patients until r=3 trauma cases
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   r, p = 3, 0.15
   # scipy NegBin counts failures before r-th success; shift for total trials
   rv = stats.nbinom(r, p)
   samples = rv.rvs(size=100_000) + r  # shift to total trials

   E_theory   = r / p
   Var_theory = r * (1 - p) / p**2
   print(f"NegBin(r={r}, p={p}) -- 'Patients until {r}rd trauma case'")
   print(f"  Theory:     E[X] = {E_theory:.2f},  Var(X) = {Var_theory:.2f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.2f}")

   # Verify NegBin(r=1,p) = Geom(p)
   negbin1 = stats.nbinom.rvs(1, p, size=100_000) + 1
   geom    = stats.geom.rvs(p, size=100_000)
   print(f"\n  NegBin(1, {p}) mean = {negbin1.mean():.2f}")
   print(f"  Geom({p})      mean = {geom.mean():.2f}")
   print(f"  These match: NegBin(r=1) = Geometric")


3.1.6 Hypergeometric Distribution
------------------------------------

Now a different sampling question.  The hospital pharmacy has :math:`N = 200`
vials of a certain drug, of which :math:`K = 50` are from a recalled batch.
A pharmacist randomly selects :math:`n = 30` vials for inspection (without
replacement).  How many recalled vials :math:`X` are in the sample?

This is sampling *without replacement*, so the draws are not independent ---
and that is exactly what the Hypergeometric models.

**PMF.**

.. math::

   p(x) = \frac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}},
   \qquad x \in \{\max(0, n+K-N), \dots, \min(n, K)\}.

**Parameters.** :math:`N` (population size), :math:`K` (number of successes in
population), :math:`n` (draw size).

**Mean.**

.. math::

   E[X] = n\,\frac{K}{N}.

*Derivation.*  Write :math:`X = \sum_{i=1}^n X_i` where :math:`X_i = 1` if
the :math:`i`-th drawn item is a success.  By symmetry,
:math:`E[X_i] = K/N` for every :math:`i`, so :math:`E[X] = nK/N`.

**Variance.**

.. math::

   \operatorname{Var}(X) = n\,\frac{K}{N}\,\frac{N-K}{N}\,\frac{N-n}{N-1}.

The factor :math:`(N-n)/(N-1)` is the **finite-population correction**;
it is close to 1 when :math:`n \ll N`, recovering the Binomial variance.

.. code-block:: python

   # Hypergeometric: recalled vials in a pharmacy sample
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   N, K, n = 200, 50, 30  # population, recalled, sample size
   rv = stats.hypergeom(N, K, n)
   samples = rv.rvs(size=100_000)

   E_theory   = n * K / N
   Var_theory = n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))
   print(f"Hypergeometric(N={N}, K={K}, n={n})")
   print(f"  Theory:     E[X] = {E_theory:.2f},  Var(X) = {Var_theory:.4f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.4f}")

   # Compare to Binomial (sampling WITH replacement)
   p_binom = K / N
   Var_binom = n * p_binom * (1 - p_binom)
   print(f"\n  Binomial Var (with replacement):       {Var_binom:.4f}")
   print(f"  Hypergeometric Var (without):           {Var_theory:.4f}")
   print(f"  Finite-population correction factor:    {(N-n)/(N-1):.4f}")
   print(f"  The correction reduces variance because drawing without")
   print(f"  replacement gives less variable results.")

**MGF.**  A closed-form MGF exists but is rarely used; the Hypergeometric is
typically handled via its PMF directly.


.. _sec_continuous_distributions:

3.2 Continuous Distributions
==============================


3.2.1 Uniform Distribution
----------------------------

Before a patient arrives, you might have no information about when they will
show up within the next hour.  If every moment is equally likely, the arrival
time (measured in minutes within the hour) follows a Uniform distribution.
Write :math:`X \sim \text{Uniform}(a, b)`.

The uniform distribution is the simplest continuous distribution --- it
represents "maximum ignorance" over an interval.

**PDF.**

.. math::

   f(x) = \frac{1}{b - a}, \qquad a \leq x \leq b.

**Parameters.** :math:`a < b` (endpoints).

**Mean derivation.**

.. math::

   E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx
        = \frac{1}{b-a}\,\frac{x^2}{2}\bigg|_a^b
        = \frac{b^2 - a^2}{2(b-a)}
        = \frac{a+b}{2}.

**Variance derivation.**

.. math::

   E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx
           = \frac{b^3 - a^3}{3(b-a)}
           = \frac{a^2 + ab + b^2}{3}.

.. math::

   \operatorname{Var}(X) = \frac{a^2+ab+b^2}{3} - \left(\frac{a+b}{2}\right)^2
   = \frac{(b-a)^2}{12}.

**MGF.**

.. math::

   M_X(t) = \frac{e^{tb} - e^{ta}}{t(b-a)}, \qquad t \neq 0.

.. code-block:: python

   # Uniform distribution: arrival time within the hour
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   a, b = 0, 60  # minutes within the hour
   rv = stats.uniform(loc=a, scale=b - a)
   samples = rv.rvs(size=100_000)

   E_theory   = (a + b) / 2
   Var_theory = (b - a)**2 / 12
   print(f"Uniform({a}, {b}) -- 'Arrival minute within the hour'")
   print(f"  Theory:     E[X] = {E_theory:.2f},  Var(X) = {Var_theory:.2f}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.2f}")

   # PDF is constant: verify
   x_test = [10, 30, 50]
   for x in x_test:
       print(f"  f({x}) = {rv.pdf(x):.6f}  (should be {1/(b-a):.6f})")


3.2.2 Normal (Gaussian) Distribution
---------------------------------------

A patient's heart rate is a continuous measurement influenced by many small,
roughly independent factors: fitness level, anxiety, caffeine intake, pain.
The Central Limit Theorem (:ref:`ch2_random_variables`) tells us that such
sums of many small effects tend toward the Normal distribution.  A healthy
resting heart rate might be modeled as :math:`X \sim N(\mu, \sigma^2)` with
:math:`\mu = 80` bpm and :math:`\sigma = 12` bpm.

**PDF.**

.. math::

   f(x) = \frac{1}{\sqrt{2\pi}\,\sigma}
          \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right),
          \qquad x \in \mathbb{R}.

**Parameters.** :math:`\mu \in \mathbb{R}` (mean), :math:`\sigma^2 > 0`
(variance).

**Support.** :math:`(-\infty, \infty)`.

.. code-block:: python

   # Normal PDF: heart rate distribution
   import numpy as np
   from scipy import stats

   mu    = 80   # mean heart rate (bpm)
   sigma = 12   # standard deviation

   # Compute PDF at several heart rate values
   print(f"Normal(mu={mu}, sigma={sigma}): heart rate PDF")
   print(f"{'HR (bpm)':>10s}  {'f(x)':>10s}")
   for hr in [50, 60, 68, 80, 92, 100, 110]:
       # PDF from the formula
       fx = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(hr - mu)**2 / (2 * sigma**2))
       print(f"{hr:10d}  {fx:10.6f}")

**Mean.**  By symmetry of the bell curve around :math:`\mu`, :math:`E[X] = \mu`.

*Formal derivation.*  Substitute :math:`z = (x-\mu)/\sigma`:

.. math::

   E[X] = \int_{-\infty}^{\infty} x \, f(x)\,dx
   = \int_{-\infty}^{\infty}(\sigma z + \mu)\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz
   = \sigma\underbrace{\int_{-\infty}^{\infty}z\,\frac{e^{-z^2/2}}{\sqrt{2\pi}}\,dz}_{= 0 \text{ (odd function)}}
     + \mu\underbrace{\int_{-\infty}^{\infty}\frac{e^{-z^2/2}}{\sqrt{2\pi}}\,dz}_{= 1}
   = \mu.

**Variance derivation.**

.. math::

   \operatorname{Var}(X) = E[(X-\mu)^2]
   = \int_{-\infty}^{\infty}(x-\mu)^2 f(x)\,dx
   = \sigma^2 \int_{-\infty}^{\infty} z^2 \frac{e^{-z^2/2}}{\sqrt{2\pi}}\,dz.

The last integral evaluates to 1 (by integration by parts or by recognizing
:math:`E[Z^2] = \operatorname{Var}(Z) + (E[Z])^2 = 1 + 0` for
:math:`Z \sim N(0,1)`).  Hence :math:`\operatorname{Var}(X) = \sigma^2`.

.. code-block:: python

   # Normal: verify moments and compute clinically useful probabilities
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   mu, sigma = 80, 12
   rv = stats.norm(loc=mu, scale=sigma)
   samples = rv.rvs(size=100_000)

   print(f"Normal(mu={mu}, sigma^2={sigma**2}) -- 'Resting heart rate'")
   print(f"  Theory:     E[X] = {mu},  Var(X) = {sigma**2}")
   print(f"  Simulation: E[X] = {samples.mean():.2f},  Var(X) = {samples.var():.2f}")
   print(f"\n  68-95-99.7 rule:")
   print(f"  P({mu-sigma} < X < {mu+sigma}) = "
         f"{rv.cdf(mu+sigma) - rv.cdf(mu-sigma):.4f}  (~0.6827)")
   print(f"  P({mu-2*sigma} < X < {mu+2*sigma}) = "
         f"{rv.cdf(mu+2*sigma) - rv.cdf(mu-2*sigma):.4f}  (~0.9545)")

   # Clinical question: what fraction of patients have HR > 100?
   P_high_hr = 1 - rv.cdf(100)
   print(f"\n  P(HR > 100 bpm) = {P_high_hr:.4f}")
   print(f"  About {P_high_hr:.1%} of healthy patients have resting HR above 100.")

**MGF.**

.. math::

   M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right).

*Derivation.*  Complete the square in the exponent of the integral
:math:`E[e^{tX}]`:

.. math::

   M_X(t) &= \int_{-\infty}^{\infty} e^{tx}\,\frac{1}{\sqrt{2\pi}\sigma}
   e^{-(x-\mu)^2/(2\sigma^2)}\,dx \\
   &= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}\sigma}
   \exp\!\left(-\frac{(x-\mu)^2 - 2\sigma^2 tx}{2\sigma^2}\right)dx.

Completing the square in the numerator:

.. math::

   (x-\mu)^2 - 2\sigma^2 tx = (x - (\mu + \sigma^2 t))^2 - 2\mu\sigma^2 t - \sigma^4 t^2.

The integral over the completed-square term equals :math:`\sqrt{2\pi}\sigma`,
leaving

.. math::

   M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right). \quad \square

The **standard normal** :math:`Z \sim N(0,1)` has :math:`\mu = 0`,
:math:`\sigma^2 = 1`.


3.2.3 Exponential Distribution
---------------------------------

If patient arrivals follow a Poisson process with rate :math:`\lambda = 4.5`
per hour, then the *time between consecutive arrivals* follows an Exponential
distribution: :math:`T \sim \text{Exp}(\lambda)`.  This is the continuous
counterpart of the Geometric: both model "waiting times," but the Exponential
works in continuous time.

**PDF.**

.. math::

   f(x) = \lambda\,e^{-\lambda x}, \qquad x \geq 0.

**Parameter.** :math:`\lambda > 0` (rate).

.. code-block:: python

   # Exponential PDF: time between patient arrivals
   import numpy as np
   from scipy import stats

   lam = 4.5  # arrivals per hour (same lambda as our Poisson)

   print(f"Exp(lambda={lam}): PDF at selected times (hours)")
   print(f"{'t (hours)':>10s}  {'t (min)':>8s}  {'f(t)':>10s}")
   for t in [0.0, 0.05, 0.10, 0.20, 0.33, 0.50, 1.0]:
       ft = lam * np.exp(-lam * t)
       print(f"{t:10.2f}  {t*60:8.1f}  {ft:10.4f}")

**Mean derivation.**

.. math::

   E[X] = \int_0^{\infty} x\,\lambda e^{-\lambda x}\,dx.

Integration by parts with :math:`u = x`, :math:`dv = \lambda e^{-\lambda x}dx`:

.. math::

   E[X] = \left[-x\,e^{-\lambda x}\right]_0^{\infty}
   + \int_0^{\infty} e^{-\lambda x}\,dx
   = 0 + \frac{1}{\lambda} = \frac{1}{\lambda}.

**Variance derivation.**

.. math::

   E[X^2] = \int_0^{\infty} x^2 \lambda e^{-\lambda x}\,dx = \frac{2}{\lambda^2}

(by two applications of integration by parts). Therefore

.. math::

   \operatorname{Var}(X) = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}.

**MGF.**

.. math::

   M_X(t) = \frac{\lambda}{\lambda - t}, \qquad t < \lambda.

.. code-block:: python

   # Exponential: verify moments and the Poisson-Exponential connection
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5  # patients per hour
   rv = stats.expon(scale=1/lam)  # scipy uses scale = 1/lambda
   samples = rv.rvs(size=100_000)

   E_theory   = 1 / lam
   Var_theory = 1 / lam**2
   print(f"Exp(lambda={lam}) -- 'Time between patient arrivals'")
   print(f"  Theory:     E[X] = {E_theory:.4f} hrs ({E_theory*60:.1f} min)")
   print(f"              Var(X) = {Var_theory:.4f}")
   print(f"  Simulation: E[X] = {samples.mean():.4f} hrs ({samples.mean()*60:.1f} min)")
   print(f"              Var(X) = {samples.var():.4f}")

   # KEY CONNECTION: Poisson count <-> Exponential inter-arrival time
   # Simulate Poisson arrivals by generating exponential gaps
   n_hours = 50_000
   gaps = stats.expon.rvs(scale=1/lam, size=n_hours * 20)  # generous
   arrival_times = np.cumsum(gaps)

   # Count arrivals in each 1-hour window
   counts = np.searchsorted(arrival_times, np.arange(1, n_hours + 1))
   counts = np.diff(np.concatenate([[0], counts]))

   print(f"\n  Poisson-Exponential connection:")
   print(f"  Generated Exp({lam}) inter-arrival times, counted per hour:")
   print(f"  Mean count per hour:     {counts.mean():.2f}  (should be ~{lam})")
   print(f"  Variance of counts:      {counts.var():.2f}  (should be ~{lam})")
   print(f"  Exponential gaps -> Poisson counts. Same process, two views.")

**Memoryless property.**  The Exponential is the only continuous distribution
with the memoryless property:

.. math::

   P(X > s + t \mid X > s) = P(X > t).

This says that given you have already waited :math:`s` units of time, the
distribution of *additional* waiting time is the same as if you had just
started waiting.

.. code-block:: python

   # Memoryless property: verify by simulation
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5
   samples = stats.expon.rvs(scale=1/lam, size=500_000)

   s = 0.1  # already waited 0.1 hours (6 minutes)

   # P(X > s + t | X > s) should equal P(X > t)
   survived = samples[samples > s]  # condition on X > s
   additional_wait = survived - s   # remaining time

   t_test = 0.2  # test at t = 0.2 hours
   P_conditional = np.mean(additional_wait > t_test)
   P_fresh       = np.mean(samples > t_test)
   P_theory      = np.exp(-lam * t_test)

   print(f"Memoryless property: P(X > s+t | X > s) = P(X > t)")
   print(f"  s = {s}, t = {t_test}")
   print(f"  P(additional > {t_test} | survived {s}) = {P_conditional:.4f}")
   print(f"  P(X > {t_test}) from scratch             = {P_fresh:.4f}")
   print(f"  Theory: exp(-{lam}*{t_test})              = {P_theory:.4f}")


3.2.4 Gamma Distribution
--------------------------

The Exponential models the wait for *one* patient.  What about the total time
to see the next :math:`\alpha = 3` patients?  If each inter-arrival time is
independent :math:`\text{Exp}(\lambda)`, the sum of :math:`\alpha` such times
follows a Gamma distribution: :math:`X \sim \text{Gamma}(\alpha, \beta)` with
:math:`\beta = \lambda`.

**PDF.**

.. math::

   f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}\,x^{\alpha - 1}\,e^{-\beta x},
   \qquad x > 0,

where :math:`\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1}e^{-t}\,dt` is the
gamma function.

**Parameters.** :math:`\alpha > 0` (shape), :math:`\beta > 0` (rate).

(Some textbooks use a scale parameterization :math:`\theta = 1/\beta`; we use
the rate form here.)

.. code-block:: python

   # Gamma PDF: total waiting time for alpha=3 patients
   import numpy as np
   from scipy import stats
   from scipy.special import gamma as gamma_func

   alpha = 3    # number of patients to wait for
   beta  = 4.5  # arrival rate (same as our Poisson lambda)

   print(f"Gamma(alpha={alpha}, beta={beta}): 'Wait for {alpha} patients'")
   print(f"{'t (hours)':>10s}  {'formula':>10s}  {'scipy':>10s}")
   for t in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
       pdf_manual = (beta**alpha / gamma_func(alpha)) * t**(alpha-1) * np.exp(-beta * t)
       pdf_scipy  = stats.gamma.pdf(t, a=alpha, scale=1/beta)
       print(f"{t:10.2f}  {pdf_manual:10.4f}  {pdf_scipy:10.4f}")

**Mean derivation.**

.. math::

   E[X] = \int_0^{\infty} x \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1}
   e^{-\beta x}\,dx
   = \frac{\beta^\alpha}{\Gamma(\alpha)} \int_0^{\infty} x^{\alpha} e^{-\beta x}\,dx.

Substituting :math:`u = \beta x`:

.. math::

   = \frac{\beta^\alpha}{\Gamma(\alpha)} \cdot \frac{\Gamma(\alpha+1)}{\beta^{\alpha+1}}
   = \frac{\Gamma(\alpha+1)}{\beta\,\Gamma(\alpha)}
   = \frac{\alpha}{\beta},

using :math:`\Gamma(\alpha+1) = \alpha\,\Gamma(\alpha)`.

**Variance derivation.**  By the same method,
:math:`E[X^2] = \alpha(\alpha+1)/\beta^2`, so

.. math::

   \operatorname{Var}(X) = \frac{\alpha(\alpha+1)}{\beta^2} - \frac{\alpha^2}{\beta^2}
   = \frac{\alpha}{\beta^2}.

**MGF.**

.. math::

   M_X(t) = \left(\frac{\beta}{\beta - t}\right)^\alpha, \qquad t < \beta.

.. code-block:: python

   # Gamma: verify moments and connection to sum of Exponentials
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   alpha, beta = 3, 4.5
   rv = stats.gamma(a=alpha, scale=1/beta)
   gamma_samples = rv.rvs(size=100_000)

   # Also build Gamma samples as sum of alpha Exponentials
   exp_sums = np.sum(
       stats.expon.rvs(scale=1/beta, size=(100_000, alpha)),
       axis=1
   )

   E_theory   = alpha / beta
   Var_theory = alpha / beta**2
   print(f"Gamma(alpha={alpha}, beta={beta}) -- 'Wait for {alpha} patients'")
   print(f"  Theory:         E[X] = {E_theory:.4f} hrs ({E_theory*60:.1f} min)")
   print(f"                  Var(X) = {Var_theory:.4f}")
   print(f"  Gamma samples:  E[X] = {gamma_samples.mean():.4f},  Var = {gamma_samples.var():.4f}")
   print(f"  Sum of 3 Exp:   E[X] = {exp_sums.mean():.4f},  Var = {exp_sums.var():.4f}")
   print(f"  The sum of {alpha} Exp({beta}) variables IS Gamma({alpha}, {beta}).")

When :math:`\alpha = 1`, the Gamma reduces to the Exponential.  This makes
the Gamma a natural generalization: one event gives Exponential, multiple events
give Gamma.


3.2.5 Beta Distribution
--------------------------

Now suppose you want to estimate the trauma probability :math:`p` itself.
Before collecting data, you might express prior uncertainty about :math:`p`
using a distribution on :math:`[0,1]`.  The Beta distribution is the natural
choice.  Write :math:`X \sim \text{Beta}(\alpha, \beta)`.

In Bayesian statistics, the Beta serves as the conjugate prior for the Bernoulli
and Binomial likelihoods: if your prior on :math:`p` is Beta and you observe
Bernoulli data, the posterior on :math:`p` is also Beta.

**PDF.**

.. math::

   f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)},
   \qquad 0 < x < 1,

where :math:`B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}`
is the beta function.

**Parameters.** :math:`\alpha > 0`, :math:`\beta > 0` (shape parameters).

**Mean derivation.**

.. math::

   E[X] = \int_0^1 x \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}\,dx
        = \frac{1}{B(\alpha,\beta)}\int_0^1 x^{\alpha}(1-x)^{\beta-1}\,dx
        = \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)}.

Using the relationship :math:`B(\alpha+1,\beta) = \frac{\alpha}{\alpha+\beta}\,B(\alpha,\beta)`:

.. math::

   E[X] = \frac{\alpha}{\alpha + \beta}.

**Variance derivation.**  Similarly,
:math:`E[X^2] = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}`, so

.. math::

   \operatorname{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.

**MGF.**  The Beta MGF does not have a simple closed form; it is given by a
confluent hypergeometric function.  In practice, one works directly with the
moments.

.. code-block:: python

   # Beta distribution: prior uncertainty about trauma rate p
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   alpha_param, beta_param = 3, 17  # prior: "3 traumas out of 20 patients seen"
   rv = stats.beta(alpha_param, beta_param)
   samples = rv.rvs(size=100_000)

   E_theory   = alpha_param / (alpha_param + beta_param)
   Var_theory = (alpha_param * beta_param) / (
       (alpha_param + beta_param)**2 * (alpha_param + beta_param + 1)
   )
   print(f"Beta(alpha={alpha_param}, beta={beta_param})")
   print(f"  Interpretation: prior belief about trauma rate p")
   print(f"  Theory:     E[X] = {E_theory:.4f},  Var(X) = {Var_theory:.6f}")
   print(f"  Simulation: E[X] = {samples.mean():.4f},  Var(X) = {samples.var():.6f}")

   # Show the shape: PDF at selected p values
   print(f"\n  PDF (shape of our belief about p):")
   for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
       bar = '#' * int(rv.pdf(p) * 5)
       print(f"    f({p:.2f}) = {rv.pdf(p):.4f}  {bar}")

   # 90% credible interval
   lo, hi = rv.ppf(0.05), rv.ppf(0.95)
   print(f"\n  90% prior interval for p: ({lo:.3f}, {hi:.3f})")


3.2.6 Chi-Squared Distribution
---------------------------------

If :math:`Z_1, \dots, Z_k` are independent standard normal random variables,
then

.. math::

   X = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2(k).

The chi-squared distribution with :math:`k` **degrees of freedom** is a
special case of the Gamma: :math:`\chi^2(k) = \text{Gamma}(k/2,\; 1/2)`.

In the hospital context, the chi-squared arises when testing whether observed
patient counts across categories (e.g., triage levels) match expected
proportions --- the classic goodness-of-fit test.

**PDF.**

.. math::

   f(x) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\,x^{k/2-1}\,e^{-x/2},
   \qquad x > 0.

**Mean** (from the Gamma):

.. math::

   E[X] = \frac{k/2}{1/2} = k.

**Variance** (from the Gamma):

.. math::

   \operatorname{Var}(X) = \frac{k/2}{(1/2)^2} = 2k.

**MGF.**

.. math::

   M_X(t) = (1 - 2t)^{-k/2}, \qquad t < \tfrac{1}{2}.

.. code-block:: python

   # Chi-squared: verify definition by summing squared normals
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   k = 5  # degrees of freedom

   # Build chi-squared from its definition
   Z = np.random.standard_normal((100_000, k))
   chi2_from_def = np.sum(Z**2, axis=1)

   # Compare with scipy
   chi2_scipy = stats.chi2.rvs(k, size=100_000)

   print(f"Chi-squared(k={k})")
   print(f"  Theory:          E[X] = {k},  Var(X) = {2*k}")
   print(f"  Sum of Z^2:      E[X] = {chi2_from_def.mean():.2f},  "
         f"Var(X) = {chi2_from_def.var():.2f}")
   print(f"  Scipy samples:   E[X] = {chi2_scipy.mean():.2f},  "
         f"Var(X) = {chi2_scipy.var():.2f}")

   # Verify it is Gamma(k/2, 1/2)
   gamma_samples = stats.gamma.rvs(a=k/2, scale=2, size=100_000)  # scale=1/beta=1/(1/2)=2
   print(f"\n  Gamma({k/2}, 1/2):   E[X] = {gamma_samples.mean():.2f},  "
         f"Var(X) = {gamma_samples.var():.2f}")
   print(f"  chi^2({k}) and Gamma({k/2}, 1/2) are the same distribution.")


3.2.7 Student-t Distribution
-------------------------------

When you estimate a patient's mean heart rate from a small sample, you do not
know the true variance either.  Replacing the true :math:`\sigma` with the
sample standard deviation introduces extra uncertainty, and the resulting
statistic follows a Student-t rather than a Normal.

If :math:`Z \sim N(0,1)` and :math:`V \sim \chi^2(k)` are
independent, then

.. math::

   T = \frac{Z}{\sqrt{V/k}} \sim t(k).

The t-distribution looks like a normal but with heavier tails.  The fewer
degrees of freedom you have, the heavier the tails.  As :math:`k \to \infty`,
the t-distribution converges to the standard normal.

**PDF.**

.. math::

   f(x) = \frac{\Gamma\!\left(\frac{k+1}{2}\right)}
   {\sqrt{k\pi}\;\Gamma\!\left(\frac{k}{2}\right)}
   \left(1 + \frac{x^2}{k}\right)^{-(k+1)/2},
   \qquad x \in \mathbb{R}.

**Parameter.** :math:`k \in \{1,2,\dots\}` (degrees of freedom).

**Mean.**

.. math::

   E[T] = 0 \qquad \text{for } k > 1.

(The density is symmetric around zero.)

**Variance.**

.. math::

   \operatorname{Var}(T) = \frac{k}{k-2} \qquad \text{for } k > 2.

Note the variance exceeds 1 (heavier tails than the normal) and approaches 1
as :math:`k \to \infty`.

**MGF.**  The Student-t does not have an MGF (its moments of sufficiently high
order do not exist for small :math:`k`).  Its characteristic function is used
instead.

.. code-block:: python

   # Student-t: heavier tails shrink toward Normal as k grows
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   print("Student-t: variance and tail behavior vs degrees of freedom")
   print(f"{'k':>6s}  {'Var (theory)':>12s}  {'Var (simul)':>12s}  {'P(|T|>3)':>10s}  {'Normal P':>10s}")
   P_normal = 2 * (1 - stats.norm.cdf(3))
   for k in [3, 5, 10, 30, 100]:
       rv = stats.t(k)
       samp = rv.rvs(size=200_000)
       var_theory = k / (k - 2)
       P_tail = np.mean(np.abs(samp) > 3)
       print(f"{k:6d}  {var_theory:12.4f}  {samp.var():12.4f}  {P_tail:10.4f}  {P_normal:10.4f}")
   print(f"\nAs k increases, t(k) -> N(0,1): variance -> 1 and tails shrink.")


3.2.8 F-Distribution
-----------------------

A ratio of two independent chi-squared variables (each divided by their
degrees of freedom).  If :math:`U \sim \chi^2(d_1)` and :math:`V \sim \chi^2(d_2)`
are independent, then

.. math::

   F = \frac{U / d_1}{V / d_2} \sim F(d_1, d_2).

**PDF.**

.. math::

   f(x) = \frac{1}{B(d_1/2,\,d_2/2)}
   \left(\frac{d_1}{d_2}\right)^{d_1/2}
   \frac{x^{d_1/2 - 1}}{\left(1 + \frac{d_1}{d_2}x\right)^{(d_1+d_2)/2}},
   \qquad x > 0.

**Parameters.** :math:`d_1, d_2 > 0` (numerator and denominator degrees of
freedom).

**Mean.**

.. math::

   E[F] = \frac{d_2}{d_2 - 2} \qquad \text{for } d_2 > 2.

**Variance.**

.. math::

   \operatorname{Var}(F) = \frac{2\,d_2^2\,(d_1 + d_2 - 2)}{d_1\,(d_2-2)^2\,(d_2-4)}
   \qquad \text{for } d_2 > 4.

**MGF.**  Like the Student-t, the F-distribution does not have a standard MGF.

**Usage.**  ANOVA, comparing variances, overall significance in linear
regression (F-test).  Note that :math:`T^2 \sim F(1, k)` if :math:`T \sim t(k)`.


.. _sec_distribution_relationships:

3.3 Relationships Between Distributions
==========================================

The distributions above do not exist in isolation.  They form a rich family
tree, and in our hospital scenario every connection has a concrete
interpretation.  Understanding these connections deepens intuition and
simplifies many derivations.


3.3.1 Poisson--Exponential Duality
-------------------------------------

This is the connection at the heart of our hospital scenario.  Patient arrivals
per hour follow :math:`\text{Pois}(\lambda)`, and the time between arrivals
follows :math:`\text{Exp}(\lambda)`.  These are two views of the same Poisson
process.

.. code-block:: python

   # Poisson-Exponential duality: two views of hospital arrivals
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5  # patients per hour

   # View 1: generate Poisson counts directly
   n_hours = 100_000
   poisson_counts = stats.poisson.rvs(lam, size=n_hours)

   # View 2: generate Exp inter-arrival times, count per hour
   # Simulate a long stream of arrivals
   total_events = int(lam * n_hours * 1.1)  # slightly more than expected
   inter_arrivals = stats.expon.rvs(scale=1/lam, size=total_events)
   arrival_times = np.cumsum(inter_arrivals)
   arrival_times = arrival_times[arrival_times < n_hours]

   # Count events in each hour
   hour_edges = np.arange(n_hours + 1)
   exp_counts, _ = np.histogram(arrival_times, bins=hour_edges)

   print(f"Poisson-Exponential duality (lambda={lam})")
   print(f"  Direct Poisson:      mean = {poisson_counts.mean():.3f}, "
         f"var = {poisson_counts.var():.3f}")
   print(f"  Exp inter-arrivals:  mean = {exp_counts.mean():.3f}, "
         f"var = {exp_counts.var():.3f}")
   print(f"  Theory:              mean = {lam:.3f}, var = {lam:.3f}")
   print(f"\n  Mean inter-arrival time: {inter_arrivals.mean():.4f} hrs "
         f"({inter_arrivals.mean()*60:.1f} min)")
   print(f"  Theory: 1/lambda = {1/lam:.4f} hrs ({60/lam:.1f} min)")


3.3.2 Binomial--Poisson Limit
-------------------------------

If :math:`X \sim \text{Bin}(n, p)` with :math:`n \to \infty` and
:math:`p \to 0` such that :math:`\lambda = np` remains constant, then

.. math::

   \binom{n}{x} p^x (1-p)^{n-x} \to \frac{e^{-\lambda}\lambda^x}{x!}.

Here is the key idea: when you have very many trials each with a very small
success probability, you are essentially counting rare events --- and rare
event counts follow the Poisson.

In the hospital, think of it this way: if the city has :math:`n = 100{,}000`
residents and each independently has a :math:`p = 0.000045` probability of
visiting the ED in a given hour, then the count of arrivals is
:math:`\text{Bin}(100000, 0.000045) \approx \text{Pois}(4.5)`.

*Derivation sketch.*  Write :math:`p = \lambda/n` and take the limit:

.. math::

   \binom{n}{x}\left(\frac{\lambda}{n}\right)^x\left(1-\frac{\lambda}{n}\right)^{n-x}
   &= \frac{n!}{x!(n-x)!}\,\frac{\lambda^x}{n^x}\,
      \left(1-\frac{\lambda}{n}\right)^n \left(1-\frac{\lambda}{n}\right)^{-x}.

As :math:`n \to \infty`:

- :math:`n! / [(n-x)!\,n^x] \to 1`,
- :math:`(1 - \lambda/n)^n \to e^{-\lambda}`,
- :math:`(1 - \lambda/n)^{-x} \to 1`.

So the limit is :math:`\frac{\lambda^x}{x!}e^{-\lambda}`, which is the Poisson
PMF.

.. code-block:: python

   # Binomial-to-Poisson limit: watch convergence as n grows
   import numpy as np
   from scipy import stats

   lam = 4.5   # fixed lambda = n*p
   x_eval = 4  # evaluate at x=4

   print(f"P(X={x_eval}): Bin(n, {lam}/n) vs Poisson({lam})")
   print(f"  Poisson:         {stats.poisson.pmf(x_eval, lam):.8f}")
   for n in [20, 50, 100, 1_000, 10_000, 100_000]:
       p = lam / n
       binom_pmf = stats.binom.pmf(x_eval, n, p)
       error = abs(binom_pmf - stats.poisson.pmf(x_eval, lam))
       print(f"  Bin({n:>7d}, {p:.6f}): {binom_pmf:.8f}  (error: {error:.2e})")
   print(f"\n  As n -> inf with np = {lam}, Binomial -> Poisson.")


3.3.3 Gamma--Exponential Relationship
----------------------------------------

- :math:`\text{Exp}(\lambda) = \text{Gamma}(1, \lambda)`.
- If :math:`X_1, \dots, X_n` are independent :math:`\text{Exp}(\lambda)`, then
  :math:`\sum X_i \sim \text{Gamma}(n, \lambda)`.

In the hospital: the wait for one patient is Exponential; the total wait for
:math:`n` patients is Gamma.

*Proof via MGFs.*

.. math::

   M_{\sum X_i}(t) = \prod_{i=1}^n \frac{\lambda}{\lambda - t}
   = \left(\frac{\lambda}{\lambda - t}\right)^n,

which is the MGF of :math:`\text{Gamma}(n, \lambda)`. :math:`\square`

.. code-block:: python

   # Gamma = sum of Exponentials: verify for several values of n
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5  # patient arrival rate

   print(f"Sum of n Exp({lam}) vs Gamma(n, {lam}):")
   print(f"{'n':>4s}  {'Sum E[X]':>10s}  {'Gamma E[X]':>12s}  {'Theory':>10s}")
   for n_wait in [1, 3, 5, 10]:
       exp_sums = np.sum(
           stats.expon.rvs(scale=1/lam, size=(100_000, n_wait)), axis=1
       )
       gamma_samp = stats.gamma.rvs(a=n_wait, scale=1/lam, size=100_000)
       theory = n_wait / lam
       print(f"{n_wait:4d}  {exp_sums.mean():10.4f}  {gamma_samp.mean():12.4f}  {theory:10.4f}")


3.3.4 Chi-Squared as a Gamma Special Case
-------------------------------------------

.. math::

   \chi^2(k) = \text{Gamma}\!\left(\frac{k}{2},\;\frac{1}{2}\right).

This follows directly from comparing PDFs.


3.3.5 Normal--Chi-Squared Relationship
-----------------------------------------

If :math:`Z \sim N(0,1)`, then :math:`Z^2 \sim \chi^2(1)`.

*Proof.*  For :math:`x > 0`,

.. math::

   P(Z^2 \leq x) = P(-\sqrt{x} \leq Z \leq \sqrt{x})
   = 2\Phi(\sqrt{x}) - 1,

where :math:`\Phi` is the standard normal CDF.  Differentiating gives the
:math:`\chi^2(1)` density.


3.3.6 Beta--Gamma Relationship
---------------------------------

If :math:`X \sim \text{Gamma}(\alpha, 1)` and :math:`Y \sim \text{Gamma}(\beta, 1)`
are independent, then

.. math::

   \frac{X}{X + Y} \sim \text{Beta}(\alpha, \beta).


3.3.7 Student-t and F-Distribution Connection
------------------------------------------------

If :math:`T \sim t(k)`, then :math:`T^2 \sim F(1, k)`.

If :math:`F \sim F(d_1, d_2)`, then :math:`1/F \sim F(d_2, d_1)`.


3.3.8 Binomial--Normal Approximation (CLT)
--------------------------------------------

By the CLT, when :math:`n` is large,

.. math::

   \frac{X - np}{\sqrt{np(1-p)}} \;\dot\sim\; N(0, 1)
   \qquad \text{for } X \sim \text{Bin}(n, p).

In the hospital, the number of trauma cases in :math:`n = 200` patients
(:math:`p = 0.15`) is :math:`\text{Bin}(200, 0.15)`, but we can approximate
it with :math:`N(30, 25.5)`.

.. code-block:: python

   # Binomial-Normal approximation: trauma counts for large n
   import numpy as np
   from scipy import stats

   n, p = 200, 0.15
   mu_approx    = n * p
   sigma2_approx = n * p * (1 - p)

   print(f"Bin({n}, {p}) vs N({mu_approx:.0f}, {sigma2_approx:.1f})")
   print(f"{'x':>4s}  {'Bin PMF':>10s}  {'Normal':>10s}  {'Error':>10s}")
   for x in range(15, 46, 3):
       bp = stats.binom.pmf(x, n, p)
       na = stats.norm.pdf(x, loc=mu_approx, scale=np.sqrt(sigma2_approx))
       print(f"{x:4d}  {bp:10.6f}  {na:10.6f}  {abs(bp-na):10.6f}")

   # Practical check: P(X > 40)
   P_binom  = 1 - stats.binom.cdf(40, n, p)
   P_normal = 1 - stats.norm.cdf(40, loc=mu_approx, scale=np.sqrt(sigma2_approx))
   print(f"\n  P(X > 40): Binomial = {P_binom:.6f},  Normal approx = {P_normal:.6f}")


3.3.9 Summary of Relationships
---------------------------------

.. list-table::
   :header-rows: 1

   * - From
     - To
     - Mechanism
   * - Bernoulli
     - Binomial
     - Sum of :math:`n` i.i.d. copies
   * - Binomial
     - Poisson
     - :math:`n \to \infty`, :math:`p \to 0`, :math:`np = \lambda`
   * - Binomial
     - Normal
     - CLT (large :math:`n`)
   * - Poisson
     - Exponential
     - Counts vs. inter-arrival times (same process)
   * - Exponential
     - Gamma
     - Sum of i.i.d. copies / shape = 1 special case
   * - Gamma
     - Chi-squared
     - :math:`\alpha=k/2`, :math:`\beta=1/2`
   * - Normal
     - Chi-squared
     - Square of standard normal
   * - Chi-squared
     - Student-t
     - :math:`Z / \sqrt{V/k}`
   * - Chi-squared
     - F
     - Ratio of independent chi-squareds
   * - Student-t
     - F
     - :math:`T^2 \sim F(1,k)`
   * - Gamma
     - Beta
     - :math:`X/(X+Y)` for independent Gammas


3.4 Connecting the Hospital Story
====================================

Let us step back and see how all these distributions fit together in our
emergency department scenario.

.. code-block:: python

   # The hospital ED: one scenario, five distributions
   import numpy as np
   from scipy import stats

   np.random.seed(42)
   lam = 4.5   # patient arrival rate (per hour)
   p   = 0.15  # probability each patient is a trauma case
   mu_hr, sigma_hr = 80, 12  # heart rate parameters

   # Simulate one 8-hour shift
   n_hours = 8

   # 1. Poisson: patient counts per hour
   hourly_counts = stats.poisson.rvs(lam, size=n_hours)
   total_patients = hourly_counts.sum()

   # 2. Exponential: inter-arrival times (for the total_patients)
   inter_arrivals = stats.expon.rvs(scale=1/lam, size=total_patients)

   # 3. Binomial: how many are trauma cases?
   n_trauma = stats.binom.rvs(total_patients, p)

   # 4. Normal: heart rates for each patient
   heart_rates = stats.norm.rvs(loc=mu_hr, scale=sigma_hr, size=total_patients)

   # 5. Gamma: total wait time for next 3 patients
   wait_for_3 = stats.gamma.rvs(a=3, scale=1/lam)

   print("=" * 60)
   print("  HOSPITAL ED: ONE SHIFT SIMULATION")
   print("=" * 60)
   print(f"\n  Arrival rate:    lambda = {lam} patients/hour")
   print(f"  Trauma prob:     p = {p}")
   print(f"  Heart rate:      N({mu_hr}, {sigma_hr}^2)")
   print(f"\n  Hourly counts (Poisson):  {hourly_counts}")
   print(f"  Total patients (8 hrs):    {total_patients}")
   print(f"  Trauma cases (Binomial):   {n_trauma} out of {total_patients}")
   print(f"  Mean inter-arrival (Exp):  {inter_arrivals.mean()*60:.1f} min "
         f"(theory: {60/lam:.1f} min)")
   print(f"  Mean heart rate (Normal):  {heart_rates.mean():.1f} bpm")
   print(f"  High HR (>100 bpm):        {np.sum(heart_rates > 100)} patients")
   print(f"  Wait for next 3 (Gamma):   {wait_for_3*60:.1f} min")
   print(f"\n  Every number above came from a distribution in this chapter.")


3.5 Summary
=============

This chapter catalogued the probability distributions that form the
workhorses of likelihood-based inference, all tied together through a single
hospital emergency department scenario:

- **Discrete distributions** (Bernoulli, Binomial, Poisson, Geometric,
  Negative Binomial, Hypergeometric) model count data --- from individual
  patient classifications to arrival counts to pharmacy sampling.
- **Continuous distributions** (Uniform, Normal, Exponential, Gamma, Beta,
  Chi-squared, Student-t, F) model measurements on the real line or subsets
  thereof --- from waiting times to vital signs to prior beliefs.
- These distributions are connected through limiting arguments (Binomial to
  Poisson, CLT), algebraic relationships (sums of Exponentials give Gammas,
  Poisson counts correspond to Exponential gaps), and transformations
  (squaring a Normal gives a Chi-squared).

In :ref:`ch4_likelihood`, we will see how these distributions give rise to
likelihood functions, transforming them from descriptions of data into tools
for inference about unknown parameters.
