.. _ch21_numerical:

==========================================
Chapter 21 -- Numerical Considerations
==========================================

A beautifully derived likelihood estimator is worthless if the computer returns
``NaN``.  This chapter covers the practical numerical issues that arise when
implementing likelihood-based methods: floating-point pitfalls, log-space
tricks, differentiation strategies, conditioning, exploiting structure, and
leveraging modern hardware.  Every practitioner who writes code for
optimization or inference will encounter these issues.

Think of this chapter as the "survival guide" for computational statistics.
The theory from earlier chapters tells you *what* to compute; this chapter
tells you *how* to compute it without your program silently producing garbage.


21.1 Floating-Point Arithmetic
---------------------------------

**Why you need to know this.**
All modern computers represent real numbers using the IEEE 754 standard.
Understanding its limitations prevents silent errors and mysterious crashes.

The IEEE 754 double-precision format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A 64-bit double-precision number stores:

* 1 **sign bit** :math:`s`.
* 11 **exponent bits** encoding :math:`e \in \{-1022, \ldots, 1023\}`.
* 52 **mantissa bits** encoding a fraction :math:`f \in [1, 2)`.

The represented value is:

.. math::

   x = (-1)^s \times 2^e \times f.

Key consequences:

* **Machine epsilon** :math:`\varepsilon_{\text{mach}} = 2^{-52} \approx
  2.22 \times 10^{-16}`.  This is the smallest number such that
  :math:`1 + \varepsilon_{\text{mach}} \neq 1` in floating-point.
  It sets the limit on *relative* precision.
* **Range:** The largest finite double is about :math:`1.8 \times 10^{308}`;
  the smallest positive normal number is about :math:`2.2 \times 10^{-308}`.
* **Overflow** occurs when a result exceeds the range (returns ``Inf``).
* **Underflow** occurs when a result is closer to zero than the smallest
  representable number (returns 0 or a *denormalized* number with reduced
  precision).
* **Catastrophic cancellation** occurs when subtracting two nearly equal
  numbers, e.g., :math:`1.0000001 - 1.0000000` loses most significant digits.

**Exploring machine epsilon and floating-point limits.**
Before we can understand *why* likelihood computations break, we need to see
what the computer is actually doing.  The following code probes the fundamental
limits of IEEE 754 double precision---the "resolution" and "range" of your
numerical ruler.

.. code-block:: python

   # Machine epsilon and floating-point limits
   import numpy as np
   import sys

   # Machine epsilon: smallest eps such that 1.0 + eps != 1.0
   eps = np.finfo(np.float64).eps
   print(f"Machine epsilon:      {eps:.2e}")
   print(f"  = 2^(-52) = {2**(-52):.2e}")
   print(f"1.0 + eps == 1.0?     {1.0 + eps == 1.0}")
   print(f"1.0 + eps/2 == 1.0?   {1.0 + eps/2 == 1.0}  <- half epsilon vanishes!")

   # Range limits
   print(f"\nLargest double:       {np.finfo(np.float64).max:.2e}")
   print(f"Smallest normal:      {np.finfo(np.float64).tiny:.2e}")
   print(f"Smallest subnormal:   {np.nextafter(0, 1):.2e}")

   # Overflow and underflow in action
   print(f"\n1e308 * 2   = {1e308 * 2}")       # overflow -> inf
   print(f"1e-323 / 10 = {1e-323 / 10}")       # underflow -> 0

   # Count the bits: how many significant digits?
   print(f"\nSignificant decimal digits: {np.finfo(np.float64).precision}")
   print(f"Number of mantissa bits:    52")
   print(f"Total bits:                 64 (1 sign + 11 exponent + 52 mantissa)")

**So what?**  Every number you compute in Python or R lives in this 64-bit box.
When likelihood values push against the edges of this box, you get silent
garbage.  The rest of this chapter teaches you how to stay safely inside.

**Catastrophic cancellation in action.**
Catastrophic cancellation is not a theoretical curiosity---it happens in
everyday statistical computations.  When you compute the variance using the
"textbook formula" :math:`E[X^2] - (E[X])^2`, you subtract two nearly equal
large numbers.  With shifted data, this can lose *all* significant digits.

.. code-block:: python

   # Catastrophic cancellation: textbook variance formula
   import numpy as np

   # Simple demonstration: 1 + 1e-16 - 1
   a = 1.0 + 1e-16
   result = a - 1.0
   print(f"(1 + 1e-16) - 1 = {result:.2e}  (true: 1e-16)")
   print(f"Relative error: {abs(result - 1e-16) / 1e-16 * 100:.0f}%\n")

   # Real-world example: computing variance of shifted data
   np.random.seed(42)
   x = np.random.normal(loc=1e8, scale=1.0, size=1000)  # mean ~ 10^8, std = 1

   # Method 1: "textbook" formula E[X^2] - E[X]^2 (BAD)
   var_textbook = np.mean(x**2) - np.mean(x)**2

   # Method 2: centered formula (GOOD)
   var_centered = np.mean((x - np.mean(x))**2)

   # Method 3: numpy's built-in (uses numerically stable algorithm)
   var_numpy = np.var(x)

   print(f"Variance estimates for data with mean ~ 1e8, true var ~ 1.0:")
   print(f"  Textbook (E[X^2]-E[X]^2): {var_textbook:.6f}")
   print(f"  Centered:                  {var_centered:.6f}")
   print(f"  NumPy built-in:            {var_numpy:.6f}")
   print(f"\nThe textbook formula lost nearly all precision due to")
   print(f"subtracting two numbers both close to {np.mean(x)**2:.2e}")

When precision matters for likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Likelihoods are products of many probabilities, each between 0 and 1.  For
:math:`n = 1000` observations:

.. math::

   L(\theta)
   = \prod_{i=1}^{1000} p(x_i \mid \theta)
   \approx (0.1)^{1000}
   = 10^{-1000},

which is far below the smallest representable double.  The likelihood
*underflows to zero*.  This is why we always work with the log-likelihood.

**Likelihood underflow: watching it happen.**
Let us generate a modest dataset and watch the raw likelihood collapse to zero
while the log-likelihood remains perfectly usable.  This is not a pathological
edge case---it happens with a few hundred observations.

.. code-block:: python

   # Likelihood underflow: raw likelihood vs log-likelihood
   import numpy as np

   np.random.seed(42)

   theta_true = 0.3

   # Show the progression: at what n does underflow occur?
   print(f"{'n':<8} {'raw likelihood':<20} {'log-likelihood':<18} {'underflow?'}")
   print("-" * 66)
   for n in [10, 50, 100, 200, 500, 1000, 2000]:
       data = np.random.binomial(1, theta_true, n)
       s = data.sum()

       # Raw likelihood
       raw = theta_true**s * (1 - theta_true)**(n - s)

       # Log-likelihood
       loglik = s * np.log(theta_true) + (n - s) * np.log(1 - theta_true)

       underflow = "YES" if raw == 0.0 else "no"
       print(f"{n:<8} {raw:<20.6e} {loglik:<18.4f} {underflow}")

   print(f"\nSmallest representable double: ~10^(-308)")
   print(f"Log-likelihood for n=2000:     corresponds to ~10^({loglik/np.log(10):.0f})")
   print(f"That is why we ALWAYS use log-likelihoods.")

.. admonition:: Real-World Example

   **Likelihood underflow in genomics.**

   In genomic studies, you routinely work with thousands or tens of thousands
   of observations (e.g., SNPs across a genome).  Even moderate per-observation
   probabilities produce a raw likelihood that is astronomically small.  Here
   is a concrete demonstration of the problem and its solution.

   *Pseudocode:*

   .. code-block:: text

      1. Generate n=2000 Bernoulli observations with theta=0.3
      2. Attempt to compute raw likelihood: prod(theta^x * (1-theta)^(1-x))
      3. Observe underflow: result = 0.0
      4. Compute log-likelihood instead: sum(x*log(theta) + (1-x)*log(1-theta))
      5. Show that log-likelihood is perfectly well-behaved

   *Python implementation:*

   .. code-block:: python

      # Likelihood underflow in genomics: raw vs log-likelihood
      import numpy as np

      np.random.seed(42)

      # Simulated genomic data: 2000 binary markers
      n = 2000
      theta_true = 0.3
      data = np.random.binomial(1, theta_true, n)

      # Attempt 1: raw likelihood (WILL underflow)
      raw_likelihood = np.prod(theta_true**data * (1 - theta_true)**(1 - data))
      print(f"Raw likelihood:  {raw_likelihood}")
      print(f"  -> Underflowed to zero!")

      # Attempt 2: log-likelihood (correct approach)
      log_lik = np.sum(data * np.log(theta_true) + (1 - data) * np.log(1 - theta_true))
      print(f"\nLog-likelihood:  {log_lik:.4f}")
      print(f"  -> Perfectly finite and usable.")

      # Show the scale of the problem
      expected_log = n * (theta_true * np.log(theta_true) + (1-theta_true) * np.log(1-theta_true))
      print(f"\nExpected log-likelihood: {expected_log:.1f}")
      print(f"That corresponds to 10^({expected_log / np.log(10):.0f}) — far below 10^(-308)!")


21.2 Working in Log-Space
----------------------------

**The golden rule: never compute a raw likelihood.  Always work with
log-likelihoods.**

Products become sums:

.. math::

   \log L(\theta)
   = \sum_{i=1}^n \log p(x_i \mid \theta).

Each :math:`\log p(x_i \mid \theta)` is a moderate negative number (e.g.,
:math:`-2.3`), and the sum is manageable even for very large :math:`n`.

Ratios become differences:

.. math::

   \log \frac{L(\theta_1)}{L(\theta_2)}
   = \ell(\theta_1) - \ell(\theta_2),

which is numerically stable even when the individual likelihoods are
astronomically small.

.. admonition:: Why does this matter?

   Working in log-space is not an optional optimization---it is a necessity.
   Without it, virtually every likelihood computation with more than a few
   hundred observations will silently produce zeros, infinities, or NaN
   values.  The log transform converts multiplicative problems (which overflow
   and underflow) into additive problems (which stay in a comfortable range).


21.3 The Log-Sum-Exp Trick
-----------------------------

**Why we need this.**
Many computations require the logarithm of a sum of exponentials, for example:

* The marginal likelihood in mixture models:
  :math:`p(\mathbf{x}) = \sum_k \pi_k\, p(\mathbf{x} \mid \theta_k)`.
* The normalizing constant in softmax / multinomial logistic models.
* Importance sampling weights.

Naively computing :math:`\log\!\bigl(\sum_k \exp(a_k)\bigr)` when some
:math:`a_k` are very large (overflow of :math:`\exp`) or very negative
(underflow to zero) gives garbage.

**Derivation.**
Let :math:`a^* = \max_k a_k`.  Then:

.. math::

   \log\!\left(\sum_{k=1}^K \exp(a_k)\right)
   &= \log\!\left(\sum_{k=1}^K \exp(a_k - a^* + a^*)\right) \\
   &= \log\!\left(\exp(a^*) \sum_{k=1}^K \exp(a_k - a^*)\right) \\
   &= a^* + \log\!\left(\sum_{k=1}^K \exp(a_k - a^*)\right).

Now :math:`a_k - a^* \le 0` for all :math:`k`, so every :math:`\exp(a_k -
a^*)` is in :math:`(0, 1]` (the term with :math:`k = k^*` contributes exactly
1), and the sum is at least 1.  There is no overflow, and the dominant term
cannot underflow.

**Implementation (pseudocode):**

.. code-block:: python

   def logsumexp(a):
       a_max = max(a)
       return a_max + log(sum(exp(a_k - a_max) for a_k in a))

Most numerical libraries (SciPy, PyTorch, JAX, Stan) provide a built-in
``logsumexp``.

**The log-sum-exp trick: naive vs stable computation.**
Here we show three scenarios where naive computation fails and the log-sum-exp
trick succeeds.  We also connect it to a practical mixture model likelihood
calculation.

.. code-block:: python

   # The log-sum-exp trick: naive vs stable computation
   import numpy as np
   from scipy.special import logsumexp

   np.random.seed(42)

   # Case 1: Large values (overflow risk)
   a_large = np.array([1000.0, 1001.0, 1002.0])
   naive_large = np.log(np.sum(np.exp(a_large)))  # overflow!
   stable_large = logsumexp(a_large)
   print("Case 1: Large values (a ~ 1000)")
   print(f"  Naive:  {naive_large}")
   print(f"  Stable: {stable_large:.6f}")

   # Case 2: Very negative values (underflow risk)
   a_small = np.array([-1000.0, -999.0, -1001.0])
   naive_small = np.log(np.sum(np.exp(a_small)))  # underflow to -inf!
   stable_small = logsumexp(a_small)
   print("\nCase 2: Very negative values (a ~ -1000)")
   print(f"  Naive:  {naive_small}")
   print(f"  Stable: {stable_small:.6f}")

   # Case 3: Mixture model log-likelihood
   log_pi = np.log([0.3, 0.7])
   log_comp = np.array([-500.5, -501.2])  # log-densities from two components
   log_mix = logsumexp(log_pi + log_comp)
   print(f"\nMixture model log-density: {log_mix:.6f}")

   # Case 4: Full mixture model with many observations
   n = 500
   K = 3
   weights = np.array([0.3, 0.5, 0.2])
   means = np.array([-2.0, 1.0, 4.0])
   stds = np.array([0.5, 1.0, 0.8])

   data = np.concatenate([
       np.random.normal(means[k], stds[k], int(n * weights[k]))
       for k in range(K)
   ])

   # Compute log-likelihood using logsumexp for each observation
   log_lik = 0.0
   for x in data:
       log_components = (np.log(weights)
                         - 0.5 * np.log(2 * np.pi * stds**2)
                         - 0.5 * ((x - means) / stds)**2)
       log_lik += logsumexp(log_components)

   print(f"\nGaussian mixture log-likelihood (n={n}, K={K}): {log_lik:.2f}")
   print("Without logsumexp, individual component densities would underflow.")

**Example: mixture model log-likelihood.**
For a two-component Gaussian mixture with log-component-densities
:math:`\ell_1(x)` and :math:`\ell_2(x)` and mixing weight :math:`\pi`:

.. math::

   \log p(x)
   = \text{logsumexp}\!\bigl(\log\pi + \ell_1(x),\;
     \log(1-\pi) + \ell_2(x)\bigr).


21.4 Numerical Differentiation
---------------------------------

**Why numerical derivatives?**
Sometimes we need derivatives of the log-likelihood but:

* The model is implemented as a black-box simulator.
* We want to check an analytical or AD gradient.

Let's explore the main approaches and their trade-offs.

Finite differences
^^^^^^^^^^^^^^^^^^^

The simplest approach.  The *forward difference* approximation for a scalar
function is:

.. math::

   f'(\theta) \approx \frac{f(\theta + h) - f(\theta)}{h}.

The error is :math:`O(h)` (first-order).  The *central difference* is more
accurate:

.. math::

   f'(\theta) \approx \frac{f(\theta + h) - f(\theta - h)}{2h}.

The error is :math:`O(h^2)` (second-order).

**Choosing** :math:`h`.
There is a trade-off: too large :math:`h` gives truncation error; too small
:math:`h` gives roundoff error (subtracting nearly equal function values).
The optimal :math:`h` for central differences is approximately:

.. math::

   h_{\text{opt}} \approx \varepsilon_{\text{mach}}^{1/3}\, |\theta|
   \approx 6 \times 10^{-6}\, |\theta|.

**Where does this come from?**  The central-difference error has two parts:
truncation error :math:`\sim h^2 f'''` (from ignoring higher-order Taylor
terms) and roundoff error :math:`\sim \varepsilon_{\text{mach}} / h` (from
subtracting two nearly equal function values).  Truncation error shrinks as
:math:`h` gets smaller, but roundoff error grows.  Setting the derivative of
the total error to zero and solving for :math:`h` gives :math:`h_{\text{opt}}
\sim \varepsilon_{\text{mach}}^{1/3}`.  With :math:`\varepsilon_{\text{mach}}
\approx 10^{-16}`, this works out to roughly :math:`10^{-5}` to
:math:`10^{-6}` times :math:`|\theta|`.

**The h trade-off: watching error shrink then grow.**
The following experiment sweeps :math:`h` from :math:`10^{-1}` down to
:math:`10^{-15}` and records the error of both forward and central differences.
You will see the characteristic V-shaped error curve: decreasing as truncation
error shrinks, then increasing as roundoff error takes over.

.. code-block:: python

   # Numerical vs analytical derivatives: the h trade-off
   import numpy as np

   np.random.seed(42)

   # Function: log-likelihood of Exponential(lambda) with data
   data = np.random.exponential(scale=2.0, size=50)

   def log_lik(lam):
       return len(data) * np.log(lam) - lam * np.sum(data)

   # Analytical derivative: n/lambda - sum(data)
   lam_val = 0.5
   analytical = len(data) / lam_val - np.sum(data)

   print(f"{'h':<14} {'Forward diff':<16} {'Central diff':<16} {'Error (central)'}")
   print("-" * 62)
   best_err = np.inf
   best_h = None
   for k in range(1, 16):
       h = 10**(-k)
       forward = (log_lik(lam_val + h) - log_lik(lam_val)) / h
       central = (log_lik(lam_val + h) - log_lik(lam_val - h)) / (2 * h)
       err = abs(central - analytical)
       if err < best_err:
           best_err = err
           best_h = h
       print(f"{h:<14.0e} {forward:<16.8f} {central:<16.8f} {err:<.2e}")

   print(f"\nAnalytical:    {analytical:.8f}")
   print(f"Best h:        {best_h:.0e} (error: {best_err:.2e})")
   print("Note: error shrinks then grows as h gets too small (roundoff).")

The complex-step derivative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A remarkable alternative that avoids subtraction altogether.  If :math:`f` is
analytic (can be extended to complex arguments), then:

.. math::

   f(\theta + ih) = f(\theta) + ih\, f'(\theta) - \frac{h^2}{2} f''(\theta) + \cdots

Taking the imaginary part:

.. math::

   f'(\theta) = \frac{\text{Im}\!\bigl[f(\theta + ih)\bigr]}{h} + O(h^2).

**Why this is so elegant.**  Ordinary finite differences compute
:math:`(f(\theta+h) - f(\theta))/h`, which requires *subtracting* two nearby
real numbers --- the source of all roundoff trouble.  The complex-step method
extracts the derivative from the *imaginary part* of a single evaluation.
There is **no subtraction** of nearly equal real numbers, so :math:`h` can be
chosen extremely small (e.g., :math:`h = 10^{-30}`) without roundoff issues,
giving effectively exact derivatives.  The only requirement is that your
function implementation supports complex-valued inputs.

**Complex-step derivative: near-exact numerical differentiation.**
The complex-step method is one of the most elegant tricks in numerical
computing.  By using the imaginary part of a complex perturbation, we avoid
subtraction entirely and get machine-precision derivatives from a single
function evaluation.

.. code-block:: python

   # Complex-step derivative: near-exact numerical differentiation
   import numpy as np

   np.random.seed(42)

   data = np.random.exponential(scale=2.0, size=50)

   def log_lik_complex(lam):
       """Log-likelihood that works with complex numbers."""
       return len(data) * np.log(lam) - lam * np.sum(data)

   lam_val = 0.5
   analytical = len(data) / lam_val - np.sum(data)

   # Compare across a range of h values
   print(f"{'h':<14} {'Complex-step':<20} {'Central diff':<20} "
         f"{'Err (complex)':<15} {'Err (central)'}")
   print("-" * 83)
   for k in [5, 10, 15, 20, 30, 50]:
       h = 10**(-k)
       # Complex step
       cx = np.imag(log_lik_complex(lam_val + 1j * h)) / h
       # Central difference
       if k <= 15:
           cd = (log_lik_complex(lam_val + h).real
                 - log_lik_complex(lam_val - h).real) / (2 * h)
           cd_err = f"{abs(cd - analytical):.2e}"
       else:
           cd = float('nan')
           cd_err = "diverged"
       print(f"{h:<14.0e} {cx:<20.15f} "
             f"{'N/A' if np.isnan(cd) else f'{cd:<20.15f}'}"
             f"{abs(cx - analytical):<15.2e} {cd_err}")

   print(f"\nAnalytical:   {analytical:.15f}")
   print("Complex-step gives machine-precision accuracy with a single eval!")

**Limitation.**  The function must be implemented in a way that supports
complex arithmetic (no ``abs``, no branching on the sign of the input, etc.).


21.5 Numerical Hessians
--------------------------

**Why Hessians?**
The Hessian of the log-likelihood is needed for standard errors (via the
observed Fisher information), Newton's method, and the Laplace approximation
(see :ref:`Chapter 17 <ch17_bayesian>`).

Finite-difference Hessian
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a function :math:`f: \mathbb{R}^p \to \mathbb{R}`, the :math:`(i,j)`
entry of the Hessian can be approximated by:

.. math::

   H_{ij}
   \approx \frac{f(\theta + h\mathbf{e}_i + h\mathbf{e}_j)
                 - f(\theta + h\mathbf{e}_i - h\mathbf{e}_j)
                 - f(\theta - h\mathbf{e}_i + h\mathbf{e}_j)
                 + f(\theta - h\mathbf{e}_i - h\mathbf{e}_j)}{4h^2},

where :math:`\mathbf{e}_i` is the :math:`i`-th standard basis vector.  This
requires :math:`O(p^2)` function evaluations.

For the diagonal, a simpler formula suffices:

.. math::

   H_{ii}
   \approx \frac{f(\theta + h\mathbf{e}_i) - 2f(\theta) + f(\theta - h\mathbf{e}_i)}{h^2}.

Richardson extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To improve accuracy without reducing :math:`h` further (which increases
roundoff), Richardson extrapolation combines estimates at two step sizes.

For a central-difference derivative with error :math:`O(h^2)`:

.. math::

   D(h) = f'(\theta) + c_1 h^2 + c_2 h^4 + \cdots

Compute :math:`D(h)` and :math:`D(h/2)`:

.. math::

   D_{\text{improved}}
   = \frac{4\, D(h/2) - D(h)}{3}
   = f'(\theta) + O(h^4).

**How this works.**  Both :math:`D(h)` and :math:`D(h/2)` have the same
leading error term :math:`c_1 h^2`, but with different coefficients (because
:math:`D(h/2)` uses half the step size, its error is :math:`c_1 (h/2)^2 =
c_1 h^2/4`).  By taking the weighted combination :math:`(4 \cdot D(h/2) -
D(h))/3`, the :math:`h^2` terms cancel exactly, leaving only the much smaller
:math:`h^4` error.  The leading error term cancels.  This can be iterated
(Romberg's method) to reach any desired order.  The same idea applies to
Hessian approximations.


21.6 Condition Numbers
------------------------

**Why conditioning matters.**
Even with perfect arithmetic, some problems are inherently sensitive to small
perturbations in the data or parameters.  Understanding condition numbers helps
you diagnose when your results might be unreliable.

Definition
^^^^^^^^^^^

The condition number of a matrix :math:`A` is:

.. math::

   \kappa(A) = \|A\| \cdot \|A^{-1}\|,

where :math:`\|\cdot\|` is a matrix norm (typically the 2-norm, giving
:math:`\kappa = \sigma_{\max}/\sigma_{\min}` in terms of singular values).

For solving :math:`A\mathbf{x} = \mathbf{b}`, a relative perturbation
:math:`\delta\mathbf{b}/\|\mathbf{b}\|` in the right-hand side causes a
relative perturbation in the solution bounded by:

.. math::

   \frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}
   \le \kappa(A)\, \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}.

In plain English: the condition number acts as an "error amplification
factor."  A 0.01% perturbation in the data, multiplied by a condition number
of :math:`10^6`, could produce up to a 10,000% error in the answer.  If
:math:`\kappa(A) = 10^k`, we lose roughly :math:`k` digits of accuracy.
Since double-precision floating point gives about 16 digits, a condition
number of :math:`10^{10}` means your answer may have only 6 reliable digits,
and :math:`10^{16}` means you may have none.

**Well-conditioned vs ill-conditioned Hessians.**
The condition number of the Hessian determines whether your standard errors
are trustworthy.  Below we construct two regression problems: one with nicely
scaled, uncorrelated predictors, and one with near-collinear predictors on
wildly different scales.

.. code-block:: python

   # Condition numbers of the Hessian: well-conditioned vs ill-conditioned
   import numpy as np

   np.random.seed(42)

   # Well-conditioned: two uncorrelated predictors on similar scales
   n = 200
   X_good = np.column_stack([np.ones(n),
                              np.random.normal(0, 1, n),
                              np.random.normal(0, 1, n)])
   H_good = X_good.T @ X_good  # proportional to Fisher information
   kappa_good = np.linalg.cond(H_good)

   # Ill-conditioned: highly correlated predictors on different scales
   x1 = np.random.normal(0, 1, n)
   x2 = x1 + np.random.normal(0, 0.01, n)  # almost identical to x1
   X_bad = np.column_stack([np.ones(n), x1, x2 * 1000])
   H_bad = X_bad.T @ X_bad
   kappa_bad = np.linalg.cond(H_bad)

   print(f"Well-conditioned Hessian:  kappa = {kappa_good:.1f}")
   print(f"Ill-conditioned Hessian:   kappa = {kappa_bad:.1e}")
   print(f"\nDigits of accuracy lost (well):  ~{np.log10(kappa_good):.1f}")
   print(f"Digits of accuracy lost (ill):   ~{np.log10(kappa_bad):.1f}")
   print("\nIf kappa > 1e6, standard errors from the Hessian may be unreliable.")

**What ill-conditioning does to parameter estimates.**
Let us go further and show that ill-conditioning causes the *solution itself*
to be unstable: tiny perturbations in the data lead to large swings in the
estimated coefficients.

.. code-block:: python

   # Ill-conditioning: small data perturbations cause large coefficient changes
   import numpy as np

   np.random.seed(42)
   n = 100

   # Create near-collinear design matrix
   x1 = np.random.normal(0, 1, n)
   x2 = x1 + np.random.normal(0, 0.001, n)  # nearly identical
   X = np.column_stack([np.ones(n), x1, x2])
   beta_true = np.array([1.0, 2.0, 3.0])
   y = X @ beta_true + np.random.normal(0, 0.1, n)

   # Solve with original data
   beta_hat_1 = np.linalg.lstsq(X, y, rcond=None)[0]

   # Perturb y by a tiny amount
   y_perturbed = y + np.random.normal(0, 1e-6, n)
   beta_hat_2 = np.linalg.lstsq(X, y_perturbed, rcond=None)[0]

   print(f"Condition number: {np.linalg.cond(X.T @ X):.1e}")
   print(f"\nbeta from original y:  {np.round(beta_hat_1, 4)}")
   print(f"beta from perturbed y: {np.round(beta_hat_2, 4)}")
   print(f"Change in beta:        {np.round(beta_hat_2 - beta_hat_1, 4)}")
   print(f"Max perturbation in y: {np.max(np.abs(y_perturbed - y)):.2e}")
   print(f"\nCoefficients changed by O(1) from a perturbation of O(1e-6)!")
   print(f"This is the numerical consequence of ill-conditioning.")

Ill-conditioned likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Hessian of the log-likelihood can be ill-conditioned when:

* **Parameters are on very different scales** (e.g., a mean near 1000 and a
  variance near 0.01).  *Fix:* standardize the data or reparameterize.
* **Parameters are highly correlated** (near-collinearity in regression).
  The Hessian has eigenvalues spanning many orders of magnitude.
  *Fix:* ridge regularization, centering covariates, or removing redundant
  variables.
* **The likelihood surface has flat ridges** --- directions along which the
  log-likelihood changes very slowly.  This indicates weak identifiability.
  *Fix:* add informative priors (Bayesian regularization), fix one parameter,
  or reparameterize.

.. admonition:: Common Pitfall

   A large condition number does not mean your *model* is wrong---it means
   the computer cannot reliably extract the information you are asking for.
   The fix is usually a reparameterization (centering, scaling) or
   regularization, not a different model.

**Practical check.**  After optimization, compute
:math:`\kappa(\mathcal{J}(\hat{\theta}))`.  If :math:`\kappa > 10^6`,
standard errors may be unreliable.


21.7 Cholesky vs Inverse for Solving Linear Systems
------------------------------------------------------

**Why this matters for likelihood computations.**
Many likelihood-based methods require solving linear systems
:math:`\boldsymbol{\Sigma} \mathbf{x} = \mathbf{b}` (e.g., computing
:math:`\boldsymbol{\Sigma}^{-1}\mathbf{y}` for the MVN log-likelihood) or
computing :math:`\log|\boldsymbol{\Sigma}|`.  You should *never* explicitly
invert the matrix.  Instead, use the Cholesky decomposition.

For a positive definite matrix :math:`\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top`:

1. Solve :math:`\boldsymbol{\Sigma}\mathbf{x} = \mathbf{b}` by two
   triangular solves: :math:`\mathbf{L}\mathbf{y} = \mathbf{b}`, then
   :math:`\mathbf{L}^\top\mathbf{x} = \mathbf{y}`.
2. Compute :math:`\log|\boldsymbol{\Sigma}| = 2\sum_i \log L_{ii}`.

**Why this is better.**  Step 1 avoids forming the inverse entirely: triangular
systems can be solved by simple forward/backward substitution in :math:`O(p^2)`
time, compared to :math:`O(p^3)` for a full inversion.  Step 2 exploits the
fact that :math:`\det(\mathbf{L}\mathbf{L}^\top) = (\det \mathbf{L})^2 =
(\prod_i L_{ii})^2`, so the log-determinant reduces to a sum of logarithms of
the diagonal entries of :math:`\mathbf{L}` --- which is both fast and
numerically stable (no risk of overflow from computing a huge determinant
directly).  Both are faster and more accurate than forming
:math:`\boldsymbol{\Sigma}^{-1}`.

.. code-block:: python

   # Cholesky vs explicit inverse: timing and accuracy
   import numpy as np
   import time

   np.random.seed(42)

   results = []
   for p in [50, 200, 500]:
       # Create a positive definite matrix
       A = np.random.randn(p, p)
       Sigma = A @ A.T + p * np.eye(p)
       b = np.random.randn(p)

       # Method 1: Explicit inverse (slow and less accurate)
       t0 = time.perf_counter()
       for _ in range(10):
           Sigma_inv = np.linalg.inv(Sigma)
           x_inv = Sigma_inv @ b
           logdet_inv = np.log(np.linalg.det(Sigma))
       t_inv = (time.perf_counter() - t0) / 10

       # Method 2: Cholesky (fast and more accurate)
       t0 = time.perf_counter()
       for _ in range(10):
           L = np.linalg.cholesky(Sigma)
           y = np.linalg.solve(L, b)              # forward substitution
           x_chol = np.linalg.solve(L.T, y)       # back substitution
           logdet_chol = 2 * np.sum(np.log(np.diag(L)))
       t_chol = (time.perf_counter() - t0) / 10

       # Compare accuracy
       residual_inv = np.linalg.norm(Sigma @ x_inv - b)
       residual_chol = np.linalg.norm(Sigma @ x_chol - b)

       results.append((p, t_inv, t_chol, residual_inv, residual_chol,
                        abs(logdet_inv - logdet_chol)))

   print(f"{'p':<6} {'Time(inv)':<12} {'Time(chol)':<12} "
         f"{'Resid(inv)':<14} {'Resid(chol)':<14} {'logdet diff'}")
   print("-" * 70)
   for p, ti, tc, ri, rc, ld in results:
       print(f"{p:<6} {ti:<12.6f} {tc:<12.6f} {ri:<14.2e} {rc:<14.2e} {ld:.2e}")

   print("\nCholesky is faster and has smaller residuals.")
   print("NEVER compute Sigma^{-1} explicitly in likelihood computations.")


21.8 Parameterization Matters
---------------------------------

**Why reparameterize?**
The same statistical model can be expressed in different coordinate systems.
Some parameterizations make the log-likelihood surface well-behaved (nearly
quadratic, well-conditioned); others create highly curved, ill-conditioned
surfaces that optimizers struggle with.

**Beta distribution on raw vs logit scale.**
Fitting a Beta(:math:`\alpha`, :math:`\beta`) distribution is notoriously
difficult when the MLE is near the boundary (e.g., :math:`\alpha` near 0).
Working on the log or logit scale transforms the constrained, non-linear
problem into an unconstrained, better-conditioned one.

.. code-block:: python

   # Parameterization matters: Beta MLE on raw vs log scale
   import numpy as np
   from scipy.optimize import minimize
   from scipy.special import gammaln

   np.random.seed(42)

   # Generate data from a Beta near the boundary
   alpha_true, beta_true = 0.3, 5.0
   n = 200
   data = np.random.beta(alpha_true, beta_true, size=n)
   log_data = np.log(data)
   log_1mdata = np.log(1 - data)

   # Negative log-likelihood
   def neg_loglik_raw(params):
       a, b = params
       if a <= 0 or b <= 0:
           return 1e10
       return -(gammaln(a + b) - gammaln(a) - gammaln(b)
                + (a - 1) * log_data.sum() + (b - 1) * log_1mdata.sum())

   def neg_loglik_log(log_params):
       """Same likelihood, but parameterized as log(alpha), log(beta)."""
       a, b = np.exp(log_params)
       return -(gammaln(a + b) - gammaln(a) - gammaln(b)
                + (a - 1) * log_data.sum() + (b - 1) * log_1mdata.sum())

   # Optimize on raw scale (constrained)
   res_raw = minimize(neg_loglik_raw, x0=[1.0, 1.0], method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10})

   # Optimize on log scale (unconstrained)
   res_log = minimize(neg_loglik_log, x0=[0.0, 0.0], method='BFGS',
                       options={'maxiter': 5000})
   est_log = np.exp(res_log.x)

   print(f"True parameters:     alpha={alpha_true}, beta={beta_true}")
   print(f"\nRaw-scale optimization (Nelder-Mead, constrained):")
   print(f"  alpha={res_raw.x[0]:.4f}, beta={res_raw.x[1]:.4f}")
   print(f"  Iterations: {res_raw.nit}, Converged: {res_raw.success}")

   print(f"\nLog-scale optimization (BFGS, unconstrained):")
   print(f"  alpha={est_log[0]:.4f}, beta={est_log[1]:.4f}")
   print(f"  Iterations: {res_log.nit}, Converged: {res_log.success}")

   # Condition number comparison at the MLE
   from scipy.optimize import approx_fprime
   h = 1e-5

   hess_raw = np.zeros((2, 2))
   for i in range(2):
       def grad_i(params, idx=i):
           return approx_fprime(params, neg_loglik_raw, h)[idx]
       hess_raw[i] = approx_fprime(res_raw.x, grad_i, h)

   hess_log = np.zeros((2, 2))
   for i in range(2):
       def grad_i_log(params, idx=i):
           return approx_fprime(params, neg_loglik_log, h)[idx]
       hess_log[i] = approx_fprime(res_log.x, grad_i_log, h)

   print(f"\nCondition number of Hessian (raw scale):  "
         f"{np.linalg.cond(hess_raw):.1f}")
   print(f"Condition number of Hessian (log scale):  "
         f"{np.linalg.cond(hess_log):.1f}")
   print("\nThe log-scale Hessian is better conditioned, letting BFGS")
   print("converge faster and produce more reliable standard errors.")


21.9 Sparse and Structured Hessians
--------------------------------------

**Why structure matters.**
For a model with :math:`p` parameters the Hessian is :math:`p \times p`.
Storing it costs :math:`O(p^2)` memory and inverting it costs :math:`O(p^3)`
time.  When :math:`p` is large (thousands to millions) this is prohibitive.
Fortunately, many statistical models produce Hessians with exploitable
structure.

Diagonal Hessians
^^^^^^^^^^^^^^^^^^^

If parameters interact only weakly, the off-diagonal entries of the Hessian
are small.  Approximating :math:`H \approx \text{diag}(H_{11}, \ldots,
H_{pp})` reduces storage to :math:`O(p)` and inversion to :math:`O(p)`.
This is common in variational inference with mean-field approximations
(see :ref:`Chapter 18 <ch18_computational>`).

Band and sparse Hessians
^^^^^^^^^^^^^^^^^^^^^^^^^^

In time series and spatial models, parameter :math:`\theta_i` may interact
only with nearby parameters :math:`\theta_{i-1}, \theta_{i+1}`.  The Hessian
is banded or sparse.  Sparse Cholesky factorization (complexity :math:`O(p)`
for banded matrices) replaces the :math:`O(p^3)` dense factorization.  This
is the key computational trick behind INLA
(see :ref:`Chapter 18 <ch18_computational>`).

Block-diagonal Hessians
^^^^^^^^^^^^^^^^^^^^^^^^^

Hierarchical models often have parameter blocks that interact only through
hyperparameters.  The Hessian is block-diagonal (possibly after reordering),
and each block can be inverted independently.

Hessian-free methods
^^^^^^^^^^^^^^^^^^^^^^

When even storing the Hessian is too expensive, we can compute
*Hessian--vector products* :math:`H\mathbf{v}` without forming :math:`H`
explicitly.  This is done via a second forward-mode AD pass or via the
finite-difference approximation:

.. math::

   H\mathbf{v} \approx \frac{\nabla f(\theta + h\mathbf{v}) - \nabla f(\theta)}{h}.

Reading this formula: to compute the product of the Hessian with a vector
:math:`\mathbf{v}`, we evaluate the gradient at two nearby points (the current
parameter and a small step in the direction :math:`\mathbf{v}`) and take the
difference.  This gives us the Hessian--vector product using just two gradient
evaluations, regardless of the dimension :math:`p` --- we never need to compute
or store the full :math:`p \times p` Hessian matrix.

Conjugate gradient methods use only Hessian--vector products to solve
:math:`H\mathbf{d} = -\nabla f`, enabling Newton-like updates in
:math:`O(p)` per iteration.


21.10 Parallel and GPU Computation
------------------------------------

**Why parallelism?**
The log-likelihood of iid data is a *sum*:

.. math::

   \ell(\theta) = \sum_{i=1}^n \ell_i(\theta),

where :math:`\ell_i(\theta) = \log p(x_i \mid \theta)`.  Sums are trivially
parallelizable: split the data across :math:`P` processors, compute partial
sums, and combine.  The gradient and (approximate) Hessian decompose similarly.

Data-parallel likelihood evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Multi-core CPU:**  Split :math:`n` observations across cores.  Frameworks
  like OpenMP, multiprocessing (Python), or ``parallel::mclapply`` (R)
  handle this transparently.
* **GPU:**  A modern GPU has thousands of small cores.  When each
  :math:`\ell_i` involves the same arithmetic (e.g., evaluating a Gaussian
  density), the GPU processes thousands of observations simultaneously.

The speedup is nearly linear in :math:`P` for the likelihood evaluation; the
overhead is the initial data transfer to the device and the final reduction
(summing partial results).

GPU considerations
^^^^^^^^^^^^^^^^^^^^

* **Memory:**  GPU memory (typically 8--80 GB) limits the batch size.  If the
  data do not fit, use mini-batch gradient methods.
* **Precision:**  Many GPUs are faster in single precision (32-bit).  For
  likelihood optimization this is often insufficient --- the Hessian and
  standard errors need double precision.  Mixed-precision strategies compute
  the forward pass in 32-bit and accumulate the loss in 64-bit.
* **Branching:**  GPUs perform best when all threads execute the same
  instruction (SIMD).  If the likelihood involves heavy branching (e.g.,
  mixture models with different component structures), performance degrades.
* **Frameworks:**  JAX, PyTorch, TensorFlow, and CuPy expose GPU-accelerated
  linear algebra and automatic differentiation, making it straightforward to
  port likelihood computations.

Mini-batch stochastic optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For very large :math:`n`, even a single pass through all data is expensive.
Stochastic gradient descent (SGD) and its variants (Adam, AdaGrad) use a
random subset (mini-batch) of size :math:`B \ll n` to estimate the gradient:

.. math::

   \nabla \ell(\theta)
   \approx \frac{n}{B} \sum_{i \in \mathcal{B}} \nabla \ell_i(\theta).

The factor :math:`n/B` rescales the mini-batch gradient to approximate the
full gradient.  This introduces noise, but the noise can actually help escape
saddle points and shallow local minima.

**Convergence.**  SGD with a decaying step size :math:`\eta_t` converges to
the MLE under standard conditions (:math:`\sum \eta_t = \infty`,
:math:`\sum \eta_t^2 < \infty`).


21.11 Software Libraries
--------------------------

A brief overview of widely used tools for likelihood optimization and
inference.

Optimization
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Library
     - Language
     - Notes
   * - ``scipy.optimize``
     - Python
     - General-purpose; ``minimize`` supports BFGS, L-BFGS-B, Nelder--Mead, trust-region
   * - ``optim`` / ``nlminb``
     - R
     - Base R optimizers; ``nlminb`` is port-bounded quasi-Newton
   * - ``NLopt``
     - C (with Python, R, Julia bindings)
     - Large collection of local and global optimizers
   * - ``Optax``
     - Python (JAX)
     - Gradient-based optimizers for ML; Adam, SGD, etc.

Bayesian inference
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Library
     - Language
     - Notes
   * - Stan (CmdStan, PyStan, RStan)
     - Stan DSL, Python, R
     - HMC/NUTS; automatic differentiation; gold standard for MCMC
   * - PyMC
     - Python
     - NUTS via PyTensor; variational inference; flexible model specification
   * - NumPyro
     - Python (JAX)
     - NUTS on GPU; fast; composable with JAX ecosystem
   * - JAGS
     - BUGS-like DSL
     - Gibbs sampling; older but still used for conjugate models
   * - R-INLA
     - R
     - INLA for latent Gaussian models; very fast
   * - ``sbi``
     - Python (PyTorch)
     - Simulation-based inference (NPE, NLE, NRE)

Automatic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Library
     - Language
     - Notes
   * - JAX
     - Python
     - Functional AD; JIT compilation; GPU/TPU support
   * - PyTorch
     - Python
     - Dynamic computation graph; dominant in deep learning
   * - TensorFlow
     - Python
     - Static or eager AD; ``tf.GradientTape``
   * - Stan Math
     - C++
     - Reverse-mode AD tuned for statistical models
   * - ForwardDiff.jl
     - Julia
     - Forward-mode AD; exact and fast for low-dimensional gradients


21.12 Summary
---------------

* IEEE 754 double precision provides about 16 significant digits and a range
  up to :math:`10^{308}`.  Raw likelihoods easily underflow; always work in
  log-space.
* Catastrophic cancellation can destroy precision when subtracting nearly
  equal numbers---use numerically stable formulas (centered variance, log-space
  computations).
* The log-sum-exp trick stabilizes :math:`\log(\sum \exp(a_k))` by
  subtracting the maximum, preventing overflow and underflow.
* Numerical differentiation by central differences has error :math:`O(h^2)`;
  the complex-step method avoids subtraction cancellation and can achieve
  near-machine-precision accuracy.
* Richardson extrapolation improves finite-difference estimates by combining
  results at multiple step sizes.
* The condition number of the Hessian determines how reliably we can compute
  standard errors; ill-conditioning signals reparameterization or
  regularization is needed.
* Cholesky factorization is faster and more accurate than explicit matrix
  inversion for solving linear systems in likelihood computations.
* Reparameterization (e.g., log or logit scale) can dramatically improve
  the conditioning and convergence behavior of likelihood optimization.
* Sparse, banded, block-diagonal, and Hessian-free techniques make
  second-order methods feasible for large models.
* Likelihoods of iid data are embarrassingly parallel; GPUs offer massive
  speedups when the per-observation computation is uniform.
* A rich ecosystem of software --- from ``scipy.optimize`` to Stan to JAX ---
  provides production-quality implementations of everything in this guide.
