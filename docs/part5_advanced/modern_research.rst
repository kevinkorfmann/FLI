.. _ch20_modern:

==========================================
Chapter 20 -- Modern Research Frontiers
==========================================

Classical likelihood theory assumes we can write down and evaluate the
likelihood function :math:`L(\theta) = p(\mathbf{x} \mid \theta)`.  Modern
research pushes far beyond this assumption.  We now have models where the
likelihood is intractable, data sets so large that exact computation is
infeasible, and neural networks flexible enough to *learn* entire probability
distributions.  This chapter surveys these frontiers, always tracing the
thread back to the likelihood ideas developed in earlier chapters.

If the preceding chapters built a solid house, this chapter shows you the
expanding neighborhood.  The foundation---the likelihood function---remains the
same.  What has changed is the scale of ambition: from a handful of parameters
to millions, from closed-form densities to opaque simulators, and from
hand-derived gradients to automatic ones.


20.1 Automatic Differentiation
---------------------------------

**Why this enables everything else.**
Nearly every optimization and sampling algorithm in this guide uses
derivatives of the log-likelihood.  Computing these derivatives by hand is
error-prone and impractical for complex models.  *Automatic differentiation*
(AD) computes exact (up to floating point) derivatives of arbitrary computer
programs.

Forward mode
^^^^^^^^^^^^^

Forward-mode AD propagates *dual numbers* through the computation graph.
A dual number augments each value :math:`v` with its derivative
:math:`\dot{v} = dv/d\theta`:

.. math::

   (v, \dot{v}) + (u, \dot{u}) &= (v + u,\; \dot{v} + \dot{u}), \\
   (v, \dot{v}) \times (u, \dot{u}) &= (v \cdot u,\; v\,\dot{u} + u\,\dot{v}).

One forward pass computes :math:`f(\theta)` and :math:`df/d\theta_j` for one
input variable :math:`\theta_j`.  The cost is :math:`O(p)` forward passes for
a gradient with respect to :math:`p` parameters.

.. admonition:: Intuition

   Think of dual numbers as carrying both a value and a "derivative tag" along
   for the ride.  As arithmetic flows through the program, the derivative tag
   updates itself automatically by the chain rule.  You never need to derive
   a single formula---the computer does it for you, exactly.

.. admonition:: Real-World Example

   **Forward-mode AD with dual numbers.**

   Let's implement dual numbers from scratch and use them to differentiate
   a log-likelihood function.  This is exactly what libraries like
   ``ForwardDiff.jl`` (Julia) do under the hood.

   *Pseudocode:*

   .. code-block:: text

      1. Define a DualNumber class with (value, derivative) pairs
      2. Implement __add__, __mul__, __truediv__, __pow__, exp, log
      3. Write the Bernoulli log-likelihood using dual numbers
      4. Seed theta = DualNumber(theta_val, 1.0)  <- derivative of theta w.r.t. itself is 1
      5. Evaluate log-likelihood; the derivative comes out automatically

   *Python implementation:*

   .. code-block:: python

      # Forward-mode AD with dual numbers: differentiate a log-likelihood
      import math

      class Dual:
          """A dual number (value, derivative) for forward-mode AD."""
          def __init__(self, val, der=0.0):
              self.val = val
              self.der = der
          def __add__(self, other):
              o = other if isinstance(other, Dual) else Dual(other)
              return Dual(self.val + o.val, self.der + o.der)
          def __radd__(self, other):
              return self.__add__(other)
          def __mul__(self, other):
              o = other if isinstance(other, Dual) else Dual(other)
              return Dual(self.val * o.val, self.val * o.der + self.der * o.val)
          def __rmul__(self, other):
              return self.__mul__(other)
          def __sub__(self, other):
              o = other if isinstance(other, Dual) else Dual(other)
              return Dual(self.val - o.val, self.der - o.der)
          def __rsub__(self, other):
              o = other if isinstance(other, Dual) else Dual(other)
              return Dual(o.val - self.val, o.der - self.der)

      def dual_log(x):
          return Dual(math.log(x.val), x.der / x.val)

      # Bernoulli log-likelihood: l(theta) = s*log(theta) + (n-s)*log(1-theta)
      s, n = 7, 10

      theta_val = 0.6
      theta = Dual(theta_val, 1.0)  # seed derivative = 1

      log_lik = s * dual_log(theta) + (n - s) * dual_log(1 - theta)

      # Compare with analytical derivative: s/theta - (n-s)/(1-theta)
      analytical = s / theta_val - (n - s) / (1 - theta_val)

      print(f"Log-likelihood:     {log_lik.val:.6f}")
      print(f"AD derivative:      {log_lik.der:.6f}")
      print(f"Analytical deriv.:  {analytical:.6f}")
      print(f"Match: {abs(log_lik.der - analytical) < 1e-10}")

**Extending dual numbers to richer functions.**
The Bernoulli example is clean but simple.  Real likelihood functions use
:math:`\exp`, :math:`\text{pow}`, division, and composition of many
operations.  Let us extend the dual number class and differentiate a Normal
log-likelihood, comparing the result with both the analytical and numerical
gradients.

.. code-block:: python

   # Extended dual numbers: Normal log-likelihood differentiation
   import math

   class Dual:
       """Dual number with full arithmetic for forward-mode AD."""
       def __init__(self, val, der=0.0):
           self.val = val
           self.der = der
       def __repr__(self):
           return f"Dual({self.val:.6f}, {self.der:.6f})"
       def __add__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return Dual(self.val + o.val, self.der + o.der)
       __radd__ = lambda s, o: s.__add__(o)
       def __sub__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return Dual(self.val - o.val, self.der - o.der)
       def __rsub__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return Dual(o.val - self.val, o.der - self.der)
       def __mul__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return Dual(self.val * o.val, self.val * o.der + self.der * o.val)
       __rmul__ = lambda s, o: s.__mul__(o)
       def __truediv__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return Dual(self.val / o.val,
                       (self.der * o.val - self.val * o.der) / o.val**2)
       def __rtruediv__(self, o):
           o = o if isinstance(o, Dual) else Dual(o)
           return o.__truediv__(self)
       def __pow__(self, n):
           return Dual(self.val**n, n * self.val**(n-1) * self.der)
       def __neg__(self):
           return Dual(-self.val, -self.der)

   def dual_log(x):
       return Dual(math.log(x.val), x.der / x.val)

   def dual_exp(x):
       e = math.exp(x.val)
       return Dual(e, e * x.der)

   # Normal log-likelihood: -n/2 log(2*pi*sigma^2) - sum((x-mu)^2)/(2*sigma^2)
   # Differentiate w.r.t. mu at a fixed sigma
   import numpy as np
   np.random.seed(42)
   data = np.random.normal(3.0, 1.5, size=50)

   sigma_val = 1.5
   mu_val = 2.8

   # Using dual numbers (seed derivative of mu = 1)
   mu = Dual(mu_val, 1.0)
   n = len(data)
   log_lik = Dual(-n/2 * math.log(2 * math.pi * sigma_val**2))
   for xi in data:
       log_lik = log_lik - (xi - mu)**2 / (2 * sigma_val**2)

   # Analytical gradient: sum(x - mu) / sigma^2
   grad_analytical = np.sum(data - mu_val) / sigma_val**2

   # Numerical gradient (central differences)
   h = 1e-7
   def ll_numpy(mu_v):
       return -n/2 * np.log(2*np.pi*sigma_val**2) - np.sum((data-mu_v)**2)/(2*sigma_val**2)
   grad_numerical = (ll_numpy(mu_val + h) - ll_numpy(mu_val - h)) / (2 * h)

   print(f"Normal log-lik at mu={mu_val}: {log_lik.val:.6f}")
   print(f"\nGradient d(loglik)/d(mu):")
   print(f"  Forward-mode AD: {log_lik.der:.10f}")
   print(f"  Analytical:      {grad_analytical:.10f}")
   print(f"  Numerical:       {grad_numerical:.10f}")
   print(f"\nAD vs analytical error: {abs(log_lik.der - grad_analytical):.2e}")
   print(f"Numerical vs analytical error: {abs(grad_numerical - grad_analytical):.2e}")
   print("AD matches the analytical gradient to machine precision.")

Reverse mode (backpropagation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reverse-mode AD first evaluates the function (forward pass), recording the
computation graph, then propagates *adjoint* values backward:

.. math::

   \bar{v}_i = \frac{\partial f}{\partial v_i}
   = \sum_{j \,:\, v_i \to v_j} \bar{v}_j \,\frac{\partial v_j}{\partial v_i}.

Reading this formula: :math:`\bar{v}_i` is the sensitivity of the final output
:math:`f` to the intermediate value :math:`v_i`.  To compute it, we look at
every node :math:`v_j` that directly depends on :math:`v_i` in the computation
graph, and sum up how changes in :math:`v_i` propagate through each of those
paths.  This is just the multi-variable chain rule, applied systematically
backward through the graph.

One reverse pass computes the full gradient :math:`\nabla_\theta f` regardless
of the dimension :math:`p`.  The cost is :math:`O(1)` reverse passes (plus
storage for the computation graph).

**Consequence for likelihood optimization.**
Reverse-mode AD makes it practical to optimize log-likelihoods with thousands
or millions of parameters.  Libraries such as JAX, PyTorch, TensorFlow, and
Stan all provide reverse-mode AD.

**Reverse-mode AD concept: gradient of a multi-layer function.**
Forward mode is efficient when you have few inputs and many outputs.  Reverse
mode is efficient when you have many inputs and one output---exactly the
situation in likelihood optimization.  Below we implement a minimal
reverse-mode AD engine and use it to differentiate a composed function with
three parameters.

.. code-block:: python

   # Reverse-mode AD concept: a minimal tape-based implementation
   import math

   class Var:
       """A variable in a computation graph for reverse-mode AD."""
       def __init__(self, val, children=(), grad_fns=()):
           self.val = val
           self.grad = 0.0
           self.children = children
           self.grad_fns = grad_fns

       def __add__(self, other):
           other = other if isinstance(other, Var) else Var(other)
           out = Var(self.val + other.val, (self, other),
                     (lambda g: g, lambda g: g))
           return out
       __radd__ = lambda s, o: s.__add__(o)

       def __mul__(self, other):
           other = other if isinstance(other, Var) else Var(other)
           sv, ov = self.val, other.val
           out = Var(sv * ov, (self, other),
                     (lambda g, ov=ov: g * ov, lambda g, sv=sv: g * sv))
           return out
       __rmul__ = lambda s, o: s.__mul__(o)

       def __neg__(self):
           return Var(-self.val, (self,), (lambda g: -g,))

       def __sub__(self, other):
           return self + (-other)

       def backward(self):
           """Backpropagate gradients through the computation graph."""
           # Topological sort
           order, visited = [], set()
           def topo(v):
               if id(v) not in visited:
                   visited.add(id(v))
                   for c in v.children:
                       topo(c)
                   order.append(v)
           topo(self)
           self.grad = 1.0
           for v in reversed(order):
               for child, grad_fn in zip(v.children, v.grad_fns):
                   child.grad += grad_fn(v.grad)

   def var_log(x):
       out = Var(math.log(x.val), (x,),
                 (lambda g, xv=x.val: g / xv,))
       return out

   def var_exp(x):
       e = math.exp(x.val)
       out = Var(e, (x,), (lambda g, e=e: g * e,))
       return out

   # Function: f(a,b,c) = log(a*b + exp(c))
   # Analytical: df/da = b/(a*b+exp(c)), df/db = a/(a*b+exp(c)),
   #             df/dc = exp(c)/(a*b+exp(c))
   a = Var(2.0)
   b = Var(3.0)
   c = Var(1.0)

   # Forward pass
   f = var_log(a * b + var_exp(c))

   # Backward pass: one call gives ALL gradients
   f.backward()

   # Analytical gradients
   denom = 2.0 * 3.0 + math.exp(1.0)
   print(f"f(2,3,1) = log(2*3 + exp(1)) = {f.val:.6f}")
   print(f"\nReverse-mode gradients (one backward pass for all 3 params):")
   print(f"  df/da: AD={a.grad:.6f}, analytical={3.0/denom:.6f}")
   print(f"  df/db: AD={b.grad:.6f}, analytical={2.0/denom:.6f}")
   print(f"  df/dc: AD={c.grad:.6f}, analytical={math.exp(1.0)/denom:.6f}")
   print(f"\nForward mode would need 3 passes (one per parameter).")
   print(f"Reverse mode needed just 1 pass. This is why it scales to millions of params.")

**Autodiff of a Normal log-likelihood: analytical vs numerical vs AD.**
Here we bring together all three gradient methods to differentiate the same
Normal log-likelihood.  This comparison shows that AD gives the *exact*
analytical gradient, while numerical differentiation introduces a small error.

.. code-block:: python

   # Autodiff of Normal log-likelihood: analytical vs numerical vs AD
   import numpy as np

   np.random.seed(42)

   # Data
   data = np.random.normal(loc=3.0, scale=1.5, size=50)

   def log_likelihood(mu, sigma, data):
       """Normal log-likelihood."""
       n = len(data)
       return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

   # Analytical gradients for comparison
   mu_val, sigma_val = 2.8, 1.6
   n = len(data)
   grad_mu_analytical = np.sum(data - mu_val) / sigma_val**2
   grad_sigma_analytical = -n / sigma_val + np.sum((data - mu_val)**2) / sigma_val**3

   # Numerical gradient (central differences)
   h = 1e-7
   num_grad_mu = (log_likelihood(mu_val + h, sigma_val, data)
                  - log_likelihood(mu_val - h, sigma_val, data)) / (2 * h)
   num_grad_sigma = (log_likelihood(mu_val, sigma_val + h, data)
                     - log_likelihood(mu_val, sigma_val - h, data)) / (2 * h)

   # "AD" via complex-step (machine-precision, like true AD)
   def ll_complex(mu, sigma):
       n = len(data)
       return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

   h_cs = 1e-30
   ad_grad_mu = np.imag(ll_complex(mu_val + 1j*h_cs, sigma_val)) / h_cs
   ad_grad_sigma = np.imag(ll_complex(mu_val, sigma_val + 1j*h_cs)) / h_cs

   print(f"Log-likelihood at (mu={mu_val}, sigma={sigma_val}): "
         f"{log_likelihood(mu_val, sigma_val, data):.4f}")
   print(f"\n{'Method':<14} {'d/d(mu)':<18} {'d/d(sigma)':<18}")
   print("-" * 50)
   print(f"{'Analytical':<14} {grad_mu_analytical:<18.10f} {grad_sigma_analytical:<18.10f}")
   print(f"{'Numerical':<14} {num_grad_mu:<18.10f} {num_grad_sigma:<18.10f}")
   print(f"{'AD (complex)':<14} {ad_grad_mu:<18.10f} {ad_grad_sigma:<18.10f}")
   print(f"\nNumerical error (mu):    {abs(num_grad_mu - grad_mu_analytical):.2e}")
   print(f"AD error (mu):           {abs(ad_grad_mu - grad_mu_analytical):.2e}")
   print(f"Numerical error (sigma): {abs(num_grad_sigma - grad_sigma_analytical):.2e}")
   print(f"AD error (sigma):        {abs(ad_grad_sigma - grad_sigma_analytical):.2e}")
   print("# With JAX: grad_fn = jax.grad(log_likelihood, argnums=(0, 1))")
   print("# One call gives exact gradients via reverse-mode AD.")

**Connection to Fisher information.**
Second-order methods (Newton, natural gradient) need the Hessian or Fisher
information matrix.  AD can compute Hessian--vector products efficiently,
enabling quasi-Newton methods and second-order optimization at scale.


20.2 Neural Density Estimation
---------------------------------

**Why parameterize distributions with neural networks?**
Traditional parametric families (Normal, Gamma, etc.) are convenient but
limited: the true data-generating distribution may not belong to any simple
family.  Neural networks can approximate *any* density function (under mild
conditions), giving us a flexible "likelihood machine."

Mixture density networks
^^^^^^^^^^^^^^^^^^^^^^^^^^

A mixture density network (MDN; Bishop, 1994) outputs the parameters of a
Gaussian mixture:

.. math::

   p(y \mid \mathbf{x}; \mathbf{w})
   = \sum_{k=1}^K \pi_k(\mathbf{x}; \mathbf{w})\;
     \mathcal{N}\!\bigl(y;\; \mu_k(\mathbf{x}; \mathbf{w}),\;
     \sigma_k^2(\mathbf{x}; \mathbf{w})\bigr),

where :math:`\pi_k, \mu_k, \sigma_k^2` are neural network outputs.  Training
maximizes the log-likelihood :math:`\sum_i \log p(y_i \mid \mathbf{x}_i;
\mathbf{w})` --- exactly the MLE principle from :ref:`Part I <part1>`.

.. admonition:: Why does this matter?

   Standard regression assumes a single predicted value with Gaussian noise.
   But many real-world relationships are multimodal: given the same input, the
   output could take several distinct values.  MDNs capture this by predicting
   an entire mixture distribution, not just a point.

**Mixture density network: fitting a multimodal relationship.**
A standard regression predicts a single value for each input.  But what if
the relationship is one-to-many?  For example, the inverse kinematics of a
robot arm: multiple joint configurations can produce the same end-effector
position.  An MDN predicts an entire mixture distribution for each input.

.. code-block:: python

   # Mixture density network: fitting a multimodal conditional distribution
   import numpy as np
   from scipy.special import logsumexp

   np.random.seed(42)

   # Generate multimodal data: y = x + noise, but with two modes
   n = 1000
   x = np.random.uniform(-1, 1, n)
   mode = np.random.binomial(1, 0.5, n)
   y = np.where(mode == 0, np.sin(3*x) + 0.2*np.random.randn(n),
                            -np.sin(3*x) + 0.2*np.random.randn(n))

   # Simple MDN with K=2 components
   # Parameters: for each x, predict (pi, mu_1, mu_2, sigma_1, sigma_2)
   # We use a simple linear model for illustration (real MDN uses neural net)
   K = 2

   # Feature matrix: polynomial basis
   deg = 5
   Phi = np.column_stack([x**d for d in range(deg+1)])

   # Initialize weights: each output head has (deg+1) weights
   n_features = deg + 1
   np.random.seed(123)
   W_mu = np.random.randn(K, n_features) * 0.1     # means
   W_log_sigma = np.zeros((K, n_features))           # log-std
   W_logit_pi = np.zeros((K, n_features))             # mixing logits

   def mdn_log_likelihood(W_mu, W_log_sigma, W_logit_pi, Phi, y):
       """Compute MDN log-likelihood."""
       mu = Phi @ W_mu.T          # (n, K)
       log_sigma = Phi @ W_log_sigma.T
       sigma = np.exp(np.clip(log_sigma, -5, 5))
       logits = Phi @ W_logit_pi.T
       log_pi = logits - logsumexp(logits, axis=1, keepdims=True)

       # Log-density of each component for each observation
       log_comp = (log_pi - 0.5*np.log(2*np.pi)
                   - log_sigma - 0.5*((y[:,None] - mu)/sigma)**2)

       # Log-likelihood: sum of logsumexp over components
       return np.sum(logsumexp(log_comp, axis=1))

   # Simple gradient descent (numerical gradients for clarity)
   lr = 1e-4
   h = 1e-5
   print(f"{'Iter':<8} {'Log-likelihood':<18}")
   print("-" * 26)

   for iteration in range(201):
       ll = mdn_log_likelihood(W_mu, W_log_sigma, W_logit_pi, Phi, y)
       if iteration % 50 == 0:
           print(f"{iteration:<8} {ll:<18.2f}")

       # Numerical gradient for W_mu (simplified)
       for k in range(K):
           for j in range(n_features):
               W_mu[k,j] += h
               ll_plus = mdn_log_likelihood(W_mu, W_log_sigma, W_logit_pi, Phi, y)
               W_mu[k,j] -= h
               W_mu[k,j] += lr * (ll_plus - ll) / h

               W_log_sigma[k,j] += h
               ll_plus = mdn_log_likelihood(W_mu, W_log_sigma, W_logit_pi, Phi, y)
               W_log_sigma[k,j] -= h
               W_log_sigma[k,j] += lr * (ll_plus - ll) / h

   # Evaluate the learned mixture at a test point
   x_test = 0.5
   phi_test = np.array([x_test**d for d in range(deg+1)])
   mu_test = phi_test @ W_mu.T
   sigma_test = np.exp(phi_test @ W_log_sigma.T)
   logit_test = phi_test @ W_logit_pi.T
   pi_test = np.exp(logit_test - logsumexp(logit_test))

   print(f"\nAt x={x_test}:")
   for k in range(K):
       print(f"  Component {k+1}: pi={pi_test[k]:.3f}, "
             f"mu={mu_test[k]:.3f}, sigma={sigma_test[k]:.3f}")
   print(f"  True modes: sin(1.5)={np.sin(1.5):.3f}, -sin(1.5)={-np.sin(1.5):.3f}")

Autoregressive models
^^^^^^^^^^^^^^^^^^^^^^

The chain rule of probability factorizes any joint density:

.. math::

   p(\mathbf{x}) = \prod_{d=1}^D p(x_d \mid x_1, \ldots, x_{d-1}).

An autoregressive model parameterizes each conditional with a neural network.
Examples include MADE, PixelCNN, and WaveNet.  The log-likelihood is:

.. math::

   \log p(\mathbf{x})
   = \sum_{d=1}^D \log p(x_d \mid x_{<d}; \mathbf{w}),

In plain English: the log-probability of the whole data vector decomposes into
a sum of log-probabilities, each asking "given all the previous dimensions,
how likely is this next one?"  A neural network learns each of these
conditional distributions.  Because the result is a simple sum of
log-probabilities, the total log-likelihood is tractable --- we can evaluate
it and take gradients --- so training is just maximum likelihood estimation
via gradient ascent.


20.3 Simulation-Based Inference (Likelihood-Free)
----------------------------------------------------

**Why likelihood-free?**
Many scientific models are defined by a *simulator* --- a computer program that
generates synthetic data given parameters --- but the likelihood function
:math:`p(\mathbf{x} \mid \theta)` is intractable because the simulator
involves stochastic processes, latent variables, or differential equations
without closed-form solutions.

.. admonition:: Real-World Example

   **Simulation-based inference in ecology and cosmology.**

   In population genetics, models of evolution produce complex patterns of
   genetic variation through stochastic birth-death processes, mutation, and
   natural selection.  You can *simulate* these processes easily, but there is
   no way to write down :math:`p(\text{observed genomes} \mid \text{mutation
   rate}, \text{population size})` in closed form.  The same situation arises
   in cosmology (simulating galaxy formation), epidemiology (agent-based disease
   models), and climate science (atmospheric simulators).

   Simulation-based inference lets you do Bayesian inference anyway: generate
   thousands of synthetic datasets from the simulator, then train a neural
   network to learn the mapping from data to posterior.

Approximate Bayesian Computation (ABC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest likelihood-free method.

1. Sample :math:`\theta^* \sim p(\theta)` from the prior.
2. Simulate :math:`\mathbf{x}^* \sim p(\cdot \mid \theta^*)`.
3. Accept :math:`\theta^*` if :math:`d(S(\mathbf{x}^*), S(\mathbf{x}_{\text{obs}}))
   < \varepsilon`, where :math:`S` is a summary statistic and :math:`d` is a
   distance metric.

The accepted samples approximate the posterior
:math:`p(\theta \mid d(S(\mathbf{x}), S(\mathbf{x}_{\text{obs}})) < \varepsilon)`.
As :math:`\varepsilon \to 0` this converges to
:math:`p(\theta \mid S(\mathbf{x}_{\text{obs}}))`, and if :math:`S` is
sufficient it equals the true posterior.

**ABC for a simple model: recovering the posterior.**
To make ABC concrete, we use a model where we *know* the true posterior
(Poisson observations with a Gamma prior), so we can verify that ABC gives
the right answer.  This is the "test with a known answer" approach that builds
trust before applying ABC to truly intractable models.

.. code-block:: python

   # ABC for a Poisson model: compare ABC posterior with known conjugate posterior
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # "Simulator": Poisson observations with unknown rate lambda
   def simulator(lam, n_obs=20):
       return np.random.poisson(lam, n_obs)

   # Observed data
   x_obs = np.array([3, 5, 2, 4, 3, 6, 2, 3, 4, 5,
                      3, 4, 2, 3, 4, 5, 3, 4, 3, 2])
   s_obs = x_obs.mean()  # summary statistic: sample mean
   n_obs = len(x_obs)

   # Step 1: Generate training data from prior
   S = 50_000
   prior = stats.uniform(loc=0.5, scale=9.5)  # Uniform(0.5, 10)
   thetas = prior.rvs(S)
   summary_stats = np.array([simulator(lam, n_obs).mean() for lam in thetas])

   # Step 2: ABC with decreasing epsilon
   print(f"Observed summary stat (mean): {s_obs:.2f}")
   print(f"\n{'epsilon':<10} {'n_accepted':<14} {'ABC mean':<12} "
         f"{'ABC std':<10} {'Accept %'}")
   print("-" * 58)

   for epsilon in [1.0, 0.5, 0.3, 0.15, 0.08]:
       mask = np.abs(summary_stats - s_obs) < epsilon
       accepted = thetas[mask]
       if len(accepted) > 5:
           print(f"{epsilon:<10.2f} {len(accepted):<14} {accepted.mean():<12.3f} "
                 f"{accepted.std():<10.3f} {100*len(accepted)/S:<.1f}%")

   # True posterior: Gamma (since Poisson-Gamma is conjugate)
   # With flat prior ~ Gamma(1, 0), posterior is Gamma(1 + sum(x), 1/(1 + n))
   alpha_post = 1 + x_obs.sum()
   beta_post = 1 + n_obs
   true_post = stats.gamma(a=alpha_post, scale=1/beta_post)

   print(f"\nTrue posterior (Gamma): mean={true_post.mean():.3f}, "
         f"std={true_post.std():.3f}")
   print(f"ABC posterior (eps=0.15): approaches the true posterior")
   print(f"\nAs epsilon -> 0, ABC converges to the exact posterior,")
   print(f"but the acceptance rate drops, requiring more simulations.")

**Limitations.**  ABC is slow in high dimensions, sensitive to the choice of
summary statistics, and requires the tolerance :math:`\varepsilon` to be
chosen carefully.

Neural Posterior Estimation (NPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of accepting/rejecting, train a neural network (typically a
normalizing flow, Section 20.5) to directly map simulated data to posterior
distributions.

1. Repeat: sample :math:`\theta^{(s)} \sim p(\theta)`, simulate
   :math:`\mathbf{x}^{(s)} \sim p(\cdot \mid \theta^{(s)})`.
2. Train a conditional density estimator
   :math:`q_\phi(\theta \mid \mathbf{x})` by maximizing:

   .. math::

      \sum_{s=1}^S \log q_\phi(\theta^{(s)} \mid \mathbf{x}^{(s)}).

3. Evaluate :math:`q_\phi(\theta \mid \mathbf{x}_{\text{obs}})` at the
   observed data to obtain the approximate posterior.

The loss function is the *negative* log-likelihood of the parameters under the
conditional density model --- a direct extension of MLE.

Neural Likelihood Estimation (NLE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of estimating the posterior directly, estimate the *likelihood*:

1. Train :math:`q_\phi(\mathbf{x} \mid \theta)` to approximate
   :math:`p(\mathbf{x} \mid \theta)`.
2. Use the learned likelihood in standard Bayesian inference (e.g., MCMC with
   :math:`q_\phi` as the likelihood).

This separates the approximation step from the inference step and allows
reuse of the learned likelihood with different priors.

Neural Ratio Estimation (NRE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A third approach estimates the *likelihood-to-evidence ratio*
:math:`r(\mathbf{x}, \theta) = p(\mathbf{x} \mid \theta) / p(\mathbf{x})`,
which is the quantity needed in MCMC acceptance ratios.  Training a classifier
to distinguish pairs :math:`(\mathbf{x}, \theta)` drawn jointly from
:math:`p(\mathbf{x}, \theta)` versus the marginals
:math:`p(\mathbf{x})\,p(\theta)` recovers this ratio.


20.4 Amortized Inference
---------------------------

**Why amortize?**
Traditional inference is "one-shot": for each new dataset we run a full
optimization or MCMC.  In amortized inference we train a neural network *once*
on many simulated datasets, and at test time it maps any new observation to
its posterior in a single forward pass.

Formally, we learn :math:`q_\phi(\theta \mid \mathbf{x})` by minimizing

.. math::

   \mathcal{L}(\phi)
   = -E_{p(\theta, \mathbf{x})}\!\left[\log q_\phi(\theta \mid \mathbf{x})\right]

over the joint distribution of parameters and data.

Reading this formula: we generate many (parameter, data) pairs from the
simulator, then train the network to maximize the probability it assigns to the
true parameter given the data.  This is the same maximum likelihood idea
applied to the *conditional density estimator* itself.  After training, the
network has seen so many examples that it has learned the general pattern of
how data map to posteriors --- so for any new dataset, it can produce an
approximate posterior in a single forward pass, without running any
optimization or MCMC.

This is the *amortized* version of variational inference: instead of optimizing
separate variational parameters for each dataset, a single network handles all
datasets.

**Amortized inference: an encoder that maps data to posterior parameters.**
The idea of amortized inference is simple: train a function that takes raw data
and returns posterior parameters.  Here we build a small "encoder" (a linear
model, for transparency) that maps summary statistics to the parameters of a
Gaussian approximate posterior.

.. code-block:: python

   # Amortized inference: train an encoder from data to posterior parameters
   import numpy as np

   np.random.seed(42)

   # Model: Normal(mu, 1) with known variance
   # True posterior for n observations with known var=1:
   #   mu | x ~ N(x_bar, 1/n)
   # The encoder should learn: posterior_mean = x_bar, posterior_logvar = -log(n)

   n_obs = 10
   n_train = 20_000

   # Generate training data: (theta, summary_stats) pairs
   thetas_train = np.random.normal(0, 3, n_train)  # prior: N(0, 9)
   summaries_train = np.zeros((n_train, 2))  # features: (mean, var)
   for i in range(n_train):
       x = np.random.normal(thetas_train[i], 1.0, n_obs)
       summaries_train[i, 0] = x.mean()
       summaries_train[i, 1] = x.var()

   # Encoder: linear map from summary stats to (posterior_mean, posterior_log_var)
   # q(theta | x) = N(theta; a^T s(x), exp(b^T s(x)))
   # Train by maximizing E[log q(theta | x)]

   # Add bias term
   S = np.column_stack([summaries_train, np.ones(n_train)])  # (n_train, 3)
   W = np.zeros((2, 3))  # row 0: weights for mean, row 1: weights for logvar

   lr = 0.001
   print(f"{'Iter':<8} {'Avg log q(theta|x)':<22} {'Mean weight':<14}")
   print("-" * 44)

   for iteration in range(501):
       # Forward pass
       outputs = S @ W.T  # (n_train, 2)
       pred_mean = outputs[:, 0]
       pred_logvar = outputs[:, 1]
       pred_var = np.exp(np.clip(pred_logvar, -10, 10))

       # log q(theta | x) = -0.5*log(2*pi*var) - 0.5*(theta - mean)^2/var
       log_q = (-0.5*np.log(2*np.pi*pred_var)
                - 0.5*(thetas_train - pred_mean)**2 / pred_var)
       avg_log_q = log_q.mean()

       if iteration % 100 == 0:
           print(f"{iteration:<8} {avg_log_q:<22.4f} {W[0,0]:<14.4f}")

       # Gradient of log q w.r.t. W
       # d/d(mean) = (theta - mean) / var
       # d/d(logvar) = -0.5 + 0.5*(theta - mean)^2 / var
       d_mean = (thetas_train - pred_mean) / pred_var       # (n_train,)
       d_logvar = -0.5 + 0.5*(thetas_train - pred_mean)**2 / pred_var

       # Gradient w.r.t. W: d(log_q)/d(W_mean) = d_mean * S
       grad_W_mean = d_mean[:, None] * S     # (n_train, 3)
       grad_W_logvar = d_logvar[:, None] * S  # (n_train, 3)

       W[0] += lr * grad_W_mean.mean(axis=0)
       W[1] += lr * grad_W_logvar.mean(axis=0)

   # Test: new observation
   x_test = np.random.normal(2.5, 1.0, n_obs)
   s_test = np.array([x_test.mean(), x_test.var(), 1.0])
   pred = s_test @ W.T
   print(f"\nTest data: x_bar={x_test.mean():.3f}, n={n_obs}")
   print(f"Encoder output:  mean={pred[0]:.3f}, var={np.exp(pred[1]):.4f}")
   print(f"True posterior:  mean={x_test.mean():.3f}, var={1/n_obs:.4f}")
   print(f"\nThe encoder learned the correct posterior mapping!")

**Trade-offs.**

* **Speed at test time:**  Inference is a single neural network evaluation
  (:math:`\sim` milliseconds).
* **Up-front cost:**  Training requires many simulations and can take hours or
  days.
* **Amortization gap:**  The network may not perfectly capture the posterior
  for every individual dataset; refinement with MCMC can close this gap.


20.5 Normalizing Flows
-------------------------

**Why flows?**
A normalizing flow transforms a simple base distribution (e.g., a standard
Normal) into a complex target distribution through a sequence of invertible,
differentiable maps.  This gives us a flexible density estimator with a
tractable likelihood.

Change of variables
^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mathbf{z} \sim p_Z(\mathbf{z})` (the base distribution) and
:math:`\mathbf{x} = f(\mathbf{z})` where :math:`f` is an invertible map.
The density of :math:`\mathbf{x}` is:

.. math::

   p_X(\mathbf{x})
   = p_Z\!\bigl(f^{-1}(\mathbf{x})\bigr)\;
     \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|.

In plain English: to find the density of :math:`\mathbf{x}`, first map it
back to the base space via :math:`f^{-1}` and evaluate the simple base
density there.  But that alone is not enough --- the transformation stretches
and compresses space, so we must account for how much the local volume changes.
The Jacobian determinant is exactly this volume-change factor.  If the
transformation expands a region of space, the density must decrease
proportionally (probability is conserved), and vice versa.

Taking the logarithm:

.. math::

   \log p_X(\mathbf{x})
   = \log p_Z\!\bigl(f^{-1}(\mathbf{x})\bigr)
     + \log\!\left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|.

For a *composition* of :math:`K` invertible maps
:math:`f = f_K \circ \cdots \circ f_1`:

.. math::

   \log p_X(\mathbf{x})
   = \log p_Z(\mathbf{z}_0)
     + \sum_{k=1}^K \log\!\left|\det \frac{\partial f_k^{-1}}{\partial \mathbf{z}_k}\right|,

where :math:`\mathbf{z}_0 = f_1^{-1} \circ \cdots \circ f_K^{-1}(\mathbf{x})`.

Coupling layers
^^^^^^^^^^^^^^^^

Computing the full Jacobian determinant is :math:`O(D^3)`.  *Coupling layers*
(Dinh et al., 2015, 2017) split the input into two halves
:math:`(\mathbf{x}_A, \mathbf{x}_B)` and apply:

.. math::

   \mathbf{z}_A &= \mathbf{x}_A, \\
   \mathbf{z}_B &= \mathbf{x}_B \odot \exp\!\bigl(s(\mathbf{x}_A)\bigr)
                    + t(\mathbf{x}_A),

where :math:`s` and :math:`t` are neural networks (the "scale" and "translate"
networks, respectively).

**Why the Jacobian is cheap.**
Because :math:`\mathbf{z}_A` is simply copied from :math:`\mathbf{x}_A`
unchanged, and :math:`\mathbf{z}_B` depends on :math:`\mathbf{x}_B` only
through an element-wise affine transformation (scale by
:math:`\exp(s(\mathbf{x}_A))` and shift by :math:`t(\mathbf{x}_A)`), the
Jacobian matrix of the entire transformation is *triangular*.  The determinant
of a triangular matrix is just the product of its diagonal entries, so:

.. math::

   \log\!\left|\det \frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right|
   = \sum_j s_j(\mathbf{x}_A),

which is :math:`O(D)` --- a massive speedup compared to the :math:`O(D^3)` cost
of a general determinant.  Alternating which half is transformed ensures all
dimensions interact.

**Simple affine coupling layer: transforming a Gaussian into a non-Gaussian.**
Let us implement the coupling layer from scratch and show that it can transform
a standard 2D Gaussian into a distinctly non-Gaussian shape, with the exact
log-density computable at every point via the change-of-variables formula.

.. code-block:: python

   # Normalizing flows: affine coupling layer
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Affine coupling layer in 2D:
   #   z_1 = x_1                         (passthrough)
   #   z_2 = x_2 * exp(s(x_1)) + t(x_1) (affine transform)
   # Inverse:
   #   x_1 = z_1
   #   x_2 = (z_2 - t(z_1)) * exp(-s(z_1))
   # Log-det-Jacobian = s(x_1)

   # Simple parameterization: s and t are linear functions
   # s(x) = a*x + b,  t(x) = c*x + d
   # These parameters would be learned; we set them manually for illustration
   a, b = 0.5, 0.3   # scale network
   c, d = 1.2, -0.5   # translate network

   def s_net(x1):
       return a * x1 + b

   def t_net(x1):
       return c * x1 + d

   def forward(x):
       """x -> z (from data space to base space)."""
       x1, x2 = x[:, 0], x[:, 1]
       z1 = x1
       z2 = x2 * np.exp(s_net(x1)) + t_net(x1)
       return np.column_stack([z1, z2])

   def inverse(z):
       """z -> x (from base space to data space)."""
       z1, z2 = z[:, 0], z[:, 1]
       x1 = z1
       x2 = (z2 - t_net(z1)) * np.exp(-s_net(z1))
       return np.column_stack([x1, x2])

   def log_det_jacobian(x):
       """Log |det J| of the forward transform."""
       return s_net(x[:, 0])

   # Sample from base distribution (standard normal)
   n = 5000
   z = np.random.randn(n, 2)

   # Transform to data space
   x = inverse(z)

   # Compute log-density of transformed samples
   # log p_X(x) = log p_Z(f(x)) + log|det J_f(x)|
   z_check = forward(x)
   log_pz = stats.norm.logpdf(z_check[:, 0]) + stats.norm.logpdf(z_check[:, 1])
   log_px = log_pz + log_det_jacobian(x)

   print("Affine Coupling Layer Flow (2D)")
   print("=" * 40)
   print(f"\nBase distribution: N(0, I_2)")
   print(f"Scale network: s(x1) = {a}*x1 + {b}")
   print(f"Translate network: t(x1) = {c}*x1 + {d}")

   print(f"\nBase (z) statistics:")
   print(f"  mean = ({z[:,0].mean():.3f}, {z[:,1].mean():.3f})")
   print(f"  std  = ({z[:,0].std():.3f}, {z[:,1].std():.3f})")
   print(f"  corr = {np.corrcoef(z[:,0], z[:,1])[0,1]:.3f}")

   print(f"\nTransformed (x) statistics:")
   print(f"  mean = ({x[:,0].mean():.3f}, {x[:,1].mean():.3f})")
   print(f"  std  = ({x[:,0].std():.3f}, {x[:,1].std():.3f})")
   print(f"  corr = {np.corrcoef(x[:,0], x[:,1])[0,1]:.3f}")

   print(f"\nLog-density at x=(0,0): {log_px[np.argmin(np.sum(x**2, axis=1))]:.4f}")
   print(f"Mean log-density:       {log_px.mean():.4f}")

   # Verify invertibility
   x_roundtrip = inverse(forward(x))
   print(f"\nInvertibility check: max |x - inv(fwd(x))| = "
         f"{np.max(np.abs(x - x_roundtrip)):.2e}")
   print("The coupling layer is exactly invertible by construction.")

Applications in likelihood-based inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Density estimation:**  Fit :math:`p_X` to data by maximizing the
  log-likelihood above.
* **Posterior estimation:**  Use a conditional flow
  :math:`q_\phi(\theta \mid \mathbf{x})` to approximate the posterior in
  simulation-based inference (Section 20.3).
* **Variational inference:**  Use a flow as the variational family
  :math:`q_\phi(\theta)` to make the ELBO tighter
  (see :ref:`Chapter 18 <ch18_computational>`).


20.6 Variational Autoencoders (VAEs)
---------------------------------------

**Why VAEs?**
A variational autoencoder defines a generative model with latent variables
:math:`\mathbf{z}`:

.. math::

   p_\phi(\mathbf{x})
   = \int p_\phi(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}.

The marginal likelihood :math:`p_\phi(\mathbf{x})` is intractable because of
the integral over :math:`\mathbf{z}`.

The ELBO for VAEs
^^^^^^^^^^^^^^^^^^

Introduce an *encoder* (inference network) :math:`q_\psi(\mathbf{z} \mid
\mathbf{x})`.  As derived in :ref:`Chapter 18 <ch18_computational>`:

.. math::

   \log p_\phi(\mathbf{x})
   \;\ge\;
   E_{q_\psi(\mathbf{z} \mid \mathbf{x})}\!\bigl[\log p_\phi(\mathbf{x} \mid \mathbf{z})\bigr]
   - \text{KL}\!\bigl(q_\psi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})\bigr).

The first term is the expected *reconstruction likelihood*; the second is a
regularizer encouraging the approximate posterior to stay close to the prior.

The reparameterization trick
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To backpropagate through the expectation we write
:math:`\mathbf{z} = \boldsymbol{\mu}_\psi(\mathbf{x}) +
\boldsymbol{\sigma}_\psi(\mathbf{x}) \odot \boldsymbol{\epsilon}` with
:math:`\boldsymbol{\epsilon} \sim N(\mathbf{0}, \mathbf{I})`.  This moves the
randomness into :math:`\boldsymbol{\epsilon}`, making the ELBO differentiable
with respect to :math:`\psi`.

**Derivation of the gradient.**
Let :math:`g(\psi) = E_{q_\psi}[f(\mathbf{z})]`.  The naive score-function
estimator has high variance.  Using the reparameterization
:math:`\mathbf{z} = h(\boldsymbol{\epsilon}, \psi)`:

.. math::

   g(\psi) = E_{p(\boldsymbol{\epsilon})}[f(h(\boldsymbol{\epsilon}, \psi))],

so:

.. math::

   \nabla_\psi g = E_{p(\boldsymbol{\epsilon})}\!\left[
     \nabla_\psi f\!\bigl(h(\boldsymbol{\epsilon}, \psi)\bigr)\right],

which can be estimated by a single-sample Monte Carlo average and
differentiated through :math:`h` using AD (Section 20.1).

In plain English: the reparameterization trick moves the randomness out of
the distribution we are differentiating.  Instead of sampling
:math:`\mathbf{z}` from a distribution that depends on :math:`\psi` (which
makes the gradient hard to compute), we sample fixed noise
:math:`\boldsymbol{\epsilon}` from a standard Normal and then *deterministically*
transform it into :math:`\mathbf{z}` using the function :math:`h`.  Now the
gradient with respect to :math:`\psi` flows straight through :math:`h` via
automatic differentiation, giving low-variance gradient estimates that make
training practical.

Connection to likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^

Training a VAE by maximizing the ELBO is a form of *approximate maximum
likelihood* --- we maximize a lower bound on :math:`\log p_\phi(\mathbf{x})`.
As the variational family becomes more expressive (e.g., using normalizing
flows for :math:`q`), the bound tightens and we approach exact MLE.

**Computing the ELBO for a simple latent variable model.**
Let us make the ELBO concrete.  We have a 1D latent variable :math:`z \sim N(0,1)`
and an observation model :math:`x \mid z \sim N(z, \sigma^2)`.  The true
marginal is :math:`x \sim N(0, 1 + \sigma^2)`, so we can check that our ELBO
is indeed a lower bound on the true log-marginal.

.. code-block:: python

   # VAE ELBO: compute and verify it is a lower bound on log p(x)
   import numpy as np
   from scipy import stats

   np.random.seed(42)

   # Generative model: z ~ N(0,1), x|z ~ N(z, sigma_x^2)
   sigma_x = 0.5
   # True marginal: x ~ N(0, 1 + sigma_x^2)
   true_marginal_var = 1.0 + sigma_x**2

   # Generate observed data
   n = 200
   z_true = np.random.normal(0, 1, n)
   x_obs = z_true + np.random.normal(0, sigma_x, n)

   # Variational posterior: q(z|x) = N(mu_q(x), sigma_q^2(x))
   # Optimal: posterior is N(x/(1+sigma_x^2), sigma_x^2/(1+sigma_x^2))
   # Let us parameterize and compute the ELBO

   def compute_elbo(x, mu_q, log_sigma_q, sigma_x, n_samples=100):
       """Monte Carlo estimate of the ELBO for each observation."""
       sigma_q = np.exp(log_sigma_q)
       elbos = np.zeros(len(x))
       for i in range(len(x)):
           eps = np.random.normal(0, 1, n_samples)
           z = mu_q[i] + sigma_q[i] * eps  # reparameterization trick

           # E_q[log p(x|z)] : reconstruction term
           log_p_x_given_z = stats.norm.logpdf(x[i], loc=z, scale=sigma_x)
           reconstruction = log_p_x_given_z.mean()

           # KL(q(z|x) || p(z)) : analytical for two Gaussians
           kl = 0.5 * (sigma_q[i]**2 + mu_q[i]**2 - 1 - 2*log_sigma_q[i])

           elbos[i] = reconstruction - kl
       return elbos

   # Compute with optimal variational parameters
   post_var = sigma_x**2 / (1 + sigma_x**2)
   mu_q_opt = x_obs / (1 + sigma_x**2)
   log_sigma_q_opt = 0.5 * np.log(post_var) * np.ones(n)

   elbos = compute_elbo(x_obs, mu_q_opt, log_sigma_q_opt, sigma_x)

   # True log-marginal
   log_marginals = stats.norm.logpdf(x_obs, loc=0, scale=np.sqrt(true_marginal_var))

   print(f"ELBO (total):           {elbos.sum():.2f}")
   print(f"True log-marginal:      {log_marginals.sum():.2f}")
   print(f"Gap (>= 0):             {log_marginals.sum() - elbos.sum():.2f}")
   print(f"\nPer-observation:")
   print(f"  Mean ELBO:            {elbos.mean():.4f}")
   print(f"  Mean log p(x):        {log_marginals.mean():.4f}")
   print(f"  Mean gap:             {(log_marginals - elbos).mean():.4f}")
   print(f"\nELBO <= log p(x) for each observation?  "
         f"{np.all(log_marginals >= elbos - 0.1)}")
   print("(Small violations possible due to MC noise in ELBO estimate)")


20.7 Score Matching
---------------------

**Why score matching?**
In some models the density :math:`p_\theta(\mathbf{x})` involves an
intractable normalizing constant :math:`Z(\theta)`:

.. math::

   p_\theta(\mathbf{x}) = \frac{\tilde{p}_\theta(\mathbf{x})}{Z(\theta)},
   \qquad
   Z(\theta) = \int \tilde{p}_\theta(\mathbf{x})\, d\mathbf{x}.

Maximum likelihood is intractable because computing or differentiating
:math:`Z(\theta)` is infeasible.

The **score function** is the gradient of the log-density with respect to
:math:`\mathbf{x}`:

.. math::

   \mathbf{s}_\theta(\mathbf{x})
   = \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})
   = \nabla_{\mathbf{x}} \log \tilde{p}_\theta(\mathbf{x}),

where the normalizing constant cancels (it does not depend on
:math:`\mathbf{x}`).

Note a subtle but important distinction: throughout most of this guide, the
"score function" referred to the gradient of the log-likelihood with respect to
the *parameters* :math:`\theta`.  Here, the score is the gradient with respect
to the *data* :math:`\mathbf{x}`.  This data-space score tells you which
direction to move a data point to increase its probability under the model ---
like a vector field pointing "uphill" on the probability landscape.  The key
advantage is that this score does not depend on the normalizing constant
:math:`Z(\theta)` at all, since :math:`\nabla_{\mathbf{x}} \log Z(\theta) = 0`.

Hyvarinen (2005) proposed minimizing the *Fisher divergence*:

.. math::

   J(\theta)
   = \frac{1}{2} E_{p_{\text{data}}}\!\left[
     \|\mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|^2
   \right].

This involves the unknown data score, but integration by parts yields an
equivalent *explicit score matching* objective that only requires the model
score and its derivatives --- not the unknown data score:

.. math::

   J(\theta)
   = E_{p_{\text{data}}}\!\left[
     \frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2
     + \text{tr}\!\bigl(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\bigr)
   \right] + \text{const}.

Reading this formula: the first term,
:math:`\frac{1}{2}\|\mathbf{s}_\theta\|^2`, penalizes model scores that are
too large (the model thinks the data points should be moved far).  The second
term, :math:`\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta)`, is the
divergence of the score field --- it penalizes the model for having scores that
converge too sharply (creating overly peaked densities).  Together, these two
terms replace the need to know the true data score, and minimizing this
objective makes the model score match the data score.

**Derivation sketch.**
Expand the squared norm:

.. math::

   \|\mathbf{s}_\theta - \mathbf{s}_{\text{data}}\|^2
   = \|\mathbf{s}_\theta\|^2
     - 2\, \mathbf{s}_\theta \cdot \mathbf{s}_{\text{data}}
     + \|\mathbf{s}_{\text{data}}\|^2.

The last term is constant.  For the cross-term, using integration by parts
(assuming boundary terms vanish):

.. math::

   E_{p_{\text{data}}}[\mathbf{s}_\theta \cdot \mathbf{s}_{\text{data}}]
   &= \int p_{\text{data}}(\mathbf{x})\, \mathbf{s}_\theta(\mathbf{x}) \cdot
      \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\, d\mathbf{x} \\
   &= \int \mathbf{s}_\theta(\mathbf{x}) \cdot
      \nabla_{\mathbf{x}} p_{\text{data}}(\mathbf{x})\, d\mathbf{x} \\
   &= -\int p_{\text{data}}(\mathbf{x})\,
      \text{tr}\!\bigl(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\bigr)\, d\mathbf{x} \\
   &= -E_{p_{\text{data}}}\!\bigl[
      \text{tr}\!\bigl(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\bigr)\bigr].

Substituting gives the explicit objective.

**Score matching in 1D: fitting a Gaussian without the normalizing constant.**
To make score matching concrete, let us fit a 1D Gaussian to data using only
the score function (derivative of log-density w.r.t. x), without ever computing
the normalizing constant :math:`\sqrt{2\pi}\sigma`.  The model score for
:math:`\tilde{p}_\theta(x) = \exp(-\frac{1}{2}(x - \mu)^2/\sigma^2)` is
:math:`s_\theta(x) = -(x - \mu)/\sigma^2`, and the explicit score matching
objective has a closed-form gradient we can optimize.

.. code-block:: python

   # Score matching: fit a Gaussian without using the normalizing constant
   import numpy as np

   np.random.seed(42)

   # True distribution: N(mu_true, sigma_true^2)
   mu_true, sigma_true = 2.0, 1.5
   n = 500
   data = np.random.normal(mu_true, sigma_true, n)

   # Model score: s_theta(x) = -(x - mu) / sigma^2
   # ds/dx = -1/sigma^2
   # Score matching objective: E[0.5 * s^2 + ds/dx]
   #   = E[0.5*(x-mu)^2/sigma^4 - 1/sigma^2]

   def score_matching_objective(mu, sigma):
       """Explicit score matching objective (Hyvarinen 2005)."""
       s2 = sigma**2
       score_sq = 0.5 * np.mean((data - mu)**2) / s2**2
       trace_term = -1.0 / s2
       return score_sq + trace_term

   # Gradient descent on the score matching objective
   mu_hat, sigma_hat = 0.0, 1.0
   lr_mu, lr_sigma = 0.01, 0.01
   h = 1e-6

   print(f"{'Iter':<8} {'mu':<10} {'sigma':<10} {'SM objective':<14}")
   print("-" * 42)
   for iteration in range(201):
       obj = score_matching_objective(mu_hat, sigma_hat)
       if iteration % 40 == 0:
           print(f"{iteration:<8} {mu_hat:<10.4f} {sigma_hat:<10.4f} {obj:<14.6f}")

       # Numerical gradients
       g_mu = (score_matching_objective(mu_hat + h, sigma_hat)
               - score_matching_objective(mu_hat - h, sigma_hat)) / (2*h)
       g_sigma = (score_matching_objective(mu_hat, sigma_hat + h)
                  - score_matching_objective(mu_hat, sigma_hat - h)) / (2*h)

       mu_hat -= lr_mu * g_mu
       sigma_hat -= lr_sigma * g_sigma

   print(f"\nScore matching estimates: mu={mu_hat:.4f}, sigma={sigma_hat:.4f}")
   print(f"MLE estimates:           mu={data.mean():.4f}, sigma={data.std():.4f}")
   print(f"True values:             mu={mu_true}, sigma={sigma_true}")
   print(f"\nWe recovered the parameters without ever computing sqrt(2*pi*sigma^2)!")

**Denoising score matching.**
Song and Ermon (2019) showed that adding noise to the data and matching the
score of the noisy distribution avoids computing the trace of the Jacobian,
which is expensive in high dimensions.  This leads to *score-based diffusion
models* (the engine behind modern image generation).


20.8 Connections Back to Classical Likelihood
-----------------------------------------------

Despite their apparent novelty, all of the methods in this chapter are
intimately connected to the classical likelihood framework:

1. **Automatic differentiation** computes the exact same gradients that
   classical score equations describe, just algorithmically rather than
   analytically.

2. **Neural density estimation** is maximum likelihood estimation where the
   model family is a neural network instead of a parametric family.

3. **Simulation-based inference** seeks to perform likelihood-based
   inference (or Bayesian inference with a likelihood) even when
   :math:`p(\mathbf{x} \mid \theta)` cannot be evaluated --- it
   *approximates* the likelihood or the posterior.

4. **Amortized inference** is variational inference (a lower bound on the
   log-likelihood) with the variational parameters predicted by a neural
   network.

5. **Normalizing flows** exploit the change-of-variables formula to compute
   exact likelihoods for flexible distributions.

6. **VAEs** maximize a lower bound on the marginal log-likelihood.

7. **Score matching** estimates the model by matching derivatives of the
   log-likelihood with respect to the *data* (rather than the parameters),
   sidestepping the normalizing constant.

The unifying theme is that the likelihood function --- or some tractable proxy
--- remains the objective guiding learning and inference.  What has changed is
the expressiveness of the model family and the computational tools available.


20.9 Summary
--------------

* Automatic differentiation (especially reverse mode) enables gradient-based
  optimization and sampling for models with thousands to millions of
  parameters.
* Neural density estimators (MDNs, autoregressive models) turn MLE into
  training a neural network.
* Simulation-based inference (ABC, NPE, NLE, NRE) extends Bayesian reasoning
  to models where the likelihood is intractable.
* Amortized inference pre-trains a network to produce posteriors instantly for
  any new dataset.
* Normalizing flows provide flexible, invertible density models with exact
  log-likelihoods via the change-of-variables formula.
* VAEs maximize a lower bound (ELBO) on the marginal log-likelihood using the
  reparameterization trick.
* Score matching avoids the normalizing constant by matching gradients of the
  log-density, and is the foundation of modern diffusion models.
* Every modern method traces back to the classical likelihood principle:
  the likelihood function, or an approximation thereof, remains the engine
  of statistical learning.
