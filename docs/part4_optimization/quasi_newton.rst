.. _ch14_quasi_newton:

========================================
Chapter 14: Quasi-Newton Methods
========================================

Newton's method (:ref:`ch13_newton`) converges quadratically but requires
computing and inverting the Hessian at every iteration --- a cost of
:math:`O(np^2 + p^3)` per step. For problems with hundreds or thousands of
parameters, this is prohibitive. Quasi-Newton methods approximate the Hessian
(or its inverse) using only gradient information, reducing the per-iteration
cost to :math:`O(p^2)` while retaining *superlinear* convergence. This chapter
derives the key ideas: the secant condition, the celebrated BFGS update and its
limited-memory variant L-BFGS, the SR1 update, and trust-region methods.

.. admonition:: Why Quasi-Newton?

   Think of Newton's method as hiring a full survey team to map the curvature
   of the landscape at every step --- accurate but expensive.  Quasi-Newton
   methods instead *learn* the curvature as they go, updating their map with
   each new gradient measurement.  The result is nearly as good as Newton in
   terms of convergence, but at a fraction of the cost.  If gradient descent is
   "cheap and slow" and Newton is "expensive and fast," quasi-Newton methods
   are the sweet spot: "affordable and fast."


14.1 Motivation
================

Recall the Newton update for minimizing :math:`f`:

.. math::

   \boldsymbol{\theta}_{k+1}
   = \boldsymbol{\theta}_k
   - \mathbf{H}_k^{-1}\nabla f_k.

Three costs dominate:

.. list-table::
   :header-rows: 1
   :widths: 30 20 40

   * - Operation
     - Cost
     - Comment
   * - Gradient :math:`\nabla f_k`
     - :math:`O(np)`
     - Unavoidable
   * - Hessian :math:`\mathbf{H}_k`
     - :math:`O(np^2)`
     - Often the bottleneck
   * - Solve :math:`\mathbf{H}_k \mathbf{d} = -\nabla f_k`
     - :math:`O(p^3)`
     - Or :math:`O(p^2)` with Cholesky

The idea of a quasi-Newton method is to maintain an approximation
:math:`\mathbf{B}_k \approx \mathbf{H}_k` (or
:math:`\mathbf{D}_k \approx \mathbf{H}_k^{-1}`) that is updated cheaply ---
in :math:`O(p^2)` --- at each iteration using only the gradient difference.

The key question is: what information can we extract from the gradients alone to
build a good Hessian approximation?  The answer is the *secant condition*.


14.2 The Secant Condition
==========================

Derivation
----------

For a smooth function, the mean-value theorem gives

.. math::

   \nabla f(\boldsymbol{\theta}_{k+1})
   - \nabla f(\boldsymbol{\theta}_k)
   \;\approx\;
   \mathbf{H}(\boldsymbol{\theta}_k)\,
   (\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k).

Here is the core insight: even though we do not know the Hessian, we *do* know
both sides of this equation --- we have the gradient at two consecutive points
and the step between them.  This gives us a constraint on what our Hessian
approximation should look like.

Define the notation

.. math::

   \mathbf{s}_k &= \boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k
     \qquad\text{(step)}, \\
   \mathbf{y}_k &= \nabla f_{k+1} - \nabla f_k
     \qquad\text{(gradient change)}.

Then the **secant condition** (also called the quasi-Newton equation) is

.. math::
   :label: secant

   \mathbf{B}_{k+1}\,\mathbf{s}_k = \mathbf{y}_k,

where :math:`\mathbf{B}_{k+1}` is our Hessian approximation. Equivalently, in
terms of the inverse Hessian approximation
:math:`\mathbf{D}_{k+1} = \mathbf{B}_{k+1}^{-1}`:

.. math::

   \mathbf{D}_{k+1}\,\mathbf{y}_k = \mathbf{s}_k.

The secant condition provides :math:`p` equations (one per component) but
:math:`\mathbf{B}_{k+1}` has :math:`p(p+1)/2` unknowns (exploiting symmetry).
The condition alone does not determine :math:`\mathbf{B}_{k+1}` uniquely; the
various quasi-Newton methods differ in *how they resolve this
under-determination*.

Verifying the Secant Condition
--------------------------------

The secant condition is the single most important equation in this chapter.
Let's make it tangible: take a concrete quadratic function where we *know*
the true Hessian, run one step of an optimizer, and then verify that
:math:`\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k` holds.

.. code-block:: python

   # Secant condition verification on a quadratic
   import numpy as np

   np.random.seed(42)
   p = 4

   # Quadratic: f(x) = 0.5 x^T A x - b^T x,  grad f = Ax - b,  Hessian = A
   A = np.array([[6, 2, 1, 0],
                 [2, 5, 1, 1],
                 [1, 1, 4, 1],
                 [0, 1, 1, 3]], dtype=float)
   b = np.array([1, 2, 3, 4], dtype=float)

   grad_f = lambda x: A @ x - b

   # Take one step from x0 to x1 (e.g., a gradient step)
   x0 = np.zeros(p)
   g0 = grad_f(x0)
   alpha = 0.05
   x1 = x0 - alpha * g0            # gradient step
   g1 = grad_f(x1)

   s_k = x1 - x0                   # step
   y_k = g1 - g0                   # gradient change

   # The true Hessian satisfies the secant condition exactly for a quadratic
   print("Secant condition: B_{k+1} s_k should equal y_k")
   print(f"  A @ s_k = {A @ s_k}")
   print(f"  y_k     = {y_k}")
   print(f"  Match?    {np.allclose(A @ s_k, y_k)}")

   # Now build a BFGS approximation starting from B0 = I
   B0 = np.eye(p)
   # BFGS Hessian update:
   # B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)
   Bs = B0 @ s_k
   B1 = B0 - np.outer(Bs, Bs) / (s_k @ Bs) + np.outer(y_k, y_k) / (y_k @ s_k)

   print(f"\nAfter BFGS update from B0=I:")
   print(f"  B1 @ s_k = {np.round(B1 @ s_k, 10)}")
   print(f"  y_k      = {np.round(y_k, 10)}")
   print(f"  Secant satisfied? {np.allclose(B1 @ s_k, y_k)}")
   print(f"\n  True Hessian A:\n{A}")
   print(f"  BFGS approx B1:\n{np.round(B1, 4)}")
   print(f"  ||B1 - A||_F = {np.linalg.norm(B1 - A, 'fro'):.4f}")

So the BFGS approximation :math:`\mathbf{B}_1` satisfies the secant condition
*exactly*, but it may still be far from the true Hessian after just one update.
It gets closer with each step.

Curvature Condition
---------------------

For :math:`\mathbf{B}_{k+1}` to be positive definite (so that the quasi-Newton
direction is a descent direction), we need

.. math::
   :label: curv_cond

   \mathbf{s}_k^{\!\top}\mathbf{y}_k > 0.

This is called the **curvature condition**. It is guaranteed to hold when the
step :math:`\alpha_k` satisfies the Wolfe conditions (Section 12.2 of
:ref:`ch12_gradient`), which is one reason the Wolfe conditions are important
in practice.


14.3 The BFGS Update
======================

The BFGS method (Broyden, Fletcher, Goldfarb, Shanno; 1970) is the most widely
used quasi-Newton method. It maintains an approximation to the *inverse*
Hessian, :math:`\mathbf{D}_k`, and updates it to satisfy the secant condition
while staying as close as possible to the previous approximation.

.. admonition:: Intuition

   BFGS asks: "Given everything I learned about the curvature so far
   (stored in :math:`\mathbf{D}_k`), what is the *smallest* change I can make
   to my approximation so that it is consistent with the new gradient
   information I just observed?"  This minimum-change philosophy keeps the
   approximation stable while steadily improving it.

Derivation
----------

We seek :math:`\mathbf{D}_{k+1}` that solves

.. math::

   \min_{\mathbf{D}} \;\|\mathbf{D} - \mathbf{D}_k\|_W
   \qquad\text{subject to}\qquad
   \mathbf{D}\,\mathbf{y}_k = \mathbf{s}_k,
   \quad \mathbf{D} = \mathbf{D}^{\!\top},

Reading this optimization problem: we want the new inverse-Hessian approximation
:math:`\mathbf{D}_{k+1}` to be as close as possible to the old one (in the
"minimum-change" sense of the weighted Frobenius norm), while satisfying the
secant condition :math:`\mathbf{D}\mathbf{y}_k = \mathbf{s}_k` and remaining
symmetric. The norm :math:`\|\cdot\|_W` uses a weight matrix
:math:`W` chosen as a certain average Hessian. The solution (which can be
verified by Lagrange multipliers on the matrix optimization) is the
**BFGS inverse-Hessian update**:

.. math::
   :label: bfgs

   \mathbf{D}_{k+1}
   = \bigl(\mathbf{I} - \rho_k\,\mathbf{s}_k\,\mathbf{y}_k^{\!\top}\bigr)\,
     \mathbf{D}_k\,
     \bigl(\mathbf{I} - \rho_k\,\mathbf{y}_k\,\mathbf{s}_k^{\!\top}\bigr)
   + \rho_k\,\mathbf{s}_k\,\mathbf{s}_k^{\!\top},

where

.. math::

   \rho_k = \frac{1}{\mathbf{y}_k^{\!\top}\mathbf{s}_k}.

The scalar :math:`\rho_k` is one over the inner product of the gradient change
and the step. It normalizes the update so that the secant condition is
satisfied exactly.

Reading the BFGS formula: the two outer factors
:math:`(\mathbf{I} - \rho_k\,\mathbf{s}_k\,\mathbf{y}_k^{\!\top})` "project
away" the component of :math:`\mathbf{D}_k` that is inconsistent with the new
secant information, and the final
:math:`\rho_k\,\mathbf{s}_k\,\mathbf{s}_k^{\!\top}` term adds back the
correct curvature along the step direction. The result automatically satisfies
:math:`\mathbf{D}_{k+1}\mathbf{y}_k = \mathbf{s}_k` (the secant condition).

This is a **rank-2 update**: :math:`\mathbf{D}_{k+1}` differs from
:math:`\mathbf{D}_k` by the addition of two rank-1 matrices, so the update
costs :math:`O(p^2)`.

Step-by-Step BFGS Update
---------------------------

Let's watch the inverse-Hessian approximation :math:`\mathbf{D}_k` evolve
over several BFGS steps on a simple quadratic, showing how each update
brings :math:`\mathbf{D}_k` closer to the true :math:`\mathbf{H}^{-1}`.

.. code-block:: python

   # BFGS update step-by-step: watching D_k evolve toward H^{-1}
   import numpy as np

   np.random.seed(42)

   # Quadratic: f(x) = 0.5 x^T A x - b^T x
   A = np.array([[4.0, 1.0],
                 [1.0, 3.0]])
   b = np.array([1.0, 2.0])
   H_inv_true = np.linalg.inv(A)

   grad_f = lambda x: A @ x - b

   D_k = np.eye(2)       # start with identity
   x = np.array([5.0, 5.0])

   print(f"True H^{{-1}}:\n{np.round(H_inv_true, 4)}\n")
   print(f"{'Step':>4s}  {'||D_k - H^-1||':>15s}  D_k")
   print("-" * 60)

   for k in range(6):
       err = np.linalg.norm(D_k - H_inv_true, 'fro')
       D_flat = D_k.flatten()
       print(f"{k:4d}  {err:15.6e}  [{D_flat[0]:7.4f} {D_flat[1]:7.4f}; "
             f"{D_flat[2]:7.4f} {D_flat[3]:7.4f}]")

       g = grad_f(x)
       d = -D_k @ g                               # search direction
       # Exact line search for quadratic: alpha = -(g^T d)/(d^T A d)
       alpha = -(g @ d) / (d @ A @ d)
       x_new = x + alpha * d

       s = x_new - x
       y = grad_f(x_new) - g
       rho = 1.0 / (y @ s)

       I = np.eye(2)
       V = I - rho * np.outer(s, y)
       D_k = V @ D_k @ V.T + rho * np.outer(s, s)   # BFGS update
       x = x_new

   print(f"\nFinal D_k:\n{np.round(D_k, 6)}")
   print(f"True H^{{-1}}:\n{np.round(H_inv_true, 6)}")

For a 2-dimensional quadratic, BFGS recovers the exact inverse Hessian in
at most 2 steps --- and then it takes one final step to the exact minimum.

Let's also see the one-step BFGS update in detail, matching every line of
the formula :eq:`bfgs`:

.. code-block:: python

   # One step of the BFGS inverse-Hessian update
   import numpy as np

   np.random.seed(42)
   p = 3
   D_k = np.eye(p)                           # initial: identity
   s_k = np.array([0.1, -0.2, 0.05])         # step taken
   y_k = np.array([0.3, -0.1, 0.15])         # gradient change

   rho_k = 1.0 / (y_k @ s_k)
   I = np.eye(p)
   V = I - rho_k * np.outer(s_k, y_k)
   D_new = V @ D_k @ V.T + rho_k * np.outer(s_k, s_k)

   print(f"rho_k = {rho_k:.4f}")
   print(f"D_k (before):\n{D_k}")
   print(f"D_{'{k+1}'} (after BFGS update):\n{np.round(D_new, 4)}")
   print(f"Secant check D_new @ y_k = s_k: {np.allclose(D_new @ y_k, s_k)}")

The Hessian-approximation form (updating :math:`\mathbf{B}_k` directly rather
than its inverse) is sometimes useful for understanding or for trust-region
methods:

.. math::

   \mathbf{B}_{k+1}
   = \mathbf{B}_k
   - \frac{\mathbf{B}_k\,\mathbf{s}_k\,\mathbf{s}_k^{\!\top}\mathbf{B}_k}
          {\mathbf{s}_k^{\!\top}\mathbf{B}_k\,\mathbf{s}_k}
   + \frac{\mathbf{y}_k\,\mathbf{y}_k^{\!\top}}
          {\mathbf{y}_k^{\!\top}\mathbf{s}_k}.

Reading this: we subtract a rank-1 correction that removes the old Hessian
approximation along the step direction, and add a rank-1 correction from the
actual gradient change. The net effect is that
:math:`\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k`, the secant condition.

Positive Definiteness
----------------------

**Theorem.** If :math:`\mathbf{D}_k` is positive definite and the curvature
condition :math:`\mathbf{s}_k^{\!\top}\mathbf{y}_k > 0` holds, then
:math:`\mathbf{D}_{k+1}` given by :eq:`bfgs` is also positive definite.

*Proof.* For any :math:`\mathbf{v} \neq \mathbf{0}`, write
:math:`\mathbf{u} = (\mathbf{I} - \rho_k\,\mathbf{y}_k\,\mathbf{s}_k^{\!\top})\mathbf{v}`.
Then

.. math::

   \mathbf{v}^{\!\top}\mathbf{D}_{k+1}\mathbf{v}
   = \mathbf{u}^{\!\top}\mathbf{D}_k\,\mathbf{u}
   + \rho_k\,(\mathbf{s}_k^{\!\top}\mathbf{v})^2.

The first term is non-negative (since :math:`\mathbf{D}_k \succ 0`); the
second is non-negative (since :math:`\rho_k > 0`). Both terms vanish
simultaneously only if :math:`\mathbf{u} = \mathbf{0}` and
:math:`\mathbf{s}_k^{\!\top}\mathbf{v} = 0`. But
:math:`\mathbf{u} = \mathbf{v} - \rho_k(\mathbf{s}_k^{\!\top}\mathbf{v})\mathbf{y}_k
= \mathbf{v}`, which contradicts :math:`\mathbf{v} \neq \mathbf{0}`. Hence
:math:`\mathbf{v}^{\!\top}\mathbf{D}_{k+1}\mathbf{v} > 0`. :math:`\square`

The BFGS Algorithm
--------------------

.. code-block:: text

   Initialize: theta_0, D_0 = I (or other SPD matrix)
   For k = 0, 1, 2, ...
       1. Compute search direction:  d_k = -D_k grad_f_k
       2. Line search: find alpha_k satisfying Wolfe conditions
       3. Update: theta_{k+1} = theta_k + alpha_k d_k
       4. Compute s_k = theta_{k+1} - theta_k,  y_k = grad_f_{k+1} - grad_f_k
       5. Update D_{k+1} via the BFGS formula (14.3)
       6. Check convergence

The cost per iteration is :math:`O(np)` for the gradient and :math:`O(p^2)` for
the matrix-vector product and the update --- a large saving over the
:math:`O(np^2 + p^3)` of Newton.

Full BFGS Optimizer on the Rosenbrock Function
-------------------------------------------------

The Rosenbrock function :math:`f(x,y) = 100(y - x^2)^2 + (1-x)^2` is the
classic test for optimizers. Its minimum sits at the bottom of a narrow,
curved valley, punishing methods that cannot adapt to anisotropic curvature.
Let's implement a full BFGS optimizer from scratch with a Wolfe-condition
line search and print a convergence table.

.. code-block:: python

   # Full BFGS optimizer on the Rosenbrock function
   import numpy as np

   np.random.seed(42)

   def rosenbrock(x):
       return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

   def rosenbrock_grad(x):
       g0 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
       g1 = 200*(x[1] - x[0]**2)
       return np.array([g0, g1])

   def wolfe_line_search(f, grad, x, d, c1=1e-4, c2=0.9, max_ls=40):
       """Backtracking line search satisfying strong Wolfe conditions."""
       alpha = 1.0
       fx = f(x)
       gx = grad(x)
       dg = gx @ d
       for _ in range(max_ls):
           if f(x + alpha*d) <= fx + c1*alpha*dg:
               if grad(x + alpha*d) @ d >= c2*dg:
                   return alpha
           alpha *= 0.5
       return alpha

   x = np.array([-1.2, 1.0])   # classic starting point
   D = np.eye(2)
   p = 2

   print(f"{'iter':>4s}  {'f(x)':>12s}  {'||grad||':>12s}  {'alpha':>8s}  {'x':>24s}")
   print("-" * 68)

   for k in range(80):
       g = rosenbrock_grad(x)
       gnorm = np.linalg.norm(g)
       print(f"{k:4d}  {rosenbrock(x):12.6f}  {gnorm:12.6e}  {'':>8s}  "
             f"[{x[0]:10.6f}, {x[1]:10.6f}]")

       if gnorm < 1e-8:
           print(f"\nConverged after {k} iterations!")
           break

       d = -D @ g                                # search direction
       alpha = wolfe_line_search(rosenbrock, rosenbrock_grad, x, d)
       x_new = x + alpha * d

       s = x_new - x
       y = rosenbrock_grad(x_new) - g
       sy = y @ s

       if sy > 1e-10:  # curvature condition
           rho = 1.0 / sy
           I = np.eye(p)
           V = I - rho * np.outer(s, y)
           D = V @ D @ V.T + rho * np.outer(s, s)

       x = x_new

   print(f"\nFinal x: {x}")
   print(f"True minimum: [1, 1], f* = 0")


14.4 Limited-Memory BFGS (L-BFGS)
===================================

Motivation
----------

BFGS stores the full :math:`p \times p` matrix :math:`\mathbf{D}_k`, requiring
:math:`O(p^2)` memory. When :math:`p` is in the tens of thousands or more (as
in many machine-learning models), this is infeasible. **L-BFGS** avoids storing
the matrix entirely; instead, it stores only the most recent :math:`m` pairs
:math:`\{(\mathbf{s}_j, \mathbf{y}_j)\}_{j=k-m}^{k-1}` and uses them to
compute the product :math:`\mathbf{D}_k\,\nabla f_k` implicitly. Typically
:math:`m = 5` to :math:`20`.

The beauty of L-BFGS is that it gets you most of the benefit of BFGS --- the
superlinear convergence, the automatic step-size scaling --- while using memory
that scales linearly in :math:`p`, not quadratically.

The Two-Loop Recursion
-----------------------

The following algorithm computes :math:`\mathbf{r} = \mathbf{D}_k\,\mathbf{q}`
where :math:`\mathbf{q} = \nabla f_k`, using only the stored pairs and a
simple initial matrix :math:`\mathbf{D}_k^{(0)} = \gamma_k\,\mathbf{I}`:

.. code-block:: text

   q <- grad_f_k
   for i = k-1, k-2, ..., k-m:
       alpha_i <- rho_i * s_i^T q
       q <- q - alpha_i y_i
   r <- D_k^(0) q          [typically gamma_k I q = gamma_k q]
   for i = k-m, k-m+1, ..., k-1:
       beta <- rho_i * y_i^T r
       r <- r + (alpha_i - beta) s_i
   return r

A good choice for the scaling is

.. math::

   \gamma_k = \frac{\mathbf{s}_{k-1}^{\!\top}\mathbf{y}_{k-1}}
                    {\mathbf{y}_{k-1}^{\!\top}\mathbf{y}_{k-1}},

which approximates the inverse Hessian's scale. To see why, note that for a
quadratic :math:`f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^{\!\top}\mathbf{H}\mathbf{x}`,
the step is :math:`\mathbf{s} = \mathbf{H}^{-1}\mathbf{y}`, so the scale of
the inverse Hessian is roughly
:math:`\|\mathbf{s}\|^2 / (\mathbf{s}^{\!\top}\mathbf{y}) = \mathbf{s}^{\!\top}\mathbf{y} / \|\mathbf{y}\|^2`.
This initial scaling ensures that the first search direction has the right
order of magnitude, which is crucial for practical convergence.

**Cost:** :math:`O(mp)` per iteration --- *linear* in :math:`p`. **Storage:**
:math:`O(mp)`.

L-BFGS is the method of choice for large-scale smooth optimization. It is the
default in many software packages (e.g., ``scipy.optimize.minimize`` with
``method='L-BFGS-B'``, R's ``optim`` with ``method="L-BFGS-B"``).

L-BFGS Two-Loop Recursion from Scratch
-----------------------------------------

The two-loop recursion is the algorithmic heart of L-BFGS. Let's implement it
line by line, then use it inside a full optimizer. The key insight is that we
never form :math:`\mathbf{D}_k` explicitly --- we compute
:math:`\mathbf{D}_k \nabla f_k` in :math:`O(mp)` time using only stored
:math:`(\mathbf{s}, \mathbf{y})` pairs.

.. code-block:: python

   # L-BFGS two-loop recursion implemented from scratch
   import numpy as np

   np.random.seed(42)

   def lbfgs_two_loop(grad, s_hist, y_hist, gamma):
       """
       Compute H_k @ grad using the L-BFGS two-loop recursion.

       Parameters
       ----------
       grad   : current gradient (length p)
       s_hist : list of recent step vectors s_j = x_{j+1} - x_j
       y_hist : list of recent gradient differences y_j = g_{j+1} - g_j
       gamma  : scalar for initial Hessian approximation H0 = gamma * I

       Returns
       -------
       r : the search direction component D_k @ grad
       """
       m = len(s_hist)
       q = grad.copy()
       alphas = np.zeros(m)
       rhos = np.zeros(m)

       # --- First loop: newest to oldest ---
       for i in range(m - 1, -1, -1):
           rhos[i] = 1.0 / (y_hist[i] @ s_hist[i])
           alphas[i] = rhos[i] * (s_hist[i] @ q)
           q = q - alphas[i] * y_hist[i]

       # --- Apply initial Hessian: r = gamma * I * q ---
       r = gamma * q

       # --- Second loop: oldest to newest ---
       for i in range(m):
           beta = rhos[i] * (y_hist[i] @ r)
           r = r + (alphas[i] - beta) * s_hist[i]

       return r

   # Test on a quadratic: f(x) = 0.5 x^T A x - b^T x
   A = np.array([[8, 1, 2],
                 [1, 6, 1],
                 [2, 1, 5]], dtype=float)
   b = np.array([1, 2, 3], dtype=float)
   x_star = np.linalg.solve(A, b)

   grad_f = lambda x: A @ x - b
   f = lambda x: 0.5 * x @ A @ x - b @ x

   # Run L-BFGS with m=3
   x = np.zeros(3)
   m = 3
   s_hist, y_hist = [], []

   print(f"{'iter':>4s}  {'f(x)':>12s}  {'||grad||':>12s}  {'||x - x*||':>12s}")
   print("-" * 48)

   for k in range(10):
       g = grad_f(x)
       gnorm = np.linalg.norm(g)
       print(f"{k:4d}  {f(x):12.6f}  {gnorm:12.6e}  {np.linalg.norm(x - x_star):12.6e}")

       if gnorm < 1e-12:
           print(f"Converged after {k} iterations.")
           break

       # Compute search direction via two-loop recursion
       if len(s_hist) == 0:
           d = -g            # steepest descent for first step
       else:
           gamma = (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
           d = -lbfgs_two_loop(g, s_hist, y_hist, gamma)

       # Exact line search for quadratic
       alpha = -(g @ d) / (d @ A @ d)
       x_new = x + alpha * d

       s = x_new - x
       y = grad_f(x_new) - g
       s_hist.append(s)
       y_hist.append(y)
       if len(s_hist) > m:
           s_hist.pop(0)
           y_hist.pop(0)

       x = x_new

   print(f"\nFinal x:    {np.round(x, 8)}")
   print(f"True x*:    {np.round(x_star, 8)}")

L-BFGS vs BFGS vs Gradient Descent
--------------------------------------

Let's put all three methods head-to-head on the same 20-dimensional
Rosenbrock function to see how they compare in iterations and function
evaluations.

.. code-block:: python

   # L-BFGS vs BFGS vs gradient descent on 20-D Rosenbrock
   import numpy as np

   np.random.seed(42)
   n_dim = 20

   def rosenbrock(x):
       return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

   def rosenbrock_grad(x):
       n = len(x)
       g = np.zeros(n)
       g[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
       for i in range(1, n - 1):
           g[i] = 200*(x[i] - x[i-1]**2) - 400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
       g[-1] = 200*(x[-1] - x[-2]**2)
       return g

   def backtrack(f, grad, x, d, c1=1e-4, rho=0.5):
       alpha = 1.0
       fx = f(x); dg = grad(x) @ d
       for _ in range(50):
           if f(x + alpha*d) <= fx + c1*alpha*dg:
               return alpha
           alpha *= rho
       return alpha

   results = {}

   # --- Gradient descent ---
   x = np.zeros(n_dim)
   fhist_gd = [rosenbrock(x)]
   for k in range(5000):
       g = rosenbrock_grad(x)
       if np.linalg.norm(g) < 1e-6:
           break
       d = -g
       alpha = backtrack(rosenbrock, rosenbrock_grad, x, d)
       x = x + alpha * d
       fhist_gd.append(rosenbrock(x))
   results['GD'] = (len(fhist_gd) - 1, fhist_gd[-1])

   # --- BFGS ---
   x = np.zeros(n_dim)
   D = np.eye(n_dim)
   fhist_bfgs = [rosenbrock(x)]
   for k in range(500):
       g = rosenbrock_grad(x)
       if np.linalg.norm(g) < 1e-6:
           break
       d = -D @ g
       alpha = backtrack(rosenbrock, rosenbrock_grad, x, d)
       x_new = x + alpha * d
       s = x_new - x
       y = rosenbrock_grad(x_new) - g
       sy = y @ s
       if sy > 1e-10:
           rho = 1.0 / sy
           I = np.eye(n_dim)
           V = I - rho * np.outer(s, y)
           D = V @ D @ V.T + rho * np.outer(s, s)
       x = x_new
       fhist_bfgs.append(rosenbrock(x))
   results['BFGS'] = (len(fhist_bfgs) - 1, fhist_bfgs[-1])

   # --- L-BFGS (m=10) ---
   x = np.zeros(n_dim)
   m_lbfgs = 10
   s_hist, y_hist = [], []
   fhist_lbfgs = [rosenbrock(x)]
   for k in range(500):
       g = rosenbrock_grad(x)
       if np.linalg.norm(g) < 1e-6:
           break
       if len(s_hist) == 0:
           d = -g
       else:
           gamma = (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
           q = g.copy()
           alphas_l = []
           rhos_l = []
           for i in range(len(s_hist) - 1, -1, -1):
               rho_i = 1.0 / (y_hist[i] @ s_hist[i])
               a_i = rho_i * (s_hist[i] @ q)
               q = q - a_i * y_hist[i]
               alphas_l.append(a_i)
               rhos_l.append(rho_i)
           alphas_l.reverse()
           rhos_l.reverse()
           r = gamma * q
           for i in range(len(s_hist)):
               beta = rhos_l[i] * (y_hist[i] @ r)
               r = r + (alphas_l[i] - beta) * s_hist[i]
           d = -r
       alpha = backtrack(rosenbrock, rosenbrock_grad, x, d)
       x_new = x + alpha * d
       s = x_new - x
       y = rosenbrock_grad(x_new) - g
       if y @ s > 1e-10:
           s_hist.append(s)
           y_hist.append(y)
           if len(s_hist) > m_lbfgs:
               s_hist.pop(0)
               y_hist.pop(0)
       x = x_new
       fhist_lbfgs.append(rosenbrock(x))
   results['L-BFGS'] = (len(fhist_lbfgs) - 1, fhist_lbfgs[-1])

   print(f"{'Method':<10s}  {'Iters':>6s}  {'Final f(x)':>14s}")
   print("-" * 36)
   for name in ['GD', 'BFGS', 'L-BFGS']:
       iters, fval = results[name]
       print(f"{name:<10s}  {iters:6d}  {fval:14.6e}")

   print(f"\n(Gradient descent capped at 5000 iterations)")
   print(f"Memory: GD uses O(p)={n_dim}, BFGS uses O(p^2)={n_dim**2}, "
         f"L-BFGS uses O(mp)={m_lbfgs*n_dim}")

Scipy Optimizer Comparison with Timing
-----------------------------------------

In practice, you rarely implement BFGS from scratch. Let's compare the
polished implementations in ``scipy.optimize.minimize`` on the same
Rosenbrock problem, with wall-clock timing.

.. code-block:: python

   # scipy.optimize.minimize comparison: BFGS vs L-BFGS-B vs Nelder-Mead
   import numpy as np
   from scipy.optimize import minimize
   import time

   np.random.seed(42)

   def rosenbrock(x):
       return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

   def rosenbrock_grad(x):
       n = len(x)
       g = np.zeros(n)
       g[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
       for i in range(1, n-1):
           g[i] = 200*(x[i]-x[i-1]**2) - 400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
       g[-1] = 200*(x[-1]-x[-2]**2)
       return g

   for n_dim in [10, 50, 200]:
       print(f"\n{'='*60}")
       print(f"Dimension p = {n_dim}")
       print(f"{'Method':>14s}  {'f*':>10s}  {'nfev':>6s}  {'nit':>5s}  "
             f"{'time(ms)':>9s}  {'success':>7s}")
       print("-" * 60)

       x0 = np.zeros(n_dim)
       for method in ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'CG']:
           t0 = time.perf_counter()
           if method in ['Nelder-Mead']:
               res = minimize(rosenbrock, x0, method=method,
                              options={'maxiter': 50000, 'maxfev': 100000})
           else:
               res = minimize(rosenbrock, x0, jac=rosenbrock_grad,
                              method=method)
           elapsed = (time.perf_counter() - t0) * 1000
           print(f"{method:>14s}  {res.fun:10.2e}  {res.nfev:6d}  {res.nit:5d}  "
                 f"{elapsed:9.1f}  {str(res.success):>7s}")

.. admonition:: Real-World Example

   **Large-scale NLP model fitting with L-BFGS.**

   A natural language processing team is training a conditional random field
   (CRF) for named-entity recognition.  The model has 500,000 features (word
   shapes, prefixes, suffixes, context patterns).  Storing a full
   :math:`500{,}000 \times 500{,}000` inverse-Hessian would require about
   2 TB of RAM --- clearly impossible.  L-BFGS with :math:`m = 10` stored
   pairs requires only :math:`10 \times 2 \times 500{,}000` floats, which is
   about 80 MB.  The CRF log-likelihood is smooth and concave, so L-BFGS
   converges in 100--200 gradient evaluations, far fewer than the thousands
   needed by plain gradient descent.


14.5 The SR1 Update
=====================

The **Symmetric Rank-1 (SR1)** update is an alternative to BFGS:

.. math::
   :label: sr1

   \mathbf{B}_{k+1}
   = \mathbf{B}_k
   + \frac{(\mathbf{y}_k - \mathbf{B}_k\mathbf{s}_k)\,
           (\mathbf{y}_k - \mathbf{B}_k\mathbf{s}_k)^{\!\top}}
          {(\mathbf{y}_k - \mathbf{B}_k\mathbf{s}_k)^{\!\top}\mathbf{s}_k}.

Reading this formula: the vector
:math:`\mathbf{y}_k - \mathbf{B}_k\mathbf{s}_k` is the *residual* --- the
discrepancy between the actual gradient change :math:`\mathbf{y}_k` and what
the current approximation :math:`\mathbf{B}_k` would predict
(:math:`\mathbf{B}_k\mathbf{s}_k`). The SR1 update adds a single rank-1
correction built from this residual, which is the simplest possible update
that eliminates the error along the step direction. You can verify directly
that :math:`\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k`.

As the name suggests, this is a rank-1 update, costing :math:`O(p^2)`.

The SR1 update has a remarkable property: if you could somehow feed it the true
secant pairs from a quadratic function, it would recover the exact Hessian
in at most :math:`p` steps.  BFGS generally cannot make this claim.

SR1 Update and Comparison with BFGS on an Indefinite Problem
---------------------------------------------------------------

The SR1 update does *not* guarantee positive definiteness, which makes it
unsuitable for line-search methods but perfect for trust-region methods
where the trust constraint regularizes the subproblem. Its big advantage
is that it can capture negative curvature, which BFGS cannot.

Consider the function :math:`f(x,y) = x^2 - y^2 + 0.05(x^4 + y^4)`, which
has a saddle point at the origin and minima away from it. The Hessian near
the saddle is indefinite. BFGS, forced to maintain positive definiteness,
cannot represent this; SR1 can.

.. code-block:: python

   # SR1 vs BFGS Hessian approximation on an indefinite-Hessian problem
   import numpy as np

   np.random.seed(42)

   # f(x,y) = x^2 - y^2 + 0.05*(x^4 + y^4)
   def f(xy):
       x, y = xy
       return x**2 - y**2 + 0.05*(x**4 + y**4)

   def grad_f(xy):
       x, y = xy
       return np.array([2*x + 0.2*x**3, -2*y + 0.2*y**3])

   def hess_f(xy):
       x, y = xy
       return np.array([[2 + 0.6*x**2, 0],
                        [0, -2 + 0.6*y**2]])

   # Collect gradient information along a path
   path = [np.array([1.5, 0.5])]
   for _ in range(4):
       x = path[-1]
       g = grad_f(x)
       path.append(x - 0.1 * g)  # small gradient steps

   # Build SR1 and BFGS approximations
   B_sr1 = np.eye(2)
   B_bfgs = np.eye(2)

   print(f"{'Step':>4s}  {'||B_sr1 - H||':>14s}  {'||B_bfgs - H||':>15s}  "
         f"{'SR1 eigvals':>20s}  {'BFGS eigvals':>20s}")
   print("-" * 82)

   for i in range(len(path) - 1):
       s = path[i+1] - path[i]
       y = grad_f(path[i+1]) - grad_f(path[i])
       H_true = hess_f(path[i+1])

       # SR1 update
       r = y - B_sr1 @ s
       denom = r @ s
       if abs(denom) > 1e-8 * np.linalg.norm(r) * np.linalg.norm(s):
           B_sr1 = B_sr1 + np.outer(r, r) / denom

       # BFGS update (Hessian form)
       Bs = B_bfgs @ s
       sy = y @ s
       if sy > 1e-10:
           B_bfgs = (B_bfgs - np.outer(Bs, Bs) / (s @ Bs)
                     + np.outer(y, y) / sy)

       eig_sr1 = np.sort(np.linalg.eigvalsh(B_sr1))
       eig_bfgs = np.sort(np.linalg.eigvalsh(B_bfgs))
       err_sr1 = np.linalg.norm(B_sr1 - H_true, 'fro')
       err_bfgs = np.linalg.norm(B_bfgs - H_true, 'fro')
       print(f"{i:4d}  {err_sr1:14.4f}  {err_bfgs:15.4f}  "
             f"[{eig_sr1[0]:8.3f}, {eig_sr1[1]:8.3f}]  "
             f"[{eig_bfgs[0]:8.3f}, {eig_bfgs[1]:8.3f}]")

   print(f"\nTrue Hessian at final point:\n{hess_f(path[-1])}")
   print(f"  eigenvalues: {np.sort(np.linalg.eigvalsh(hess_f(path[-1])))}")
   print(f"\nSR1 can have negative eigenvalues (capturing indefiniteness).")
   print(f"BFGS eigenvalues are always positive (forced positive definiteness).")

Comparison with BFGS
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Property
     - BFGS
     - SR1
   * - Update rank
     - 2
     - 1
   * - Positive definiteness
     - Guaranteed (with Wolfe)
     - **Not guaranteed**
   * - Hessian approximation quality
     - Good for positive-definite Hessians
     - Can capture indefinite curvature
   * - Use case
     - Standard line-search methods
     - Trust-region methods, saddle points

Because SR1 does not guarantee positive definiteness, it is typically used
inside a trust-region framework (Section 14.6) rather than with a line search.
Its advantage is that it can model negative curvature, which is useful for
finding saddle points or for problems where the Hessian is indefinite.


14.6 Trust-Region Methods
==========================

Line-search methods choose a direction and then decide how far to go.
**Trust-region methods** reverse this: they fix a region around the current
point within which the quadratic model is trusted, and find the best step
within that region.

.. admonition:: Intuition

   With a line search, you pick a direction and then ask "how far?"  With a
   trust region, you draw a circle around yourself and ask "where is the
   best place to step within this circle?"  If the model is accurate (the
   function really does decrease as predicted), you widen the circle next time.
   If the model was misleading, you shrink the circle.  This self-regulating
   behavior makes trust-region methods remarkably robust.

The Trust-Region Subproblem
----------------------------

At iteration :math:`k`, solve

.. math::
   :label: tr_sub

   \min_{\mathbf{d}} \;
   m_k(\mathbf{d})
   = f_k + \nabla f_k^{\!\top}\mathbf{d}
   + \tfrac{1}{2}\,\mathbf{d}^{\!\top}\mathbf{B}_k\,\mathbf{d}
   \qquad\text{subject to}\qquad
   \|\mathbf{d}\| \leq \Delta_k,

where :math:`\Delta_k > 0` is the **trust-region radius** and
:math:`\mathbf{B}_k` is a (possibly indefinite) Hessian approximation.

Reading this: :math:`m_k(\mathbf{d})` is the same quadratic model that Newton
uses, but now we minimize it only within a ball of radius :math:`\Delta_k`
around the current point. The model :math:`m_k` says "if the world were
quadratic, moving by :math:`\mathbf{d}` would change the function value by
the gradient term plus the curvature term." We only trust this prediction
within a certain distance, so we restrict :math:`\|\mathbf{d}\| \leq \Delta_k`.

This subproblem always has a solution, even when :math:`\mathbf{B}_k` is not
positive definite --- the constraint :math:`\|\mathbf{d}\| \leq \Delta_k`
regularizes the problem. In contrast, the unconstrained Newton step
:math:`-\mathbf{B}_k^{-1}\nabla f_k` may not even exist when
:math:`\mathbf{B}_k` is singular.

Updating the Trust-Region Radius
---------------------------------

After computing the trial step :math:`\mathbf{d}_k`, evaluate the **ratio of
actual to predicted reduction**:

.. math::

   r_k = \frac{f(\boldsymbol{\theta}_k) - f(\boldsymbol{\theta}_k + \mathbf{d}_k)}
              {m_k(\mathbf{0}) - m_k(\mathbf{d}_k)}.

The numerator is how much the function *actually* decreased; the denominator is
how much the quadratic model *predicted* it would decrease. When
:math:`r_k \approx 1`, the model is trustworthy. When :math:`r_k` is small or
negative, the model was misleading.

- If :math:`r_k > \eta_2` (e.g., :math:`\eta_2 = 0.75`): the model is very
  accurate; **expand** the trust region: :math:`\Delta_{k+1} = 2\Delta_k`.
- If :math:`r_k > \eta_1` (e.g., :math:`\eta_1 = 0.25`): acceptable;
  **keep** the trust region: :math:`\Delta_{k+1} = \Delta_k`.
- If :math:`r_k \leq \eta_1`: poor prediction; **shrink** the trust region:
  :math:`\Delta_{k+1} = \Delta_k / 2`; reject the step
  (:math:`\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k`).

Trust-Region Demo: Radius Expanding and Contracting
------------------------------------------------------

Let's implement a trust-region optimizer using the Cauchy point (steepest
descent constrained to the trust region) and watch the radius
:math:`\Delta_k` adapt based on the agreement ratio :math:`r_k`.

.. code-block:: python

   # Trust-region demo: radius expanding/contracting based on agreement ratio
   import numpy as np

   np.random.seed(42)

   # Rosenbrock function
   def f(xy):
       x, y = xy
       return 100*(y - x**2)**2 + (1 - x)**2

   def grad_f(xy):
       x, y = xy
       return np.array([-400*x*(y - x**2) - 2*(1 - x),
                        200*(y - x**2)])

   def hess_f(xy):
       x, y = xy
       return np.array([[-400*(y - x**2) + 800*x**2 + 2, -400*x],
                        [-400*x, 200]])

   def cauchy_point(g, B, delta):
       """Compute the Cauchy point: minimizer of model along -g, within trust region."""
       gBg = g @ B @ g
       gnorm = np.linalg.norm(g)
       if gBg <= 0:
           # Negative curvature: go to boundary
           return -(delta / gnorm) * g
       tau = min(1.0, gnorm**3 / (delta * gBg))
       return -tau * (delta / gnorm) * g

   def model_reduction(g, B, d):
       """Predicted reduction: m(0) - m(d) = -(g^T d + 0.5 d^T B d)"""
       return -(g @ d + 0.5 * d @ B @ d)

   x = np.array([-1.2, 1.0])
   delta = 1.0        # initial trust-region radius
   eta1 = 0.25
   eta2 = 0.75

   print(f"{'iter':>4s}  {'f(x)':>12s}  {'Delta':>8s}  {'r_k':>8s}  "
         f"{'||d||':>8s}  {'action':>12s}")
   print("-" * 62)

   for k in range(60):
       g = grad_f(x)
       B = hess_f(x)
       gnorm = np.linalg.norm(g)

       if gnorm < 1e-6:
           print(f"\nConverged after {k} iterations!")
           break

       # Compute trial step (Cauchy point)
       d = cauchy_point(g, B, delta)
       dnorm = np.linalg.norm(d)

       # Evaluate agreement ratio
       actual_red = f(x) - f(x + d)
       pred_red = model_reduction(g, B, d)

       if pred_red < 1e-16:
           rk = 1.0
       else:
           rk = actual_red / pred_red

       # Decide: accept/reject step and adjust radius
       if rk > eta2:
           action = "EXPAND"
           x = x + d
           delta = min(2 * delta, 10.0)
       elif rk > eta1:
           action = "keep"
           x = x + d
       else:
           action = "SHRINK/reject"
           delta = delta / 2

       print(f"{k:4d}  {f(x):12.4f}  {delta:8.4f}  {rk:8.3f}  "
             f"{dnorm:8.4f}  {action:>12s}")

   print(f"\nFinal x: {np.round(x, 6)}")
   print(f"f(x):    {f(x):.6e}")

Watch the ``Delta`` column: it expands when the quadratic model predicts
the function well (:math:`r_k > 0.75`) and shrinks when the prediction
is poor (:math:`r_k < 0.25`), with rejected steps where the iterate does
not move.

The Cauchy Point
-----------------

The **Cauchy point** is the minimizer of the quadratic model along the
steepest-descent direction, subject to the trust-region constraint. It provides
a lower bound on the reduction and is used as a benchmark: any step that
achieves at least as much reduction as the Cauchy point is acceptable.

The Cauchy step is

.. math::

   \mathbf{d}_k^C = -\tau_k\,\frac{\Delta_k}{\|\nabla f_k\|}\,\nabla f_k,

where

.. math::

   \tau_k = \begin{cases}
   1 & \text{if } \nabla f_k^{\!\top}\mathbf{B}_k\,\nabla f_k \leq 0, \\
   \min\!\left(1,\;\frac{\|\nabla f_k\|^3}
     {\Delta_k\,\nabla f_k^{\!\top}\mathbf{B}_k\,\nabla f_k}\right)
   & \text{otherwise.}
   \end{cases}

The factor :math:`\tau_k` is a step-size along the steepest-descent direction.
If the curvature along the gradient is negative (the function curves *down*
in the gradient direction), then the quadratic model says "go as far as you
can" --- so :math:`\tau_k = 1` and we step all the way to the trust-region
boundary. If the curvature is positive, the model has a finite minimizer along
the gradient direction and we take the shorter of that minimizer and the
trust-region boundary.

The Dogleg Method
------------------

When :math:`\mathbf{B}_k` is positive definite, the **dogleg method** provides
an efficient approximate solution to the trust-region subproblem. It
interpolates between the Cauchy point :math:`\mathbf{d}^C` and the full Newton
step :math:`\mathbf{d}^N = -\mathbf{B}_k^{-1}\nabla f_k`:

- If :math:`\|\mathbf{d}^N\| \leq \Delta_k`: use the full Newton step.
- If :math:`\|\mathbf{d}^C\| \geq \Delta_k`: use the constrained
  steepest-descent step :math:`(\Delta_k / \|\nabla f_k\|)(-\nabla f_k)`.
- Otherwise: move along the "dogleg path" --- from the origin to the Cauchy
  point, then from the Cauchy point to the Newton point --- and stop where
  this path intersects the trust-region boundary.

The intersection point is found by solving a scalar quadratic equation:

.. math::

   \|\mathbf{d}^C + t(\mathbf{d}^N - \mathbf{d}^C)\|^2 = \Delta_k^2

for :math:`t \in [0, 1]`.


14.7 Comparison of Methods
============================

.. list-table::
   :header-rows: 1
   :widths: 18 18 16 16 16 16

   * - Method
     - Per-iteration cost
     - Storage
     - Convergence rate
     - Robustness
     - Best for
   * - Gradient descent
     - :math:`O(np)`
     - :math:`O(p)`
     - Linear
     - High
     - Very large :math:`p`
   * - Newton
     - :math:`O(np^2 + p^3)`
     - :math:`O(p^2)`
     - Quadratic
     - Medium
     - Small :math:`p`, smooth
   * - BFGS
     - :math:`O(np + p^2)`
     - :math:`O(p^2)`
     - Superlinear
     - High
     - Moderate :math:`p`
   * - L-BFGS
     - :math:`O(np + mp)`
     - :math:`O(mp)`
     - Superlinear
     - High
     - Large :math:`p`, smooth
   * - SR1 + trust region
     - :math:`O(np + p^2)`
     - :math:`O(p^2)`
     - Superlinear
     - High
     - Indefinite Hessian
   * - Fisher scoring
     - :math:`O(np^2 + p^3)`
     - :math:`O(p^2)`
     - Quadratic
     - High (for MLE)
     - Small :math:`p`, stat models

**Superlinear convergence** of BFGS means

.. math::

   \lim_{k\to\infty}
   \frac{\|\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}^*\|}
        {\|\boldsymbol{\theta}_k - \boldsymbol{\theta}^*\|}
   = 0.

Reading this formula: the ratio of successive errors goes to zero. For linear
convergence, this ratio would settle at some fixed constant :math:`c < 1` (you
gain the same number of digits each iteration). For quadratic convergence, the
errors shrink much faster (digits double each iteration). Superlinear
convergence sits between the two: the error-reduction factor *improves* with
every step, approaching zero, but it does not square the error as Newton does.
In practice, BFGS often converges nearly
as fast as Newton in the number of iterations, at a fraction of the cost.

Let's see this convergence hierarchy in action by comparing all three methods on
the same problem.

.. code-block:: python

   # Convergence comparison: gradient descent vs BFGS vs Newton
   import numpy as np

   np.random.seed(42)
   A = np.array([[10.0, 2.0], [2.0, 3.0]])
   b = np.array([1.0, 2.0])
   x_star = np.linalg.solve(A, b)

   def f(x): return 0.5 * x @ A @ x - b @ x
   def grad(x): return A @ x - b

   # Gradient descent with optimal step size
   x_gd = np.array([5.0, 5.0])
   L = np.max(np.linalg.eigvalsh(A))
   for k in range(50):
       x_gd = x_gd - (1.0/L) * grad(x_gd)
   err_gd = np.linalg.norm(x_gd - x_star)

   # BFGS
   x_bfgs = np.array([5.0, 5.0])
   D = np.eye(2)
   for k in range(50):
       g = grad(x_bfgs)
       d = -D @ g
       alpha = -(g @ d) / (d @ A @ d)       # exact line search for quadratic
       x_new = x_bfgs + alpha * d
       s = x_new - x_bfgs
       y = grad(x_new) - g
       rho = 1.0 / (y @ s)
       I = np.eye(2)
       D = (I - rho * np.outer(s, y)) @ D @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
       x_bfgs = x_new
   err_bfgs = np.linalg.norm(x_bfgs - x_star)

   # Newton (exact for quadratic -- 1 step)
   x_newton = np.array([5.0, 5.0])
   x_newton = x_newton - np.linalg.solve(A, grad(x_newton))
   err_newton = np.linalg.norm(x_newton - x_star)

   print(f"After 50 iters -- GD error:     {err_gd:.2e}")
   print(f"After 50 iters -- BFGS error:   {err_bfgs:.2e}")
   print(f"After  1 iter  -- Newton error: {err_newton:.2e}")

.. admonition:: Common Pitfall

   When using ``scipy.optimize.minimize`` with ``method='BFGS'``, make sure to
   supply an analytic gradient via the ``jac`` argument whenever possible.
   Finite-difference gradients introduce noise that can degrade the BFGS
   Hessian approximation over time, leading to slower convergence or even
   failure.  For L-BFGS-B, this is even more critical because the limited
   memory means there is less "buffer" to absorb noisy gradient information.


14.8 Summary
==============

Quasi-Newton methods occupy the sweet spot between the cheap but slow gradient
methods of :ref:`ch12_gradient` and the fast but expensive Newton methods of
:ref:`ch13_newton`. The BFGS update builds an increasingly accurate
inverse-Hessian approximation using only gradient evaluations, achieving
superlinear convergence with :math:`O(p^2)` cost per iteration. L-BFGS drops
this to :math:`O(mp)` by storing only the :math:`m` most recent
gradient-difference pairs, making it the default for large-scale smooth
optimization. Trust-region methods, combined with SR1 or BFGS updates, provide
robust global convergence even for non-convex problems. These methods are the
backbone of most general-purpose numerical optimizers used in statistical
computing.
