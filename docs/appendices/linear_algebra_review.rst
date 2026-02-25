.. _appendix_linalg:

====================================
Appendix A: Linear Algebra Review
====================================

This appendix collects the linear algebra results that appear most frequently
in likelihood-based inference. The emphasis is on matrix calculus and quadratic
forms, since these are the tools needed to derive maximum-likelihood estimators
for multivariate models.

If you have taken a standard linear algebra course, most of this material will
be familiar. What may be new is seeing how each concept connects directly to
something you need when writing down, differentiating, or optimizing a
likelihood function. We will highlight those connections throughout.


Vectors and Matrices
====================

Notation
--------

We write column vectors in bold lowercase and matrices in bold uppercase:

.. math::

   \mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
   \in \mathbb{R}^n,
   \qquad
   \mathbf{A} = \begin{pmatrix}
   a_{11} & a_{12} & \cdots & a_{1m} \\
   a_{21} & a_{22} & \cdots & a_{2m} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{n1} & a_{n2} & \cdots & a_{nm}
   \end{pmatrix}
   \in \mathbb{R}^{n \times m}.

The entry in the :math:`i`-th row and :math:`j`-th column of :math:`\mathbf{A}`
is denoted :math:`a_{ij}` or :math:`[\mathbf{A}]_{ij}`.

Let's see how NumPy represents these objects. A vector is simply a
one-dimensional array, while a matrix is two-dimensional.

.. code-block:: python

   # Creating vectors and matrices in NumPy
   import numpy as np
   np.random.seed(42)

   x = np.array([1.0, 2.0, 3.0])          # column vector (shape (3,))
   A = np.array([[1, 2, 3],
                 [4, 5, 6]])               # 2x3 matrix

   print("Vector x:", x, " shape:", x.shape)
   print("Matrix A:\n", A, " shape:", A.shape)
   print("Entry A[0,2]:", A[0, 2])         # row 0, column 2 -> 3

Basic Operations
----------------

**Addition.** Two matrices of the same dimensions are added element-wise:

.. math::

   [\mathbf{A} + \mathbf{B}]_{ij} = a_{ij} + b_{ij}.

**Scalar multiplication.** Every entry is multiplied by the scalar:

.. math::

   [c\,\mathbf{A}]_{ij} = c \, a_{ij}.

**Inner product.** For two vectors in :math:`\mathbb{R}^n`, the inner product
(or dot product) gives you a single number that measures how aligned the two
vectors are. Think of it as a "similarity score" weighted by magnitude:

.. math::

   \mathbf{x}^\top \mathbf{y}
   = \sum_{i=1}^{n} x_i \, y_i.

You will see this constantly in likelihood work: every time you compute a
linear predictor :math:`\mathbf{x}^\top \boldsymbol{\beta}` in regression,
you are computing an inner product.

**Outer product.** For :math:`\mathbf{x} \in \mathbb{R}^n` and
:math:`\mathbf{y} \in \mathbb{R}^m`:

.. math::

   \mathbf{x}\,\mathbf{y}^\top \in \mathbb{R}^{n \times m},
   \qquad
   [\mathbf{x}\,\mathbf{y}^\top]_{ij} = x_i \, y_j.

While the inner product collapses two vectors into a scalar, the outer product
expands them into a full matrix. The sample covariance matrix, for instance,
is built from outer products of centered observations.

.. code-block:: python

   # Inner product vs outer product
   import numpy as np
   np.random.seed(42)

   x = np.array([1.0, 2.0, 3.0])
   y = np.array([4.0, 5.0, 6.0])

   inner = x @ y                       # dot product: scalar
   outer = np.outer(x, y)              # outer product: 3x3 matrix

   print("Inner product x^T y:", inner)
   print("Outer product x y^T:\n", outer)

.. admonition:: Real-World Connection

   In logistic regression the log-likelihood involves sums of terms like
   :math:`y_i \, \mathbf{x}_i^\top \boldsymbol{\beta}`. Each of these is an
   inner product between a feature vector and the parameter vector. Fast
   linear algebra for inner products is therefore the computational backbone
   of fitting generalized linear models.


Matrix Multiplication, Transpose, Trace, and Determinant
=========================================================

Matrix Multiplication
---------------------

If :math:`\mathbf{A} \in \mathbb{R}^{n \times p}` and
:math:`\mathbf{B} \in \mathbb{R}^{p \times m}`, the product
:math:`\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{n \times m}` has
entries:

.. math::

   c_{ij} = \sum_{k=1}^{p} a_{ik}\,b_{kj}.

Matrix multiplication is associative and distributive but, in general, **not
commutative**: :math:`\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}`.

Here is a quick check that NumPy's ``@`` operator implements this definition:

.. code-block:: python

   # Verifying matrix multiplication entry by entry
   import numpy as np
   np.random.seed(42)

   A = np.random.randn(3, 4)
   B = np.random.randn(4, 2)
   C = A @ B                          # 3x2 result

   # Check one entry manually: C[1,0] = sum_k A[1,k]*B[k,0]
   manual = sum(A[1, k] * B[k, 0] for k in range(4))
   print("C[1,0] via @  :", C[1, 0])
   print("C[1,0] manual :", manual)
   print("Match:", np.isclose(C[1, 0], manual))

Transpose
---------

The transpose :math:`\mathbf{A}^\top` is obtained by reflecting entries across
the main diagonal:

.. math::

   [\mathbf{A}^\top]_{ij} = a_{ji}.

Key properties:

.. math::

   (\mathbf{A}\mathbf{B})^\top &= \mathbf{B}^\top \mathbf{A}^\top, \\
   (\mathbf{A}^\top)^\top &= \mathbf{A}, \\
   (\mathbf{A} + \mathbf{B})^\top &= \mathbf{A}^\top + \mathbf{B}^\top.

The reversal rule :math:`(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top`
may seem like a small detail, but it shows up every time you transpose a
likelihood expression involving matrix products -- get it wrong and your
derivation breaks.

Trace
-----

The trace of a square matrix :math:`\mathbf{A} \in \mathbb{R}^{n \times n}` is
the sum of its diagonal entries:

.. math::

   \operatorname{tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}.

The trace has the **cyclic property**: for conformable matrices
:math:`\mathbf{A}`, :math:`\mathbf{B}`, :math:`\mathbf{C}`,

.. math::

   \operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C})
   = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{A})
   = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B}).

A useful special case: for vectors :math:`\mathbf{x}, \mathbf{y} \in
\mathbb{R}^n`,

.. math::

   \mathbf{x}^\top \mathbf{y}
   = \operatorname{tr}(\mathbf{x}^\top \mathbf{y})
   = \operatorname{tr}(\mathbf{y}\,\mathbf{x}^\top).

.. admonition:: Why Does This Matter?

   The cyclic trace property is the workhorse identity behind nearly every
   matrix-calculus derivation in multivariate statistics. Whenever you need to
   differentiate an expression like
   :math:`\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{S})` with respect
   to :math:`\boldsymbol{\Sigma}`, you first rearrange using the cyclic
   property to put the variable matrix in a convenient position.

.. code-block:: python

   # Demonstrating the cyclic trace property
   import numpy as np
   np.random.seed(42)

   A = np.random.randn(3, 4)
   B = np.random.randn(4, 2)
   C = np.random.randn(2, 3)

   tr_ABC = np.trace(A @ B @ C)
   tr_BCA = np.trace(B @ C @ A)
   tr_CAB = np.trace(C @ A @ B)

   print(f"tr(ABC) = {tr_ABC:.6f}")
   print(f"tr(BCA) = {tr_BCA:.6f}")
   print(f"tr(CAB) = {tr_CAB:.6f}")
   print("All equal:", np.allclose([tr_ABC, tr_BCA], [tr_BCA, tr_CAB]))

Determinant
-----------

The determinant of a square matrix :math:`\mathbf{A}` is denoted
:math:`\det(\mathbf{A})` or :math:`|\mathbf{A}|`. For a
:math:`2 \times 2` matrix:

.. math::

   \det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc.

Important properties:

.. math::

   \det(\mathbf{A}\mathbf{B}) &= \det(\mathbf{A})\,\det(\mathbf{B}), \\
   \det(\mathbf{A}^\top) &= \det(\mathbf{A}), \\
   \det(c\,\mathbf{A}) &= c^n \det(\mathbf{A})
   \quad \text{for } \mathbf{A} \in \mathbb{R}^{n \times n}, \\
   \det(\mathbf{A}^{-1}) &= \frac{1}{\det(\mathbf{A})}.

Think of the determinant as a measure of the "volume scaling factor" of the
linear map defined by the matrix. In the context of the multivariate normal,
:math:`\det(\boldsymbol{\Sigma})` tells you the volume of the probability
ellipsoid -- which is why :math:`\log|\boldsymbol{\Sigma}|` appears in the
log-likelihood.

.. code-block:: python

   # Determinant properties
   import numpy as np
   np.random.seed(42)

   A = np.random.randn(3, 3)
   B = np.random.randn(3, 3)

   print(f"det(A)         = {np.linalg.det(A):.6f}")
   print(f"det(A^T)       = {np.linalg.det(A.T):.6f}")
   print(f"det(AB)        = {np.linalg.det(A @ B):.6f}")
   print(f"det(A)*det(B)  = {np.linalg.det(A) * np.linalg.det(B):.6f}")

   c = 2.0
   n = 3
   print(f"\ndet(cA)        = {np.linalg.det(c * A):.6f}")
   print(f"c^n * det(A)   = {c**n * np.linalg.det(A):.6f}")


Inverse Matrices and Positive Definiteness
============================================

Inverse Matrices
----------------

A square matrix :math:`\mathbf{A}` is **invertible** (non-singular) if there
exists a matrix :math:`\mathbf{A}^{-1}` such that:

.. math::

   \mathbf{A}\,\mathbf{A}^{-1} = \mathbf{A}^{-1}\,\mathbf{A} = \mathbf{I}.

A matrix is invertible if and only if :math:`\det(\mathbf{A}) \neq 0`. Key
identities:

.. math::

   (\mathbf{A}\mathbf{B})^{-1} &= \mathbf{B}^{-1}\mathbf{A}^{-1}, \\
   (\mathbf{A}^\top)^{-1} &= (\mathbf{A}^{-1})^\top.

**Woodbury identity.** A workhorse for likelihood computations:

.. math::

   (\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1}
   = \mathbf{A}^{-1}
   - \mathbf{A}^{-1}\mathbf{U}
     (\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}
     \mathbf{V}\mathbf{A}^{-1}.

Here is the intuition: suppose you already know :math:`\mathbf{A}^{-1}` and
you want the inverse of :math:`\mathbf{A}` plus a low-rank correction
:math:`\mathbf{U}\mathbf{C}\mathbf{V}`. Instead of inverting a full
:math:`n \times n` matrix from scratch, the Woodbury identity lets you do it
via a smaller inversion (the size of :math:`\mathbf{C}`). In mixed-effects
models and Gaussian process regression, this saves enormous computation.

.. code-block:: python

   # Verifying the Woodbury identity
   import numpy as np
   np.random.seed(42)

   n, k = 5, 2
   A = np.eye(n) * 3.0                         # easy to invert
   U = np.random.randn(n, k)
   C = np.random.randn(k, k)
   V = np.random.randn(k, n)

   # Direct inversion
   direct = np.linalg.inv(A + U @ C @ V)

   # Woodbury formula
   A_inv = np.linalg.inv(A)
   inner_inv = np.linalg.inv(np.linalg.inv(C) + V @ A_inv @ U)
   woodbury = A_inv - A_inv @ U @ inner_inv @ V @ A_inv

   print("Direct and Woodbury agree:",
         np.allclose(direct, woodbury))

Positive Definite Matrices
--------------------------

A symmetric matrix :math:`\mathbf{A}` is **positive definite** (written
:math:`\mathbf{A} \succ 0`) if, for every non-zero vector :math:`\mathbf{x}`:

.. math::

   \mathbf{x}^\top \mathbf{A}\,\mathbf{x} > 0.

It is **positive semi-definite** (:math:`\mathbf{A} \succeq 0`) if the
inequality is non-strict. Covariance matrices are always positive
semi-definite and, under mild regularity conditions, positive definite.

What is the intuition? A positive definite matrix defines a "bowl" in
:math:`n`-dimensional space -- every direction curves upward. This is exactly
what we need for a quadratic form to have a unique minimum, or equivalently,
for a covariance matrix to define a proper probability distribution.

Equivalent characterizations of positive definiteness:

- All eigenvalues of :math:`\mathbf{A}` are strictly positive.
- All leading principal minors are strictly positive.
- There exists a non-singular matrix :math:`\mathbf{L}` such that
  :math:`\mathbf{A} = \mathbf{L}\mathbf{L}^\top` (Cholesky decomposition).

.. code-block:: python

   # Checking positive definiteness three ways
   import numpy as np
   np.random.seed(42)

   # Construct a positive definite matrix from random data
   M = np.random.randn(4, 4)
   A = M.T @ M + 0.1 * np.eye(4)       # guaranteed PD

   # Method 1: eigenvalues
   eigenvalues = np.linalg.eigvalsh(A)
   print("Eigenvalues:", eigenvalues)
   print("All positive:", np.all(eigenvalues > 0))

   # Method 2: Cholesky (succeeds only if PD)
   L = np.linalg.cholesky(A)
   print("Cholesky succeeded: matrix is PD")

   # Method 3: quadratic form with random vectors
   x = np.random.randn(4)
   print("x^T A x =", x @ A @ x, " (should be > 0)")

.. admonition:: Real-World Connection

   In maximum likelihood estimation, the negative Hessian of the
   log-likelihood at the MLE (the observed information matrix) should be
   positive definite. If it is not, the MLE may be a saddle point or the
   model may be misspecified. Checking positive definiteness of the
   information matrix is therefore a standard diagnostic after fitting a model.


Eigenvalues and Eigenvectors
==============================

Definition and Properties
-------------------------

Here is the key idea: an eigenvector of a matrix is a direction that the
matrix simply stretches (or shrinks) without rotating. The eigenvalue tells
you the stretching factor. Formally, a scalar :math:`\lambda` and a non-zero
vector :math:`\mathbf{v}` satisfying

.. math::

   \mathbf{A}\,\mathbf{v} = \lambda\,\mathbf{v}

are called an **eigenvalue** and **eigenvector** of :math:`\mathbf{A}`,
respectively. The eigenvalues are the roots of the **characteristic
polynomial**:

.. math::

   \det(\mathbf{A} - \lambda\,\mathbf{I}) = 0.

For a real symmetric matrix :math:`\mathbf{A} \in \mathbb{R}^{n \times n}`:

1. All eigenvalues are real.
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3. :math:`\mathbf{A}` can be diagonalized by an orthogonal matrix.

These properties are not just mathematical niceties. Because covariance
matrices and Hessians are symmetric, we always get real eigenvalues, and the
eigenvectors give us an orthogonal coordinate system aligned with the
principal axes of the distribution or the curvature directions of the
log-likelihood.

Spectral properties linked to the trace and determinant:

.. math::

   \operatorname{tr}(\mathbf{A}) &= \sum_{i=1}^{n} \lambda_i, \\
   \det(\mathbf{A}) &= \prod_{i=1}^{n} \lambda_i.

.. admonition:: Intuition

   The trace tells you the "total stretch" of a matrix -- the sum of all
   eigenvalues. The determinant tells you the "volume stretch" -- the product
   of all eigenvalues. If even one eigenvalue is zero, the determinant is zero
   and the matrix is singular. This is why a singular covariance matrix means
   your data live in a lower-dimensional subspace.

.. code-block:: python

   # Eigenvalues, trace, and determinant
   import numpy as np
   np.random.seed(42)

   M = np.random.randn(4, 4)
   A = M.T @ M                          # symmetric PSD matrix

   eigenvalues = np.linalg.eigvalsh(A)
   print("Eigenvalues:", eigenvalues)
   print(f"Sum of eigenvalues:  {eigenvalues.sum():.6f}")
   print(f"Trace of A:          {np.trace(A):.6f}")
   print(f"Prod of eigenvalues: {eigenvalues.prod():.6f}")
   print(f"Determinant of A:    {np.linalg.det(A):.6f}")


Matrix Decompositions
======================

Eigendecomposition (Spectral Decomposition)
--------------------------------------------

For a real symmetric matrix :math:`\mathbf{A}`, the eigendecomposition is:

.. math::

   \mathbf{A}
   = \mathbf{Q}\,\boldsymbol{\Lambda}\,\mathbf{Q}^\top
   = \sum_{i=1}^{n} \lambda_i\,\mathbf{q}_i\,\mathbf{q}_i^\top,

where :math:`\mathbf{Q} = (\mathbf{q}_1, \ldots, \mathbf{q}_n)` is orthogonal
(:math:`\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}`) and
:math:`\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)`.

This decomposition is central to understanding covariance matrices and the
geometry of multivariate normal distributions. Think of it this way: the
eigendecomposition tells you the principal axes (directions :math:`\mathbf{q}_i`)
and the variances along each axis (eigenvalues :math:`\lambda_i`). The
probability contours of a multivariate normal are ellipsoids whose axes are
precisely the eigenvectors of the covariance matrix.

.. code-block:: python

   # Eigendecomposition and reconstruction
   import numpy as np
   np.random.seed(42)

   # Create a symmetric matrix (a covariance matrix)
   M = np.random.randn(4, 4)
   A = M.T @ M

   # Eigendecomposition
   eigenvalues, Q = np.linalg.eigh(A)   # eigh for symmetric matrices
   Lambda = np.diag(eigenvalues)

   # Reconstruct: A = Q Lambda Q^T
   A_reconstructed = Q @ Lambda @ Q.T
   print("Reconstruction matches:", np.allclose(A, A_reconstructed))

   # Verify orthogonality: Q^T Q = I
   print("Q is orthogonal:", np.allclose(Q.T @ Q, np.eye(4)))

   # The outer-product form: sum of lambda_i * q_i * q_i^T
   A_outer = sum(eigenvalues[i] * np.outer(Q[:, i], Q[:, i])
                 for i in range(4))
   print("Outer-product form matches:", np.allclose(A, A_outer))

.. admonition:: Real-World Connection

   Hessian matrices of the log-likelihood are symmetric, so
   eigendecomposition tells us about the curvature of the log-likelihood
   surface. Large eigenvalues of :math:`-\mathbf{H}` (the information matrix)
   mean the likelihood is sharply peaked in that direction -- the data are
   very informative about that parameter combination. Small eigenvalues mean
   the likelihood is flat and the corresponding parameter combination is
   poorly identified.

Cholesky Decomposition
-----------------------

Every positive definite matrix :math:`\mathbf{A}` can be written as:

.. math::

   \mathbf{A} = \mathbf{L}\,\mathbf{L}^\top,

where :math:`\mathbf{L}` is a lower triangular matrix with strictly positive
diagonal entries. The Cholesky factor is unique.

In practice, the Cholesky decomposition is the preferred method for:

- Solving linear systems involving covariance matrices.
- Computing log-determinants:
  :math:`\log\det(\mathbf{A}) = 2\sum_i \log L_{ii}`.
- Sampling from multivariate normal distributions.

Why is Cholesky preferred over a general inverse? It exploits the positive
definite structure to be about twice as fast as LU decomposition and is
numerically very stable. Let's see all three applications in action:

.. code-block:: python

   # Cholesky decomposition: solve, log-det, and sample
   import numpy as np
   np.random.seed(42)

   # A positive definite covariance matrix
   M = np.random.randn(3, 3)
   Sigma = M.T @ M + 0.5 * np.eye(3)

   L = np.linalg.cholesky(Sigma)
   print("Cholesky factor L:\n", L)

   # 1. Log-determinant via Cholesky
   logdet_chol = 2.0 * np.sum(np.log(np.diag(L)))
   logdet_direct = np.log(np.linalg.det(Sigma))
   print(f"\nlog|Sigma| via Cholesky:  {logdet_chol:.6f}")
   print(f"log|Sigma| via det:       {logdet_direct:.6f}")

   # 2. Solve Sigma @ x = b  via Cholesky (forward + back substitution)
   from scipy.linalg import cho_solve, cho_factor
   b = np.array([1.0, 2.0, 3.0])
   x_chol = cho_solve(cho_factor(Sigma), b)
   x_direct = np.linalg.solve(Sigma, b)
   print(f"\nSolve agrees: {np.allclose(x_chol, x_direct)}")

   # 3. Sample from N(mu, Sigma): z ~ N(0,I), then x = mu + L @ z
   mu = np.array([1.0, -1.0, 0.5])
   z = np.random.randn(3)
   sample = mu + L @ z
   print(f"\nMultivariate normal sample: {sample}")

.. admonition:: Common Pitfall

   If ``np.linalg.cholesky`` raises a ``LinAlgError``, your matrix is not
   positive definite. In practice this can happen when a covariance matrix
   estimate is computed from too few observations (or with collinear features).
   A common fix is to add a small ridge term: :math:`\mathbf{A} + \epsilon\mathbf{I}`.

Singular Value Decomposition (SVD)
-----------------------------------

Any matrix :math:`\mathbf{A} \in \mathbb{R}^{n \times m}` can be decomposed
as:

.. math::

   \mathbf{A} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top,

where :math:`\mathbf{U} \in \mathbb{R}^{n \times n}` and
:math:`\mathbf{V} \in \mathbb{R}^{m \times m}` are orthogonal, and
:math:`\boldsymbol{\Sigma} \in \mathbb{R}^{n \times m}` is diagonal with
non-negative entries :math:`\sigma_1 \geq \sigma_2 \geq \cdots \geq 0` (the
**singular values**).

For a symmetric positive semi-definite matrix, the singular values equal the
eigenvalues, and :math:`\mathbf{U} = \mathbf{V} = \mathbf{Q}`.

The SVD provides the best low-rank approximation to a matrix (Eckart--Young
theorem), which is useful in dimensionality reduction and principal component
analysis.

What makes the SVD so versatile is that it works for *any* matrix -- not just
square or symmetric ones. When you run PCA on a data matrix, you are
implicitly computing an SVD.

.. code-block:: python

   # SVD: decompose, verify, and low-rank approximation
   import numpy as np
   np.random.seed(42)

   A = np.random.randn(5, 3)
   U, s, Vt = np.linalg.svd(A, full_matrices=False)

   print("Singular values:", s)
   print("Reconstruction matches:",
         np.allclose(A, U @ np.diag(s) @ Vt))

   # Best rank-1 approximation
   A_rank1 = s[0] * np.outer(U[:, 0], Vt[0, :])
   error_full = np.linalg.norm(A)
   error_rank1 = np.linalg.norm(A - A_rank1)
   print(f"\n||A||         = {error_full:.4f}")
   print(f"||A - A_1||   = {error_rank1:.4f}")
   print(f"Captured fraction: {1 - error_rank1/error_full:.2%}")

.. admonition:: Real-World Connection

   In high-dimensional settings (e.g., genomics with many more variables than
   observations), the design matrix :math:`\mathbf{X}` is rank-deficient and
   the usual :math:`(\mathbf{X}^\top\mathbf{X})^{-1}` does not exist. The
   SVD of :math:`\mathbf{X}` leads directly to the Moore--Penrose
   pseudoinverse and to ridge regression, both of which are regularized
   alternatives to ordinary least squares.


Matrix Calculus Essentials
===========================

In likelihood-based inference we routinely differentiate scalar-valued
functions of vectors or matrices. We adopt the **numerator layout** convention.

If this section feels more abstract than the previous ones, that is because
matrix calculus is genuinely the most algebra-intensive part of multivariate
MLE derivations. The good news is that a handful of identities cover nearly
every case you will encounter in practice.

Derivative of a Scalar with Respect to a Vector
-------------------------------------------------

If :math:`f : \mathbb{R}^n \to \mathbb{R}`, its gradient is the column vector:

.. math::

   \frac{\partial f}{\partial \mathbf{x}}
   = \nabla_{\mathbf{x}} f
   = \begin{pmatrix}
   \partial f / \partial x_1 \\
   \partial f / \partial x_2 \\
   \vdots \\
   \partial f / \partial x_n
   \end{pmatrix}.

Common results:

.. math::

   \frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^\top \mathbf{x})
   &= \mathbf{a}, \\[6pt]
   \frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top \mathbf{A}\,\mathbf{x})
   &= (\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x}, \\[6pt]
   \frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top \mathbf{x})
   &= 2\,\mathbf{x}.

When :math:`\mathbf{A}` is symmetric, the quadratic form derivative simplifies
to:

.. math::

   \frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top \mathbf{A}\,\mathbf{x})
   = 2\,\mathbf{A}\,\mathbf{x}.

Let's verify these identities numerically. The idea is simple: compare the
analytical gradient with a finite-difference approximation.

.. code-block:: python

   # Numerical verification of matrix calculus identities
   import numpy as np
   np.random.seed(42)

   n = 4
   A_sym = np.random.randn(n, n)
   A_sym = A_sym + A_sym.T                 # make symmetric
   a = np.random.randn(n)
   x = np.random.randn(n)

   # Gradient of a^T x  should be  a
   def f_linear(x):
       return a @ x
   grad_numerical = np.zeros(n)
   eps = 1e-7
   for i in range(n):
       x_plus = x.copy(); x_plus[i] += eps
       grad_numerical[i] = (f_linear(x_plus) - f_linear(x)) / eps
   print("d/dx(a^T x): analytical =", a)
   print("             numerical  =", grad_numerical)

   # Gradient of x^T A x  should be  2 A x  (A symmetric)
   def f_quad(x):
       return x @ A_sym @ x
   grad_numerical_q = np.zeros(n)
   for i in range(n):
       x_plus = x.copy(); x_plus[i] += eps
       grad_numerical_q[i] = (f_quad(x_plus) - f_quad(x)) / eps
   grad_analytical = 2 * A_sym @ x
   print("\nd/dx(x^T A x): analytical =", grad_analytical)
   print("               numerical  =", grad_numerical_q)
   print("Match:", np.allclose(grad_analytical, grad_numerical_q, atol=1e-5))

Derivative of a Vector with Respect to a Vector
-------------------------------------------------

If :math:`\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m`, the **Jacobian** is the
:math:`m \times n` matrix:

.. math::

   \mathbf{J}
   = \frac{\partial \mathbf{f}}{\partial \mathbf{x}^\top}
   = \begin{pmatrix}
   \partial f_1 / \partial x_1 & \cdots & \partial f_1 / \partial x_n \\
   \vdots & \ddots & \vdots \\
   \partial f_m / \partial x_1 & \cdots & \partial f_m / \partial x_n
   \end{pmatrix}.

An important special case:

.. math::

   \frac{\partial}{\partial \mathbf{x}} (\mathbf{A}\,\mathbf{x})
   = \mathbf{A}.

The **Hessian** of a scalar function :math:`f` is the matrix of second
derivatives:

.. math::

   \mathbf{H}
   = \frac{\partial^2 f}{\partial \mathbf{x}\,\partial \mathbf{x}^\top}
   = \begin{pmatrix}
   \partial^2 f / \partial x_1^2 & \cdots & \partial^2 f / \partial x_1 \partial x_n \\
   \vdots & \ddots & \vdots \\
   \partial^2 f / \partial x_n \partial x_1 & \cdots & \partial^2 f / \partial x_n^2
   \end{pmatrix}.

For the quadratic form :math:`f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A}\,\mathbf{x}`
with symmetric :math:`\mathbf{A}`:

.. math::

   \mathbf{H} = 2\,\mathbf{A}.

Notice what this means: the Hessian of a quadratic form is a constant matrix.
This is why quadratic approximations to the log-likelihood (Taylor expansions)
are so tractable -- the curvature is captured entirely by the Hessian at the
expansion point.

.. code-block:: python

   # Computing the Hessian numerically for a quadratic form
   import numpy as np
   np.random.seed(42)

   n = 3
   A_sym = np.random.randn(n, n)
   A_sym = A_sym + A_sym.T

   def f_quad(x):
       return x @ A_sym @ x

   # Numerical Hessian via finite differences
   x0 = np.random.randn(n)
   eps = 1e-5
   H_num = np.zeros((n, n))
   for i in range(n):
       for j in range(n):
           xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
           xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
           xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
           xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
           H_num[i, j] = (f_quad(xpp) - f_quad(xpm)
                          - f_quad(xmp) + f_quad(xmm)) / (4*eps**2)

   print("Analytical Hessian = 2A:\n", 2 * A_sym)
   print("\nNumerical Hessian:\n", H_num)
   print("Match:", np.allclose(2 * A_sym, H_num, atol=1e-4))


Key Matrix Calculus Identities
================================

The following identities appear repeatedly when deriving MLEs for multivariate
normal and related models. We present each one with enough context so you can
see not just *what* the identity is, but *where* it comes from and *when* you
will need it.

Derivatives Involving the Log-Determinant
------------------------------------------

For a non-singular matrix :math:`\mathbf{A}` (where the derivative is taken
with respect to the entries of :math:`\mathbf{A}`):

.. math::

   \frac{\partial}{\partial \mathbf{A}} \log|\mathbf{A}|
   = \mathbf{A}^{-\top}
   = (\mathbf{A}^{-1})^\top.

When :math:`\mathbf{A}` is symmetric, this simplifies to:

.. math::

   \frac{\partial}{\partial \mathbf{A}} \log|\mathbf{A}|
   = \mathbf{A}^{-1}.

This identity is used every time we differentiate the multivariate normal
log-likelihood with respect to the covariance matrix. It is worth memorizing:
"the derivative of the log-determinant is the inverse."

.. code-block:: python

   # Numerical check: d/dA log|A| = A^{-1} for symmetric A
   import numpy as np
   np.random.seed(42)

   n = 3
   M = np.random.randn(n, n)
   A = M.T @ M + np.eye(n)              # symmetric PD

   eps = 1e-7
   grad_num = np.zeros((n, n))
   logdet_A = np.log(np.linalg.det(A))
   for i in range(n):
       for j in range(n):
           A_pert = A.copy()
           A_pert[i, j] += eps
           grad_num[i, j] = (np.log(np.linalg.det(A_pert)) - logdet_A) / eps

   print("Analytical (A^{-1}):\n", np.linalg.inv(A))
   print("\nNumerical gradient:\n", grad_num)
   print("Match:", np.allclose(np.linalg.inv(A), grad_num, atol=1e-5))

Derivatives Involving the Trace
--------------------------------

.. math::

   \frac{\partial}{\partial \mathbf{A}} \operatorname{tr}(\mathbf{A}\mathbf{B})
   &= \mathbf{B}^\top, \\[6pt]
   \frac{\partial}{\partial \mathbf{A}} \operatorname{tr}(\mathbf{A}^\top \mathbf{B})
   &= \mathbf{B}, \\[6pt]
   \frac{\partial}{\partial \mathbf{A}} \operatorname{tr}(\mathbf{A}^{-1}\mathbf{B})
   &= -\mathbf{A}^{-\top}\mathbf{B}^\top\mathbf{A}^{-\top}.

When :math:`\mathbf{A}` is symmetric, the last identity becomes:

.. math::

   \frac{\partial}{\partial \mathbf{A}} \operatorname{tr}(\mathbf{A}^{-1}\mathbf{B})
   = -\mathbf{A}^{-1}\mathbf{B}\,\mathbf{A}^{-1}.

These identities may look abstract, but they are precisely the tools you need
to differentiate the quadratic form
:math:`\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{S})` that appears in
the multivariate normal log-likelihood.

Combined Identity for Multivariate Normal
-------------------------------------------

In the multivariate normal log-likelihood, we encounter
:math:`-\tfrac{1}{2}\log|\boldsymbol{\Sigma}| - \tfrac{1}{2}\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{S})`
where :math:`\mathbf{S}` is the sample covariance matrix. Differentiating with
respect to :math:`\boldsymbol{\Sigma}`:

.. math::

   \frac{\partial}{\partial \boldsymbol{\Sigma}}
   \left[
   -\tfrac{1}{2}\log|\boldsymbol{\Sigma}|
   - \tfrac{1}{2}\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{S})
   \right]
   = -\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}
   + \tfrac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{S}\,\boldsymbol{\Sigma}^{-1}.

Setting this to zero yields the MLE :math:`\hat{\boldsymbol{\Sigma}} = \mathbf{S}`.

This is one of the most satisfying results in multivariate statistics: the
maximum likelihood estimator of the covariance matrix is simply the sample
covariance matrix. Let's verify this numerically by showing that the gradient
vanishes when :math:`\boldsymbol{\Sigma} = \mathbf{S}`.

.. code-block:: python

   # MLE for the covariance matrix of a multivariate normal
   import numpy as np
   np.random.seed(42)

   # Generate data from a known multivariate normal
   n, p = 200, 3
   true_Sigma = np.array([[2.0, 0.5, 0.3],
                          [0.5, 1.0, 0.2],
                          [0.3, 0.2, 1.5]])
   mu = np.zeros(p)
   X = np.random.multivariate_normal(mu, true_Sigma, size=n)

   # Sample covariance (the MLE)
   S = (X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / n

   # Evaluate the gradient at Sigma = S
   S_inv = np.linalg.inv(S)
   grad = -0.5 * S_inv + 0.5 * S_inv @ S @ S_inv
   print("Gradient at MLE (should be ~0):\n", grad)
   print("Max absolute gradient entry:", np.abs(grad).max())

.. admonition:: Real-World Connection

   This derivation generalizes to structured covariance models. In factor
   analysis, the covariance has the form
   :math:`\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top + \boldsymbol{\Psi}`,
   so the derivative above is the starting point, but the chain rule must be
   applied further to find the MLE for :math:`\mathbf{L}` and
   :math:`\boldsymbol{\Psi}` separately.


Quadratic Forms and Their Optimization
========================================

A quadratic form in :math:`\mathbf{x} \in \mathbb{R}^n` with symmetric matrix
:math:`\mathbf{A}` is:

.. math::

   Q(\mathbf{x}) = \mathbf{x}^\top \mathbf{A}\,\mathbf{x}
   = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}\,x_i\,x_j.

You can think of a quadratic form as the matrix generalization of
:math:`ax^2`. Just as the sign of :math:`a` tells you whether the parabola
opens up or down, the eigenvalues of :math:`\mathbf{A}` tell you whether the
surface curves up or down in each direction.

The general quadratic function including linear and constant terms is:

.. math::

   f(\mathbf{x})
   = \tfrac{1}{2}\,\mathbf{x}^\top \mathbf{A}\,\mathbf{x}
   - \mathbf{b}^\top \mathbf{x}
   + c.

Taking the gradient and setting it to zero:

.. math::

   \nabla f = \mathbf{A}\,\mathbf{x} - \mathbf{b} = \mathbf{0}
   \quad \Longrightarrow \quad
   \mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}.

The Hessian is :math:`\mathbf{A}`. When :math:`\mathbf{A}` is positive
definite, the critical point :math:`\mathbf{x}^*` is a global minimum. When
:math:`\mathbf{A}` is negative definite, it is a global maximum.

.. code-block:: python

   # Minimizing a quadratic function
   import numpy as np
   np.random.seed(42)

   n = 3
   M = np.random.randn(n, n)
   A = M.T @ M + np.eye(n)              # positive definite
   b = np.random.randn(n)
   c = 1.0

   def f(x):
       return 0.5 * x @ A @ x - b @ x + c

   # Analytical minimizer
   x_star = np.linalg.solve(A, b)
   print("Optimal x*:", x_star)
   print("f(x*) =", f(x_star))

   # Verify gradient is zero at x*
   eps = 1e-7
   grad = np.zeros(n)
   for i in range(n):
       x_plus = x_star.copy(); x_plus[i] += eps
       grad[i] = (f(x_plus) - f(x_star)) / eps
   print("Gradient at x* (should be ~0):", grad)

**Rayleigh quotient.** For symmetric :math:`\mathbf{A}`, the Rayleigh quotient

.. math::

   R(\mathbf{x}) = \frac{\mathbf{x}^\top \mathbf{A}\,\mathbf{x}}
                         {\mathbf{x}^\top \mathbf{x}}

satisfies :math:`\lambda_{\min} \leq R(\mathbf{x}) \leq \lambda_{\max}`, with
the bounds achieved by the corresponding eigenvectors.

**Constrained optimization.** The problem of maximizing
:math:`\mathbf{x}^\top \mathbf{A}\,\mathbf{x}` subject to
:math:`\mathbf{x}^\top \mathbf{x} = 1` is solved by the eigenvector
corresponding to the largest eigenvalue of :math:`\mathbf{A}`. This is the
mathematical foundation of principal component analysis.

Let's see the Rayleigh quotient in action and verify that it is bounded by
the extreme eigenvalues:

.. code-block:: python

   # Rayleigh quotient bounds and PCA connection
   import numpy as np
   np.random.seed(42)

   M = np.random.randn(4, 4)
   A = M.T @ M                          # symmetric PSD

   eigenvalues, eigenvectors = np.linalg.eigh(A)
   lam_min, lam_max = eigenvalues[0], eigenvalues[-1]

   # Evaluate Rayleigh quotient for many random directions
   rayleigh_values = []
   for _ in range(10000):
       x = np.random.randn(4)
       R = (x @ A @ x) / (x @ x)
       rayleigh_values.append(R)

   print(f"lambda_min = {lam_min:.4f}")
   print(f"lambda_max = {lam_max:.4f}")
   print(f"min R(x) observed = {min(rayleigh_values):.4f}")
   print(f"max R(x) observed = {max(rayleigh_values):.4f}")

   # Verify: R(q_max) = lambda_max
   q_max = eigenvectors[:, -1]
   R_at_qmax = (q_max @ A @ q_max) / (q_max @ q_max)
   print(f"\nR(q_max) = {R_at_qmax:.4f}  (should equal lambda_max)")
