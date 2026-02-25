.. _ch1_probability:

====================================
Chapter 1: Probability Basics
====================================

Probability is the language we use to talk about uncertainty.  Before we can
understand likelihoods, estimators, or any of the machinery of statistical
inference, we need a solid, careful foundation in how probability works.  This
chapter builds that foundation from the ground up, assuming only basic
familiarity with sets and arithmetic.

Two running examples will follow us through the chapter.  The first is
**medical screening**: a disease that affects 1 % of the population, detected
by a test with 95 % sensitivity and 97 % specificity.  The second is
**factory quality control**: two assembly lines with different defect rates.
Every new concept will be applied to one or both of these scenarios so you can
see the formulas doing real work.

By the end you will be able to compute conditional probabilities, apply Bayes'
theorem to real screening problems, and understand why independence is the
assumption that makes likelihood-based inference possible.

.. code-block:: python

   # ------------------------------------------------------------------
   # Setup: parameters for our two running examples
   # ------------------------------------------------------------------
   import numpy as np

   np.random.seed(42)

   # --- Medical screening scenario ---
   prevalence    = 0.01   # P(Disease)
   sensitivity   = 0.95   # P(Test+ | Disease)      — true positive rate
   specificity   = 0.97   # P(Test- | No Disease)    — true negative rate
   false_pos_rate = 1 - specificity  # P(Test+ | No Disease) = 0.03

   # --- Factory quality-control scenario ---
   P_line_A       = 0.60   # Line A produces 60 % of output
   P_line_B       = 0.40   # Line B produces 40 % of output
   defect_rate_A  = 0.02   # 2 % defect rate on Line A
   defect_rate_B  = 0.05   # 5 % defect rate on Line B

   print("=== Medical screening parameters ===")
   print(f"  Prevalence  P(D)        = {prevalence}")
   print(f"  Sensitivity P(T+|D)     = {sensitivity}")
   print(f"  Specificity P(T-|D^c)   = {specificity}")
   print(f"  False-pos   P(T+|D^c)   = {false_pos_rate}")
   print()
   print("=== Factory QC parameters ===")
   print(f"  P(Line A) = {P_line_A},  P(Defect|A) = {defect_rate_A}")
   print(f"  P(Line B) = {P_line_B},  P(Defect|B) = {defect_rate_B}")

.. contents:: Chapter Contents
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here


1.1 Sample Spaces and Events
=============================

You are about to roll two dice in a board game.  Before the dice leave your
hand you already know, intuitively, what *could* happen --- every pair of faces
from (1,1) to (6,6).  Probability theory demands that we write this intuition
down precisely, because no number can be attached to an outcome until the full
menu of possibilities is on the table.

**Sample space.**  The **sample space**, usually written :math:`\Omega`, is the
set of all possible outcomes of a random experiment.  Every probability question
begins by specifying :math:`\Omega`.

*Example 1 (coin toss).*  Toss a fair coin once.

.. math::

   \Omega = \{H, T\}

*Example 2 (two dice).*  Roll two distinguishable six-sided dice.

.. math::

   \Omega = \{(i, j) : i, j \in \{1,2,3,4,5,6\}\}

This sample space has :math:`6 \times 6 = 36` outcomes.

*Example 3 (continuous).*  Measure the lifetime (in hours) of a light bulb.

.. math::

   \Omega = [0, \infty)

Here the sample space is uncountably infinite --- there are infinitely many
possible lifetimes.

.. code-block:: python

   # Enumerate the sample space for two six-sided dice
   Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
   print(f"|Omega| = {len(Omega)}")
   print(f"First 6 outcomes: {Omega[:6]}")
   print(f"Last  6 outcomes: {Omega[-6:]}")

**Events.**  An **event** is any subset :math:`A \subseteq \Omega`.  We say the
event *occurs* when the outcome of the experiment falls inside :math:`A`.

*Example.*  In the two-dice experiment, the event "the sum is 7" is

.. math::

   A = \{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\}

.. code-block:: python

   # Event "sum is 7" — enumerate and compute its probability
   A_sum7 = [(i, j) for (i, j) in Omega if i + j == 7]
   P_sum7 = len(A_sum7) / len(Omega)

   print(f"A = {A_sum7}")
   print(f"|A| = {len(A_sum7)}")
   print(f"P(sum = 7) = |A|/|Omega| = {len(A_sum7)}/{len(Omega)} = {P_sum7:.4f}")

.. code-block:: python

   # Verify P(sum=7) by simulation
   rolls = np.random.randint(1, 7, size=(1_000_000, 2))
   sums = rolls.sum(axis=1)
   print(f"Formula:   P(sum = 7) = {6/36:.4f}")
   print(f"Simulated: P(sum = 7) = {(sums == 7).mean():.4f}")

*Medical screening — the sample space.*  For our running example, a single
patient can fall into one of four categories:

.. code-block:: python

   # Sample space for one patient in the screening scenario
   Omega_screening = [
       ("Disease",    "Test+"),
       ("Disease",    "Test-"),
       ("No Disease", "Test+"),
       ("No Disease", "Test-"),
   ]
   for outcome in Omega_screening:
       print(outcome)

The collection of all events we wish to assign probabilities to is called a
**sigma-algebra** (or :math:`\sigma`-algebra), denoted :math:`\mathcal{F}`.
For finite sample spaces we usually take :math:`\mathcal{F}` to be the power
set of :math:`\Omega` (every subset is an event).  For continuous sample
spaces, some care is needed to avoid paradoxes; the standard choice is the
**Borel** :math:`\sigma`-algebra, which contains all intervals and their
countable unions, intersections, and complements.


1.2 The Kolmogorov Axioms
==========================

Imagine two friends arguing about a coin toss.  One says "heads is 60 %
likely," the other says "no, 40 %."  How do we decide what rules
*any* probability assignment must follow?  In 1933, Andrey Kolmogorov proposed
three axioms --- deliberately minimal, yet strong enough to derive every
formula in this book.

A **probability measure** is a function :math:`P : \mathcal{F} \to \mathbb{R}`
satisfying:

**Axiom 1 (Non-negativity).**

.. math::

   P(A) \geq 0 \quad \text{for every event } A \in \mathcal{F}.

*Plain English:*  Probabilities are never negative.  An event either can happen
(positive probability) or cannot happen (zero probability); it never has a
"negative chance."

**Axiom 2 (Normalization).**

.. math::

   P(\Omega) = 1.

*Plain English:*  Something must happen.  The probability that *some* outcome
in the sample space occurs is exactly 1 (certainty).

**Axiom 3 (Countable Additivity).**  If :math:`A_1, A_2, \dots` are pairwise
disjoint events (meaning :math:`A_i \cap A_j = \emptyset` for :math:`i \neq j`),
then

.. math::

   P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i).

*Plain English:*  If several events can never happen at the same time
(mutually exclusive), the probability that at least one of them happens is the
sum of their individual probabilities.

Together, the triple :math:`(\Omega, \mathcal{F}, P)` is called a
**probability space**.

.. code-block:: python

   # Verify the three axioms on the two-dice sample space
   Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]

   # Assign uniform probability: P({omega}) = 1/36 for each outcome
   P = {omega: 1/36 for omega in Omega}

   # Axiom 1: all probabilities >= 0
   assert all(p >= 0 for p in P.values()), "Axiom 1 violated!"

   # Axiom 2: P(Omega) = 1
   P_Omega = sum(P.values())
   print(f"Axiom 2 check: P(Omega) = {P_Omega:.4f}  (should be 1.0)")

   # Axiom 3: disjoint events — "sum <= 4" and "sum >= 10"
   A = {omega for omega in Omega if sum(omega) <= 4}
   B = {omega for omega in Omega if sum(omega) >= 10}
   assert A & B == set(), "A and B are not disjoint!"

   P_A = sum(P[omega] for omega in A)
   P_B = sum(P[omega] for omega in B)
   P_A_union_B = sum(P[omega] for omega in A | B)
   print(f"P(A) = {P_A:.4f},  P(B) = {P_B:.4f}")
   print(f"P(A) + P(B) = {P_A + P_B:.4f}")
   print(f"P(A ∪ B)     = {P_A_union_B:.4f}")
   print(f"Axiom 3 check: {abs(P_A + P_B - P_A_union_B) < 1e-12}")

.. admonition:: Intuition

   Why only three axioms?  Because they are *exactly* what you need to ensure
   probabilities behave like proportions: they are non-negative, they add up to
   one in total, and you can break disjoint pieces apart and recombine them
   freely.  Every "rule" you learned in an introductory course --- complement
   rules, inclusion-exclusion, Bayes' theorem --- is a logical consequence of
   these three statements.


1.2.1 Consequences of the Axioms
----------------------------------

From these three axioms we can derive every probability rule you have ever
seen.  Let us prove the most important ones.

**Complement Rule.**  For any event :math:`A`,

.. math::

   P(A^c) = 1 - P(A).

*Proof.*  Since :math:`A` and :math:`A^c` are disjoint and
:math:`A \cup A^c = \Omega`, Axiom 3 gives

.. math::

   P(\Omega) = P(A) + P(A^c).

By Axiom 2, :math:`P(\Omega) = 1`, so :math:`P(A^c) = 1 - P(A)`. :math:`\square`

This is surprisingly useful in practice: sometimes it is much easier to compute
the probability that something does *not* happen and then subtract from one.

.. code-block:: python

   # Complement rule with our medical screening numbers
   # P(No Disease) = 1 - P(Disease)
   P_D = prevalence          # 0.01
   P_Dc = 1 - P_D            # P(D^c)
   print(f"P(Disease)    = {P_D}")
   print(f"P(No Disease) = 1 - P(D) = 1 - {P_D} = {P_Dc}")

**Probability of the Empty Set.**

.. math::

   P(\emptyset) = 0.

*Proof.*  :math:`\emptyset = \Omega^c`, so :math:`P(\emptyset) = 1 - P(\Omega) = 1 - 1 = 0`.
:math:`\square`

**Monotonicity.**  If :math:`A \subseteq B`, then :math:`P(A) \leq P(B)`.

*Proof.*  Write :math:`B = A \cup (B \setminus A)`.  These two sets are
disjoint, so by Axiom 3,

.. math::

   P(B) = P(A) + P(B \setminus A) \geq P(A)

where the inequality uses Axiom 1 (:math:`P(B \setminus A) \geq 0`).
:math:`\square`

**Inclusion--Exclusion (two events).**  For any events :math:`A` and :math:`B`,

.. math::

   P(A \cup B) = P(A) + P(B) - P(A \cap B).

Here is the key idea: when you add :math:`P(A)` and :math:`P(B)`, the overlap
:math:`A \cap B` gets counted twice, so you subtract it once to correct.

*Proof.*  Write the union as three disjoint pieces:

.. math::

   A \cup B = (A \setminus B) \;\cup\; (A \cap B) \;\cup\; (B \setminus A).

By Axiom 3,

.. math::

   P(A \cup B) = P(A \setminus B) + P(A \cap B) + P(B \setminus A).

Also :math:`A = (A \setminus B) \cup (A \cap B)` (disjoint), so
:math:`P(A) = P(A \setminus B) + P(A \cap B)`, giving
:math:`P(A \setminus B) = P(A) - P(A \cap B)`.  Similarly
:math:`P(B \setminus A) = P(B) - P(A \cap B)`.  Substituting,

.. math::

   P(A \cup B) &= [P(A) - P(A \cap B)] + P(A \cap B) + [P(B) - P(A \cap B)] \\
               &= P(A) + P(B) - P(A \cap B). \quad \square

.. code-block:: python

   # Inclusion-exclusion with the factory scenario
   # A = "item from Line A", B = "item is defective"
   # These are NOT mutually exclusive — an item can be both from A and defective.
   #
   # P(from A ∪ defective) = P(from A) + P(defective) - P(from A ∩ defective)

   P_defect = defect_rate_A * P_line_A + defect_rate_B * P_line_B
   P_A_and_defect = defect_rate_A * P_line_A  # P(Defect ∩ Line A)

   P_A_or_defect = P_line_A + P_defect - P_A_and_defect

   print(f"P(Line A)            = {P_line_A:.4f}")
   print(f"P(Defective)         = {P_defect:.4f}")
   print(f"P(Line A ∩ Defective) = {P_A_and_defect:.4f}")
   print(f"P(Line A ∪ Defective) = P(A) + P(Def) - P(A∩Def)")
   print(f"                      = {P_line_A} + {P_defect:.4f} - {P_A_and_defect:.4f}")
   print(f"                      = {P_A_or_defect:.4f}")

**General Inclusion--Exclusion.**  For events :math:`A_1, \dots, A_n`,

.. math::

   P\!\left(\bigcup_{i=1}^{n} A_i\right)
   = \sum_{i} P(A_i)
     - \sum_{i < j} P(A_i \cap A_j)
     + \sum_{i < j < k} P(A_i \cap A_j \cap A_k)
     - \cdots
     + (-1)^{n+1} P(A_1 \cap \cdots \cap A_n).

The proof proceeds by induction using the two-event formula.


1.3 Conditional Probability
=============================

A medical lab just told you your test was positive.  Should you panic?  The
answer depends on what "positive" really means for *your* situation --- your
age, the disease prevalence in your demographic, and the test's error rates.
To figure out what a positive result actually implies, we need a way to update
probabilities when we learn new information.  That tool is **conditional
probability**.

**Definition.**  The **conditional probability** of :math:`A` given :math:`B`,
written :math:`P(A \mid B)`, is defined when :math:`P(B) > 0` by

.. math::

   P(A \mid B) = \frac{P(A \cap B)}{P(B)}.

*Plain English:*  To find the probability of :math:`A` given that :math:`B`
has occurred, we look only at outcomes that are in :math:`B` and ask what
fraction of them are also in :math:`A`.

Think of it this way: conditioning is like putting on a pair of glasses that
filters out all outcomes inconsistent with the new information.  You then
re-normalize so the remaining outcomes add up to one.

*Example.*  Fair die.  Let :math:`A = \{6\}`, :math:`B = \{2, 4, 6\}`.

.. math::

   P(A \mid B) = \frac{P(\{6\})}{P(\{2,4,6\})} = \frac{1/6}{3/6} = \frac{1}{3}.

.. code-block:: python

   # Conditional probability: P(A|B) = P(A ∩ B) / P(B)
   # Die example
   Omega_die = {1, 2, 3, 4, 5, 6}
   A = {6}
   B = {2, 4, 6}

   P_A_and_B = len(A & B) / len(Omega_die)     # P(A ∩ B) = 1/6
   P_B       = len(B) / len(Omega_die)          # P(B)     = 3/6
   P_A_given_B = P_A_and_B / P_B               # P(A|B)

   print(f"P(A ∩ B) = {P_A_and_B:.4f}")
   print(f"P(B)     = {P_B:.4f}")
   print(f"P(A|B)   = P(A∩B)/P(B) = {P_A_and_B:.4f}/{P_B:.4f} = {P_A_given_B:.4f}")

Now let us apply the same formula to our medical screening scenario.  We know
the test's sensitivity --- :math:`P(T^+ \mid D) = 0.95` --- but that *is*
already a conditional probability.  We can also ask the reverse question:
among all people who test positive, what fraction truly has the disease?
That requires Bayes' theorem, which we derive in Section 1.5.  First, though,
we need two building blocks: the multiplication rule and the law of total
probability.

.. code-block:: python

   # Conditional probability in the screening context
   # We KNOW these conditional probabilities from clinical trials:
   P_Tpos_given_D  = sensitivity        # P(T+ | D)  = 0.95
   P_Tpos_given_Dc = false_pos_rate     # P(T+ | D^c) = 0.03
   P_D             = prevalence          # P(D) = 0.01
   P_Dc            = 1 - P_D             # P(D^c) = 0.99

   # The joint probabilities follow from the definition:
   # P(T+ ∩ D)  = P(T+|D)  * P(D)
   P_Tpos_and_D  = P_Tpos_given_D * P_D
   # P(T+ ∩ D^c) = P(T+|D^c) * P(D^c)
   P_Tpos_and_Dc = P_Tpos_given_Dc * P_Dc

   print(f"P(T+ ∩ D)   = P(T+|D)·P(D)     = {P_Tpos_given_D}×{P_D}   = {P_Tpos_and_D:.4f}")
   print(f"P(T+ ∩ D^c) = P(T+|D^c)·P(D^c) = {P_Tpos_given_Dc}×{P_Dc} = {P_Tpos_and_Dc:.4f}")


1.3.1 The Multiplication Rule
-------------------------------

Rearranging the definition of conditional probability gives the
**multiplication rule**:

.. math::

   P(A \cap B) = P(A \mid B)\,P(B).

By symmetry we also have :math:`P(A \cap B) = P(B \mid A)\,P(A)`.

.. code-block:: python

   # Multiplication rule: compute joint probabilities for the factory
   # P(Defective ∩ Line A) = P(Defective | Line A) * P(Line A)
   P_Def_and_A = defect_rate_A * P_line_A
   P_Def_and_B = defect_rate_B * P_line_B

   print(f"P(Defective ∩ Line A) = P(Def|A)·P(A) = {defect_rate_A}×{P_line_A} = {P_Def_and_A:.4f}")
   print(f"P(Defective ∩ Line B) = P(Def|B)·P(B) = {defect_rate_B}×{P_line_B} = {P_Def_and_B:.4f}")

For more than two events, the rule extends by chaining:

.. math::

   P(A_1 \cap A_2 \cap A_3)
   = P(A_1)\;P(A_2 \mid A_1)\;P(A_3 \mid A_1 \cap A_2).

*Derivation.*  Apply the definition twice:

.. math::

   P(A_1 \cap A_2 \cap A_3)
   &= P(A_3 \mid A_1 \cap A_2) \; P(A_1 \cap A_2) \\
   &= P(A_3 \mid A_1 \cap A_2) \; P(A_2 \mid A_1) \; P(A_1). \quad \square

The general form for :math:`n` events is

.. math::

   P\!\left(\bigcap_{i=1}^n A_i\right)
   = P(A_1) \prod_{k=2}^{n} P\!\left(A_k \;\Big|\; \bigcap_{j=1}^{k-1} A_j\right).

.. code-block:: python

   # Chain rule example: 3 patients tested sequentially
   # Suppose each patient is drawn independently with P(D)=0.01
   # P(all 3 have disease) = P(D1) * P(D2|D1) * P(D3|D1∩D2)
   # Under independence, this simplifies to the product:
   P_all_3_disease = prevalence ** 3
   print(f"P(all 3 patients have disease) = {prevalence}^3 = {P_all_3_disease:.8f}")
   print(f"That is about 1 in {1/P_all_3_disease:,.0f}")


1.4 The Law of Total Probability
==================================

You are the epidemiologist who designed the screening program.  A politician
asks: "What fraction of people who take this test will get a positive result?"
You know the true-positive and false-positive rates from clinical trials, and
you know the disease prevalence.  But the positive rate depends on whether
the person is sick or healthy --- two mutually exclusive scenarios.  The
**law of total probability** tells you how to combine the scenario-specific
rates into a single overall number.

**Setup.**  Suppose :math:`B_1, B_2, \dots, B_n` form a **partition** of
:math:`\Omega`:

1. They are mutually exclusive: :math:`B_i \cap B_j = \emptyset` for
   :math:`i \neq j`.
2. They cover the sample space: :math:`B_1 \cup B_2 \cup \cdots \cup B_n = \Omega`.
3. Each :math:`P(B_i) > 0`.

**Statement.**

.. math::

   P(A) = \sum_{i=1}^{n} P(A \mid B_i)\,P(B_i).

**Derivation.**  Because the :math:`B_i` partition :math:`\Omega`, the event
:math:`A` can be split into disjoint pieces:

.. math::

   A = A \cap \Omega = A \cap (B_1 \cup \cdots \cup B_n)
     = (A \cap B_1) \cup (A \cap B_2) \cup \cdots \cup (A \cap B_n).

Since the :math:`B_i` are disjoint, so are the :math:`A \cap B_i`.  By
Axiom 3 (countable additivity),

.. math::

   P(A) = \sum_{i=1}^{n} P(A \cap B_i).

Applying the multiplication rule :math:`P(A \cap B_i) = P(A \mid B_i)\,P(B_i)` gives

.. math::

   P(A) = \sum_{i=1}^{n} P(A \mid B_i)\,P(B_i). \quad \square

**Medical screening --- computing P(Test positive).**  Let :math:`D` = has
disease, :math:`T^+` = tests positive.  The partition is
:math:`\{D, D^c\}`.

.. math::

   P(T^+)
   &= P(T^+ \mid D)\,P(D) + P(T^+ \mid D^c)\,P(D^c) \\
   &= (0.95)(0.01) + (0.03)(0.99) \\
   &= 0.0095 + 0.0297 \\
   &= 0.0392.

So about 3.92 % of the population tests positive.

.. code-block:: python

   # Law of total probability: P(T+) for our screening test
   # P(T+) = P(T+|D)·P(D) + P(T+|D^c)·P(D^c)
   P_Tpos = P_Tpos_given_D * P_D + P_Tpos_given_Dc * P_Dc

   print("Law of Total Probability — Medical Screening")
   print(f"  P(T+|D)·P(D)     = {P_Tpos_given_D}×{P_D}   = {P_Tpos_given_D * P_D:.4f}")
   print(f"  P(T+|D^c)·P(D^c) = {P_Tpos_given_Dc}×{P_Dc} = {P_Tpos_given_Dc * P_Dc:.4f}")
   print(f"  P(T+)             = {P_Tpos:.4f}")
   print(f"  About {P_Tpos*100:.2f}% of the population tests positive.")

.. code-block:: python

   # Verify by simulation: screen 1 million people
   n_people = 1_000_000
   has_disease = np.random.rand(n_people) < prevalence
   # Generate test results conditional on disease status
   test_positive = np.where(
       has_disease,
       np.random.rand(n_people) < sensitivity,      # true positives
       np.random.rand(n_people) < false_pos_rate     # false positives
   )

   P_Tpos_sim = test_positive.mean()
   print(f"Formula:   P(T+) = {P_Tpos:.4f}")
   print(f"Simulated: P(T+) = {P_Tpos_sim:.4f}  (n = {n_people:,})")

**Factory quality control --- computing P(Defective).**  The partition is
:math:`\{\text{Line A}, \text{Line B}\}`.

.. math::

   P(\text{Defect})
   &= P(\text{Defect} \mid A)\,P(A) + P(\text{Defect} \mid B)\,P(B) \\
   &= (0.02)(0.60) + (0.05)(0.40) \\
   &= 0.012 + 0.020 = 0.032.

.. code-block:: python

   # Law of total probability: P(Defective) for the factory
   P_defect = defect_rate_A * P_line_A + defect_rate_B * P_line_B

   print("Law of Total Probability — Factory QC")
   print(f"  P(Def|A)·P(A) = {defect_rate_A}×{P_line_A} = {defect_rate_A * P_line_A:.4f}")
   print(f"  P(Def|B)·P(B) = {defect_rate_B}×{P_line_B} = {defect_rate_B * P_line_B:.4f}")
   print(f"  P(Defective)   = {P_defect:.4f}")
   print(f"  That is, {P_defect*100:.1f}% of all items are defective.")


1.5 Bayes' Theorem
====================

A defective widget just rolled off the line.  Which machine probably made it?
Or: a patient's test came back positive --- what is the probability they
actually have the disease?  In both cases we know the "forward" conditional
probability (defect rate given the machine, or positive rate given the disease
status), but we need the "reverse" --- the probability of the *cause* given
the *effect*.  Bayes' theorem is the bridge.

**Derivation.**  Start from the two expressions for :math:`P(A \cap B)`:

.. math::

   P(A \cap B) = P(B \mid A)\,P(A) = P(A \mid B)\,P(B).

Solving for :math:`P(A \mid B)`:

.. math::

   P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}.

We can substitute the law of total probability into the denominator.  If
:math:`A` and :math:`A^c` partition :math:`\Omega`,

.. math::

   P(A \mid B)
   = \frac{P(B \mid A)\,P(A)}{P(B \mid A)\,P(A) + P(B \mid A^c)\,P(A^c)}.

.. code-block:: python

   # Bayes' theorem — the general formula in code
   def bayes(P_B_given_A, P_A, P_B):
       """Compute P(A|B) = P(B|A) * P(A) / P(B)."""
       return (P_B_given_A * P_A) / P_B

   # Quick check with the die example:
   # P(even | >= 4) = P(>= 4 | even) * P(even) / P(>= 4)
   # even = {2,4,6}, >=4 = {4,5,6}
   P_ge4_given_even = 2/3   # {4,6} out of {2,4,6}
   P_even = 3/6
   P_ge4  = 3/6
   P_even_given_ge4 = bayes(P_ge4_given_even, P_even, P_ge4)
   print(f"P(even | X>=4) = {P_even_given_ge4:.4f}")  # {4,6} out of {4,5,6} = 2/3

**General form.**  If :math:`A_1, \dots, A_n` partition :math:`\Omega`, then

.. math::

   P(A_i \mid B)
   = \frac{P(B \mid A_i)\,P(A_i)}{\displaystyle\sum_{j=1}^{n} P(B \mid A_j)\,P(A_j)}.


1.5.1 Interpretation
---------------------

Bayes' theorem has a beautiful interpretation in the language of *updating
beliefs*:

- :math:`P(A)` is the **prior** --- our belief about :math:`A` before seeing
  data.
- :math:`P(B \mid A)` is the **likelihood** --- how probable the observed data
  :math:`B` is if :math:`A` is true.
- :math:`P(A \mid B)` is the **posterior** --- our updated belief about
  :math:`A` after seeing the data.

In words:

.. math::

   \text{posterior} \propto \text{likelihood} \times \text{prior}.

This "update" interpretation is the foundation of Bayesian statistics and will
recur throughout this book.  The concept of "likelihood" appearing here is the
first hint of the likelihood function we formalize in :ref:`ch4_likelihood`.

.. admonition:: Intuition

   Imagine you are a detective.  Your *prior* is your initial suspicion about
   who committed a crime.  The *likelihood* is how well the new evidence fits
   each suspect.  The *posterior* is your updated suspicion after considering
   the evidence.  Bayes' theorem tells you exactly how to combine your prior
   suspicion with new evidence to get a rational updated belief.


1.5.2 Example: Disease Screening (continued)
----------------------------------------------

Continuing the screening example from Section 1.4, suppose a person tests
positive.  What is the probability they actually have the disease?

.. math::

   P(D \mid T^+)
   = \frac{P(T^+ \mid D)\,P(D)}{P(T^+)}
   = \frac{(0.95)(0.01)}{0.0392}
   = \frac{0.0095}{0.0392}
   \approx 0.242.

Despite the test being 95 % accurate, a positive result only means a 24.2 %
chance of actually having the disease.  This perhaps surprising result is
because the disease is rare (low prior), so most positives are false positives.

.. code-block:: python

   # Bayes' theorem: P(D | T+) — the question the patient actually asks
   # P(D|T+) = P(T+|D) * P(D) / P(T+)
   numerator   = P_Tpos_given_D * P_D   # = sensitivity * prevalence
   denominator = P_Tpos                  # computed via total probability
   P_D_given_Tpos = numerator / denominator

   print("Bayes' Theorem — Medical Screening")
   print(f"  Numerator:   P(T+|D)·P(D) = {P_Tpos_given_D}×{P_D} = {numerator:.4f}")
   print(f"  Denominator: P(T+)         = {denominator:.4f}")
   print(f"  P(D|T+)    = {numerator:.4f} / {denominator:.4f} = {P_D_given_Tpos:.4f}")
   print()
   print(f"  Only {P_D_given_Tpos*100:.1f}% of positive tests are true positives!")

.. code-block:: python

   # Verify P(D|T+) by simulation (reusing the simulated population above)
   P_D_given_Tpos_sim = has_disease[test_positive].mean()
   print(f"Formula:   P(D|T+) = {P_D_given_Tpos:.4f}")
   print(f"Simulated: P(D|T+) = {P_D_given_Tpos_sim:.4f}")

.. admonition:: Common Pitfall

   Many people --- including medical professionals --- fall into the "base-rate
   neglect" trap.  They hear "95 % accurate test" and assume a positive result
   means 95 % chance of disease.  But when the disease is rare, the
   denominator :math:`P(T^+)` is dominated by false positives from the large
   healthy population.  Always check the prevalence (base rate).

**How does prevalence affect the posterior?**  Let us sweep prevalence from
0.1 % to 50 % and watch how :math:`P(D \mid T^+)` changes.

.. code-block:: python

   # Sensitivity analysis: P(D|T+) as a function of prevalence
   prevalence_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

   print(f"{'Prevalence':>12s}  {'P(T+)':>8s}  {'P(D|T+)':>9s}")
   print("-" * 35)
   for prev in prevalence_values:
       p_tpos = sensitivity * prev + false_pos_rate * (1 - prev)
       p_d_given_tpos = (sensitivity * prev) / p_tpos
       print(f"{prev:12.3f}  {p_tpos:8.4f}  {p_d_given_tpos:9.4f}")

   print()
   print("Notice: even with a good test, rare diseases yield many false positives.")
   print("At 1% prevalence, ~76% of positive results are FALSE positives.")


1.5.3 Example: Factory Quality Control (continued)
----------------------------------------------------

A defective widget appears on the factory floor.  Which line most likely
produced it?

.. math::

   P(A \mid \text{Defect})
   = \frac{P(\text{Defect} \mid A)\,P(A)}{P(\text{Defect})}
   = \frac{(0.02)(0.60)}{0.032}
   = \frac{0.012}{0.032}
   = 0.375.

.. math::

   P(B \mid \text{Defect}) = 1 - 0.375 = 0.625.

So Line B is the more likely source, despite producing fewer items overall,
because its defect rate is more than double that of Line A.

.. code-block:: python

   # Bayes' theorem: P(Line A | Defective) and P(Line B | Defective)
   P_A_given_defect = (defect_rate_A * P_line_A) / P_defect
   P_B_given_defect = (defect_rate_B * P_line_B) / P_defect

   print("Bayes' Theorem — Factory QC")
   print(f"  P(A|Def) = P(Def|A)·P(A) / P(Def) = ({defect_rate_A}×{P_line_A}) / {P_defect:.4f} = {P_A_given_defect:.4f}")
   print(f"  P(B|Def) = P(Def|B)·P(B) / P(Def) = ({defect_rate_B}×{P_line_B}) / {P_defect:.4f} = {P_B_given_defect:.4f}")
   print()
   print(f"  Line B is more likely ({P_B_given_defect:.1%} vs {P_A_given_defect:.1%}),")
   print(f"  despite producing only {P_line_B:.0%} of the output.")

.. code-block:: python

   # Verify factory Bayes by simulation
   n_items = 1_000_000
   from_line_A = np.random.rand(n_items) < P_line_A
   is_defective = np.where(
       from_line_A,
       np.random.rand(n_items) < defect_rate_A,
       np.random.rand(n_items) < defect_rate_B,
   )

   # Among defective items, what fraction came from Line A?
   P_A_given_def_sim = from_line_A[is_defective].mean()
   P_B_given_def_sim = 1 - P_A_given_def_sim

   print(f"Formula:   P(A|Def) = {P_A_given_defect:.4f},  P(B|Def) = {P_B_given_defect:.4f}")
   print(f"Simulated: P(A|Def) = {P_A_given_def_sim:.4f},  P(B|Def) = {P_B_given_def_sim:.4f}")


1.6 Independence
=================

You are designing a clinical trial with 500 patients.  Each patient takes the
screening test independently.  Can you analyze each result separately, or must
you model all 500 jointly?  If the tests are **independent** --- meaning one
patient's result tells you nothing about another's --- the joint probability of
all 500 results is simply the product of the individual probabilities.  This
factorization is what makes the likelihood function tractable.

**Definition.**  Events :math:`A` and :math:`B` are **independent** if

.. math::

   P(A \cap B) = P(A)\,P(B).

Equivalently (when :math:`P(B) > 0`),

.. math::

   P(A \mid B) = P(A).

*Plain English:*  Learning :math:`B` happened does not change the probability
of :math:`A`.

.. code-block:: python

   # Testing independence: are "Disease" and "Test+" independent?
   # If independent: P(D ∩ T+) should equal P(D) * P(T+)
   P_D_and_Tpos = P_Tpos_given_D * P_D         # 0.95 * 0.01 = 0.0095
   product_P_D_P_Tpos = P_D * P_Tpos            # 0.01 * 0.0392

   print("Independence check: Disease and Test+")
   print(f"  P(D ∩ T+)     = {P_D_and_Tpos:.6f}")
   print(f"  P(D)·P(T+)    = {P_D}×{P_Tpos:.4f} = {product_P_D_P_Tpos:.6f}")
   print(f"  Equal?  {abs(P_D_and_Tpos - product_P_D_P_Tpos) < 1e-10}")
   print(f"  --> They are NOT independent (the test is informative about disease).")

**Mutual independence.**  Events :math:`A_1, \dots, A_n` are **mutually
independent** if for every subset :math:`S \subseteq \{1, \dots, n\}`,

.. math::

   P\!\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i).

Note that *pairwise* independence (every pair satisfies the product rule) does
**not** imply mutual independence.  All subsets must factor.

.. admonition:: Common Pitfall

   Pairwise independence does not guarantee mutual independence.  A classic
   counterexample: toss two fair coins.  Let :math:`A` = "first coin heads",
   :math:`B` = "second coin heads", :math:`C` = "the two coins match."  Any two
   of these are pairwise independent, but :math:`P(A \cap B \cap C) = 1/4 \neq
   1/8 = P(A)P(B)P(C)`.


1.6.1 Why Independence Matters
-------------------------------

Independence is the key assumption that lets us multiply probabilities and,
later, construct likelihoods from individual data points.  When we observe
data :math:`x_1, x_2, \dots, x_n` and assume the observations are
**independent and identically distributed (i.i.d.)**, we can write the joint
probability as

.. math::

   P(x_1, x_2, \dots, x_n) = \prod_{i=1}^{n} P(x_i).

This product form is exactly what gives rise to the likelihood function in
:ref:`ch4_likelihood`.

**Medical screening --- independent patients.**  Suppose we screen 5 patients
independently, each with :math:`P(T^+) = 0.0392`.  What is the probability
that *none* of them tests positive?

.. math::

   P(\text{all } T^-) = (1 - 0.0392)^5 = (0.9608)^5 \approx 0.8171.

.. code-block:: python

   # Independence in action: probability that k out of 5 patients test positive
   from math import comb

   n_patients = 5
   p = P_Tpos  # probability any one patient tests positive

   print(f"P(T+) per patient = {p:.4f}")
   print(f"Assuming independence across {n_patients} patients:\n")
   print(f"  k   P(exactly k positives)")
   for k in range(n_patients + 1):
       # Binomial: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
       P_k = comb(n_patients, k) * p**k * (1 - p)**(n_patients - k)
       print(f"  {k}   {P_k:.6f}")

   P_none = (1 - p) ** n_patients
   print(f"\n  P(no positives) = (1 - {p:.4f})^{n_patients} = {P_none:.4f}")

.. code-block:: python

   # Verify with simulation: screen 5 patients, repeat 100,000 times
   n_trials = 100_000
   positives_per_trial = np.random.binomial(n_patients, p, size=n_trials)

   print(f"  k   Formula     Simulated")
   for k in range(n_patients + 1):
       P_k_formula = comb(n_patients, k) * p**k * (1 - p)**(n_patients - k)
       P_k_sim = (positives_per_trial == k).mean()
       print(f"  {k}   {P_k_formula:.6f}    {P_k_sim:.6f}")

**Non-independence example.**  Draw one card from a standard deck.  Let
:math:`A` = "the card is a heart", :math:`B` = "the card is red."

.. math::

   P(A) = \tfrac{1}{4}, \quad P(B) = \tfrac{1}{2}, \quad
   P(A \cap B) = P(A) = \tfrac{1}{4} \neq \tfrac{1}{4} \cdot \tfrac{1}{2} = \tfrac{1}{8}.

So :math:`A` and :math:`B` are *not* independent --- knowing the card is red
makes it more likely to be a heart.

.. code-block:: python

   # Independence check: Hearts vs Red in a deck of cards
   P_heart = 13/52          # P(A)
   P_red   = 26/52          # P(B)
   P_heart_and_red = 13/52  # P(A ∩ B) — every heart IS red

   print(f"P(Heart)     = {P_heart:.4f}")
   print(f"P(Red)       = {P_red:.4f}")
   print(f"P(Heart∩Red) = {P_heart_and_red:.4f}")
   print(f"P(Heart)·P(Red) = {P_heart * P_red:.4f}")
   print(f"Independent? {abs(P_heart_and_red - P_heart * P_red) < 1e-10}")
   print("Heart and Red are NOT independent.")

.. code-block:: python

   # Simulation verification: two fair coin flips (independent)
   n_sims = 100_000
   coin1 = np.random.choice(['H', 'T'], size=n_sims)
   coin2 = np.random.choice(['H', 'T'], size=n_sims)

   P_A = np.mean(coin1 == 'H')
   P_B = np.mean(coin2 == 'H')
   P_A_and_B = np.mean((coin1 == 'H') & (coin2 == 'H'))

   print("Independence check (coin flips):")
   print(f"  P(A)       = {P_A:.4f}  (expect 0.5)")
   print(f"  P(B)       = {P_B:.4f}  (expect 0.5)")
   print(f"  P(A ∩ B)   = {P_A_and_B:.4f}  (expect 0.25)")
   print(f"  P(A)·P(B)  = {P_A * P_B:.4f}")
   print(f"  Independent? {abs(P_A_and_B - P_A * P_B) < 0.01}")


1.6.2 Tying It Together: Full Screening Simulation
----------------------------------------------------

Let us bring together every concept from this chapter --- sample spaces,
conditional probability, total probability, Bayes' theorem, and independence
--- in a single, complete simulation of the screening program.

.. code-block:: python

   # ============================================================
   # Full simulation: 1 million patients through the screening program
   # ============================================================
   n_population = 1_000_000

   # Step 1: Each person does or does not have the disease (prevalence = 0.01)
   disease = np.random.rand(n_population) < prevalence

   # Step 2: Test result depends on disease status (conditional probability)
   test_pos = np.where(
       disease,
       np.random.rand(n_population) < sensitivity,      # P(T+|D)  = 0.95
       np.random.rand(n_population) < false_pos_rate     # P(T+|D^c) = 0.03
   )

   # --- Compare simulation to formulas ---
   print("=== Full Screening Simulation (n = 1,000,000) ===\n")

   # Total probability: P(T+)
   P_Tpos_formula = sensitivity * prevalence + false_pos_rate * (1 - prevalence)
   P_Tpos_sim = test_pos.mean()
   print(f"P(T+):    formula = {P_Tpos_formula:.4f},  simulated = {P_Tpos_sim:.4f}")

   # Bayes: P(D|T+)
   P_D_given_Tpos_formula = (sensitivity * prevalence) / P_Tpos_formula
   P_D_given_Tpos_sim = disease[test_pos].mean()
   print(f"P(D|T+):  formula = {P_D_given_Tpos_formula:.4f},  simulated = {P_D_given_Tpos_sim:.4f}")

   # Bayes: P(D|T-)  — what if you test NEGATIVE?
   P_Tneg_given_D = 1 - sensitivity           # false negative rate
   P_Tneg_given_Dc = specificity               # true negative rate
   P_Tneg_formula = P_Tneg_given_D * prevalence + P_Tneg_given_Dc * (1 - prevalence)
   P_D_given_Tneg_formula = (P_Tneg_given_D * prevalence) / P_Tneg_formula
   P_D_given_Tneg_sim = disease[~test_pos].mean()
   print(f"P(D|T-):  formula = {P_D_given_Tneg_formula:.6f},  simulated = {P_D_given_Tneg_sim:.6f}")

   # Counts
   true_pos  = (disease & test_pos).sum()
   false_pos = (~disease & test_pos).sum()
   true_neg  = (~disease & ~test_pos).sum()
   false_neg = (disease & ~test_pos).sum()

   print(f"\nConfusion matrix (out of {n_population:,}):")
   print(f"  True positives:  {true_pos:>7,}")
   print(f"  False positives: {false_pos:>7,}")
   print(f"  True negatives:  {true_neg:>7,}")
   print(f"  False negatives: {false_neg:>7,}")


1.7 Summary
============

This chapter established the core language of probability:

- **Sample spaces** describe what can happen; **events** are the subsets we
  assign probabilities to.
- The **Kolmogorov axioms** (non-negativity, normalization, countable
  additivity) pin down how probabilities behave, and every useful rule
  (complement, inclusion--exclusion) follows from them.
- **Conditional probability** and the **multiplication rule** handle "given
  that" reasoning.
- The **law of total probability** decomposes an event's probability over a
  partition.
- **Bayes' theorem** reverses conditioning, enabling us to update beliefs ---
  a theme central to likelihood-based inference.
- **Independence** lets us multiply probabilities for separate experiments,
  foreshadowing the product structure of the likelihood function.

Throughout the chapter we applied every formula to two concrete scenarios ---
medical screening and factory quality control --- and verified the results
with simulation.  The tight match between formula and simulation is not a
coincidence: it is the Kolmogorov axioms at work.

In :ref:`ch2_random_variables`, we move from events to *numbers*: random
variables let us do arithmetic with uncertainty.
