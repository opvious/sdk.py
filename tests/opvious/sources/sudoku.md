# Sudoku model

We start by defining the sets of possible values
$\S^a: N \doteq \{1 \ldots 9\}$ and possible
positions $\S^a: P \doteq \{0 \ldots 8\}$. With
these two, we can define the variable we will optimize over:
$\S^v_{decisions[row,column,value]}: \alpha \in \{0,1\}^{P \times P \times N}$.

The first constraint uses input hints
$\S^p_{hints[row,column,value]}: h \in \{0,1\}^{P \times P \times N}$,
enforcing that our choice always matches the hint:

$$
  \S^c_{hintsObserved}:
  \forall v \in N, i,j \in P, \alpha_{i,j,v} \geq h_{i,j,v}
$$

Then there are three types of unicity constraints, all enforcing that there must
be one of each value.

+ First, per row:

  $$
    \S^c_{onePerRow}:
    \forall v \in N, i \in P,
      \sum_{j \in P} \alpha_{i,j,v} = 1
  $$

+ Then, per column:

  $$
    \S^c_{onePerColumn}:
    \forall v \in N, j \in P,
      \sum_{i \in P} \alpha_{i,j,v} = 1
  $$

+ Finally, per box:

  $$
    \S^c_{onePerBox}:
      \forall v \in N, k^{b} \in P,
      \sum_{k^{c} \in P}
      \alpha_{
          3 \left\lfloor \frac{k^{b}}{3} \right\rfloor
          + \left\lfloor \frac{k^{c}}{3} \right\rfloor,
          3 (k^{b} \bmod 3) + (k^{c} \bmod 3),
          v
      }
      = 1
  $$
