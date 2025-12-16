[//]: # (# Numerical Methods and Stability)

[//]: # ()
[//]: # (This document explains the numerical methods used to solve least squares)

[//]: # (problems in this project, and why certain approaches may fail in practice)

[//]: # (despite being correct in theory.)

[//]: # ()
[//]: # (## `np.linalg.inv` and `np.linalg.solve`)

[//]: # ()
[//]: # (### np.linalg.solve and np.linalg.inv work on the same principle)

[//]: # ()
[//]: # (Assuming that matrix $A$ is not singular &#40;else NumPy will throw ```LinAlgError: )

[//]: # (Singular matrix```&#41;, NumPy finds the inverse of a matrix in the following steps:)

[//]: # ()
[//]: # (1. Factorize matrix A using **LU decomposition** &#40;any matrix A can be decomposed)

[//]: # (into the product of a lower triangular matrix and an upper triangular matrix&#41;)

[//]: # ()
[//]: # ($$)

[//]: # (A \cdot x = B)

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (&#40;L \cdot U&#41; \cdot x = B)

[//]: # ($$)

[//]: # ()
[//]: # (Let)

[//]: # ($$)

[//]: # (U \cdot x = y)

[//]: # ($$)

[//]: # ()
[//]: # (2. Using forward and backward substitution:)

[//]: # (   - Forward Substitution:)

[//]: # (     - First find $y$:)

[//]: # ()
[//]: # ($$)

[//]: # (L \cdot y = B)

[//]: # ($$)

[//]: # (     - Which is easy since $L$ is a lower triangular matrix)

[//]: # (   - Backward Substitution:)

[//]: # (     - Find $x$:)

[//]: # ()
[//]: # ($$)

[//]: # (U \cdot x = y)

[//]: # ($$)

[//]: # (     - Since $U$ is a upper triangular matrix, we have $x$ !)

[//]: # ()
[//]: # (<h3 align="center">BUT</h3>)

[//]: # ()
[//]: # (If A is singular or nearly singular, NumPy errors or unstable results)

[//]: # ()
[//]: # (**When singular:**)

[//]: # (- That means: The rows/columns are linearly dependent)

[//]: # (- There is no unique solution &#40;OR&#41; infinite solutions)

[//]: # (- NumPy cannot decide a single correct solution so throws an error)

[//]: # ()
[//]: # (**When nearly singular:**)

[//]: # (- The matrix is almost dependent:)

[//]: # (  - Very tiny pivot elements &#40;the first non-zero element in a row of an )

[//]: # (  Echelon form matrix&#41;)

[//]: # (  - Division by very tiny numbers leads to huge floating-point error)

[//]: # (  - So, small noise in input gives a big error in output)

[//]: # (  - Even if NumPy can compute something, the result may be wildly inaccurate)

[//]: # ()
[//]: # ()
[//]: # (**Example:**)

[//]: # ()
[//]: # (```commandline)

[//]: # (A = np.array&#40;[[1.0, 1.0],)

[//]: # (              [1.0, 1.0000001]&#41;)

[//]: # (              )
[//]: # (B1 = np.array&#40;[2.0, 2.0000001]&#41;)

[//]: # (B2 = np.array&#40;[2.0, 2.0000002]&#41;)

[//]: # ()
[//]: # (x1 = np.linalg.solve&#40;A, B1&#41;)

[//]: # (x2 = np.linalg.solve&#40;A, B2&#41;)

[//]: # (```)

[//]: # ()
[//]: # ($$)

[//]: # (\Delta B = \begin{bmatrix})

[//]: # (0 \\)

[//]: # (10^{-6})

[//]: # (\end{bmatrix})

[//]: # ($$)

[//]: # ()
[//]: # (<h3 align="center">BUT</h3>)

[//]: # ()
[//]: # ($$)

[//]: # (\Delta x = \begin{bmatrix})

[//]: # (-1\\)

[//]: # (1)

[//]: # (\end{bmatrix})

[//]: # ($$)

[//]: # ()
[//]: # (That is a huge change in $x$ compared to a tiny change in $B$.)

[//]: # (This is what happens when A is nearly singular.)

[//]: # ()
[//]: # (## Pseudoinverse &#40;`np.linalg.pinv`&#41;)

[//]: # ()
[//]: # (By Singular Value Decomposition &#40;SVD&#41;, we know that)

[//]: # ($$)

[//]: # (A = U \Sigma V^T)

[//]: # ($$)

[//]: # (Now,)

[//]: # ($$)

[//]: # (Ax = &#40;U \Sigma V^T&#41;x = B)

[//]: # ($$)

[//]: # (We will try to isolate x:)

[//]: # ($$)

[//]: # (U^{-1} U \Sigma V^T x = U^{-1} B)

[//]: # ($$)

[//]: # ($$)

[//]: # (\Sigma V^T x = U^{-1} B)

[//]: # ($$)

[//]: # ($$)

[//]: # (\Sigma^{-1} \Sigma V^T x = \Sigma^{-1} U^{-1} B)

[//]: # ($$)

[//]: # ($$)

[//]: # (V^T x = \Sigma^{-1} U^{-1} B)

[//]: # ($$)

[//]: # ($$)

[//]: # (&#40;V^T&#41;^{-1} V^T x = &#40;V^T&#41;^{-1} \Sigma^{-1} U^{-1} B)

[//]: # ($$)

[//]: # ($$)

[//]: # (x = &#40;V^T&#41;^{-1} \Sigma^{-1} U^{-1} B)

[//]: # ($$)

[//]: # ()
[//]: # (Since $U$ and $V$ are orthogonal matrices, their transpose )

[//]: # (is equal to their inverse)

[//]: # ()
[//]: # (Hence,)

[//]: # ($$)

[//]: # (x = V \Sigma^{-1} U^T B)

[//]: # ($$)

[//]: # ()
[//]: # (In short, we write)

[//]: # ($$)

[//]: # (x = A^\dagger B)

[//]: # ($$)

[//]: # ()
[//]: # ($A^\dagger$ is called the **Moore–Penrose pseudoinverse**.)

[//]: # ()
[//]: # ()
[//]: # (**Why this is more stable**)

[//]: # (- We easily get $U$, $V$ and $\Sigma$ from SVD)

[//]: # (- $\Sigma$ is diagonal &#40;possibly rectangular&#41;)

[//]: # (- Its pseudoinverse $\Sigma^\dagger$ is formed by taking reciprocals )

[//]: # (of the non-zero singular values)

[//]: # ()
[//]: # (However, reciprocals of extremely small singular values are scary, )

[//]: # (so we use a trick:)

[//]: # ()
[//]: # ($$)

[//]: # (\frac{1}{\sigma_i} \approx 0 \text{  when } \sigma_i \approx 0)

[//]: # ($$)

[//]: # ()
[//]: # (This is called ***Truncation***, and it makes the solution stable )

[//]: # (even when the matrix is ill-conditioned.)

[//]: # ()
[//]: # (This works because the exact solution for an ill-conditioned or )

[//]: # (singular matrix is meaningless anyway. So we intentionally )

[//]: # (avoid chasing nonsense and instead compute the most stable, )

[//]: # (best-fit solution.)

[//]: # ()
[//]: # (If we do not truncate, we get a “solution” that changes dramatically )

[//]: # (if a single data point changes slightly. Which is mathematically )

[//]: # (correct, but numerically useless!)

[//]: # ()
[//]: # (Mathematically it is not at all convincing, but the geometric meaning)

[//]: # (has significance.)

[//]: # ()
[//]: # (### Geometric Interpretation: ###)

[//]: # ()
[//]: # (If a singular value is tiny, it means the data tells us: That direction in solution space is totally unreliable!)

[//]: # ()
[//]: # (Having a small singular value for a particular eigenvector just means that after the)

[//]: # (rotation, when we scale the space, the space gets **COMPLETELY** squished along the)

[//]: # (direction of the eigen vector! So in a way, we are losing data. And when we want to)

[//]: # (calculate the inverse, we are technically trying to apply something to the space so that )

[//]: # (we can get the original data back, which is impossible.)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (## Least Squares Solver &#40;`np.linalg.lstsq`&#41;)

[//]: # ()
[//]: # (Least squares is solved using the Moore–Penrose pseudoinverse so has the same math as above.)

[//]: # ()
[//]: # ()
[//]: # (## Regularization)

[//]: # ()
[//]: # (- This is a way of limiting the flexibility of the model)

[//]: # (- We do this by lowering the weights)

[//]: # (- That seems like a completely unrelated way to limit the flexibility,)

[//]: # (but essentially, weights are the importance of the output of a certain)

[//]: # (feature.)

[//]: # (When we are overfitting, we are exaggerating the importance of a certain )

[//]: # (input/data point)

[//]: # (- So, we will try to add the weights in the loss calculation, penalizing )

[//]: # (the model for high weights)

[//]: # (- For the regularization I used &#40;L2 Regularization &#40;OR&#41; Ridge Regularization&#41;,)

[//]: # (we define the loss function as)

[//]: # ($$)

[//]: # (L_{total} = L_{data} + \lambda \sum_i w_i^2)

[//]: # ($$)

[//]: # (- The constant $\lambda$ is a like a knob controlling model complexity)

[//]: # (- Small $\lambda$ could lead to overfitting, Large $\lambda$ could lead to underfitting)

# Numerical Methods and Stability

This document explains the numerical methods used to solve least squares
problems in this project, and why certain approaches may fail in practice
despite being correct in theory.

## `np.linalg.inv` and `np.linalg.solve`

### np.linalg.solve and np.linalg.inv work on the same principle

Assuming that matrix $A$ is not singular (else NumPy will throw ```LinAlgError: 
Singular matrix```), NumPy finds the inverse of a matrix in the following steps:

1. Factorize matrix A using **LU decomposition** (any matrix A can be decomposed
into the product of a lower triangular matrix and an upper triangular matrix)

$$
A \cdot x = B
$$

$$
(L \cdot U) \cdot x = B
$$

Let

$$
U \cdot x = y
$$

2. Using forward and backward substitution:
- Forward Substitution:
  - First find $y$:

$$
L \cdot y = B
$$

  - Which is easy since $L$ is a lower triangular matrix

- Backward Substitution:
  - Find $x$:

$$
U \cdot x = y
$$

  - Since $U$ is a upper triangular matrix, we have $x$ !
<h3 align="center">BUT</h3>

If A is singular or nearly singular, NumPy errors or unstable results

**When singular:**
- That means: The rows/columns are linearly dependent
- There is no unique solution (OR) infinite solutions
- NumPy cannot decide a single correct solution so throws an error

**When nearly singular:**
- The matrix is almost dependent:
  - Very tiny pivot elements (the first non-zero element in a row of an 
  Echelon form matrix)
  - Division by very tiny numbers leads to huge floating-point error
  - So, small noise in input gives a big error in output
  - Even if NumPy can compute something, the result may be wildly inaccurate


**Example:**
```commandline
A = np.array([[1.0, 1.0],
              [1.0, 1.0000001])
              
B1 = np.array([2.0, 2.0000001])
B2 = np.array([2.0, 2.0000002])

x1 = np.linalg.solve(A, B1)
x2 = np.linalg.solve(A, B2)
```

$$
\Delta B = \begin{bmatrix}
0 \\
10^{-6}
\end{bmatrix}
$$

<h3 align="center">BUT</h3>

$$
\Delta x = \begin{bmatrix}
-1\\
1
\end{bmatrix}
$$

That is a huge change in $x$ compared to a tiny change in $B$.
This is what happens when A is nearly singular.

## Pseudoinverse (`np.linalg.pinv`)

By Singular Value Decomposition (SVD), we know that

$$
A = U \Sigma V^T
$$

Now,

$$
Ax = (U \Sigma V^T)x = B
$$

We will try to isolate x:

$$
U^{-1} U \Sigma V^T x = U^{-1} B
$$

$$
\Sigma V^T x = U^{-1} B
$$

$$
\Sigma^{-1} \Sigma V^T x = \Sigma^{-1} U^{-1} B
$$

$$
V^T x = \Sigma^{-1} U^{-1} B
$$

$$
(V^T)^{-1} V^T x = (V^T)^{-1} \Sigma^{-1} U^{-1} B
$$

$$
x = (V^T)^{-1} \Sigma^{-1} U^{-1} B
$$

Since $U$ and $V$ are orthogonal matrices, their transpose 
is equal to their inverse

Hence,

$$
x = V \Sigma^{-1} U^T B
$$

In short, we write

$$
x = A^\dagger B
$$

$A^\dagger$ is called the **Moore–Penrose pseudoinverse**.


**Why this is more stable**
- We easily get $U$, $V$ and $\Sigma$ from SVD
- $\Sigma$ is diagonal (possibly rectangular)
- Its pseudoinverse $\Sigma^\dagger$ is formed by taking reciprocals 
of the non-zero singular values

However, reciprocals of extremely small singular values are scary, 
so we use a trick:

$$
\frac{1}{\sigma_i} \approx 0 \text{  when } \sigma_i \approx 0
$$

This is called ***Truncation***, and it makes the solution stable 
even when the matrix is ill-conditioned.

This works because the exact solution for an ill-conditioned or 
singular matrix is meaningless anyway. So we intentionally 
avoid chasing nonsense and instead compute the most stable, 
best-fit solution.

If we do not truncate, we get a "solution" that changes dramatically 
if a single data point changes slightly. Which is mathematically 
correct, but numerically useless!

Mathematically it is not at all convincing, but the geometric meaning
has significance.

### Geometric Interpretation:

If a singular value is tiny, it means the data tells us: That direction in solution space is totally unreliable!

Having a small singular value for a particular eigenvector just means that after the
rotation, when we scale the space, the space gets **COMPLETELY** squished along the
direction of the eigen vector! So in a way, we are losing data. And when we want to
calculate the inverse, we are technically trying to apply something to the space so that 
we can get the original data back, which is impossible.




## Least Squares Solver (`np.linalg.lstsq`)

Least squares is solved using the Moore–Penrose pseudoinverse so has the same math as above.


## Regularization

- This is a way of limiting the flexibility of the model
- We do this by lowering the weights
- That seems like a completely unrelated way to limit the flexibility,
but essentially, weights are the importance of the output of a certain
feature.
When we are overfitting, we are exaggerating the importance of a certain 
input/data point
- So, we will try to add the weights in the loss calculation, penalizing 
the model for high weights
- For the regularization I used (L2 Regularization (OR) Ridge Regularization),
we define the loss function as

$$
L_{total} = L_{data} + \lambda \sum_i w_i^2
$$

- The constant $\lambda$ is a like a knob controlling model complexity
- Small $\lambda$ could lead to overfitting, Large $\lambda$ could lead to underfitting