# TODO

## Upcoming

### More Expressions

    * where( condition, left_expression, right_expression)
        * Evaluate condition.
        * If it returns true, evaluate and return left expression.
        * If it returns false, return right expression
    * is_nan/is_finite etc.
    * square/cube
    * reciprocal
    * to_radians/to_degrees
    * signbit/copysign
    * trunc (like ceil and floor, always towards zero)
    * include mathematical constants

### Special consideration for std::complex

    * need new type trait to fill the role of is_arithmetic. is_scalar?
    * new expressions: real, imag, arg, norm, conj
    * hermitian transpose (will need an eval)

### Linear algebra

    * Implement matrix generators (such as 'eye', 'identity')
    * trace
    * dot
    * matmul
    * GaussianElimination solver
    * LU factorisation
    * Set BLAS/LAPACK usage at compile time

## Wishlist

### OpenMP

    * Depends on successful implementation of striping expressions.
    * Requires custom copy construct/assigns.
    * Will allow most expressions to be calculated in parallel. The exceptions are those that
      must be stepped through linearly, and a special case must be made for these. Examples include
      cumulative sum/product. Perhaps these should be evaluating expressions if OpenMP is activated.
    * Do not have this switched on by default. The end user may wish to use this library alongside
      their own parallelisation strategy, and therefore this would interfere. However, signpost it
      well!

### Further linear algebra

    * inner, outer
    * cond
    * rank
    * kronecker product
    * QR decomposition
    * SVD decomposition

### Iterative linear algebra

    * Jacobi
    * Gauss-Seidel
    * Richardson
    * SOR/SSOR
    * Multigrid methods
    * Krylov methods

### Optimisation

    * Gradient Descent
    * Stochastic Gradient Descent
    * Conjugate Gradient Descent
    * Preconditioning

### FFT

    * Optional functionality if fftw3 is installed

### Sparse Matrices

    * compressed sparse row/col mainly
    * not technically 'sparse' but consider types for packed and banded matrices

### Automatic Broadcasting

    * Requires a complete rethink of striped iteration to do this efficiently

### Distributed

    * Oh boy
    * Here it is
    * The big one
    * Distributed arrays, expressions, and linear algebra.
