# TODO

## Upcoming

    * Implement pairwise summation, Kahan summation, and apply to sum
        * Requires implementation of 'ComplexFold', which copies a stripe into an
          appropriately sized std::vector, and then carries out an arbitrary fold over
          that. This will also be useful for the later implementation of median.
        * perhaps have pairwise as default and kahan_sum as a high-precision version
        * implement naive sum as fast_sum
        * have also fast/kahan versions of mean, stddev, var

### Special consideration for std::complex

    * need new type trait to fill the role of is_arithmetic. is_scalar?
    * new expressions: real, imag, arg, norm, conj
    * complex sqrt, complex log, complex pow, complex acos/asin/atanh
        * std:: already deals with most complex variants on cmath functions. However, this deals only
          with the case f(Complex) -> Complex. Special versions are necessary when f(Real) -> Complex.
    * include abs in var/stddev
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

### Improve expressions


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

### Statistics

    * covariance/correlation
    * weighted average
    * median
    * histogram
    * PCA
    * random chi squared/poisson/etc

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
