# TODO

## Upcoming

### General

    * unit test in-place scalar update
    * unit test reshape from expression
    * unit test new random generators
    * unit test arange/regspace

### Containers
    
    * Vector class, including fixed variety.
    * Matrix class, including fixed variety.
    * Include matrix generators (eye)
    * Must allow easy conversion to/from Arrays.

## Wishlist

### LinearAlgebra

    * MatMul
    * GaussianElimination solver
    * LU factorisation
    * Set BLAS/LAPACK usage at compile time
    * This list will go MUCH deeper as time goes on...

### OpenMP

    * Depends on successful implementation of striping expressions.
    * Requires custom copy construct/assigns.
    * Will allow most expressions to be calculated in parallel. The exceptions are those that
      must be stepped through linearly, and a special case must be made for these. Examples include
      cumulative sum/product. Perhaps these should be evaluating expressions if OpenMP is activated.
    * Do not have this switched on by default. The end user may wish to use this library alongside
      their own parallelisation strategy, and therefore this would interfere. However, signpost it
      well!

### WhereExpression

    * where( left_expression, right_expression, default_val)
        * Evaluate left expression.
        * If it returns true, evaluate and return right expression.
        * If it returns false, return default_val

### ForEachExpression

    * apply function to each arg in turn

### FFT

    * Optional functionality if fftw3 is installed

### Sparse Matrices

    * compressed sparse row/col mainly
    * not technically 'sparse' but consider types for packed and banded matrices

### Distributed

    * Oh boy
    * Here it is
    * The big one
    * Distributed arrays, expressions, and linear algebra.
