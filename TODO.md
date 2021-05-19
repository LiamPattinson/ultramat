# TODO

## Upcoming

### Containers
    
    * Vector class, including fixed variety.
        * Must allow easy conversion to/from Arrays.
    * Matrix class, including fixed variety.

### Expressions

    * fold expressions
        - test general fold
        - implement boolean folds
    * Cumulative should specify direction, default to 0.
        * should be evaluating
        * test mixed row/col major
    * Rework Generator expressions
    * Methods to broadcast scalars.
        * scalar_as_dense function that returns fixed-size Array<T,1>;
        * May just require a _lot_ of function overloading.
    * Take views, reshapes, permutations, transposes from expressions.
        * implement as ReinterpretExpression. Perform eval, then apply function.

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
