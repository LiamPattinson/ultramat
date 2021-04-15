# TODO

## Upcoming

### Containers
    
    * Vector class, including fixed variety.
        * Must allow easy conversion to/from Arrays.
    * Matrix class, including fixed variety.

### StripingExpressions

    * Overhaul expressions to make use of striping rather than standard iteration
    * Top-level expression (the target) will send a stripe dimension and number down the stack,
      and each lower expression will access that stripe or delegate further.
    * Dimension-reducing expressions will complete the operation over their given dimension only
      when called, rather than evaluating their lower expressions.
    * Pros:
        * Faster for many use cases i.e. adding two non-contiguous views
        * Much more OpenMP friendly
            * At the target, there is a loop over all stripes, and a loop over each individual
              stripe. If individual stripes can be guaranteed random-access, can parallelise
              both loops to handle edge cases
        * Can do dim-wise reductions etc without building an intermediate
            * full reductions can be constructed at run time from a series of
              dim-wise reductions.
        * Offers a means to have row_major and col_major entities work together, as when the
          target is col_major, it will default to striping in the 0th dimension. This will be
          naturally be passed down to any row major entities, which will handle it just fine.
    * Cons:
        * Slightly slower for some common operations, i.e. adding two contiguous arrays
        * Not clear how well this will apply to generator expressions.
            * Perhaps limit them to 1D objects. Maybe just accept that this isn't going to end
              well and perform an eval.
        * Dim-wise cumulative expressions present a serious issue. May have to make them eval.
            * Operations must be performed in a linear order: not parallelisable
            * Stripe direction matters, and cannot be controlled by the target.
            * Might just remove them... they're a seriously niche application.

### Expressions

    * Rework Generator expressions
    * New expression types
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
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
