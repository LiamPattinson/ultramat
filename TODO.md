# TODO

### Documentation

    * Finish the docs!
        - Related pages
            - DenseObjects              I
                - Striping              O
                - Broadcasting          O
                - Vector/Matrix stacks  O
            - ExpressionTemplates O
        - Dense
            - DenseFixed   X
            - Dense        X
            - DenseImpl    X
            - DenseStripe  I
            - DenseUtils   X
            - DenseView    O
            - Expressions
                - DenseCumulativeExpression  O
                - DenseElementWiseExpression O
                - DenseExpression            I
                - DenseFoldExpression        O
                - DenseGeneratorExpression   O
                - DenseWhereExpression       O
            - Math
                - DenseArithmetic O
                - DenseCumulative O
                - DenseFolds      O
                - DenseGenerators O
                - DenseMath       O
        - Utils
            - Utils         I
            - IteratorTuple O

### Linear algebra

    * DenseLinearAlgebraMethod
        * Vector-Vector
            * Given a row_major arg of size (N_a,N_b,...,N_w,N_x), any second arg must also have size (N_a,N_b,...,N_w,N_x).
            * Given a col_major arg of size (N_x,N_a,N_b,...,N_w), any second arg must also have size (N_x,N_a,N_b,...,N_w).
            * Used to implement dot/inner, cross, and outer.
            * If it acts effectively as a reduction operation, such as a dot product, the result will have size (N_a,N_b,...,N_w),
              as the preceding dimensions are treated as a 'stack' of vectors of size N_x.
            * If the operation instead preserves the vector size, such as the cross product, we require N_x==3, and it produces an 
              object of the same size.
            * Outer product should accept row_major arrays of size (N_a,N_b,...,N_w,N_x) and (N_a,N_b,...,N_w,N_y).
              It will produce an array of size (N_a,N_b,...,N_w,N_x,N_y)
        * Matrix-Matrix Operations
            * Similar to vector operations, but the first/last two dimensions are considered when col/row_major.
            * Should tolerate different dims. If the first arg has one less dim, we interpret this as a stack of
              row vectors. If the second arg has one less dim, we interpret this as a stack of column vectors.
              No need for Matrix-Vector operations as this will fill that requirement.
            * Used to implement matmul, gauss_elim_solve, lu_solve
        * Unary Operations
            * May require a second expression type
    * Should implement 'refocus' method for DenseView, allowing a single DenseView to reference a given vector/matrix within
      a stack, and then to efficiently reference the next one.
    * Dot product should be non-evaluating. Probably best to actually implement it using existing expressions.
    * Set BLAS/LAPACK usage at compile time. Include a USE_EIGEN option maybe?

## Wishlist

### Finite Differences

    * Define a 'Stencil' expression, which is given template params `<unsigned int (dim), int, int, int...>` to
      represent a finite difference stencil. For example, a forward/backward first-order difference over the
      0 axis would have args <0, -1, 1>. A second-order centered difference over the 3 axis would be <3,-1,0,1>.
      The numerator of a second order differential over the 1 axis would be <1,1,-2,1>, etc.
    * Act almost like an element-wise expression, only it results in a partial fold reducing the operating dimension
      by stencil_size-1.
    * Pretty much impossible to avoid evaluating some elements more than once. Will lead to weird results when
      applied to random generators.

### Further linear algebra

    * outer product
    * det (uses LU factorisation, unless input matrix is particularly small, in which case it's hardcoded)
    * norm (perhaps rename complex norm function to abs2?)
    * trace (unary matrix operation, tricky iteration method required)
    * cond
    * rank
    * kronecker product
    * QR decomposition
    * SVD decomposition

### Further OpenMP

    * parallel within each stripe
    * benchmark testing, optimise scheduling

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

    * compressed sparse row/col
    * not technically 'sparse' but consider types for packed and banded matrices

### Distributed

    * Oh boy
    * Here it is
    * The big one
    * Distributed arrays, expressions, and linear algebra.
