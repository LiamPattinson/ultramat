# TODO

### Cleanup

    * Finish the docs!
        - Related pages
            - DenseObjects        X
            - ExpressionTemplates O
            - Broadcasting        O
        - Dense
            - DenseFixed   X
            - Dense        X
            - DenseImpl    I
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

### Slicing update

    * Slice::min should alias 0
    * Slice::max should alias maximum std::ptrdiff_t
    * Slice::all should refer to Slice{Slice::min,Slice::max};
    * Slicing via operator()


### Linear algebra

    * Would be handy if some sort of DenseLinearAlgebraExpression could be defined, but it'll be tricky
      to generalise it.
        * Vector Operations
            * If operating on vectors, and given a row_major object of size (N_a,N_b,...,N_w,N_x), any second
              args must also have size (N_a,N_b,...,N_w,N_x). If it acts effectively as a reduction operation,
              such as a dot product, the result will have size (N_a,N_b,...,N_w), as the preceding dimensions
              are treated as a 'stack' of vectors of size N_x. If the operation instead preserves the vector
              size, such as the cross product, we require N_x==3, and it produces an object of the same size.
            * If instead we operate on col_major objects, the first dimension is considered to be the 'vector'
              dimension, so we instead have objects of size (N_x,N_a,N_b,...,N_w). They otherwise behave the
              same. Mixed row_major/col_major operations should respect this.
            * Unary vector operations will operate on the last row if row_major, or first row if col_major.
            * Outer product should accept row_major arrays of size (N_a,N_b,...,N_w,N_x) and (N_a,N_b,...,N_w,N_y).
              It will produce an array of size (N_a,N_b,...,N_w,N_x,N_y)
        * Matrix Operations
            * Similar to vector operations, but the first/last two dimensions are considered when col/row_major.
            * Special case of matrix-vector multiplication. Matmul should handle both matrix-matrix and matrix-
              vector cases, depending on whether the args have the same number of dims or one of them has
              one less dim than the other. If the first arg has one less dim, we interpret this as a stack of
              row vectors. If the second arg has one less dim, we interpret this as a stack of column vectors.
        * A generic DenseLinearAlgebraExpression should be defined, with variances between different operation
          types implemented via CRTP policy classes.
        * Will need a way to iterate over vector/matrix stacks as if they're a regular array.
            * Implement DenseLinearAlgebraStack, which is a variant on a DenseView
            * Holds a DenseView over the vector/matrix, and on iterating it 'refocuses' the view with a new
              data pointer while keeping its shape and stride.
            * It should be broadcastable as usual, and unlike most expressions it should do this by default.

    * dot
        * Give two vectors, return scalar. 
        * Does not do matmul as other libraries do, with some exceptions where it happens to give the same result.
            * dot(3x5,5) interpretted as a stack of 3 5-vectors, dotted with a single 5-vector using broadcasting.
            * dot(5,5x3) intepretted as a single 5-vector dotted with a stack of 5 3-vectors, which is an error.
            * When using col_major, the former will fail while the latter will succeed.
    * cross
        * Give two vectors of size 3, return vector of size 3.
    * matmul
        * should handle matrix-matrix, matrix-vector, and vector-matrix multiplication.
    * GaussianElimination solver
    * LU factorisation
    * Set BLAS/LAPACK usage at compile time

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

    * inner, outer
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
