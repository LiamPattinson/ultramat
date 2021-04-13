# TODO

## Containers
    
    * Vector class, including fixed variety.
        * Must allow easy conversion to/from Arrays.
    * Matrix class, including fixed variety.

## Expressions

    * Rework to 'DenseExpression' (mostly a naming change)
    * Methods to broadcast scalars.
        * scalar_as_dense function that returns fixed-size Array<T,1>;
        * May just require a _lot_ of function overloading.
    * New expression types
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
    * Take views, reshapes, permutations, transposes from expressions.
        * implement as ReinterpretExpression. Perform eval, then apply function.
    * Faster iteration strategy
        * Rather than using standard begin/end, perhaps look into striped iteration
          as a general standard.
