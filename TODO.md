# TODO

## Containers
    
    * 'as_row_major' and 'as_col_major'
    * `+=`, `*=` etc from expressions.
    * Vector class, including fixed variety.
        * Must allow easy conversion to/from Arrays.
    * Matrix class, including fixed variety.

## Expressions

    * Faster iteration strategy
        * Rather than using standard begin/end, perhaps look into striped iteration
          as a general standard. stripe_iterator can be implemented in place of
          fast_iterator, with the difference being that stripe_iterator can have
          a stride other than 1, including negatives.
    * New expression types
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
    * Take views, reshapes, permutations, transposes from expressions.
