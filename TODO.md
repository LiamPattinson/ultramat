# TODO

## Containers
    
    * broadcasting
        * Return non-writeable view with some strides set to zero
        * May run into issues where broadcast iterators never reach 'end'.
          Shouldn't be a problem using Expressions, as only the destination
          array worries about end, and you never write to a broadcast.
    * transpose
        * Can use this to reinterpret row major as col major, and vice versa.
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
    * Take views, reshapes, etc from expressions.
