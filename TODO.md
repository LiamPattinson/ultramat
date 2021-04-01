# TODO

## Array.hpp
    
    * Make View class. Should be able to reuse a lot of current Array implementation
      details.
        * Keep status, remove semi-contiguous, add writeable.
    * Remove non-owning arrays, clean up internals to use standard library containers
        * std::vector for data, shape, stride. Remove everything else.
        * Row/col major should be a template arg. See FixedArray.
    * reshape
        * Arrays and contiguous views only.
        * return *this.
    * broadcasting
        * Return non-writeable view with some strides set to zero
        * May run into issues where broadcast iterators never reach 'end'.
          Shouldn't be a problem using Expressions, as only the destination
          array worries about end, and you never write to a broadcast.
    * transpose
        * Can use this to reinterpret row major as col major, and vice versa.
    * `+=`, `*=` etc from expressions.

## Expressions.hpp

    * Faster iteration strategy
        * Rather than using standard begin/end, perhaps look into striped iteration
          as a general standard. stripe_iterator can be implemented in place of
          fast_iterator, with the difference being that stripe_iterator can have
          a stride other than 1, including negatives.
    * New expression types
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
