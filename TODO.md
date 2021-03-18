# TODO

## Array.hpp

* Array upgrades
    * reshape
    * broadcasting
        * May run into issues where broadcast iterators never reach 'end'.
        * shouldn't be a problem using Expressions, as only the destination
          array worries about end.
        * Can use broadcasting to handle issues of `Expression<ScalarType>`
* Template expressions
    * CumulativeExpression. For cumsum and cumprod.
    * ReductionExpression. Can be used for sum, prod, max, etc.
    * DimWiseExpression. Reduction in one dimension only.
    * GeneratorExpression. Can be used for linspace, logspace, etc.
* Implement fixed size arrays
    * Variadic templates, first element of type, the rest as `std::size_t`
    * Partial specialisation for dynamic length
    * Implement common functionality using CRTP
