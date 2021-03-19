# TODO

## Array.hpp

* Array upgrades
    * reshape
    * broadcasting
        * May run into issues where broadcast iterators never reach 'end'.
        * shouldn't be a problem using Expressions, as only the destination
          array worries about end, and you never write to a broadcast.
        * Can use broadcasting to handle issues of `Expression<ScalarType>`
    * `+=`, `*=` etc from expressions. Can use regular assignment, and delegate
      to `+`, `*`, etc, i.e. `return (*this) + expression;`
* Template expressions
    * Further non-blocking Expressions, that properly implement lazy evaluation
      like most traditional expression template schemes.
        * CumulativeExpression. For cumsum and cumprod.
        * GeneratorExpression. Can be used for linspace, logspace, etc.
    * Blocking expression, which force evalution of the expressions ahead of them,
      store the result in a temporary, and then pass this on. This is necessary to
      achieve high performance for more complex operations. These will be used later
      to implement BLAS calls etc.
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
        * ViewExpression. Permits reinterpretation of args. Examples include transpose,
          as_row_major, as_col_major. Could implement reshape this way? Could call it
          ReinterpretExpression?
* Implement fixed size arrays
    * Variadic templates, first element of type, the rest as `std::size_t`
    * Partial specialisation for dynamic length
    * Implement common functionality using CRTP
