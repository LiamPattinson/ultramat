# TODO

## Array.hpp

    * resize
        * Will have to look into allocators. Should this be a template
          argument? How will it fit alongside plans for fixed arrays?
    * reshape
        * Contiguous arrays only.
        * Could be done inplace by just updating dim/shape/stride, or
          as a blocking expression.
    * broadcasting
        * Return view
        * May run into issues where broadcast iterators never reach 'end'.
          Shouldn't be a problem using Expressions, as only the destination
          array worries about end, and you never write to a broadcast.
        * Can use broadcasting to handle issues of `Expression<ScalarType>`
    * `+=`, `*=` etc from expressions. Can use regular assignment, and delegate
      to `+`, `*`, etc, i.e. `return (*this) + expression;`

## Expressions.hpp

    * Evaluating expression, which force evalution of the expressions ahead of them,
      store the result in a temporary, and then pass this on. This is necessary to
      achieve high performance for more complex operations. These will be used later
      to implement BLAS calls etc.
        * eval. Simple function that forces evaluation and provides an interface to a temporary.
        * ReductionExpression. Can be used for sum, prod, max, etc.
        * DimWiseExpression. Reduction in one dimension only.
        * ViewExpression. Permits reinterpretation of args. Examples include transpose,
          as_row_major, as_col_major. Could implement reshape this way? Could call it
          ReinterpretExpression?
        * copy/assignment functions should be specialised for these expressions so they can
          avoid copying at the very end.

## FixedArray.hpp

    * Variadic templates, first element of type, the rest as `std::size_t`
    * Partial specialisation for dynamic length
    * Implement common functionality using CRTP
