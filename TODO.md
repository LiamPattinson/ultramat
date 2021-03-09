# TODO

## Array.hpp

* Implement slicing/views
    * test constructors/attributes
    * test iterators for the following cases:
        - contiguous view
        - semi contiguous view
        - non contiguous view
        - non contiguous view with negative stride
* Implement broadcasting
* Implement fixed size
    * Variadic templates, first element of type, the rest as `std::size_t`
    * Partial specialisation for dynamic length
    * Implement common functionality using CRTP
* Begin template expressions
