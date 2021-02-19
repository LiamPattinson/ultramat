# TODO

## Array.hpp

* Implement iterators
    * Need to test
* Implementing assignment etc
    * need casting copy constructor and assignment. Not move though.
    * need to test after implementing slicing/broadcast etc.
* Implement slicing
* Implement broadcasting
* Implement fixed size
    * Variadic templates, first element of type, the rest as `std::size_t`
    * Partial specialisation for dynamic length
    * Implement common functionality using CRTP
* Begin template expressions
