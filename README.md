# ultramat

An N-dimensional array and linear algebra library in C++.

Makes use of sophisticated expression templates to optimise operations, while providing a simple Numpy-inspired interface.

Currently a work in progress. Proper versioning will begin after the features listed under 'Coming Soon' are implemented.

## Features

* N-dimensional arrays, either dynamically allocated or fixed-size, row-major ordered (C-style) or column-major ordered
(Fortran-style), all completely interoperable.
* A large collection of mathematical functions, including N-dimensional versions of every function in the cmath library (and more),
reduction operations (min, max, sum, etc), and simple arithmetic operations. All are optimised using expression templates to
ensure no temporary arrays are allocated unnecessarily.
* Take non-contiguous 'views' of arrays using Python-inspired slicing operations. Views may be used in place of arrays in all
operations.
* Automatic Numpy-style broadcasting, allowing significant memory savings and permitting the inclusion of scalars in all
array operations.
* Built-in OpenMP parallelisation, allowing significant performance improvements on multicore machines.

## Examples

The following example showcases the syntax of ultramat.

```
#include <ultramat>
using namespace ultra;
using shape = std::vector<std::size_t>;

// Create a linearly spaced array
Array<double> a = linspace(-50.,49.,100);

// Create a fixed-size 3-dimensional array, initialised with random numbers from a normal
// distribution. Fixed size arrays look similar to dynamically sized arrays, though have
// their dimensions specified in their template parameter lists.
double mean = 5; 
double stddev = 2;
Array<double,10,25,100> b;
b = random_normal( mean, stddev, b.shape());

// Add the two together, raise to the power 3, and divide by 2.
// Automatically 'broadcasts' the scalars and the first array to the size (10,25,100)
// The resulting array will have shape (10,25,100)
Array<double> c = pow(a + b,3)/2;

// Avoid creating any intermediates, and do all of this in one step.
// No temporary arrays are allocated. The linspace and random_normal generators produce
// elements lazily, only when they're needed.
Array<double> d = pow(linspace(-50.,49.,100) + random_normal(mean,stddev,shape{10,25,100}),3)/2;
```

## Features Coming Soon

* Foundational linear algebra operations (dot, cross, matmul, lu_solve), with BLAS and LAPACK support.
* Proper documentation using Doxygen.

## Features Coming at Some Point, Probably

* Further linear algebra, including:
    * Other basics, such as `det`, `norm`, `trace`, `outer`, etc.
    * Various decompositions, such as QR and SVD
    * Iterative solvers
* Finite difference methods
* Fast Fourier Transforms
* Statistics functions
* Sparse arrays and associated methods
* Distributed N-dimensional arrays (though that's a long way off...)

## Requirements

* A C++ compiler supporting the C++20 standard and OpenMP
* cmake

## Installation

Requires cmake. Simply enter the `build` directory and call the following:

```
cmake ..
cmake --build . --target install
```

The installation directory can be specified by instead calling:

```
cmake .. -DCMAKE_INSTALL_PREFIX_PATH=/your/install/path
```

To build unit tests, enter the `unit_testing` directory and call:

```
cmake .
cmake --build .
```

You can also build in debug mode by calling:

```
cmake . -DCMAKE_BUILD_TYPE=DEBUG
```

Unit tests can be run by calling:

```
ctest --verbose
```

from the `unit_testing` directory.

## Tips

### Broadcasting

Broadcasting doesn't work exactly like Numpy. Say you're trying to add a
(3,3) array to a (5,3,3) array:

```
using shape = std::vector<std::size_t>;
ultra::Array<int> x( shape{3,3} );
ultra::Array<int> y( shape{5,3,3} );

/* Fill x and y with something... */

ultra::Array<int> z = x+y;
```

An extra dimension will be effectively _prepended_ onto `x` in order to complete the
additon to `y`. So far, so numpy. However, ultramat deviates when column-major ordering
is used instead:

```
using shape = std::vector<std::size_t>;
ultra::Array<int>::col_major x( shape{3,3} );
ultra::Array<int>::col_major y1( shape{5,3,3} );
ultra::Array<int>::col_major y2( shape{3,3,5} );

/* Fill x, y1, and y2 with something... */

ultra::Array<int>::col_major z1 = x+y1; // FAILS
ultra::Array<int>::col_major z2 = x+y2; // SUCCEEDS

```

As the first dimension is the 'fast' dimension with column-major ordering, it was deemed
inappropriate to prepend new dimensions. Instead, new dimensions are _appended_
onto the shape of column-major objects. 
Consider that, in both cases, `x` is a (3,3) matrix. In the row-major case, `y` is a stack of 5 (3,3) matrices.
However, in the col-major case, `y1` is a stack of 3 (5,3) matrices, while `y2` is a stack of
5 (3,3) matrices as expected.

As broadcasting rules differ for row-major and column-major ordering, they may not be broadcast
together under any circumstances, though they can be used in the same expressions provided their
dimensions match. This may arise in unexpected places, as many objects such as generators are
row-major by default and make use of broadcasting internally.

### Avoid auto

The `auto` keyword is extremely powerful in modern C++, and it is used heavily within
the internals of ultramat. However, in most cases it should be avoided by the end-user,
and typenames should be written out in full. This is because most operations between
ultramat objects return expression objects rather than directly returning the results of
a computation:

```
using shape = std::vector<std::size_t>;
Array<float> p( shape{5,6,7,8} );
Array<float> q( shape{5,6,7,8} );

/* Fill p and q with something interesting */

// Adding two ultramat arrays the correct way:

Array<float> r1 = p + q; // r1 has shape {5,6,7,8}

// Adding two ultrmat arrays incorrectly
auto r2 = p + q;
``` 

In the example above, `r2` is not of type `Array<float>`, as expected, but rather is an
`ElementWiseDenseExpression`, templated over the types it has been provided -- likely not
what the user actually wants.

## Licensing

This project is licensed under the MIT License -- see the [LICENSE.md](License.md) file for details. If you would like to use
ultramat in your own work, there is no need to provide credit, though it would be highly appreciated. Note that this project
is currently a work-in-progress, and liable to change significantly until the first stable build is released.
