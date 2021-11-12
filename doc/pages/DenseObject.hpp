// A Doxygen file containing generic information related to dense objects

/*! \page DenseObject Dense object
 *  \tableofcontents
 *
 *  Within the context of Ultramat, a 'dense object' is an array-like data structure in which every element is stored in a \link dense_semicontiguous (semi-)contiguous \endlink fashion.
 *  This is in contrast with sparse representations, in which only non-zero elements are stored. 
 *  Dense objects have many features in common with the 1-dimensional `std::vector` and `std::array` within the C++ standard template library, and to
 *  the N-dimensional arrays found within MatLab or the Python NumPy library.
 *  All Ultramat dense objects are N-dimensional, meaning the  user may choose an arbitrary number of dimensions.
 *  
 *  By default, all dense objects are stored in \link dense_order row-major ordering \endlink,
 *  but \link dense_order column-major ordering \endlink is supported for most operations.
 *
 *  The primary dense classes within Ultramat are #ultra::Dense, #ultra::DenseFixed, and #ultra::DenseView.
 *  The preferred interface is the ultra::Array alias, which may be used in the following ways to create a \f$3\times3\f$ matrix-like array:
 *
 *  ```
 *  Array<double> row_major_array( Shape{3,3} );
 *  // Or:
 *  Array<double>::row_major row_major_array( Shape{3,3} );
 *  // Equivalent to:
 *  Dense<double,DenseOrder::row_major> row_major_array_2( Shape{3,3} );
 *
 *  Array<double>::col_major column_major_array( Shape{3,3} );
 *  // Equivalent to:
 *  Dense<double,DenseOrder::col_major_major> column_major_array_2( Shape{3,3} );
 *
 *  Array<double,3,3> row_major_fixed_size_array;
 *  // Or:
 *  Array<double,3,3>::row_major row_major_fixed_size_array;
 *  // Equivalent to:
 *  DenseFixed<double,DenseOrder::row_major,3,3> row_major_fixed_size_array_2;
 *
 *  Array<double,3,3>::col_major column_major_fixed_size_array;
 *  // Equivalent to:
 *  DenseFixed<double,DenseOrder::col_major,3,3> column_major_fixed_size_array_2;
 *  ```
 *
 *  This page details some of the concepts related to dense objects.
 *
 *  \section dense_shape Shape
 *
 *  The shape of a dense object describes its length in each dimension. It is represented internally by a 1D array-like object 
 *  -- typically `std::vector<std::size_t>` or `std::array<std::size_t,N>`, as encapsulated by the #ultra::shapelike concept.
 *  As examples, a \f$3\times3\f$ matrix has a shape of `[3,3]`, while a vector in 4D space has a shape of `[4]`.
 *  A \f$100\times100\times50\f$ grid, as one may use for a finite-difference computation, has a shape of `[100,100,50]`.
 *
 *  To avoid confusion (and keep the API pretty), Ultramat \ref DenseObject%s may not be used to represent shapes.
 *
 *  \section dense_order Order
 *
 *  Throughout Ultramat, 'Order' refers to row/column-major ordering. To understand this, consider the following \f$3\times3\f$ matrix:
 *
 *  \f$ \left[\matrix{ a_{11} & a_{12} & a_{13} \cr a_{21} & a_{22} & a_{23} \cr a_{31} & a_{32} & a_{33} }\right] \f$
 *
 *  For performance reasons, it is beneficial to store all 9 elements in a contiguous block of memory. There are two reasonable ways
 *  of achieving this:
 *
 *  <table>
 *  <tr><th> Row-Major Ordering <th> Column-Major Ordering
 *  <tr>
 *  <td> \f$ \left[ [ a_{11}, a_{12}, a_{13} ] , [ a_{21} , a_{22} , a_{23} ],[ a_{31} , a_{32} , a_{33} ] \right] \f$
 *  <td> \f$ \left[ [ a_{11}, a_{21}, a_{31} ] , [ a_{12} , a_{22} , a_{32} ],[ a_{13} , a_{23} , a_{33} ] \right] \f$
 *  </table>
 *
 *  The choice of whether to use row-major or column-major ordering is largely arbitrary, and as a result it is common to find that libraries and
 *  languages are at odds with each other. Famously, a 2D array declared in C via `float A[3][3];` is row-major, while a 2D array declared in
 *  Fortran via `real A(3,3)` is column-major -- a common source of contention between two otherwise highly -compatible languages. 
 *  For this reason, it is common to see row-major ordering referred to as 'C order' while column-major ordering is referred to as 'Fortran order' 
 *  or 'F order'. 
 *
 *  The concepts of row- and column-major ordering may be generalised beyond 2D arrays by instead considering them to mean 'last dimension order'
 *  and 'first dimension order'. For example, a \f$6\times5\times4\f$ array may be represented in row-major order as 6 \f$5\times4\f$ row-major 
 *  matrices lain end-to-end in memory, while it may be represented in column-major order as 4 \f$6\times5\f$ column-major matrices lain end-to-end.
 *
 *  Ultramat is row-major ordered by default, in part due to its roots in C/C++, and in part due to its similarity to NumPy (which
 *  is also row-major by default). This choice means Ultramat often will not play nicely with column-major libraries, and some linear algebra
 *  routines optimised for column-major ordering will not run as quickly. This may be rectified by ensuring all dense objects intended
 *  for use with column-major libraries are explicitly declared as column-major.
 *
 *  The ordering of a dense object is determined by the ultra::DenseOrder enum class provided as a template argument. The end-user should
 *  not have to make use of this enum class directly, and should instead use the ultra::Array alias as follows:
 *
 *  ```
 *  Array<double>::row_major A( Shape{4,5,6} );
 *  Array<double>::col_major B( Shape{4,5,6} );
 *  ```
 *  For more information, see the relevant [Wikipedia page](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
 *
 *  \section dense_stride Stride
 *
 *  The stride of a dense object defines the number of memory locations correponding to a single step in each dimension. Similarly to \ref dense_shape,
 *  it is represented internally by a 1D array-like object, but its size is given by the number of dimensions plus 1. To understand this, it
 *  may be helpful to first understand the concept of \link dense_order row- and column-major ordering \endlink.
 *
 *  For an N-dimensional column-major ordered dense object, the n'th element of the stride gives the number of memory positions one must step
 *  through in order to increment 1 in the n'th dimension, where \f$0\le n < N\f$. The N'th element gives the number of memory positions between
 *  the first element and the element 1 position off the end of the dense object. For contiguous objects such as #ultra::Dense and
 *  #ultra::DenseFixed, this is simply the total number of elements in the container, though this is not necessarily true for 
 *  \link dense_semicontiguous semi-contiguous \endlink objects.
 *
 *  Conversely, for an N-dimensional row-major ordered dense object, the n'th element of the stride gives the number of memory positions one must step
 *  through in order to increment 1 in the (n-1)'th dimension, where \f$1\le n \le N\f$. The 0'th element gives the number of memory positions between
 *  the first element and the element 1 position off the end.
 *
 *  To give this some context, consider a \f$4\times5\times6\f$ column-major array. This array would have a stride of `[1,4,20,120]`. If one wishes
 *  to access the element at coordinate `(x,y,z)`, this may achieved by locating the memory address of the first element, and jumping in memory by:
 *
 *  \f$(1\times x)+(4\times y)+(20\times z)\f$
 *
 *  In effect, the memory jump is computed by a simple dot product between the coordinate and the first 3 elements of the stride.
 *
 *  Consider instead a \f$4\times5\times6\f$ row-major array. This array would instead have a stride of `[120,30,6,1]`, and the memory jump
 *  corresponding to the coordinate `(x,y,z)` is given by:
 *
 *  \f$(30\times x)+(6\times y)+(1\times z)\f$
 *
 *  In this case, the memory jump is given by the dot product between the coordinate and the last 3 elements of the stride.
 *
 *  \section dense_view Views
 *
 *  Views are dense data structures that do not 'own' data, but instead refer to existing objects. They may refer to a dense object in its entirety,
 *  or instead may refer to only a subset of it. For instance:
 *
 *  ```
 *  Array<double> grid( Shape{100,100} );
 *  auto grid_without_boundary_elements = grid.view( Slice{1,-1}, Slice{1,-1} );
 *  ```
 *
 *  #ultra::Slice is a lightweight object that permits Python-style slicing. The first element denotes the start of the slice, the second denotes
 *  the end of the slice (more accurately, one position off from the end), and the optional third element denotes the step size, which can be negative.
 *
 *  Views are also used for more complex manipulations. For instance, they may provide the #ultra::transpose() of a matrix in constant time, simply by swapping
 *  strides, and this can be taken further to a general transpose with the #ultra::permute() function.
 *
 *  A common pitfall associated with views is that of dangling references. If the object that a view refers to goes out of scope, the resulting view
 *  will be undefined, and it's unlikely that a compiler will notice. When returning views from functions, it is recommended to make a call to
 *  #ultra::eval() and convert to a fully-fledged dense object unless you can be confident that the referenced object will always remain in scope.
 *
 *  Views are considered to be  \link dense_semicontiguous semi-contiguous \endlink objects.
 *
 *  \section dense_semicontiguous What do you mean by semi-contiguous?
 *
 *  A contiguous data structure is one in which every element is immediately adjacent to its neighbours in memory. Within Ultramat, a semi-contiguous
 *  data structure is one that can be represented as a collection of strided 1D arrays, also known as _stripes_. For instance, a view over
 *  a subset of an N-dimensional array may exclude the majority of elements, leaving large gaps in memory between each row, but each row of this 
 *  subset may be contiguous, or at least strided (elements aren't adjacent, but are separated by a consistent distance in memory).
 *
 *  \section striped_iteration Striped Iteration
 *
 *  The C++ standard library is heavily dependent on iterators, but as iteration is inherently a 1D operation, it does not
 *  always apply well to N-dimensional objects. The concept applies well when operations occur between
 *  \ref DenseObject%s of the same \ref dense_shape and \link dense_order row/column-major ordering \endlink, 
 *  provided the operation is 'simple' (such as element-wise arithmetic), but falls down once we start taking views, broadcasting,
 *  operating between different orders, or performing more complex operations such as reductions. Though it is possible to design a
 *  1D iterator to deal with most of these problems -- see #ultra::DenseView::Iterator as an example -- it will tend to result in
 *  poor performance due to the large number of checks performed at each increment.
 *
 *  Striped iteration takes advantage of the \link dense_semicontiguous semi-contiguous \endlink nature of \ref DenseObject%s by breaking
 *  them down into a series of 1D strided arrays which can be iterated over trivially. The class #ultra::DenseStripeIndex is used to
 *  iterate over the stripes themselves, and so all N-dimensional operations can be completed in just two nested for-loops: first iterating
 *  over all stripes, and then iterating across each stripe.
 *
 *  \section dense_broadcasting Broadcasting
 *
 *  Much like NumPy, Ultramat supports automatic 'broadcasting' when performing operations between objects with non-matching 
 *  \link dense_shape shapes \endlink. This is primarily achieved via \ref striped_iteration. If a \ref DenseObject is asked to provide
 *  a stripe but is passed a #ultra::DenseStripeIndex with a shape that does not match its own, it may produce a #ultra::DenseStripe
 *  with zero stride in order to simulate an extended dimension. The rules for broadcasting are as follows:
 *
 *  - When provided with two objects with the same number of dimensions, and in a given dimension one object has size N but the other
 *    has size 1, the dimension of size 1 will be 'broadcasted' to size N.
 *  - When provided with two objects with different numbers of dimensions, new dimensions of size 1 may be *prepended* to row-major
 *    objects until the number of dimensions match. These dimensions of length 1 may then be broadcast to the required size. If using
 *    column-major objects, new dimensions are *appended* instead.
 *  - You may not broadcast between mixed \link dense_order row/column major orderings \endlink.
 *
 *  \section linalg_stacks Vector/Matrix Stacking
 *
 *  TODO
 */
