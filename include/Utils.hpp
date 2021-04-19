#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

// Utils.hpp
//
// Defines any utility functions/structs deemed useful.
// Mostly a lot of template hacking.

#include <cstdlib>
#include <cmath>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <type_traits>
#include <concepts>
#include <ranges>
#include <compare>

namespace ultra {

// Tag declarations

class DenseTag{};

// Type determination

template<class T>
struct types {
    static const bool is_dense = std::is_base_of<DenseTag,T>::value;
};

// Define row/col order enum class

enum class RCOrder { row_major, col_major, mixed_order };
const RCOrder default_rc_order = RCOrder::row_major;


// Declare Array types and preferred interface

template<class T, RCOrder Order> class ArrayImpl;
template<class T, RCOrder Order, std::size_t... Dims> class FixedArrayImpl;

template<class T,std::size_t... Dims>
using Array = std::conditional_t<sizeof...(Dims),FixedArrayImpl<T,default_rc_order,Dims...>,ArrayImpl<T,default_rc_order>>;

// Define read-only enum class

enum class ReadWrite { writeable, read_only };

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type, so should have a relatively intuitive interface by default.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};

} // namespace
#endif
