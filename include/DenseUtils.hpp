#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

// DenseUtils.hpp
//
// Defines any utility functions/structs deemed useful.

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

// Define row/col order enum class

enum class DenseOrder { row_major, col_major };
const DenseOrder default_order = DenseOrder::row_major;

// get_common_order
// return row_major if all row_major, col_major if all col_major, and default otherwise

template<class... Ts> struct GetCommonOrderImpl;

template<class T1, class T2, class... Ts>
struct GetCommonOrderImpl<T1,T2,Ts...> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order() == GetCommonOrderImpl<T2,Ts...>::order() ? std::remove_cvref_t<T1>::order() : default_order;
    static constexpr DenseOrder order() { return Order; }
};

template<class T1>
struct GetCommonOrderImpl<T1> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order();
    static constexpr DenseOrder order() { return Order; }
};

template<class... Ts>
struct get_common_order {
    static constexpr DenseOrder Order = GetCommonOrderImpl<Ts...>::order();
    static constexpr DenseOrder order(){ return Order; }
    static constexpr DenseOrder value = Order;
};

// Define 'DenseType' enum class.
// Used by Dense to determine whether to store shape/stride as std::vector or std::array, and whether to restrict reshaping

enum class DenseType : std::size_t { vec=1, mat=2, nd=0 };

// Define read-only enum class

enum class ReadWrite { writeable, read_only };

// Declare Array types and preferred interface

template<class, DenseType, DenseOrder> class Dense;
template<class, DenseOrder, std::size_t...> class DenseFixed;
template<class, ReadWrite> class DenseView;

template<class T,std::size_t... Dims>
using Array = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::nd,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 1)
using Vector = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::vec,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 2)
using Matrix = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::mat,default_order>>;

// is_dense type trait
template<class T> struct is_dense { static constexpr bool value = false; };
template<class T, DenseType Type, DenseOrder Order> struct is_dense<Dense<T,Type,Order>> { static constexpr bool value = true; };
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
template<class T, ReadWrite RW> struct is_dense<DenseView<T,RW>> { static constexpr bool value = true; };

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type, so should have a relatively intuitive interface by default.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};

// Bool class
// Looks like a bool, acts like a bool. Used in place of regular bool in dynamic dense objects to avoid the horrors of std::vector<bool>

class Bool {
    bool _x;
    public:
    Bool() = default;
    Bool( const Bool& ) = default;
    Bool( Bool&& ) = default;
    Bool& operator=( const Bool& ) = default;
    Bool& operator=( Bool&& ) = default;
    inline Bool( bool x) : _x(x) {}
    inline operator bool() const { return _x; }
    inline operator bool&() { return _x; }
};

// Arithmetic concept

template<class T>
concept arithmetic = std::is_arithmetic<T>::value;

} // namespace
#endif
