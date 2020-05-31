#ifndef _CHECKS_HPP_
#define _CHECKS_HPP_

#include <boost/python/numpy.hpp>

namespace woods {

namespace check {
    namespace dims {
        namespace np = boost::python::numpy;

        inline bool is_matrix(const np::ndarray& x) {
            return static_cast<int>(x.get_nd()) == 2;
        }

        inline bool is_vector(const np::ndarray& y) {
            return static_cast<int>(y.get_nd()) == 1;
        }

        inline bool compatible_lengths(const np::ndarray& x, const np::ndarray& y) {
            return static_cast<int>(x.shape(0)) == static_cast<int>(y.shape(0));
        }
    }
} // namespace checks

} // namespace woods

#endif // _CHECKS_HPP_