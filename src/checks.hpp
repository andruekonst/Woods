#ifndef _CHECKS_HPP_
#define _CHECKS_HPP_

#include <boost/python/numpy.hpp>

namespace woods {

namespace checks {
    namespace dims {
        namespace np = boost::python::numpy;

        inline bool matrix_input_vector_output(const np::ndarray& x, const np::ndarray& y) {
            return static_cast<int>(x.get_nd()) == 2 && static_cast<int>(y.get_nd()) == 1;
        }

        inline bool compatible_lengths(const np::ndarray& x, const np::ndarray& y) {
            return static_cast<int>(x.shape(0)) == static_cast<int>(y.shape(0));
        }
    }
} // namespace checks

} // namespace woods

#endif // _CHECKS_HPP_