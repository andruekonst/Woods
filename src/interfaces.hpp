#ifndef _INTERFACES_HPP_
#define _INTERFACES_HPP_

#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;

namespace woods {

namespace interface {
    class Fit {
    public:
        void fit();
    };

    class Predict {
    public:
        void predict();
    };
} // namespace interface

} // namespace woods

#endif // _INTERFACES_HPP_