#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;

char const* greet() {
   return "hello, world";
}

double mean(np::ndarray &arr) {
    double res = 0.0;
    double *elems = reinterpret_cast<double*>(arr.get_data());
    for (int i = 0; i < arr.shape(0); i++) {
        res += elems[i];
    }
    return res / arr.shape(0);
}

BOOST_PYTHON_MODULE(woods) {
    using namespace boost::python;
    np::initialize();
    def("greet", greet);
    def("mean", mean, (arg("x")));
}