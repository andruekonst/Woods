#include <boost/python.hpp>

char const* greet() {
   return "hello, world";
}

BOOST_PYTHON_MODULE(woods) {
    using namespace boost::python;
    def("greet", greet);
}