/*
 * Entry point for python lib.
 */
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <tree/tree.hpp>
#include <ensemble/boosting.hpp>

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
    // def("fit", woods::fit<double>, (arg("x"), arg("y")));
    using RandomDecisionRule = woods::tree::DecisionRule<double, woods::tree::SplitType::Uniform>;
    class_<RandomDecisionRule>("RandomDecisionRule")
        .def("fit", &RandomDecisionRule::fit)
        .def("predict", &RandomDecisionRule::predict)
        .def("get_split", &RandomDecisionRule::get_split);

    using Tree = woods::tree::DecisionTree<double, RandomDecisionRule>;
    class_<Tree>("RandomizedDecisionTree")
        .def("set_depth", &Tree::set_depth)
        .def("fit", &Tree::fit)
        .def("predict", &Tree::predict);

    using GradientBoosting = woods::ensemble::GradientBoosting<double, Tree>;
    class_<GradientBoosting>("RandomizedGradientBoosting")
        .def("set_depth", &GradientBoosting::set_depth)
        .def("set_learning_rate", &GradientBoosting::set_learning_rate)
        .def("set_iterations", &GradientBoosting::set_iterations)
        .def("fit", &GradientBoosting::fit)
        .def("predict", &GradientBoosting::predict);
}