#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace py = boost::python;
namespace np = boost::python::numpy;

namespace woods {

namespace utils {

    template<class DType>
    inline std::vector<std::vector<DType>> matrix_to_columns(const np::ndarray& x) {
        const int n_samples  = static_cast<int>(x.shape(0));
        const int n_features = static_cast<int>(x.shape(1));
        const long n_elements = n_samples * n_features;

        std::vector<std::vector<DType>> columns(n_features, std::vector<DType>(n_samples));
        const DType* x_data = reinterpret_cast<DType*>(x.get_data());
        
        for (long i = 0; i < n_elements; i++) {
            columns[i % n_features][i / n_features] = x_data[i];
        }
        return columns;
    }

    template<class DType>
    inline std::vector<DType> to_column(const np::ndarray& y) {
        const int n_samples = static_cast<int>(y.shape(0));
        const DType* data = reinterpret_cast<DType*>(y.get_data());
        std::vector<DType> column(n_samples);
        for (int i = 0; i < n_samples; i++) {
            column[i] = data[i];
        }
        return column;
    }

    template<class DType>
    inline np::ndarray to_ndarray(const std::vector<DType> &column) {
        int n_elements = static_cast<int>(column.size());
        py::tuple shape = py::make_tuple(n_elements);
        py::tuple stride = py::make_tuple(sizeof(DType));
        np::dtype dt = np::dtype::get_builtin<DType>();
        np::ndarray output = np::from_data(&column[0], dt, shape, stride, py::object());
        return output.copy();
    }

} // namespace utils

} // namespace woods

#endif // _UTILS_HPP_