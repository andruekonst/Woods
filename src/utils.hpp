#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;

namespace woods {

namespace utils {

namespace checks {
    namespace dims {
        inline bool matrix_input_vector_output(const np::ndarray& x, const np::ndarray& y) {
            return static_cast<int>(x.get_nd()) == 2 && static_cast<int>(y.get_nd()) == 1;
        }

        inline bool compatible_lengths(const np::ndarray& x, const np::ndarray& y) {
            return static_cast<int>(x.shape(0)) == static_cast<int>(y.shape(0));
        }
    }
} // namespace checks


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

} // namespace utils

} // namespace woods

#endif // _UTILS_HPP_