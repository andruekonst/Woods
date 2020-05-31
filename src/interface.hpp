#ifndef _INTERFACES_HPP_
#define _INTERFACES_HPP_

#include <vector>
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;

namespace woods {
namespace interface {

    template<class DType, class EstimatorImplementation>
    class EstimatorInterface : public EstimatorImplementation {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;
    public:
        void fit(const np::ndarray &x, const np::ndarray &y, const unsigned random_seed) {
            assert(check::dims::is_matrix(x));
            assert(check::dims::is_vector(y));
            assert(check::dims::compatible_lengths(x, y));

            // make array column-wise
            Matrix columns = utils::matrix_to_columns<DType>(x);
            Column target  = utils::to_column<DType>(y);

            fit_impl(columns, target, random_seed);
        }

        np::ndarray predict(const np::ndarray &x) {
            assert(checks::dims::is_matrix(x));

            Matrix columns = utils::matrix_to_columns<DType>(x);
            Column predictions = predict_impl(columns);

            return utils::to_ndarray(predictions);
        }

        // virtual void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) {}
        // virtual Column predict_impl(const Matrix &columns) = 0;
    };


    template<class DType>
    class Estimator {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;
    public:
        void fit(const np::ndarray &x, const np::ndarray &y, const unsigned random_seed) {
            assert(checks::dims::is_matrix(x));
            assert(checks::dims::is_vector(y));
            assert(checks::dims::compatible_lengths(x, y));

            // make array column-wise
            Matrix columns = utils::matrix_to_columns<DType>(x);
            Column target  = utils::to_column<DType>(y);

            fit_impl(columns, target, random_seed);
        }

        np::ndarray predict(const np::ndarray &x) {
            assert(checks::dims::is_matrix(x));

            Matrix columns = utils::matrix_to_columns<DType>(x);
            Column predictions = predict_impl(columns);

            return utils::to_ndarray(predictions);
        }

        virtual void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) {}
        virtual Column predict_impl(const Matrix &columns) = 0;
    };

} // namespace interface
} // namespace woods

#endif // _INTERFACES_HPP_