#ifndef _TREES_IMPURITY_HPP_
#define _TREES_IMPURITY_HPP_

namespace woods {
namespace tree {

    template<class DType>
    struct VariancePartialImpurity {
        template<class It>
        inline std::tuple<DType, DType, DType, DType> operator() (It begin, It end, DType threshold) {
            size_t n_left = 0, n_right = 0;
            DType left_mean = 0, right_mean = 0;
            DType left_var = 0, right_var = 0;
            for (auto it = begin; it != end; ++it) {
                const DType target = it->second;
                if (it->first <= threshold) {
                    left_mean += target;
                    left_var += target * target;
                    n_left++;
                } else {
                    right_mean += target;
                    right_var += target * target;
                    n_right++;
                }
            }
            if (n_left > 0) {
                left_mean /= n_left;
                left_var /= n_left;
            }
            if (n_right > 0) {
                right_mean /= n_right;
                right_var /= n_right;
            }

            left_var -= left_mean * left_mean;
            right_var -= right_mean * right_mean;

            // multiply variances by number of samples (impurity = n_left/N * left_var + n_right / N * right_var)
            left_var *= n_left;
            right_var *= n_right;

            return { left_mean, left_var, right_mean, right_var };
        }
    };

} // namespace tree
} // namespace woods

#endif // _TREES_IMPURITY_HPP_