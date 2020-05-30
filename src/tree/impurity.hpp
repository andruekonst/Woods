#ifndef _TREES_IMPURITY_HPP_
#define _TREES_IMPURITY_HPP_

namespace woods {
namespace tree {

    template<class DType>
    struct VariancePartialImpurity {
        template<class It>
        DType operator() (It begin, It end) {
            size_t n = std::distance(begin, end);
            DType mean = 0;
            for (auto it = begin; it != end; ++it) {
                mean += it->second;
            }
            if (n > 0)
                mean /= n;
            DType var = 0;
            for (auto it = begin; it != end; ++it) {
                const DType dist = it->second - mean;
                var += dist * dist;
            }
            return var;
        }
    };

} // namespace tree
} // namespace woods

#endif // _TREES_IMPURITY_HPP_