
#ifndef _TREES_SPLIT_HPP_
#define _TREES_SPLIT_HPP_

namespace woods {
namespace tree {

    enum class SplitType {
        Mean, Uniform, TruncatedNormal
    };

    template<class DType>
    struct Split {
        size_t feature;
        DType  threshold;
        DType  impurity;
        DType  left_value;
        DType  right_value;
    };

} // namespace tree
} // namespace woods

#endif // _TREES_SPLIT_HPP_