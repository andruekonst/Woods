#ifndef _TREES_TREE_HPP_
#define _TREES_TREE_HPP_

#include <interfaces.hpp>
#include <checks.hpp>
#include <utils.hpp>
#include <tree/impurity.hpp>
#include <vector>
#include <boost/python.hpp>
#include <boost/random.hpp>


namespace woods {

namespace tree {

    enum class SplitType {
        Mean, Uniform, TruncatedNormal
    };

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class GreedyDecisionRule {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        std::pair<DType, DType> find_split(const Column &column, const Column &target, boost::random::mt19937 &rng) {
            std::vector<std::pair<DType, DType>> pairs(column.size());

            // zip-like construction
            for (int i = 0; i < pairs.size(); i++) {
                pairs[i] = { column[i], target[i] };
            }

            // maybe it is better to pre-sort (?)
            // std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b) {
            //     return a.first > b.first;
            // });

            // DType min = pairs.front().first;
            // DType max = pairs.back().first;
            // DType median = pairs[pairs.size() / 2].first; // there is a faster way to compute median
            // DType split = median;

            DType split;
            auto min_max = std::minmax_element(column.begin(), column.end());
            DType min = *min_max.first, max = *min_max.second;

            switch (type) {
            case SplitType::Mean:
                // mean split
                split = (min + max) / 2;
                break;
            case SplitType::Uniform:
                if (max > min) {
                    boost::random::uniform_real_distribution<> uniform(min, max);
                    split = uniform(rng);
                } else {
                    split = min;
                }
                break;
            case SplitType::TruncatedNormal:
                if (max > min) {
                    DType mean = (min + max) / 2;
                    DType sigma = (max - min) / 3; // inverse three-sigma rule
                    boost::random::normal_distribution<> normal(mean, sigma);
                    // roll the dice until fall into [min, max]
                    do {
                        split = normal(rng);
                    } while (split < min || split > max);
                } else {
                    split = min;
                }
                break;
            }

            auto middle = std::partition(pairs.begin(), pairs.end(), [split](auto &p) {
                return p.first <= split;
            });

            PartialImpurity partialImpurity;
            DType left_impurity = partialImpurity(pairs.begin(), middle);
            DType right_impurity = partialImpurity(middle, pairs.end());

            // variance is multiplied by number of "left" elements
            // left_var /= n_left;
            DType impurity = left_impurity + right_impurity;

            return { split, impurity };
        }
    public:
        boost::python::tuple fit(const np::ndarray &x, const np::ndarray &y, const unsigned random_seed) {
            assert(checks::dims::matrix_input_vector_output(x, y));
            assert(checks::dims::compatible_lengths(x, y));

            // make array column-wise
            Matrix columns = utils::matrix_to_columns<DType>(x);
            Column target  = utils::to_column<DType>(y);

            auto result = fit_impl(columns, target, random_seed);
            return boost::python::make_tuple(result.first, result.second);
        }

        std::pair<DType, DType> fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) {
            const size_t n_features = columns.size();

            DType bestImpurity = std::numeric_limits<DType>::max();
            DType bestSplit = 0;
            size_t bestFeature = 0;

            boost::random::mt19937 rng(static_cast<unsigned>(random_seed));

            DType split, impurity;
            for (size_t j = 0; j < n_features; j++) {
                std::tie(split, impurity) = find_split(columns[j], target, rng);
                if (impurity < bestImpurity) {
                    bestFeature = j;
                    bestSplit = split;
                    bestImpurity = impurity;
                }
            }

            return { bestSplit, bestFeature };
        }
    };

    template<class FitOperation, class PredictOperation>
    class GreedyRule : public FitOperation, public PredictOperation {

    };

    template<class FitOperation, class PredictOperation>
    class DecisionTree : public FitOperation, public PredictOperation {

    };

} // namespace tree

} // namespace woods

#endif // _TREES_TREE_HPP_