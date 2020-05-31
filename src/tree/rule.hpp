
#ifndef _TREES_RULE_HPP_
#define _TREES_RULE_HPP_

#include <interface.hpp>
#include <tree/split.hpp>
#include <tree/impurity.hpp>
#include <boost/random.hpp>

namespace woods {
namespace tree {

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class DecisionRuleImpl {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;
        using FitData = std::pair<Matrix, Column>;

        template<class I>
        Split<DType> find_split(const Column &column, const Column &target, boost::random::mt19937 &rng,
                                std::vector<I> * indices = nullptr) {
            size_t pairs_size = (indices == nullptr) ? column.size() : indices->size();
            std::vector<std::pair<DType, DType>> pairs(pairs_size);

            DType min, max;

            if (!indices) {
                // zip-like construction
                for (int i = 0; i < pairs.size(); i++) {
                    pairs[i] = { column[i], target[i] };
                }
                auto min_max = std::minmax_element(column.begin(), column.end());
                min = *min_max.first;
                max = *min_max.second;
            } else {
                min = std::numeric_limits<DType>::max();
                max = std::numeric_limits<DType>::min();
                int pair_index = 0;
                for (int i : *indices) {
                    pairs[pair_index++] = { column[i], target[i] };
                    if (column[i] < min)
                        min = column[i];
                    if (column[i] > max)
                        max = column[i];
                }
            }

            DType threshold;

            switch (type) {
            case SplitType::Mean:
                // mean split
                threshold = (min + max) / 2;
                break;
            case SplitType::Uniform:
                if (max > min) {
                    boost::random::uniform_real_distribution<> uniform(min, max);
                    threshold = uniform(rng);
                } else {
                    threshold = min;
                }
                break;
            case SplitType::TruncatedNormal:
                if (max > min) {
                    DType mean = (min + max) / 2;
                    DType sigma = (max - min) / 3; // inverse three-sigma rule
                    boost::random::normal_distribution<> normal(mean, sigma);
                    // roll the dice until fall into [min, max]
                    do {
                        threshold = normal(rng);
                    } while (threshold < min || threshold > max);
                } else {
                    threshold = min;
                }
                break;
            }

            PartialImpurity partialImpurity;
            DType left_impurity, right_impurity;
            DType left_value, right_value;
            std::tie(left_value, left_impurity, right_value, right_impurity) = partialImpurity(pairs.begin(), pairs.end(), threshold);

            // variance is multiplied by number of "left" elements
            // left_var /= n_left;
            DType impurity = left_impurity + right_impurity;

            return Split<DType> {
                0,           // size_t feature; // dummy value here
                threshold,   // DType  threshold;
                impurity,    // DType  impurity;
                // left_value,  // DType  left_value;
                // right_value, // DType  right_value
                {
                    left_value, // index 0
                    right_value // index 1
                }
            };
        }

    public:
        Split<DType> split_info;

        boost::python::tuple get_split() {
            return boost::python::make_tuple(split_info.threshold, split_info.feature);
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) { // override {
            const size_t n_features = columns.size();

            DType bestImpurity = std::numeric_limits<DType>::max();
            Split<DType> bestSplit = {0};

            boost::random::mt19937 rng(static_cast<unsigned>(random_seed));

            for (size_t j = 0; j < n_features; j++) {
                Split<DType> current = find_split<short>(columns[j], target, rng);
                if (current.impurity < bestImpurity) {
                    bestSplit = current;
                    bestSplit.feature = j;
                }
            }

            // return { bestSplit, bestFeature };
            split_info = bestSplit;
        }
        
        template<class I>
        void fit_by_indices(const Matrix &columns, const Column &target, const unsigned random_seed, std::vector<I> * indices = nullptr) {
            if (!indices) {
                fit_impl(columns, target, random_seed);
            }

            const size_t n_features = columns.size();

            DType bestImpurity = std::numeric_limits<DType>::max();
            Split<DType> bestSplit = {0};

            boost::random::mt19937 rng(static_cast<unsigned>(random_seed));

            for (size_t j = 0; j < n_features; j++) {
                Split<DType> current = find_split(columns[j], target, rng, indices);
                if (current.impurity < bestImpurity) {
                    bestImpurity = current.impurity;
                    bestSplit = current;
                    bestSplit.feature = j;
                }
            }

            // return { bestSplit, bestFeature };
            split_info = bestSplit;
        }

        virtual Column predict_impl(const Matrix &columns) {
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                // predictions[i] = (columns[split_info.feature][i] <= split_info.threshold) ?
                //                     split_info.left_value : split_info.right_value;
                predictions[i] = split_info.values[(columns[split_info.feature][i] > split_info.threshold)];
            }
            return predictions;
        }

        template<class I> // index type
        std::pair<std::vector<I>, std::vector<I>> split_indices(const Matrix &columns, const Column &target) {
            std::vector<I> left_indices, right_indices;
            for (int i = 0; i < target.size(); i++) {
                if (columns[split_info.feature][i] <= split_info.threshold) {
                    // go left
                    left_indices.push_back(i);
                } else {
                    // go right
                    right_indices.push_back(i);
                }
            }
            return { left_indices, right_indices };
        }
        template<class I> // index type
        std::pair<std::vector<I>, std::vector<I>> split_indices(const Matrix &columns, const Column &target,
                                                                const std::vector<I> indices) {
            std::vector<I> left_indices, right_indices;
            for (int i : indices) {
                if (columns[split_info.feature][i] <= split_info.threshold) {
                    // go left
                    left_indices.push_back(i);
                } else {
                    // go right
                    right_indices.push_back(i);
                }
            }
            return { left_indices, right_indices };
        }

        // std::pair<FitData, FitData> split(const Matrix &columns, const Column &target) {
        //     Matrix left_columns(columns.size());
        //     Matrix right_columns(columns.size());
        //     Column left_target, right_target;

        //     for (int i = 0; i < target.size(); i++) {
        //         if (columns[split_info.feature][i] <= split_info.threshold) {
        //             // go left
        //             for (int j = 0; j < columns.size(); j++) {
        //                 left_columns[j].push_back(columns[j][i]);
        //             }
        //             left_target.push_back(target[i]);
        //         } else {
        //             // go right
        //             for (int j = 0; j < columns.size(); j++) {
        //                 right_columns[j].push_back(columns[j][i]);
        //             }
        //             right_target.push_back(target[i]);
        //         }
        //     }
        //     return {
        //         { left_columns, left_target },
        //         { right_columns, right_target }
        //     };
        // }
    };


    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    using DecisionRule = interface::EstimatorInterface<DType, DecisionRuleImpl<DType, type, PartialImpurity>>;

} // namespace tree
} // namespace woods

#endif // _TREES_RULE_HPP_