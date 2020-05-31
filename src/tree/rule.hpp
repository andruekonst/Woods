
#ifndef _TREES_RULE_HPP_
#define _TREES_RULE_HPP_

#include <interface.hpp>
#include <tree/split.hpp>
#include <tree/impurity.hpp>
#include <boost/random.hpp>

#include <iostream>

namespace woods {
namespace tree {

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class DecisionRuleImpl {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;
        using FitData = std::pair<Matrix, Column>;
        using PairsVector = std::vector<std::pair<DType, DType>>;

        DType find_threshold(PairsVector &pairs, const DType min, const DType max, const int num,
                             boost::random::mt19937 &rng) {
            /*
             * Convention:
             * Each expensive operation (like sorting) should be performed once,
             * when `num` == 0.
             */
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
            case SplitType::Median:
                if (num == 0) {
                    std::nth_element(pairs.begin(), pairs.begin() + pairs.size() / 2, pairs.end(), [](const auto &a, const auto &b) {
                        return a.first < b.first;
                    });
                }
                // else assume half of vector is sorted
                
                threshold = pairs[pairs.size() / 2].first;
                if (pairs.size() % 2 == 0) {
                    threshold += pairs[pairs.size() / 2 - 1].first;
                    threshold /= 2;
                }
                break;
            case SplitType::Best:
                if (num == 0) {
                    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
                        return a.first < b.first;
                    });
                }
                // else assume vector is sorted

                threshold = (pairs[num].first + pairs[num + 1].first) / 2;
                break;
            }

            return threshold;
        }

        template<class I>
        Split<DType> find_split(const Column &column, const Column &target, boost::random::mt19937 &rng,
                                std::vector<I> * indices = nullptr) {
            size_t pairs_size = (indices == nullptr) ? column.size() : indices->size();
            PairsVector pairs(pairs_size);

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
            PartialImpurity partialImpurity;
            DType left_impurity, right_impurity;
            DType left_value, right_value;

            Split<DType> bestSplit = {0};
            bestSplit.impurity = std::numeric_limits<DType>::max();

            // std::cout << "start search(" << split_iterations << ")" << std::endl;
            for (int iter = 0; iter < split_iterations; iter++) {
                threshold = find_threshold(pairs, min, max, iter, rng);
                std::tie(left_value, left_impurity, right_value, right_impurity) = partialImpurity(pairs.begin(), pairs.end(), threshold);
                // variance is multiplied by number of "left" elements
                DType impurity = left_impurity + right_impurity;

                // choose the best threshold
                if (impurity < bestSplit.impurity) {
                    bestSplit.threshold = threshold;
                    bestSplit.impurity = impurity;
                    bestSplit.values[0] = left_value;
                    bestSplit.values[1] = right_value;
                    // std::cout << "  new split is better: " << threshold << " / " << impurity << std::endl;
                } else {
                    // std::cout << "  old split is ok: " << threshold << " / " << impurity << std::endl;
                }
            }

            return bestSplit;
        }

        // parameters
        int split_iterations = 1;
    public:
        Split<DType> split_info;

        boost::python::tuple get_split() {
            return boost::python::make_tuple(split_info.threshold, split_info.feature);
        }

        void set_split_iterations(int iter) {
            split_iterations = iter;
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) {
            fit_by_indices<short>(columns, target, random_seed, nullptr);
        }
        
        template<class I>
        void fit_by_indices(const Matrix &columns, const Column &target, const unsigned random_seed, std::vector<I> * indices = nullptr) {
            const size_t n_features = columns.size();

            DType bestImpurity = std::numeric_limits<DType>::max();
            Split<DType> bestSplit = {0};

            if (type == SplitType::Best) {
                if (!indices) {
                    split_iterations = target.size() - 1;
                } else {
                    split_iterations = indices->size() - 1;
                }
            }

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

        Column predict_impl(const Matrix &columns) {
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                predictions[i] = split_info.values[(columns[split_info.feature][i] > split_info.threshold)];
            }
            return predictions;
        }

        Column predict_impl_rowwise(const Matrix &rows) {
            Column predictions(columns.size());
            for (int i = 0; i < predictions.size(); i++) {
                predictions[i] = split_info.values[(columns[i][split_info.feature] > split_info.threshold)];
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