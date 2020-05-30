#ifndef _TREES_TREE_HPP_
#define _TREES_TREE_HPP_

#include <interfaces.hpp>
#include <checks.hpp>
#include <utils.hpp>
#include <tree/impurity.hpp>
#include <vector>
#include <unordered_map>
#include <boost/python.hpp>
#include <boost/random.hpp>

#include <iostream>


namespace woods {

namespace tree {

    enum class SplitType {
        Mean, Uniform, TruncatedNormal
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

    template<class DType>
    struct Split {
        size_t feature;
        DType  threshold;
        DType  impurity;
        DType  left_value;
        DType  right_value;
    };

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class GreedyDecisionRule : public Estimator<DType> {
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

            auto middle = std::partition(pairs.begin(), pairs.end(), [threshold](auto &p) {
                return p.first <= threshold;
            });

            PartialImpurity partialImpurity;
            DType left_impurity, right_impurity;
            DType left_value, right_value;
            std::tie(left_value, left_impurity) = partialImpurity(pairs.begin(), middle);
            std::tie(right_value, right_impurity) = partialImpurity(middle, pairs.end());

            // variance is multiplied by number of "left" elements
            // left_var /= n_left;
            DType impurity = left_impurity + right_impurity;

            return Split<DType> {
                0,           // size_t feature; // dummy value here
                threshold,   // DType  threshold;
                impurity,    // DType  impurity;
                left_value,  // DType  left_value;
                right_value, // DType  right_value
            };
        }

    public:
        Split<DType> split_info;

        boost::python::tuple get_split() {
            return boost::python::make_tuple(split_info.threshold, split_info.feature);
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) override {
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

        virtual Column predict_impl(const Matrix &columns) override {
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                predictions[i] = (columns[split_info.feature][i] <= split_info.threshold) ?
                                    split_info.left_value : split_info.right_value;
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


    template<class Value>
    struct TreeNode {
        Value value;
        TreeNode *left;
        TreeNode *right;
    };

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class RandomizedDecisionTree : public Estimator<DType> {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        // std::vector<GreedyDecisionRule<DType, type, PartialImpurity>> splitters;
        // std::vector<std::pair<int, int>> routes;
        using Splitter = GreedyDecisionRule<DType, type, PartialImpurity>;
        TreeNode<Splitter> * tree = nullptr;
        std::vector<Splitter> splitters;
        std::unordered_map<int, int> route_left;
        std::unordered_map<int, int> route_right;

        // parameters
        int depth = 1;
    public:
        void set_depth(int new_depth) {
            depth = new_depth;
        }

        template<class I>
        TreeNode<Splitter> * make_node(const Matrix &columns, const Column &target, const unsigned random_seed,
                                       int inv_depth, std::vector<I> indices, bool is_first = false) {
            if (inv_depth == 0 || target.size() == 0)
                return nullptr;

            if (!is_first && indices.size() == 0)
                return nullptr;

            // prepare seeds for next nodes
            boost::random::mt19937 rng(static_cast<unsigned>(random_seed));
            boost::random::uniform_int_distribution<> seeds(0);
            
            int left_seed  = seeds(rng);
            int right_seed = seeds(rng);

            GreedyDecisionRule<DType, type, PartialImpurity> splitter;
            if (is_first)
                splitter.fit_impl(columns, target, random_seed);
            else
                splitter.fit_by_indices<I>(columns, target, random_seed, &indices);
            // auto left_right = splitter.split(columns, target);
            auto left_right = is_first ? splitter.split_indices<I>(columns, target) :
                                         splitter.split_indices<I>(columns, target, indices);

            // std::cout << "New node on inverted depth(" << inv_depth << ") = " << left_right.first.size() << " : " << left_right.second.size() << std::endl;
            TreeNode<Splitter> * node = new TreeNode<Splitter>;
            node->value = splitter;
            // node->left = make_node(left_right.first.first, left_right.first.second, left_seed, inv_depth - 1);
            // node->right = make_node(left_right.second.first, left_right.second.second, right_seed, inv_depth - 1);
            node->left = make_node<I>(columns, target, left_seed, inv_depth - 1, left_right.first, false);
            node->right = make_node<I>(columns, target, right_seed, inv_depth - 1, left_right.second, false);

            return node;
        }

        void clean_node(TreeNode<Splitter> * node) {
            if (!node)
                return;
            clean_node(node->left);
            clean_node(node->right);
            delete node;
        }

        int flatten_tree(TreeNode<Splitter> *node) {
            if (!node)
                return -1;
            splitters.push_back(node->value);
            int index = static_cast<int>(splitters.size() - 1);
            // std::cout << "Push splitter[" << index << "] = " << node->value.split_info.threshold << std::endl;
            int left_index = flatten_tree(node->left);
            int right_index = flatten_tree(node->right);
            // route_left.push_back(left_index);
            // route_right.push_back(right_index);
            route_left[index] = left_index;
            route_right[index] = right_index;
            // std::cout << "  splitter[" << index << "]: " << left_index << " " << right_index << std::endl;
            return index;
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) override {
            const size_t n_features = columns.size();
            // splitters.push_back(GreedyDecisionRule<DType, type, PartialImpurity>());
            // splitters.back().fit_impl(columns, target, seed);
            // auto split = splitters.back().split_info;
            using Ind = short;

            if (tree)
                clean_node(tree);
            tree = make_node<Ind>(columns, target, random_seed, depth, std::vector<Ind>(), true);

            // flatten tree
            splitters.clear();
            route_left.clear();
            route_right.clear();

            flatten_tree(tree);
            clean_node(tree);
            tree = nullptr;
        }

        virtual Column predict_impl(const Matrix &columns) override {
            // return splitters.back().predict_impl(columns);
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                // TreeNode<Splitter> * cur = tree;
                int cur = 0;
                DType val;
                // std::cout << "Predict " << i << ": " << std::endl;
                do {
                    auto &split_info = splitters[cur].split_info;
                    // std::cout << "  " << cur << ": " << split_info.feature << " / " << split_info.threshold << std::endl;
                    // auto &split_info = cur->value.split_info;
                    if (columns[split_info.feature][i] <= split_info.threshold) {
                        // cur = cur->left;
                        cur = route_left[cur];
                        val = split_info.left_value;
                    } else {
                        // cur = cur->right;
                        cur = route_right[cur];
                        val = split_info.right_value;
                    }
                    // std::cout << cur << std::endl;
                } while (cur > 0);
                predictions[i] = val;
            }
            return predictions;
        }

        ~RandomizedDecisionTree() {
            if (tree)
                clean_node(tree);
        }
    };



    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class RandomizedGradientBoosting : public Estimator<DType> {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        using Tree = RandomizedDecisionTree<DType, type, PartialImpurity>;
        std::vector<Tree> trees;
        DType mean = 0;

        // parameters
        DType learning_rate = 0.1;
        int depth = 1;
        int iterations = 100;
    public:
        void set_depth(int new_depth) {
            depth = new_depth;
        }

        void set_learning_rate(DType lr) {
            learning_rate = lr;
        }
        
        void set_iterations(int new_iterations) {
            iterations = new_iterations;
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) override {
            const size_t n_features = columns.size();

            boost::random::mt19937 rng(static_cast<unsigned>(random_seed));
            boost::random::uniform_int_distribution<> seeds(0);

            // compute mean
            // mean = std::accumulate(target.begin(), target.end(), 0) / target.size(); // bug
            mean = 0;
            for (int i = 0; i < target.size(); i++)
                mean += target[i];
            mean /= target.size();
            trees.clear();
            // std::cout << "MEAN: " << mean << std::endl;
            // std::cout << "target size: " << target.size() << std::endl;
            // return;

            Column cur_target(target.size());
            std::transform(target.begin(), target.end(), cur_target.begin(), [this](const DType &el) {
                return el - this->mean;
            });

            for (int iter = 0; iter < iterations; iter++) {
                Tree t;
                // set tree parameters
                t.set_depth(depth);
                t.fit_impl(columns, cur_target, seeds(rng));
                auto tree_predictions = t.predict_impl(columns);
                trees.emplace_back(t);
                for (int i = 0; i < cur_target.size(); i++) {
                    cur_target[i] -= learning_rate * tree_predictions[i];
                }
            }
        }

        virtual Column predict_impl(const Matrix &columns) override {
            // return splitters.back().predict_impl(columns);
            Column predictions(columns[0].size(), mean);
            for (int iter = 0; iter < iterations; iter++) {
                auto tree_predictions = trees[iter].predict_impl(columns);
                for (int i = 0; i < predictions.size(); i++) {
                    predictions[i] += learning_rate * tree_predictions[i];
                }
            }
            return predictions;
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