#ifndef _TREES_TREE_HPP_
#define _TREES_TREE_HPP_

#include <interface.hpp>
#include <check.hpp>
#include <utils.hpp>
#include <tree/split.hpp>
#include <tree/impurity.hpp>
#include <tree/rule.hpp>
#include <vector>
#include <unordered_map>
#include <boost/python.hpp>
#include <boost/random.hpp>

#include <iostream>


namespace woods {
namespace tree {
    template<class Value>
    struct TreeNode {
        Value value;
        TreeNode *left;
        TreeNode *right;
    };

    using namespace interface;

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    class RandomizedDecisionTreeImpl {
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

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) { // override {
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

        virtual Column predict_impl(const Matrix &columns) { // override {
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

        ~RandomizedDecisionTreeImpl() {
            if (tree)
                clean_node(tree);
        }
    };

    template<class DType, SplitType type = SplitType::Mean, class PartialImpurity = VariancePartialImpurity<DType>>
    using RandomizedDecisionTree = interface::EstimatorInterface<DType, RandomizedDecisionTreeImpl<DType, type, PartialImpurity>>;

} // namespace tree
} // namespace woods

#endif // _TREES_TREE_HPP_