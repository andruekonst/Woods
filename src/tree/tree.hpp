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
#include <array>
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

    template<class DType, class SplitRule>
    class DecisionTreeImpl {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        using Splitter = SplitRule;
        TreeNode<Splitter> * tree = nullptr;
        std::vector<Splitter> splitters;
        std::unordered_map<int, int> route_left;
        std::unordered_map<int, int> route_right;

        // std::array<std::vector<int>, 2> routes;
        std::vector<std::array<int, 2>> routes;

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

            Splitter splitter;
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
            int left_index = flatten_tree(node->left);
            int right_index = flatten_tree(node->right);
            route_left[index] = left_index;
            route_right[index] = right_index;
            return index;
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) { // override {
            const size_t n_features = columns.size();
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

            // routes[0].resize(route_left.size());
            // routes[1].resize(route_right.size());
            // std::fill(routes[0].begin(), routes[0].end(), -1);
            // std::fill(routes[1].begin(), routes[1].end(), -1);
            // for (auto &p : route_left) {
            //     routes[0][p.first] = p.second;
            // }
            // for (auto &p : route_right) {
            //     routes[1][p.first] = p.second;
            // }
            routes.resize(std::max(route_left.size(), route_right.size()));
            std::fill(routes.begin(), routes.end(), std::array<int, 2>{-1, -1});
            for (auto &p : route_left) {
                routes[p.first][0] = p.second;
            }
            for (auto &p : route_right) {
                routes[p.first][1] = p.second;
            }
            route_left.clear();
            route_right.clear();

            tree = nullptr;
        }

        virtual Column predict_impl(const Matrix &columns) {
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                int cur = 0;
                DType val;
                // std::cout << "Predict " << i << ": " << std::endl;
                do {
                    auto &split_info = splitters[cur].split_info;
                    // std::cout << "  " << cur << ": " << split_info.feature << " / " << split_info.threshold << std::endl;
                    bool cond = columns[split_info.feature][i] > split_info.threshold;
                    // cur = routes[cond][cur];
                    cur = routes[cur][cond];
                    val = split_info.values[cond];
                    // std::cout << cur << std::endl;
                } while (cur > 0);
                predictions[i] = val;
            }
            return predictions;
        }

        ~DecisionTreeImpl() {
            if (tree)
                clean_node(tree);
        }
    };

    template<class DType, class SplitRule>
    using DecisionTree = interface::EstimatorInterface<DType, DecisionTreeImpl<DType, SplitRule>>;

} // namespace tree
} // namespace woods

#endif // _TREES_TREE_HPP_