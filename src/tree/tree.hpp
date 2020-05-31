#ifndef _TREE_TREE_HPP_
#define _TREE_TREE_HPP_

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
    using namespace interface;

    template<class DType, class SplitRule>
    class DecisionTreeImpl {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        using Splitter = SplitRule;
        std::vector<Splitter> splitters;
        std::vector<std::array<int, 2>> routes;

        // parameters
        int depth = 1;
    public:
        void set_depth(int new_depth) {
            depth = new_depth;
        }

        template<class I>
        int build_tree(const Matrix &columns, const Column &target, const unsigned random_seed,
                       int inv_depth, std::vector<I> indices, bool is_first = false) {
            if (inv_depth == 0 || target.size() == 0)
                return -1;

            if (!is_first && indices.size() == 0)
                return -1;

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
            auto left_right = is_first ? splitter.split_indices<I>(columns, target) :
                                         splitter.split_indices<I>(columns, target, indices);

            splitters.emplace_back(splitter);
            int index = static_cast<int>(splitters.size() - 1);
            int left_index = build_tree<I>(columns, target, left_seed, inv_depth - 1, left_right.first, false);
            int right_index = build_tree<I>(columns, target, right_seed, inv_depth - 1, left_right.second, false);
            
            routes[index][0] = left_index;
            routes[index][1] = right_index;

            return index;
        }

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) { // override {
            const size_t n_features = columns.size();
            using Ind = short;

            splitters.clear();
            routes.clear();

            const size_t n_nodes = static_cast<size_t>(std::pow(2, depth));
            routes.resize(n_nodes, std::array<int, 2>{-1, -1});
            build_tree<Ind>(columns, target, random_seed, depth, std::vector<Ind>(), true);
        }

        Column predict_impl(const Matrix &columns) {
            Column predictions(columns[0].size());
            for (int i = 0; i < predictions.size(); i++) {
                int cur = 0;
                DType val;
                // std::cout << "Predict " << i << ": " << std::endl;
                do {
                    auto &split_info = splitters[cur].split_info;
                    // std::cout << "  " << cur << ": " << split_info.feature << " / " << split_info.threshold << std::endl;
                    bool cond = columns[split_info.feature][i] > split_info.threshold;
                    cur = routes[cur][cond];
                    val = split_info.values[cond];
                    // std::cout << cur << std::endl;
                } while (cur > 0);
                predictions[i] = val;
            }
            return predictions;
        }
        
        Column predict_impl_rowwise(const Matrix &rows) {
            Column predictions(rows.size());
            for (int i = 0; i < predictions.size(); i++) {
                int cur = 0;
                DType val;
                const auto &cur_row = rows[i];
                do {
                    auto &split_info = splitters[cur].split_info;
                    bool cond = cur_row[split_info.feature] > split_info.threshold;
                    cur = routes[cur][cond];
                    val = split_info.values[cond];
                } while (cur > 0);
                predictions[i] = val;
            }
            return predictions;
        }
    };

    template<class DType, class SplitRule>
    using DecisionTree = interface::EstimatorInterface<DType, DecisionTreeImpl<DType, SplitRule>>;

} // namespace tree
} // namespace woods

#endif // _TREE_TREE_HPP_