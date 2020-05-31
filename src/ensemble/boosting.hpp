#ifndef _ENSEMBLE_BOOSTING_HPP_
#define _ENSEMBLE_BOOSTING_HPP_

namespace woods {
namespace ensemble {

    template<class DType, class BaseEstimator>
    // class RandomizedGradientBoosting : public Estimator<DType> {
    class GradientBoostingImpl {
        using Column = std::vector<DType>;
        using Matrix = std::vector<Column>;

        using Tree = BaseEstimator;
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

        void fit_impl(const Matrix &columns, const Column &target, const unsigned random_seed) { // override {
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

        Column predict_impl(const Matrix &columns) { // override {
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

    template<class DType, class BaseEstimator>
    using GradientBoosting = interface::EstimatorInterface<DType, GradientBoostingImpl<DType, BaseEstimator>>;


} // namespace ensemble
} // namespace woods

#endif // _ENSEMBLE_BOOSTING_HPP_