#ifndef _TREES_TREE_HPP_
#define _TREES_TREE_HPP_

class FitStrategy {
public:
    void fit();
};

class PredictStrategy {
public:
    void predict();
};

template<class FitOperation, class PredictOperation>
class DecisionTree : public FitOperation, public PredictOperation {

};

#endif // _TREES_TREE_HPP_