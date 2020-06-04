use ndarray::{ArrayView2, ArrayView1, Array1, Array};
use crate::rule::{DecisionRuleImpl, D};

pub struct TreeParameters {
    pub depth: u8,
    pub min_samples_split: usize,
}

pub struct DecisionTreeImpl<Splitter> {
    params: TreeParameters,
    splitters: Vec<Splitter>,
    routes: Vec<[i64; 2]>,
}

/*

int build_tree(const Matrix &columns, const Column &target, const unsigned random_seed,
                int inv_depth, std::vector<I> indices, bool is_first = false) {
    if (inv_depth == 0 || target.size() == 0)
        return -1;

    if (!is_first && indices.size() < min_samples_split)
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

*/

impl<S> DecisionTreeImpl<S> {
    pub fn new(params: TreeParameters) -> Self {
        DecisionTreeImpl {
            params,
            splitters: vec![],
            routes: vec![],
        }
    }

    fn build_tree(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) -> usize {
        0
    }

    pub fn fit(&mut self, columns: ArrayView2<'_, D>, target: ArrayView1<'_, D>) {
        // self.fit_by_indices(columns, target, None);
        self.splitters.clear();
        self.routes.clear();

        let n_nodes = 2usize.pow(self.params.depth.into());
        self.routes.resize(n_nodes, [-1i64; 2]);

    }

    pub fn predict(&self, columns: ArrayView2<'_, D>) -> Array1<D> {
        // columns.row(self.split_info.as_ref().unwrap().feature).iter().map(|val| {
        //     let cond = *val > self.split_info.as_ref().unwrap().threshold;
        //     let index = cond as usize;
        //     self.split_info.as_ref().unwrap().values[index]
        // }).collect::<Array1<D>>()
        columns.row(0).to_owned()
    }
}