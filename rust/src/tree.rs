use ndarray::{ArrayView2, ArrayView1, Array1, Array, Axis};
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

impl DecisionTreeImpl<DecisionRuleImpl> {
    pub fn new(params: TreeParameters) -> Self {
        DecisionTreeImpl {
            params,
            splitters: vec![],
            routes: vec![],
        }
    }

    fn build_tree(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>,
                  indices: Option<&Vec<usize>>, inv_depth: u8) -> i64 {
        if inv_depth == 0 || target.dim() == 0 {
            return -1;
        }
        if let Some(ind) = indices {
            if ind.len() < self.params.min_samples_split {
                return -1;
            }
        }

        let mut splitter = DecisionRuleImpl::new();
        if let None = splitter.fit_by_indices(columns, target, indices) {
            return -1;
        }
        let split_indices = splitter.split_indices(columns, target, indices);
        self.splitters.push(splitter);
        let id = self.splitters.len() - 1;
        let left_id = self.build_tree(columns, target, Some(&split_indices.left), inv_depth - 1);
        let right_id = self.build_tree(columns, target, Some(&split_indices.right), inv_depth - 1);

        self.routes[id][0] = left_id;
        self.routes[id][1] = right_id;

        id as i64
    }

    pub fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        // self.fit_by_indices(columns, target, None);
        self.splitters.clear();
        self.routes.clear();

        let n_nodes = 2usize.pow(self.params.depth.into());
        self.routes.resize(n_nodes, [-1i64; 2]);

        self.build_tree(columns, target, None, self.params.depth);
        // println!("Routes: {:?}", self.routes);
        // println!("Splits: {:?}", self.splitters);
    }

    pub fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        // columns.row(self.split_info.as_ref().unwrap().feature).iter().map(|val| {
        //     let cond = *val > self.split_info.as_ref().unwrap().threshold;
        //     let index = cond as usize;
        //     self.split_info.as_ref().unwrap().values[index]
        // }).collect::<Array1<D>>()
        columns.axis_iter(Axis(1)).map(|features| {
            let mut cur: i64 = 0;
            let mut val = D::default();
            loop {
                let split_info = self.splitters[cur as usize].split_info.as_ref().unwrap();
                let cond: bool = features[split_info.feature] > split_info.threshold;
                cur = self.routes[cur as usize][cond as usize] as i64;
                val = split_info.values[cond as usize];
                if cur < 0 {
                    break;
                }
            }
            val
        }).collect::<Array1<D>>()
        /*
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
        */
    }
}