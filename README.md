# Woods: Decision Tree Ensembles

Currently implemented algorithms:
1. Partially randomized decision tree (variance minimization).
2. Gradient Boosting of decision trees (MSE minimization).
3. Average ensemble of GBM.
4. Deep Gradient Boosting (of Average ensembles of GBM).

## TODO
* Implement median-split, best-split decision tree;
* Provide optional min&max search based on pre-sorting (find min&max of `array[indices]`);
* Add different loss-functions, ranking support.

## Installation

### Build environment
1. Install `rustup`.
2. Set up nightly toolchain:
```
rustup toolchain install nightly
rustup default nightly
```
### Install Python extension
Run `setup.py`:
```
python setup.py install --user
```

Note that `--user` option is used to install package locally.

### Build documentation
Go to `rust` dir and run:
```
cargo doc --lib
```

Docs will be placed in `target/doc/woods/index.html`.
