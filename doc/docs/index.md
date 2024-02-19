# QWGraph Documentation

This is the documentation for the python package `qwgraph`. 
This package's aim is to provide an efficient implementation of quantum walks on graphs. 

The quantum walk model implemented in this package follows the paper https://arxiv.org/abs/2310.10451 . This model is specifically designed for searching.

Most of the critical functions of this package are implemented and compiled in rust, providing an efficient simulation.

An example notebook can be found at https://github.com/mroget/qwgraph/blob/main/demo.ipynb

## Installation
### 1. From pip
`pip install qwgraph`
For windows users, it might be necessary to install the rust toolchain cargo.

### 2. From source
```
git clone git@github.com:mroget/qwgraph.git
cd qwgraph
./build.sh
```
The script `build.sh` will compile everything and install it as a python package locally on your machine. It requires to install maturin via
`pip install maturin`

## Dependencies
+ `numpy`
+ `matplotlib`
+ `networkx`
+ `pandas`
+ `tqdm`