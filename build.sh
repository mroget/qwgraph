#!/bin/bash

pip uninstall qwgraph -y
maturin build --release
pip install target/wheels/qwgraph-0.1.7-cp310-cp310-manylinux_2_35_x86_64.whl
