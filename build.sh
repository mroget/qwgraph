#!/bin/bash

pip uninstall qwgraph -y
rm -rf target/wheels/*
maturin build --release
pip install target/wheels/*.whl
