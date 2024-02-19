#!/bin/bash

cd ~/qwgraph
rm -rf target/wheels
/usr/local/bin/maturin build --release