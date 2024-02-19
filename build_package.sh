#!/bin/bash

pip uninstall qwgraph -y
rm -rf target/wheels
maturin build --release
maturin build --release --sdist
echo "Open the MacOS VM through the sosumi command and press any key to continue ..."
read 
zip -r qwgraph.zip ../qwgraph  
scp -P10022 qwgraph.zip localhost:~/qwgraph.zip
ssh -p10022 localhost "unzip qwgraph.zip"
ssh -p10022 localhost "sh ~/qwgraph/build_vm.sh"
scp -r -P10022 localhost:~/qwgraph/target/wheels target/wheels/mac
cp target/wheels/mac/* target/wheels/
rm -rf target/wheels/mac
ssh -p10022 localhost "rm qwgraph.zip"
ssh -p10022 localhost "rm -rf qwgraph"
rm qwgraph.zip
twine upload -r pypi target/wheels/* 
echo "Ensure that the doc has been updated (version included) and press any key to continue ..."
cd doc
mkdocs gh-deploy