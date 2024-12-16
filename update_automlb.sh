cd automlbenchmark
git stash
git pull origin master
git submodule update
cd ..
cp config/frameworks.yaml automlbenchmark/resources/frameworks.yaml
cp config/constraints.yaml automlbenchmark/resources/constraints.yaml