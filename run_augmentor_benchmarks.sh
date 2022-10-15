#!/bin/bash

echo "" > out.txt
mkdir -p results
ARGS="EdgeAddition EdgeDropping NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion rLap"
DATASETS="CORA WIKI-CS COAUTHOR-CS COAUTHOR-PHY"
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
for count in {1..50}
do
   python -m memory_profiler  augmentor_benchmarks.py $aug $dataset >> "results/$aug-$dataset.txt"
done
done
done
