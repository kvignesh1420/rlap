#!/bin/bash

mkdir -p results/overheads
ARGS="EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion rLap"
DATASETS="CORA WIKI-CS COAUTHOR-CS COAUTHOR-PHY"
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
for count in {1..10}
do
   python -m memory_profiler  augmentor_benchmarks.py $aug $dataset >> "results/overheads/$aug-$dataset.txt"
done
done
done
