#!/bin/bash

mkdir -p results/GRACE
ARGS="EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion rLap"
DATASETS="CORA WIKI-CS COAUTHOR-CS COAUTHOR-PHY"
NUM_LAYERS=(2 4 8)
LR=(0.01 0.001 0.0001)
WD=(0.001 0.0001 0.00001)
for aug in $ARGS
do
for dataset in $DATASETS
do
for num_layers in $NUM_LAYERS[@]
do
for lr in $LR[@]
do
for wd in $WD[@]
do
echo "$aug $dataset $num_layers $lr $wd"
for count in {1..10}
do
   python GRACE.py $aug $dataset $num_layers $lr $wd >> "results/GRACE/$aug-$dataset-$num_layers-$lr-$wd.txt"
done # main loop
done # wd
done # lr
done # num_layers
done # dataset
done #aug
