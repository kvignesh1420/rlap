#!/bin/bash

mkdir -p results/final/node/shared
ARGS="rLap EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion"
DATASETS="CORA COAUTHOR-CS"
NUM_LAYERS=(2)
WD=(0.00001)
LR=(0.0001)
DIMS=(256)
MODES=('L2L')
for aug in $ARGS
do
for dataset in $DATASETS
do
for num_layers in ${NUM_LAYERS[@]}
do
for lr in ${LR[@]}
do
for wd in ${WD[@]}
do
for dim in ${DIMS[@]}
do
for mode in ${MODES[@]}
do
echo "$aug $dataset $num_layers $lr $wd $dim $mode"
FILE="results/final/node/shared/$aug-$dataset-$num_layers-$lr-$wd-$dim-$mode.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
python shared.py $aug $dataset $num_layers $lr $wd $dim $mode >> "$FILE"
fi
done # modes
done # dims
done # wd
done # lr
done # num_layers
done # dataset
done # aug
