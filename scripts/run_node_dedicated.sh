#!/bin/bash

mkdir -p results/final/node/dedicated
ARGS="NodeDropping EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR RandomWalkSubgraph PPRDiffusion MarkovDiffusion EdgeDroppingEVC rLap rLapRandomDesc rLapRandomRandom rLapDegree rLapDegreeDesc rLapDegreeRandom"
DATASETS="CORA AMAZON-PHOTO PUBMED COAUTHOR-CS COAUTHOR-PHY"
NUM_LAYERS=(2 4 8)
WD=(0.00001)
LR=(0.01 0.001 0.0001)
DIMS=(128 256 512)
MODES=('G2L')
FRAC1=(0.0 0.1 0.2 0.3 0.4 0.5)
FRAC2=(0.0 0.1 0.2 0.3 0.4 0.5)
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
FILE="results/final/node/dedicated/$aug-$dataset-$num_layers-$lr-$wd-$dim-$mode.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
for frac1 in ${FRAC1[@]}
do
for frac2 in ${FRAC2[@]}
do
echo "f1=$frac1 f2=$frac2" >> "$FILE"
python3 node_dedicated.py $aug $dataset $num_layers $lr $wd $dim $mode $frac1 $frac2 >> "$FILE"
done # frac1
done # frac2
fi
done # modes
done # dims
done # wd
done # lr
done # num_layers
done # dataset
done # aug
