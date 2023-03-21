#!/bin/bash

mkdir -p results
AUGS="RLAP EA ED ED_DEG ED_PPR ED_EVC MARKOVD ND PPRD RWS"

# CORA
mkdir -p results/CORA
for aug in $AUGS
do
for count in {1..10}
do
echo "CORA : AUG $aug : RUN $count"
FILE="results/CORA/$aug-$count.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
python main.py --aug $aug --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4 > $FILE 
fi
done
done

# CITESEER
mkdir -p results/CITESEER
for aug in $AUGS
do
for count in {1..10}
do
echo "CITESEER : AUG $aug : RUN $count"
FILE="results/CITESEER/$aug-$count.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
python main.py --aug $aug --dataname citeseer --epochs 20 --n_layers 1 --lambd 5e-4 --dfr 0.0 --der 0.4 --lr2 1e-2 --wd2 1e-2 > $FILE 
fi
done
done

# AMAZON-COMPUTERS
mkdir -p results/COMPUTERS
for aug in $AUGS
do
for count in {1..10}
do
echo "COMPUTERS : AUG $aug : RUN $count"
FILE="results/COMPUTERS/$aug-$count.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
python main.py --aug $aug --dataname comp --epochs 50 --lambd 5e-4 --dfr 0.1 --der 0.3 --lr2 1e-2 --wd2 1e-4 > $FILE 
fi
done
done