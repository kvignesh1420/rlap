#!/bin/bash

# GPU Disabled 
## Node classification datasets

mkdir -p results/overheads/cpu
ARGS="rLap EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion"
DATASETS="CORA AMAZON-PHOTO PUBMED COAUTHOR-CS COAUTHOR-PHY"
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
FILE="results/overheads/cpu/$aug-$dataset.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
for count in {1..10}
do
   python3 -m memory_profiler augmentor_benchmarks.py "node" $aug $dataset "cpu" >> "results/overheads/cpu/$aug-$dataset.txt"
done
fi
done
done

## Graph classification datasets

ARGS="rLap EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion"
DATASETS="PROTEINS MUTAG IMDB-BINARY IMDB-MULTI NCI1"
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
FILE="results/overheads/cpu/$aug-$dataset.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
for count in {1..10}
do
   python3 -m memory_profiler augmentor_benchmarks.py "graph" $aug $dataset "cpu" >> "results/overheads/cpu/$aug-$dataset.txt"
done
fi
done
done


# GPU Enabled
## Node classification datasets

mkdir -p results/overheads/gpu
ARGS="rLap EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion"
DATASETS="CORA AMAZON-PHOTO PUBMED COAUTHOR-CS COAUTHOR-PHY" 
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
FILE="results/overheads/gpu/$aug-$dataset.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
for count in {1..10}
do
   python3 augmentor_benchmarks.py "node" $aug $dataset "cuda" >> "results/overheads/gpu/$aug-$dataset.txt"
done
fi
done
done

## Graph classification datasets

ARGS="rLap EdgeAddition EdgeDropping EdgeDroppingDegree EdgeDroppingPR EdgeDroppingEVC NodeDropping RandomWalkSubgraph PPRDiffusion MarkovDiffusion"
DATASETS="PROTEINS MUTAG IMDB-BINARY IMDB-MULTI NCI1"
for aug in $ARGS
do
for dataset in $DATASETS
do
echo "$aug $dataset"
FILE="results/overheads/gpu/$aug-$dataset.txt"
if test -f "$FILE"; then
    echo "$FILE exists."
else
for count in {1..10}
do
   python3 augmentor_benchmarks.py "graph" $aug $dataset "cuda" >> "results/overheads/gpu/$aug-$dataset.txt"
done
fi
done
done
