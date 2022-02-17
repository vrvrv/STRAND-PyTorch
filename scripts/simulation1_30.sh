#!/bin/bash

for r in 5 10 20 30
do
  for n in 50 100 1000 3000
  do
    for m in 50 100 1000 3000
    do
      python run_sim.py experiment=simulation_5 n=$n m=$m true_rank=$r model.rank=20 trainer.gpus=2
    done
  done
done