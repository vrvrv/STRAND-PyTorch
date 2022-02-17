#!/bin/bash

for n in 50 100 1000 2000
do
  for m in 50 100 1000 2000
  do
    python run_sim.py experiment=simulation_2 n=$n m=$m true_rank=5 model.rank=5 trainer.gpus=0
  done
done
