#!/bin/bash

for n in 50 100 1000 2000
do
  for m in 50 100 1000 2000
  do
    python run_sim.py experiment=simulation sid=1 n=$n m=$m true_rank=$r model.rank=5 trainer.gpus=0
  done
done
