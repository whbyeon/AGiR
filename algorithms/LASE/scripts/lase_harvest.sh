#!/bin/bash


for seed in 10 20 30 40 50
do
  python algorithms/LASE/lase_harvest_4a.py SEED=$seed
done
