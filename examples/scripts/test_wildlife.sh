#!/bin/bash

python wildlife_ppo.py --network eqgraph --cuda_idx 0 --lr 0.0001 --n_agents 2 --run_ID 0 --grid_size 5 --filters 7 5 --strides 2 1 --fcs 64 64 --n_steps 100000
