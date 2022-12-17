#!/bin/bash

python traffic_ppo.py --network eqgraph --cuda_idx 0 --lr 0.0001 --run_ID 0 --filters 7 5 --strides 2 1 --fcs 64 64 --n_steps 100000
