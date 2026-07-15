#!/bin/bash
cd /home/scratch.ameyn_gpu_2/flashinfer_state_output_optimization || exit 1
source env.sh
python _bench_split_streams.py 2>&1 | tee _bench_split_streams.out
