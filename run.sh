#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --hidden_units 128  --num_blocks 8 --num_heads 4 --lr 0.001 --norm_first --dropout_rate 0.5
