#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --hidden_units 64  --num_blocks 4 --num_heads 4
