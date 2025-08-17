#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter infer workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u infer.py --hidden_units 64  --num_blocks 4 --num_heads 4 
