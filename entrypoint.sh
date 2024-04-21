#!/bin/bash

python -m pip install -r requirements.txt
python src/main.py --wandb --model llama2 --lr 1e-3 --n_layer 10 --batch_size 30 --grad_clip 1.0 --iterations 23000