#!/bin/bash

# Restrict JAX to a single GPU (device 0)
#export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_VISIBLE_DEVICES=1
# Run the Python script (adjust the path to point to `teng.py`)
python teng_2d_dirichlet_circ.py --D 0.1 --equation heat --nb_steps 800 --nb_iters_per_step 5 --dt 0.001 --integrator euler --save_dir None --load_model_state_from model_state_circ_2d.pickle --model_seed 1234 --nb_samples 65536 --sampler_seed 4321 --policy_grad_nb_params 1536 --policy_grad_seed 8844 --policy_grad2_nb_params 1024 --policy_grad2_seed 8848
