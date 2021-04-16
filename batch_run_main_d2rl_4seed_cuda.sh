#!/bin/bash
ENV=$1
policy=$2
CudaNum=$3

# Script to reproduce results
for ((i=24;i<28;i+=4))
do 
	CUDA_VISIBLE_DEVICES=$CudaNum python main_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $i \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+1] \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+2] \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+3] \
		--exp_name "${policy}-${ENV}"
done
