#!/bin/bash
ENV=$1
policy=$2
CudaNum=$3
layer_norm=$4

# Script to reproduce results
for ((i=24;i<28;i+=4))
do 
	CUDA_VISIBLE_DEVICES=$CudaNum python main_ofe.py \
		--policy ${policy} \
		--env $ENV \
		--seed $i \
		--layer_norm $layer_norm \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_ofe.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+1] \
		--layer_norm $layer_norm \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_ofe.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+2] \
		--layer_norm $layer_norm \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_ofe.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+3] \
		--layer_norm $layer_norm \
		--exp_name "${policy}-${ENV}"
done
