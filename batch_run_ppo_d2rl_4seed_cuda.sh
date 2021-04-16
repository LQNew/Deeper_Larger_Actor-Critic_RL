#!/bin/bash
ENV=$1
policy=$2
CudaNum=$3

# Script to reproduce results
for ((i=20;i<24;i+=4))
do 
	CUDA_VISIBLE_DEVICES=$CudaNum python main_pg_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $i \
		--cpus 1 \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_pg_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+1] \
		--cpus 1 \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_pg_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+2] \
		--cpus 1 \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_pg_d2rl.py \
		--policy ${policy} \
		--env $ENV \
		--seed $[$i+3] \
		--cpus 1 \
		--exp_name "${policy}-${ENV}"
done
