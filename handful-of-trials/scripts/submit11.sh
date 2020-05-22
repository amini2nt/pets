#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/numnets15 -wandb reacher_numnets15 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 15
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/numnets15 -wandb pusher_numnets15 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 15
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/numnets15 -wandb cartpole_numnets15 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 15
done