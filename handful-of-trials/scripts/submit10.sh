#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/numnets10 -wandb reacher_numnets10 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 10
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/numnets10 -wandb pusher_numnets10 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 10
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/numnets10 -wandb cartpole_numnets10 -o ctrl_cfg.prop_cfg.model_init_cfg.num_nets 10
done