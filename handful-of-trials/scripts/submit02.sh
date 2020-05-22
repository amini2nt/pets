#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/epochs15 -wandb reacher_epochs15 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 15
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/epochs15 -wandb pusher_epochs15 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 15
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/epochs15 -wandb cartpole_epochs15 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 15
done