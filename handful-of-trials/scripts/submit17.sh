#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/epochs200 -wandb pusher_epochs200 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 200
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/epochs200 -wandb reacher_epochs200 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 200
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/epochs200 -wandb cartpole_epochs200 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 200
done