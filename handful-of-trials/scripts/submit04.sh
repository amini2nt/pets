#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/epochs25 -wandb reacher_epochs25 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 25
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/epochs25 -wandb pusher_epochs25 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 25
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/epochs25 -wandb cartpole_epochs25 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 25
done