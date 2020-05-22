#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/epochs40 -wandb reacher_epochs40 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 40
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/epochs40 -wandb pusher_epochs40 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 40
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/epochs40 -wandb cartpole_epochs40 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 40
done