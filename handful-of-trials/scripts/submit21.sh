#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/popsize1000epoch100cem15 -wandb pusher_popsize1000epoch100cem15 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.opt_cfg.cfg.num_elites 100 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 100 -o ctrl_cfg.opt_cfg.cfg.max_iters 15 
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/popsize1000epoch100cem15 -wandb reacher_popsize1000epoch100cem15 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.opt_cfg.cfg.num_elites 100 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 100 -o ctrl_cfg.opt_cfg.cfg.max_iters 15
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/popsize1000epoch100cem15 -wandb cartpole_popsize1000epoch100cem15 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.prop_cfg.model_train_cfg.epochs 100 -o ctrl_cfg.opt_cfg.cfg.max_iters 15
done
