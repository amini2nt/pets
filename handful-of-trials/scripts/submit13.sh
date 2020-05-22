#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/popsize200 -wandb reacher_popsize200 -o ctrl_cfg.opt_cfg.cfg.popsize 200
done

for i in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/popsize800 -wandb reacher_popsize800 -o ctrl_cfg.opt_cfg.cfg.popsize 800
done

for i in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/popsize1000 -wandb reacher_popsize1000 -o ctrl_cfg.opt_cfg.cfg.popsize 1000
done
    