#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/popsize200 -wandb pusher_popsize200 -o ctrl_cfg.opt_cfg.cfg.popsize 200
done

for i in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/popsize800 -wandb pusher_popsize800 -o ctrl_cfg.opt_cfg.cfg.popsize 800
done

for i in 1 2 3 4 5 6 7 8 9 10
do
    CUA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/popsize1000 -wandb pusher_popsize1000 -o ctrl_cfg.opt_cfg.cfg.popsize 1000
done