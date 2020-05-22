#!/bin/sh

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/popsize1000elite100 -wandb pusher_popsize1000elite100 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.opt_cfg.cfg.num_elites 100 
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/popsize1000elite100 -wandb reacher_popsize1000elite100 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.opt_cfg.cfg.num_elites 100
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=0 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/popsize1000elite100 -wandb cartpole_popsize1000elite100 -o ctrl_cfg.opt_cfg.cfg.popsize 1000 -o ctrl_cfg.opt_cfg.cfg.num_elites 100
done
