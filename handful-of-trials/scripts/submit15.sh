#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env reacher -exp_id $i -logdir ./logs/reacher/max_iters10 -wandb reacher_max_iters10 -o ctrl_cfg.opt_cfg.cfg.max_iters 10
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env pusher -exp_id $i -logdir ./logs/pusher/max_iters10 -wandb pusher_max_iters10 -o ctrl_cfg.opt_cfg.cfg.max_iters 10
done

for i in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=1 python mbexp.py -env cartpole -exp_id $i -logdir ./logs/cartpole/max_iters10 -wandb cartpole_max_iters10 -o ctrl_cfg.opt_cfg.cfg.max_iters 10
done
