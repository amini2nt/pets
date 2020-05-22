import numpy as np
import pickle


def stats(split, num_traj, max_steps, traj_type):
	with open("trajectories/{}/rewards_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		rewards_traj = pickle.load(f)

	successful = 0.0
	min_length = max_steps + 1
	max_length = -1
	avg_length = 0.0

	for i in range(len(rewards_traj)):
		if rewards_traj[i][-1] == 1:
			successful += 1
		if len(rewards_traj[i]) < min_length:
			min_length = len(rewards_traj[i])
		if len(rewards_traj[i]) > max_length:
			max_length = len(rewards_traj[i])
		avg_length += len(rewards_traj[i])

	print("For {} {}".formt(traj_type, split))
	print("Fraction of successful trajectories : ", successful/len(rewards_traj)) 
	print("Min trajectory length : ", min_length)
	print("Max trajectory length : ", max_length)
	print("Avg trajectory length : ", avg_length/len(rewards_traj))

if __name__ == '__main__':
	stats("train", 10000, 100, "random")
	stats("valid", 2000, 100, "random")
	stats("train", 10000, 100, "expert")
	stats("valid", 2000, 100, "expert")


