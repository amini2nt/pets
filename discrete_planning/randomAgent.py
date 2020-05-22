from simpleGridworld import GridWorld
import numpy as np
import pickle


def random_agent(grid_size, obstacle_percentage, max_steps, num_traj, split):

	x = GridWorld(size=grid_size, obstacle_percentage=obstacle_percentage, time_limit=max_steps)
	obs = x.reset()
	actionsdict = {0:"n", 1:"s", 2:"e", 3:"w"}
	traj = 0
	states_traj = []
	actions_traj = []
	rewards_traj = []

	while traj < num_traj:

		obs = x.reset()

		traj +=1
		print(traj)
		done = False
		states = []
		actions= []
		rewards = []
		states.append(obs)
		while done!=True:
			action = actionsdict[np.random.randint(4,size=(1))[0]]
			obs, reward, done = x.step(action)
			states.append(obs)
			actions.append(action)
			rewards.append(reward)
		states_traj.append(states)
		actions_traj.append(actions)
		rewards_traj.append(rewards)
	with open("trajectories/random/states_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(states_traj, f)
	with open("trajectories/random/actions_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(actions_traj, f)
	with open("trajectories/random/rewards_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(rewards_traj, f)

if __name__ == '__main__':
	random_agent(8, 10, 100, 10000, "train")
	random_agent(8, 10, 100, 2000, "valid")