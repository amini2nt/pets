from simpleGridworld import GridWorld
import numpy as np
import pickle


def np_to_str(vector):
	return str(vector[0]) + "#" + str(vector[1])

def str_to_np(string):
	myList = string.split("#")
	return np.array([int(myList[0]), int(myList[1])])

def bfs_shortest_distance_to_goal(myGridWorld):
	 # keep track of all visited nodes
	explored = []
	# keep track of nodes to be checked
	start = myGridWorld.get_agent_location()
	start = np_to_str(start)
	queue = [start]
	goal = myGridWorld.get_goal_location()

	# keep looping until there are nodes still to be checked
	paths = {}
	paths[start] = ""
	while queue:
		# pop shallowest node (first node) from queue
		node = queue.pop(0)
		if node not in explored:
				# add node to list of checked nodes
				explored.append(node)
				current_node = str_to_np(node)
				neighbours, actions = myGridWorld.get_neighbours_and_corresponding_actions(current_node)
				# add neighbours of node to queue
				for i in range(len(neighbours)):
						queue.append(np_to_str(neighbours[i]))

						paths[np_to_str(neighbours[i])] = paths[node] + actions[i]

						if np_to_str(neighbours[i]) == np_to_str(goal):
								return paths[np_to_str(neighbours[i])]  
	return False

def generate_expert_trajectory(grid_size, obstacle_percentage, max_steps, num_traj, split):
	
	x = GridWorld(size=grid_size, time_limit=max_steps, obstacle_percentage=obstacle_percentage)
	obs = x.reset()
	traj = 0
	states_traj = []
	actions_traj = []
	rewards_traj = []
	
	while traj < num_traj:
		
		obs = x.reset()
		plan = bfs_shortest_distance_to_goal(x)
		if plan == False:
			continue
		traj += 1
		states = []
		actions= []
		rewards = []
		states.append(obs)
		for action in plan:
			obs, reward, done = x.step(action)
			states.append(obs)
			actions.append(action)
			rewards.append(reward)
		states_traj.append(states)
		actions_traj.append(actions)
		rewards_traj.append(rewards)

	with open("trajectories/expert/states_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(states_traj, f)
	with open("trajectories/expert/actions_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(actions_traj, f)
	with open("trajectories/expert/rewards_{}_{}_traj.pkl".format(num_traj, split), 'wb') as f:
		pickle.dump(rewards_traj, f)

if __name__ == '__main__':
	generate_expert_trajectory(8, 10, 100, 10000, "train")
	generate_expert_trajectory(8, 10, 100, 2000, "valid")



