from simpleGridworld import GridWorld
import ipdb
import numpy as np
from state_encoder import Net
from visualize import visualize
import torch
import torch.nn as nn
import pickle


def pretty_print(obs):

	for i in range(obs.shape[1]):
		print(obs[0][i] + obs[1][i]*2 + obs[2][i]*3)

def torch_pretty_print(obs):
	print(obs[0])
	print(obs[1] * 2)
	print(obs[2] * 3)

def make_mini_batch(states, rewards, actions):
	maximum_length = 0
	batch_size = len(states)
	for i in range(len(states)):
		if len(states[i]) > maximum_length:
			maximum_length = len(states[i])

	states_torch = torch.zeros(batch_size, maximum_length, 3, 8, 8)
	actions_torch = torch.zeros(batch_size, maximum_length-1, 4)
	rewards_torch = torch.zeros(batch_size, maximum_length-1)
	states_mask = torch.zeros(batch_size, maximum_length)
	rewards_mask = torch.zeros(batch_size, maximum_length-1)

	for i in range(batch_size):
		states_torch[i][:len(states[i])] = torch.FloatTensor(states[i])
		states_mask[i][:len(states[i])] = 1
		actions_torch[i][:len(actions[i])] = torch.FloatTensor(vectorized_actions(actions[i]))
		rewards_torch[i][:len(rewards[i])] = torch.FloatTensor(rewards[i])
		rewards_mask[i][:len(rewards[i])] = 1
	states_torch.transpose_(0,1)
	states_mask.transpose_(0,1)
	rewards_mask.transpose_(0,1)
	actions_torch.transpose_(0,1)
	rewards_torch.transpose_(0,1)
	return states_torch.cuda(), actions_torch.cuda(), rewards_torch.cuda(), states_mask.cuda(), rewards_mask.cuda()


def vectorized_actions(actions):
	actionsdict = {"n":0, "s":1, "e":2, "w":3} ## This is reverse of w
	e = torch.zeros([1, len(actions), 4], dtype = torch.float)
	for j in range(len(actions)):
			e[0][j][actionsdict[actions[j]]] = 1
	return e

x = GridWorld(size=8, time_limit=100, obstacle_percentage=15)
initial_state = x.reset()	
initial_state = torch.FloatTensor(initial_state)

PATH = 'overfitted_model.pt'
batch_size = 16

forward_model = Net(batch_size).cuda()
forward_model.load_state_dict(torch.load(PATH))
forward_model.eval()

#forward_model.retain_graph = True
for param in forward_model.parameters():
    param.requires_grad = False
split = "train"
num_traj = 10000
traj_type = "expert"
with open("trajectories/{}/rewards_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		rewards_traj = pickle.load(f)
with open("trajectories/{}/states_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
	states_traj = pickle.load(f)
with open("trajectories/{}/actions_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
	actions_traj = pickle.load(f)
batch = 1
batch_size = 16
train_data_size = len(rewards_traj)
random_shuffle_indices = np.arange(train_data_size)
start_index = batch * batch_size
end_index = (batch+1) * batch_size
if(end_index > train_data_size):
	end_index = train_data_size
index_list = random_shuffle_indices[start_index:end_index]
states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([states_traj[i] for i in index_list], [rewards_traj[i] for i in index_list], [actions_traj[i] for i in index_list])
make_positive = torch.nn.ReLU()
for j in range(actions_torch.shape[0]):
	##lets assume Im only visualizing the first trajectory of this batch
	predicted_next_state, predicted_reward = forward_model(states_torch[j], actions_torch[j])
	_ = input("press enter : ")
	##compare by visualizing both states_torch and predicted_next_state
	if(rewards_mask[j, 0] ==1):
		print("ground truth")
		torch_pretty_print(states_torch[j+1][0])
		print("predicted world")
		torch_pretty_print(make_positive (predicted_next_state[0].round()))
		



