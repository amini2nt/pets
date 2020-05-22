
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import ipdb
import pickle
import math
from scipy.stats import bernoulli


class Net(nn.Module):
	def __init__(self, horizon=10, batch_size = 16):
		super(Net,self).__init__()

		self._conv_layers = nn.Sequential(
										nn.Conv2d(3,16,2,padding=1), 
										nn.BatchNorm2d(16),
										nn.ReLU(),
										nn.Conv2d(16,16,2,padding=1),
										nn.BatchNorm2d(16), 
										nn.ReLU(),
										nn.Conv2d(16,20,2,padding=0),
										nn.BatchNorm2d(20), 
										nn.ReLU()
										)
		self._fc_layer = nn.Sequential(
										nn.Linear(20*9*9, 400),
										nn.BatchNorm1d(400), 
										nn.ReLU(),
										nn.Linear(400, 50),
										nn.BatchNorm1d(50),
										nn.ReLU(),
										nn.Linear(50, 16),
										nn.BatchNorm1d(16), 
										nn.ReLU()
										)

		self._fc_layer_for_concatenated = nn.Sequential(
										nn.Linear(32, 24), 
										nn.BatchNorm1d(24),
										nn.ReLU(),
										nn.Linear(24,16),
										nn.BatchNorm1d(16),
										nn.ReLU(),
										nn.Linear(16,1)
										)
		self._next_state = nn.Sequential(
										nn.Linear(20, 16),
										nn.BatchNorm1d(16),
										nn.ReLU()
										)


												
		self._batch_size = batch_size
		self._horizon = horizon
	
	def state_encoder(self, x):
		
		out = self._conv_layers(x)
		out = out.view(len(x), -1) 
		out = self._fc_layer(out)
		return out

	def action_encoder(self, current_action):
		out = self._fc_layer_for_action_for_SD(current_action)
		return out

	def planner(self, initial_obs, goal_obs, plan, learning_rate):
		#ipdb.set_trace()
		soft = nn.Softmax(2)
		soft_plan = soft(plan)
		latent_initial_state = self.state_encoder(initial_obs)
		latent_goal_state = self.state_encoder(goal_obs)
		latent_state = latent_initial_state
		for i in range(self._horizon):
			cat_state_action = torch.cat((latent_state, soft_plan[i]), 1)
			next_state = self._next_state(cat_state_action)
			latent_state = next_state
		loss = torch.sum(nn.SmoothL1Loss(reduction='none')(latent_state, latent_goal_state)) 
		loss.backward()
		plan  = plan - (plan.grad * learning_rate) ### this ruins the gradients for some reason
		return plan.detach()

	def forward(self, initial_obs, goal_obs, plan, n_planner_updates, learning_rate):
		for i in range(n_planner_updates):
			plan = self.planner(initial_obs, goal_obs, plan, learning_rate)
			plan.requires_grad = True
		return plan
		



def vectorized_actions(actions):
	actionsdict = {"n":0, "s":1, "e":2, "w":3} ## This is reverse of w
	e = torch.zeros([1, len(actions), 4], dtype = torch.float)
	for j in range(len(actions)):
			e[0][j][actionsdict[actions[j]]] = 1
	return e

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



def train(model, plan, expert_plan, initial_obs, goal_obs, optimizer, planner_learning_rate, n_planner_updates):
	optimizer.zero_grad()
	plan = model(initial_obs, goal_obs, plan, n_planner_updates, planner_learning_rate)
	loss = torch.sum(nn.NLLLoss(reduction='none')(plan.view(plan.shape[0] * plan.shape[1], -1), expert_plan))
	loss.backward()
	optimizer.step()


def main(opt='adam', training_mode='tf', epochs=10):
	m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/10k_states.pkl', 'rb') as f:
		states = pickle.load(f)
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/10k_actions.pkl', 'rb') as f:
		actions = pickle.load(f)
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/10k_rewards.pkl', 'rb') as f:
		rewards = pickle.load(f)
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/2k_states.pkl', 'rb') as f:
		validation_states = pickle.load(f)
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/2k_actions.pkl', 'rb') as f:
		validation_actions = pickle.load(f)
	with open('/mnt/data/mamini/discrete_planning/discrete_planning/expert/2k_rewards.pkl', 'rb') as f:
		validation_rewards = pickle.load(f) 

	
	myNet = Net(horizon, batch_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	train(model, plan, expert_plan, initial_obs, goal_obs, optimizer, planner_learning_rate, n_planner_updates)
	myNet.cuda()
	data_size = len(states)
	validation_data_size = len(validation_states)

	if opt == "adam":
		optimizer = torch.optim.Adam(myNet.parameters(), lr=0.001)
	elif opt == "rmsprop":
		 optimizer = torch.optim.RMSprop(myNet.parameters())
	
	total_loss = 0
	validation_best_loss = float('Inf')
	for epoch in range(epochs):
		total_loss = 0
		total_state_loss = 0
		total_reward_loss = 0 
		validation_total_loss = 0
		validation_total_state_loss = 0
		validation_total_reward_loss = 0 
		if training_mode =='tf':
			p = 1
		elif training_mode =='no_tf':
			p = 0
		else:
			p = 0.5
		random_shuffle_indices = np.arange(data_size)
		np.random.shuffle(random_shuffle_indices)
		for batch in range(math.ceil(data_size/batch_size)):
		#for batch in range(1):
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > data_size):
				end_index = data_size
			index_list = random_shuffle_indices[start_index:end_index]
			#index_list = np.arange(start_index,end_index).astype('int32')
			states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([states[i] for i in index_list], [rewards[i] for i in index_list], [actions[i] for i in index_list])
			import ipdb
			ipdb.set_trace()
	
			if noise==True:
				little_noise = m.sample(sample_shape=actions_torch.shape).cuda()
				actions_torch = actions_torch + little_noise.squeeze(3)
			my_loss, my_reward_loss, my_state_loss = train(myNet, states_torch, actions_torch, rewards_torch, states_mask, rewards_mask, optimizer, p)

			total_loss += my_loss
			total_state_loss += my_state_loss
			total_reward_loss += my_reward_loss
		for batch in range(math.ceil(validation_data_size/batch_size)):
		#for batch in range(1):
			print(batch)
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > validation_data_size):
				end_index = validation_data_size
			index_list = np.arange(start_index,end_index).astype('int32')
			
			
			validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask = make_mini_batch([validation_states[i] for i in index_list], [validation_rewards[i] for i in index_list], [validation_actions[i] for i in index_list])

			#validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask = make_mini_batch([states[i] for i in index_list], [rewards[i] for i in index_list], [actions[i] for i in index_list])

			###the randomIndices
			##in next line randomIndices[trajectory]
			#states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([states[i] for i in index_list], [rewards[i] for i in index_list], [actions[i] for i in index_list])
			my_validation_loss, my_validation_reward_loss, my_validation_state_loss = test(myNet, validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask)
			#my_validation_loss, my_validation_reward_loss, my_validation_state_loss = test(myNet, validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask)

			validation_total_loss += my_validation_loss
			validation_total_state_loss += my_validation_state_loss
			validation_total_reward_loss += my_validation_reward_loss  
		print("end of epoch")
		total_loss /= math.ceil(data_size/batch_size)
		total_state_loss /= math.ceil(data_size/batch_size)
		total_reward_loss /= math.ceil(data_size/batch_size)
		validation_total_loss /= math.ceil(validation_data_size/batch_size)
		validation_total_state_loss /= math.ceil(validation_data_size/batch_size)
		validation_total_reward_loss /= math.ceil(validation_data_size/batch_size)
		if validation_total_loss < validation_best_loss:
			torch.save(myNet.state_dict(), PATH )
		experiment.log_metric("total loss",total_loss, step=epoch)
		experiment.log_metric("reward loss", total_reward_loss, step=epoch)
		experiment.log_metric("state loss", total_state_loss, step=epoch)
		experiment.log_metric("total validation", validation_total_loss, step=epoch)
		experiment.log_metric("reward validation loss", validation_total_reward_loss, step=epoch)
		experiment.log_metric("state validation", validation_total_state_loss, step=epoch)









initial_obs = torch.zeros(2,3,8,8)
goal_obs = torch.zeros(2,3,8,8)
horizon = 10
batch_size = 2
plan = torch.randn(size=[horizon, batch_size , 4], dtype=torch.float, requires_grad = True)
expert_plan = torch.randint(low=0, high=4, size=[horizon, batch_size], dtype=torch.long, requires_grad = False).flatten()
planner_learning_rate = 0.01
n_planner_updates = 2
model = Net(horizon, batch_size)

main(opt='adam', training_mode='tf', epochs=10)






