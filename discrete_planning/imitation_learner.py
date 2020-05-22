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
	def __init__(self, batch_size = 16, batch_norm = False):
		super(Net,self).__init__()
		self._batch_size = batch_size
		self._conv_layers = nn.Sequential()
		self.imitation_block = nn.Sequential()
		if batch_norm == True:
				self._conv_layers = nn.Sequential(
												nn.Conv2d(3,16,2,padding=1), 
												nn.ReLU(),
												nn.BatchNorm2d(16),# batch x 16 x 8 x 8
												nn.Conv2d(16,16,2,padding=1),  # batch x 16 x 8 x 8
												nn.ReLU(),
												nn.BatchNorm2d(16)
												)
		else:

				self._conv_layers = nn.Sequential(
												nn.Conv2d(3,16,2,padding=1), 
												nn.ReLU(),
												#nn.BatchNorm2d(16),# batch x 16 x 8 x 8
												nn.Conv2d(16,16,2,padding=1),  # batch x 16 x 8 x 8
												nn.ReLU()
												#nn.BatchNorm2d(16)
												)
		self._fc_layer_for_state = nn.Sequential(
										nn.Linear(16*10*10, 16), # batch x 16
										nn.ReLU()
										)
		self._a_couple_fcs_pre_imitation = nn.Sequential(
											nn.Linear(20, 20),
											nn.ReLU(),
											# nn.BatchNorm1d(25),
											# nn.Linear(25, 20),
											# nn.ReLU(),
											nn.BatchNorm1d(20)
											)

		self.imitation_lstm1 = nn.LSTMCell(20, 26)
		self._h1 = None
		self._c1 = None
		self.imitation_lstm2 = nn.LSTMCell(26, 32)
		self._h2 = None
		self._c2 = None
		self.imitiation_fc = nn.Linear(32, 4)
		self.imitation_sigmoid = nn.Sigmoid()
											
	
	def state_encoder(self, x):
		out = self._conv_layers(x)
		out = out.view(len(x), -1) ## batch x 1600 
		out = self._fc_layer_for_state(out)
		return out  
	def imitation_learner(self, x):
		#ipdb.set_trace()
		#x = self._a_couple_fcs_pre_imitation(x)

		if self._h1 is not None:
			out, self._c1 = self.imitation_lstm1(x, (self._h1, self._c1))
		else:
			out, self._c1 = self.imitation_lstm1(x, None)
		self._h1 = out

		if self._h2 is not None:
			out, self._c2 = self.imitation_lstm2(out, (self._h2, self._c2))
		else:
			out, self._c2 = self.imitation_lstm2(out, None)
		self._h2 = out
		out = self.imitiation_fc(out)
		out = self.imitation_sigmoid(out)
		### add sigmoid 
		return out

	def forward(self, image, action):
		enc_state = self.state_encoder(image)
		concatenated_state_action = torch.cat((enc_state, action) , dim=1 )
		next_action = self.imitation_learner(concatenated_state_action)
		return next_action

def linearly_decaying_epsilon(step, decay_period=500, warmup_steps=20, epsilon=0):
	steps_left = decay_period + warmup_steps - step
	bonus = (1.0 - epsilon) * steps_left / decay_period
	bonus = np.clip(bonus, 0., 1. - epsilon)
	return epsilon + bonus




def train(model, states, actions, rewards, states_mask, rewards_mask, optimizer):
	current_state  = states[0]
	prev_action = torch.zeros(size=actions[0].shape).cuda()
	avg_action_loss = 0
	optimizer.zero_grad()
	model._h1 = None
	model._c1 = None
	model._h2 = None
	model._c2 = None
	accuracy = 0
	for j in range(actions.shape[0]):
		#summary(model, [current_state.shape, actions[j].shape])
		predicted_next_action = model(current_state, prev_action)
		
		action_loss = nn.BCELoss(reduction='none')(predicted_next_action , actions[j])
		#print(action_loss)
		avg_action_loss += torch.sum(action_loss * rewards_mask[j].unsqueeze(1) )

		current_state = states[j+1]
		prev_action = actions[j]
	#ipdb.set_trace()
	avg_action_loss /= torch.sum(rewards_mask)
	#print(avg_action_loss)
	avg_action_loss.backward(retain_graph = True)

	optimizer.step()
	return avg_action_loss

def vectorized_actions(actions):
	actionsdict = {"n":0, "s":1, "e":2, "w":3} ## This is reverse of w
	e = torch.zeros([1, len(actions), 4], dtype = torch.float)
	for j in range(len(actions)):
			e[0][j][actionsdict[actions[j]]] = 1
	return e

def eval(model, states, actions, rewards, states_mask, rewards_mask):
	current_state  = states[0]
	prev_action = torch.zeros(size=actions[0].shape).cuda()
	accuracy = 0
	model._h1 = None
	model._c1 = None
	model._h2 = None
	model._c2 = None
	#ipdb.set_trace()
	for j in range(actions.shape[0]):
		predicted_next_action = model(current_state, prev_action)
		_, pred = torch.max(predicted_next_action, 1)
		_, target = torch.max(actions[j], 1)
		current_state = states[j+1]
		prev_action = actions[j]
		#ipdb.set_trace()
		accuracy += ((pred==target).type(torch.FloatTensor).cuda() * rewards_mask[j]).sum()
	#print(accuracy)
	#print(torch.sum(rewards_mask))

	accuracy /= torch.sum(rewards_mask)
	return accuracy


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

def main(epochs, batch_size, learning_rate, batch_norm, opt):
	m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
	#PATH = "the_best_model for "+ " training loss with noise " + str(noise) + " and batch norm "+ str(batch_norm)+ " and optimizer  " + opt ".pt"
	PATH = ""
	PATH += "_2000_epochs_"
	PATH += "_imitation_with_1_fc_"

	PATH += "_With_batch_size_" + str(batch_size)+"_"
	if batch_norm == True:
		PATH += "_With_batch_norm_"
	else:
		PATH += "_Without_batch_norm_"
	PATH += ""
	experiment.add_tag(PATH)
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

	validation_data_size = len(validation_states)
	data_size = len(states)
	myNet = Net(batch_size, batch_norm)
	myNet.cuda()

	validation_states = states
	validation_actions = actions
	validation_rewards = rewards


	best_model = myNet
	if opt == "adam":
		optimizer = torch.optim.Adam(myNet.parameters())
	elif opt == "rmsprop":
		 optimizer = torch.optim.RMSprop(myNet.parameters())
	
	#total_loss = 0
	best_loss = float('Inf')
	#total_accuracy = 0
	best_accuracy = - float('Inf')
	for epoch in range(epochs):
		total_loss = 0
		my_action_accuracy = 0.0
		
		total_loss = 0 ##overfitting
		total_accuracy = 0 ###overfitting
		random_shuffle_indices = np.arange(data_size)
		np.random.shuffle(random_shuffle_indices)
		for batch in range(math.ceil(data_size/batch_size)):

		#for batch in range(1):
			#print(batch)
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > data_size):
				end_index = data_size
			index_list = random_shuffle_indices[start_index:end_index]
			#index_list = np.arange(start_index,end_index).astype('int32') ##overfitting the same batch
			
			states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([states[i] for i in index_list], [rewards[i] for i in index_list], [actions[i] for i in index_list])
			###the randomIndices
			##in next line randomIndices[trajectory]

			my_action_loss = train(myNet, states_torch, actions_torch, rewards_torch, states_mask, rewards_mask, optimizer)
			total_loss += my_action_loss
		for batch in range(math.ceil(validation_data_size/batch_size)):
		#for batch in range(1):
			#print(batch)
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > data_size):
				end_index = data_size

			#index_list = np.arange(start_index,end_index).astype('int32')
			index_list = random_shuffle_indices[start_index:end_index]


			states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([validation_states[i] for i in index_list], [validation_rewards[i] for i in index_list], [validation_actions[i] for i in index_list])

			my_action_accuracy = eval(myNet, states_torch, actions_torch, rewards_torch, states_mask, rewards_mask)
			print(my_action_accuracy)
			total_accuracy += my_action_accuracy
			#print(total_accuracy)
		print("end of epoch")
		total_loss /= math.ceil(data_size/batch_size)
		total_accuracy /= math.ceil(validation_data_size/batch_size)
		if total_accuracy < best_accuracy:
			torch.save(myNet.state_dict(), PATH )
		experiment.log_metric("training loss", total_loss, step=epoch)
		experiment.log_metric("validation accuracy", total_accuracy, step=epoch)


if __name__=="__main__":

	# commenting the following two lines because i want to import the above to my planner.py
	'''
	batch_norm = [True, False]
	opt = ["adam", "rmsprop"]
	noise = [True, False]
	'''
	experiment = Experiment(api_key = "V1ZVYej7DxAyiXRVoeAs4JWZb", project_name = "discrete planning", workspace = "amini2nt", auto_metric_logging = False)

	# for bn in batch_norm:
	# 	for optimix in opt:
	# 		for n in noise:
	# 			main(batch_size = 1, learning_rate = 0.01, epochs = 50, noise = n, batch_norm =bn, opt = optimix) 
	## training mode can be 'tf', 'no_tf', 'ss'
	main(batch_size = 32, learning_rate = 0.01, epochs = 2000, batch_norm = True, opt = "adam")

	# ##the_best_model = Net(batch_size=1)
	##the_best_model.load_state_dict(torch.load(PATH))



