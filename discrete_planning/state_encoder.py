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
	def __init__(self, batch_size = 16):
		super(Net,self).__init__()

		self._conv_layers_for_SD = nn.Sequential(
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
		self._fc_layer_for_state_for_SD = nn.Sequential(
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

		self._fc_layer_for_action_for_SD = nn.Sequential(
										nn.Linear(4, 16),
										nn.BatchNorm1d(16),
										nn.ReLU() 
										) 

		self._conv_layers_for_RD = nn.Sequential(
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

		self._fc_layer_for_state_for_RD = nn.Sequential(
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

		self._fc_layer_for_action_for_RD = nn.Sequential(
										nn.Linear(4, 16),
										nn.BatchNorm1d(16),
										nn.ReLU() 
										) 



		self._deconv_layers = nn.Sequential(
										nn.ConvTranspose2d(32, 16, 4), 
										nn.BatchNorm2d(16),
										nn.ReLU(),
										nn.ConvTranspose2d(16, 3, 5),

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
		self._batch_size = batch_size

	
	def state_encoder_for_SD(self, x):
		out = self._conv_layers_for_SD(x)
		out = out.view(x.shape[0], -1) ## batch x 1600 
		out = self._fc_layer_for_state_for_SD(out)
		return out

	def action_encoder_for_SD(self, current_action):
		out = self._fc_layer_for_action_for_SD(current_action)
		return out

	def state_encoder_for_RD(self, x):
		out = self._conv_layers_for_RD(x)
		out = out.view(len(x), -1) ## batch x 1600 
		out = self._fc_layer_for_state_for_RD(out)
		return out

	def action_encoder_for_RD(self, current_action):
		out = self._fc_layer_for_action_for_RD(current_action)
		return out

	def state_decoder(self,encoded_state, encoded_action):
		concatenated_action_state = torch.cat((encoded_state, encoded_action), 1)

		out = concatenated_action_state.unsqueeze(-1).unsqueeze(-1)
		
		out = self._deconv_layers(out)
		return out

	def reward_decoder(self,encoded_state, encoded_action):
		concatenated_action_state = torch.cat((encoded_state, encoded_action), 1)
		out = self._fc_layer_for_concatenated(concatenated_action_state)
		return out

	def forward(self, image, action):
		enc_state_for_SD = self.state_encoder_for_SD(image)
		enc_action_for_SD = self.action_encoder_for_SD(action)

		enc_state_for_RD = self.state_encoder_for_RD(image)
		enc_action_for_RD = self.action_encoder_for_RD(action)




		predicted_reward = self.reward_decoder(enc_state_for_RD, enc_action_for_RD)
		predicted_state = self.state_decoder(enc_state_for_SD, enc_action_for_SD)

		return predicted_state, predicted_reward


def train(model, states, actions, rewards, states_mask, rewards_mask, optimizer, scheduling_probability):
	current_state  = states[0]
	avg_reward_loss = 0
	avg_state_loss = 0
	optimizer.zero_grad()
	for j in range(actions.shape[0]):
		#summary(model, [current_state.shape, actions[j].shape])
		predicted_next_state, predicted_reward = model(current_state, actions[j])
		#ipdb.set_trace()
		reward_loss = nn.SmoothL1Loss(reduction='none')(predicted_reward.squeeze(1), rewards[j]) * rewards_mask[j]
		reward_loss = torch.sum(reward_loss)
		state_loss = nn.SmoothL1Loss(reduction='none')(predicted_next_state, states[j+1])
		state_loss = torch.sum(state_loss * states_mask[j+1].unsqueeze(1).unsqueeze(2).unsqueeze(3))	
		avg_reward_loss += reward_loss
		avg_state_loss += state_loss
		teacher_forcing = bernoulli.rvs(scheduling_probability)

		if teacher_forcing == 0:
			current_state = predicted_next_state.detach()
			return
		else:
			current_state = states[j+1]

	avg_reward_loss /= torch.sum(rewards_mask)
	avg_state_loss /=  torch.sum(rewards_mask)
	avg_loss = avg_reward_loss + avg_state_loss
	print(avg_loss)
	avg_loss.backward()
	optimizer.step()
	return avg_loss, avg_reward_loss, avg_state_loss

def test(model, states, actions, rewards, states_mask, rewards_mask):
	current_state  = states[0]
	avg_reward_loss = 0
	avg_state_loss = 0
	for j in range(actions.shape[0]):
		predicted_next_state, predicted_reward = model(current_state, actions[j])
		reward_loss = nn.SmoothL1Loss(reduction='none')(predicted_reward.squeeze(1), rewards[j]) * rewards_mask[j]
		reward_loss = torch.sum(reward_loss)
		state_loss = nn.SmoothL1Loss(reduction='none')(predicted_next_state, states[j+1])
		state_loss = torch.sum(state_loss * states_mask[j+1].unsqueeze(1).unsqueeze(2).unsqueeze(3))	
		avg_reward_loss += reward_loss
		avg_state_loss += state_loss

		# no teacher forcing
		#current_state = predicted_next_state.detach()
		current_state = states[j+1]
		

	avg_reward_loss /= torch.sum(rewards_mask)
	avg_state_loss /=  torch.sum(rewards_mask)
	avg_loss = avg_reward_loss + avg_state_loss
	print(avg_loss)
	return avg_loss, avg_reward_loss, avg_state_loss


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

def main(batch_size,  epochs, noise,  opt,  training_mode, traj_type):

	experiment = Experiment(api_key = "V1ZVYej7DxAyiXRVoeAs4JWZb", project_name = "discrete_planning", workspace = "amini2nt", auto_metric_logging = False)
	
	noise_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
	PATH = "the_best_model for " + " training loss with noise " + str(noise) + " and optimizer  " + opt + ".pt"
	
	experiment.add_tag(PATH)
	
	split = "train"
	num_traj = 10000
	with open("trajectories/{}/rewards_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		rewards_traj = pickle.load(f)
	with open("trajectories/{}/states_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		states_traj = pickle.load(f)
	with open("trajectories/{}/actions_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		actions_traj = pickle.load(f)
	
	split = "valid"
	num_traj = 2000
	with open("trajectories/{}/rewards_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		rewards_traj_valid = pickle.load(f)
	with open("trajectories/{}/states_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		states_traj_valid = pickle.load(f)
	with open("trajectories/{}/actions_{}_{}_traj.pkl".format(traj_type, num_traj, split), 'rb') as f:
		actions_traj_valid = pickle.load(f)

	train_data_size = len(rewards_traj)
	valid_data_size = len(rewards_traj_valid)

	myNet = Net(batch_size=batch_size).cuda()

	if opt == "adam":
		optimizer = torch.optim.Adam(myNet.parameters(), lr=0.001)
	elif opt == "rmsprop":
		 optimizer = torch.optim.RMSprop(myNet.parameters())
	
	valid_best_loss = float('Inf')
	for epoch in range(epochs):
		train_total_loss = 0.0
		train_total_state_loss = 0.0
		train_total_reward_loss = 0.0 
		valid_total_loss = 0.0
		valid_total_state_loss = 0.0
		valid_total_reward_loss = 0.0 
		if training_mode =='tf':
			p = 1
		elif training_mode =='no_tf':
			p = 0
		else:
			p = 0.5
		random_shuffle_indices = np.arange(train_data_size)
		#np.random.shuffle(random_shuffle_indices)
		myNet.train()
		#for batch in range(math.ceil(train_data_size/batch_size)):
		for batch in range(1):
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > train_data_size):
				end_index = train_data_size
			index_list = random_shuffle_indices[start_index:end_index]
			states_torch, actions_torch, rewards_torch, states_mask, rewards_mask = make_mini_batch([states_traj[i] for i in index_list], [rewards_traj[i] for i in index_list], [actions_traj[i] for i in index_list])
			
			if noise == True:
				little_noise = noise_dist.sample(sample_shape=actions_torch.shape).cuda()
				actions_torch = actions_torch + little_noise.squeeze(3)
			
			my_loss, my_reward_loss, my_state_loss = train(myNet, states_torch, actions_torch, rewards_torch, states_mask, rewards_mask, optimizer, p)

			train_total_loss += my_loss
			train_total_state_loss += my_state_loss
			train_total_reward_loss += my_reward_loss

		myNet.eval()
		#for batch in range(math.ceil(valid_data_size/batch_size)):
		for batch in range(1):
			start_index = batch * batch_size
			end_index = (batch+1) * batch_size
			if(end_index > valid_data_size):
				end_index = valid_data_size
			index_list = np.arange(start_index, end_index).astype('int32')
						
			validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask = make_mini_batch([states_traj[i] for i in index_list], [rewards_traj[i] for i in index_list], [actions_traj[i] for i in index_list])

			my_validation_loss, my_validation_reward_loss, my_validation_state_loss = test(myNet, validation_states_torch, validation_actions_torch, validation_rewards_torch, validation_states_mask, validation_rewards_mask)

			valid_total_loss += my_validation_loss
			valid_total_state_loss += my_validation_state_loss
			valid_total_reward_loss += my_validation_reward_loss
		
		print("end of epoch")
		#train_total_loss /= math.ceil(train_data_size/batch_size)
		print(train_total_loss)
		#train_total_state_loss /= math.ceil(train_data_size/batch_size)
		#train_total_reward_loss /= math.ceil(train_data_size/batch_size)
		
		#valid_total_loss /= math.ceil(valid_data_size/batch_size)
		#valid_total_state_loss /= math.ceil(valid_data_size/batch_size)
		#valid_total_reward_loss /= math.ceil(valid_data_size/batch_size)
		if valid_total_loss < valid_best_loss:
			torch.save(myNet.state_dict(), PATH)
		experiment.log_metric("total loss",train_total_loss, step=epoch)
		experiment.log_metric("reward loss", train_total_reward_loss, step=epoch)
		experiment.log_metric("state loss", train_total_state_loss, step=epoch)
		experiment.log_metric("total validation", valid_total_loss, step=epoch)
		experiment.log_metric("reward validation loss", valid_total_reward_loss, step=epoch)
		experiment.log_metric("state validation", valid_total_state_loss, step=epoch)


if __name__=="__main__":
	## training mode can be 'tf', 'no_tf', 'ss'
	main(batch_size = 16,  epochs = 4000, noise = True,  opt = "adam",  training_mode = 'tf', traj_type = "expert")


