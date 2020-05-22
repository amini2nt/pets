#goal : find the accuracy of the reward predictor from forward model
## also round up the shit and see how it is working
from simpleGridworld import GridWorld
import ipdb
import numpy as np
from state_encoder import Net
import torch
import torch.nn as nn


def vectorized_action(action_index):
	e = torch.zeros([1, 4], dtype = torch.float)
	e[0][action_index] = 1
	return e

def round_state(state):
	t = torch.Tensor([0.5])  # threshold
	out = (state > t).float() * 1
	return out

def pretty_print(obs):
	print(obs[0]+obs[1]*2+obs[2]*3)


x = GridWorld(size=8, time_limit=100, obstacle_percentage=15)
initial_state = x.reset()	
initial_state = torch.FloatTensor(initial_state)
#initial_state.requires_grad = False

PATH = '/mnt/data/mamini/discrete_planning/discrete_planning/With_Validation_2000_epochs_tf_With_noise__With_batch_size_32__with_losses_of_bce_for_state_and_huber_for_rewards'

forward_model = Net(batch_size=1)
forward_model.load_state_dict(torch.load(PATH))
forward_model.eval()

#forward_model.retain_graph = True
for param in forward_model.parameters():
    param.requires_grad = False
## lets feed them the same actions
current_state = initial_state
pretty_print(initial_state)
actionsdict = {0:"n", 1:"s", 2:"e", 3:"w"}
rev_actionsdict = {"n":0, "s":1, "e":2, "w":3} ## This is reverse of w

done = False
while(done!=True):
	x_action = input("next action: ")
	random_action_index = rev_actionsdict[x_action]
	forward_model_action = vectorized_action(random_action_index)
	obs, reward, done = x.step(x_action) ###ground truth
	print(x_action)
	#pretty_print(obs)

	enc_state_for_SD = forward_model.state_encoder_for_SD(current_state.view(1,3,8,8))
	enc_action_for_SD = forward_model.action_encoder_for_SD(forward_model_action)

	enc_state_for_RD = forward_model.state_encoder_for_RD(current_state.view(1,3,8,8))
	enc_action_for_RD = forward_model.action_encoder_for_RD(forward_model_action)

	predicted_reward = forward_model.reward_decoder(enc_state_for_RD, enc_action_for_RD)
	predicted_state = forward_model.state_decoder(enc_state_for_SD, enc_action_for_SD)

	'''

	encoded_current_state = forward_model.state_encoder_for_SD(current_state.view(1,3,8,8))
	encoded_current_action = forward_model.action_encoder_for_RD(forward_model_action)
	predicted_reward = forward_model.reward_decoder(encoded_current_state, encoded_current_action)
	predicted_state = forward_model.state_decoder(encoded_current_state, encoded_current_action)
	
	'''


	pretty_print(round_state(predicted_state.squeeze(0)))
	print(predicted_reward, "    ", reward)

	current_state = torch.FloatTensor(obs)


