from simpleGridworld import GridWorld
import ipdb
import numpy as np
from state_encoder import Net
import torch
import torch.nn as nn

soft = nn.Softmax(2)
x = GridWorld(size=8, time_limit=100, obstacle_percentage=15)
initial_state = x.reset()    
initial_state = torch.FloatTensor(initial_state)
#initial_state.requires_grad = False
PATH = "/mnt/data/mamini/discrete_planning/discrete_planning/With_Validation_2000_epochs_tf_With_noise__With_batch_size_32__with_losses_of_bce_for_state_and_huber_for_rewards"
forward_model = Net(batch_size=1)
forward_model.load_state_dict(torch.load(PATH))
forward_model.eval()

#forward_model.retain_graph = True
for param in forward_model.parameters():
    param.requires_grad = False


learning_rate = 1
K = 10 ##number of rollouts
N = 50##number of gradient updates
T = 50 ##number of time steps
best_reward = - float('Inf')
for kk in range(K):
    random_actions = torch.randn(size=[1, T, 4], dtype=torch.float, requires_grad = True)

    for n in range(N):
        soft_random_actions = soft(random_actions)
        print("n :", n)
        #print(soft_random_actions)
        current_state = initial_state
        total_rewards = 0.0
        for t in range(T):

            encoded_current_state_SD = forward_model.state_encoder_for_SD(current_state.view(1,3,8,8))
            encoded_current_action_SD = forward_model.action_encoder_for_SD(soft_random_actions[0][t].unsqueeze(0))

            encoded_current_state_RD = forward_model.state_encoder_for_RD(current_state.view(1,3,8,8))
            encoded_current_action_RD = forward_model.action_encoder_for_RD(soft_random_actions[0][t].unsqueeze(0))

            predicted_next_state = forward_model.state_decoder(encoded_current_state_SD, encoded_current_action_SD)
            predicted_reward = forward_model.reward_decoder(encoded_current_state_RD, encoded_current_action_RD)

            total_rewards += predicted_reward
            current_state = predicted_next_state

        loss =  total_rewards * (-1)
        print(-loss)
        ##second for loop and all there is to it.
        ## i think this is the problem 
        #ipdb.set_trace()
        loss.backward() 
        print(random_actions.grad)
        
        #print(random_actions.grad)
        random_actions  = random_actions - (random_actions.grad * learning_rate)
        #random_actions.grad.zero_()
        random_actions = random_actions.detach()

        random_actions.requires_grad = True
        print(random_actions.grad)
    if total_rewards > best_reward:
        best_reward = total_rewards
        the_best_plan = random_actions







