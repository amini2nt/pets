import numpy as np
import ipdb


class GridWorld(object):
	
	def __init__(self, size=8, time_limit=100, obstacle_percentage=20):
		
		self._size = size
		self._time_limit = time_limit
		self._obstacle_percentage = obstacle_percentage

		self._number_of_obstacles = np.ceil(self._size * self._size * self._obstacle_percentage / 100).astype(int)
		self._obstacles = np.zeros(shape=(self._size, self._size), dtype=np.uint8)
		self._agent = np.zeros(shape=(self._size, self._size), dtype=np.uint8)
		self._goal = np.zeros(shape=(self._size, self._size), dtype=np.uint8)
		self._agent_index = np.zeros(shape=(2), dtype=np.int)
		self._goal_index = np.zeros(shape=(2), dtype=np.int)
		self._time_alive = 0 

	def get_agent_location(self):
		return self._agent_index	

	def get_goal_location(self):
		return self._goal_index

	def get_current_observation(self):
		return np.asarray([self._obstacles, self._agent, self._goal])

	def reset(self):

		positions = np.random.choice(self._size * self._size, size=(self._number_of_obstacles+2), replace=False)
		
		self._obstacles *= 0
		for i in range(self._number_of_obstacles):
			self._obstacles[positions[i]//self._size][positions[i]%self._size] = 1

		self._agent *= 0
		self._agent_index[0] = positions[self._number_of_obstacles]//self._size 
		self._agent_index[1] = positions[self._number_of_obstacles]%self._size
		self._agent[self._agent_index[0]][self._agent_index[1]] = 1

		self._goal *= 0
		self._goal_index[0] = positions[self._number_of_obstacles+1]//self._size  
		self._goal_index[1] = positions[self._number_of_obstacles+1]%self._size
		self._goal[self._goal_index[0]][self._goal_index[1]] = 1

		self._time_alive = 0

		observation = np.asarray([self._obstacles, self._agent, self._goal])
		
		return observation

		
	def get_neighbours_and_corresponding_actions(self, index):
		
		neighbours = []
		actions = []
		#left
		if index[1] != 0: 
			if self._obstacles[index[0]][index[1]-1]==0:
				neighbours.append([index[0], index[1]-1])
				actions.append("w")
		#right
		if index[1] != self._size - 1: 
			if self._obstacles[index[0]][index[1]+1]==0:
				neighbours.append([index[0], index[1]+1])
				actions.append("e")

		#up
		if index[0] != 0: 
			if self._obstacles[index[0]-1][index[1]]==0:
				neighbours.append([index[0]-1, index[1]])
				actions.append("n")

		#down
		if index[0] != self._size - 1: 
			if self._obstacles[index[0]+1][index[1]]==0:
				neighbours.append([index[0]+1, index[1]])
				actions.append("s")

		return neighbours, actions

	def step(self, action):

		self._time_alive += 1
		if self._time_alive >= self._time_limit:
			done = True
			reward = -1

		else:
			if action == "s":
				if self._agent_index[0] != self._size - 1:
					new_agent_x = self._agent_index[0] + 1
					new_agent_y = self._agent_index[1]
				else:
					new_agent_x = self._agent_index[0]
					new_agent_y = self._agent_index[1]

			elif action == "n":
				if self._agent_index[0] != 0:
					new_agent_x = self._agent_index[0] - 1
					new_agent_y = self._agent_index[1]
				else:
					new_agent_x = self._agent_index[0]
					new_agent_y = self._agent_index[1]
					
			elif action == "w":
				if self._agent_index[1] != 0:
					new_agent_x = self._agent_index[0] 
					new_agent_y = self._agent_index[1] - 1
				else:
					new_agent_x = self._agent_index[0]
					new_agent_y = self._agent_index[1]

			elif action == "e":
				if self._agent_index[1] != self._size - 1 :
					new_agent_x = self._agent_index[0] 
					new_agent_y = self._agent_index[1] + 1
				else:
					new_agent_x = self._agent_index[0]
					new_agent_y = self._agent_index[1]
					
			else:
				print("gimme a valid action")
				return

			if self._obstacles[new_agent_x][new_agent_y] == 1:
				reward = -1
				done = True
			
			elif self._goal_index[0] == new_agent_x and self._goal_index[1] == new_agent_y:	
				reward = 1
				done = True
				self._agent[self._agent_index[0]][self._agent_index[1]] = 0
				self._agent_index[0] = new_agent_x
				self._agent_index[1] = new_agent_y
				self._agent[self._agent_index[0]][self._agent_index[1]] = 1
				self._goal[self._agent_index[0]][self._agent_index[1]] = 0
			
			else:
				reward = -0.01
				done = False
				self._agent[self._agent_index[0]][self._agent_index[1]] = 0
				self._agent_index[0] = new_agent_x
				self._agent_index[1] = new_agent_y
				self._agent[self._agent_index[0]][self._agent_index[1]] = 1

		observation = np.asarray([self._obstacles, self._agent, self._goal])

		return observation, reward, done

def play_game(mode="interactive"):

	x = GridWorld(size=8, time_limit=100, obstacle_percentage=15)
	obs = x.reset()	
	print(obs[0]+obs[1]*2+obs[2]*3)
	actionsdict = {0:"n", 1:"s", 2:"e", 3:"w"}
	done = False
	while done!=True:
		if mode == "random":
			action = actionsdict[np.random.randint(4,size=(1))[0]]
		elif mode == "interactive":
			action = input("next action: ")
		obs, reward, done = x.step(action)
		print("action: ", action)
		print("Reward: ",reward)
		print(obs[0]+obs[1]*2+obs[2]*3)

if __name__ == '__main__':
	play_game(mode="interactive")