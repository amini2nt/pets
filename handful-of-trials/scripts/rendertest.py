import gym
import numpy as np
from PIL import Image



env = gym.make("CartPole-v1")
observation = env.reset()
pic = env.render(mode="rgb_array", close=False)
im = Image.fromarray(pic)
im.save("cartpole.jpeg")

env.close()

env = gym.make("Reacher-v1")
observation = env.reset()
pic = env.render(mode="rgb_array", close=False)
im = Image.fromarray(pic)
im.save("reacher.jpeg")

env.close()

env = gym.make("Pusher-v1")
observation = env.reset()
pic = env.render(mode="rgb_array", close=False)
im = Image.fromarray(pic)
im.save("pusher.jpeg")

env.close()

"""env = gym.make("Reacher-v2")
observation = env.reset()
pic = env.render(mode="rgb_array", close=False)
png.from_array(pic, 'L').save("Reacher_v2_pic")
env.close()

env = gym.make("Pusher-v2")
observation = env.reset()
pic = env.render(mode="rgb_array", close=False)
png.from_array(pic, 'L').save("Pusher_v2_pic")
env.close()"""