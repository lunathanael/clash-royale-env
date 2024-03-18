from envs import ClassicEnv
import jax
import time
import pickle
import numpy as np


BUFFER_SIZE = 0

class Buffer():
    def __init__(self):
        self.games_rgb = []
        self.games_actions = []
        self.games_results = []
        
    def add_game(self, rgb_frames, actions, result):
        self.games_rgb.append(rgb_frames)
        self.games_actions.append(actions)
        self.games_results.append(result)

    def buffer_full(self):
        return len(self.games_results) == self.buffer_size

buffer = Buffer()



env = ClassicEnv(host=True)
key = jax.random.PRNGKey(0)
num_actions = env.num_actions()

for i in range(BUFFER_SIZE):
    key, subkey = jax.random.split(key)
    env.reset()

    rgb_frames = []
    actions = []
    while not env.terminal():
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, num_actions)
        rgb_frames.append(env.get_observation())
        env.apply(action)
        actions.append(action)
    
    result = env.result()

    buffer.add_game(rgb_frames, actions, result)

timestr = time.strftime("%m%d-%H%M")
with open(f'buffers/game_buffer_uniform_{timestr}.pkl', 'wb') as file_handle:
	pickle.dump(buffer, file_handle)
