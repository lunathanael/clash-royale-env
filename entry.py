import time
import pickle

import jax 
from jax import numpy as jnp 
jax.config.update('jax_platform_name', 'cpu')
from mctx import gumbel_muzero_policy

import muax
from muax import nn 

from envs import ClanClassicEnv


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
        return len(self.games_results) == 64
    

rng_key = jax.random.PRNGKey(42)

support_size = 10
embedding_size = 64
num_actions = 2305
full_support_size = int(support_size * 2 + 1)
discount = 0.99
num_simulations = 1

repr_fn = nn._init_representation_func(nn.ResNetRepresentation, embedding_size)
pred_fn = nn._init_prediction_func(nn.ResNetPrediction, num_actions, full_support_size)
dy_fn = nn._init_dynamic_func(nn.ResNetDynamic, embedding_size, num_actions, full_support_size)

gradient_transform = muax.model.optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)

model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                    optimizer=gradient_transform, support_size=support_size)

sample_input = jnp.expand_dims(jnp.zeros((2336, 1080, 3)), axis=0)
rng_key, subkey = jax.random.split(rng_key)
model.init(subkey, sample_input)


buffer = Buffer()

while True: 
    try:
        env = ClanClassicEnv(serial="RFCWC04A2VY", host=True)
        break
    except:
        time.sleep(1)

for i in range(1):
    rgb_frames = []
    actions = []
    #env.reset()
    print(f"Game {i} began.")
    while True: #env.in_game():
        obs = env.get_observation()
        print(obs.shape)
        rng_key, subkey = jax.random.split(rng_key)
        a, pi, v = model.act(subkey, obs, 
                        with_pi=True, 
                        with_value=True, 
                        obs_from_batch=False,
                        num_simulations=num_simulations,
                        temperature=0.)
        print(a)
        # selected_action = policy_output.action[0]
        # rng_key, act_key = jax.random.split(rng_key)
        # if jax.random.uniform(act_key) < 0.2:
        #     selected_action = 2304
        # env.apply(selected_action)
        # actions.append(selected_action)
        # rgb_frames.append(obs)
    
    print(f"Game {i} over with {len(rgb_frames)} frames.")
    result = env.await_result()
    print(f"Game {i} result: {result}")

    buffer.add_game(rgb_frames, actions, result)

del env

timestr = time.strftime("%m%d-%H%M")
timestr += '_H'

with open(f'buffers/game_buffer_uniform_{timestr}.pkl', 'wb') as file_handle:
    pickle.dump(buffer, file_handle)
