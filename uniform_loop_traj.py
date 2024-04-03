import time
import pickle

import cv2
import jax
from mctx import gumbel_muzero_policy
from absl import app
from absl import flags

from envs import ClanClassicEnv
from network import uniform_recurrentfn, uniform_rootfn
import jax.numpy as jnp

import muax


FLAGS = flags.FLAGS
flags.DEFINE_bool("host", None, "If the user is the host.")
flags.DEFINE_string("serial", "RFCWC04A2VY", "Serial ID of the Android Device")
flags.DEFINE_integer("buffer_size", 10, "Size of the buffer.")

flags.DEFINE_integer("num_simulations", 16, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("max_depth", None, "The maximum search depth.")

flags.DEFINE_integer("seed", 42, "Random seed.")

discount = 0.997


def main(_):
    tracer = muax.PNStep(10, discount, 0.5) 
    buffer = muax.TrajectoryReplayBuffer(FLAGS.buffer_size)

    env = ClanClassicEnv(serial=FLAGS.serial, host=FLAGS.host)
    rng_key = jax.random.PRNGKey(FLAGS.seed)

    for i in range(FLAGS.buffer_size):
        trajectory = muax.Trajectory()
        env.reset()
        obs = env.get_observation()
        obs = cv2.resize(obs, dsize=(135, 240), interpolation=cv2.INTER_CUBIC)
        print(f"Game {i} began.")
        while True:
            rng_key, gmp_key = jax.random.split(rng_key)
            selected_action = jax.random.randint(shape=(1,), key=gmp_key, minval=0, maxval=2305, dtype=int)[0]
            rng_key, act_key = jax.random.split(rng_key)
            if jax.random.uniform(act_key) < 0.95:
                selected_action = 2304
            env.apply(selected_action)

            if not env.in_game():
                break

            tracer.add(obs, selected_action, 0, False, v=0, pi=jnp.full([1, 2305,], (1/2305), dtype=jnp.float32))
            while tracer:
                trans = tracer.pop()
                trajectory.add(trans)
            

        
        print(f"Game {i} over with {len(trajectory)} frames.")
        result = env.await_result()
        tracer.add(obs, selected_action, result, True, v=0.5, pi=jnp.full([1, 2305,], (1/2305), dtype=jnp.float32))
        while tracer:
                trans = tracer.pop()
                trajectory.add(trans)
        trajectory.finalize()
        buffer.add(trajectory, trajectory.batched_transitions.w.mean())
        print(f"Game {i} result: {result}")

    del env

    timestr = time.strftime("%m%d-%H%M")
    if FLAGS.host:
        timestr += '_H'

    with open(f'buffers/game_buffer_S{len(buffer)}_uniform_{timestr}.pkl', 'wb') as file_handle:
        pickle.dump(buffer, file_handle)

if __name__ == "__main__":
  app.run(main)