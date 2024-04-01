from envs import ClanClassicEnv
import jax
import time
import pickle
from mctx import gumbel_muzero_policy
from network import uniform_recurrentfn, uniform_rootfn

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_simulations", 16, "Number of simulations.")
flags.DEFINE_integer("buffer_size", 10, "Size of the buffer.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("max_depth", None, "The maximum search depth.")
flags.DEFINE_bool("host", None, "If the user is the host.")



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
        return len(self.games_results) == FLAGS.buffer_size


def main(_):
    buffer = Buffer()

    env = ClanClassicEnv(host=FLAGS.host)
    rng_key = jax.random.PRNGKey(FLAGS.seed)

    for i in range(FLAGS.buffer_size):
        rgb_frames = []
        actions = []
        env.reset()
        print(f"Game {i} began.")
        while env.in_game():
            obs = env.get_observation()
            root = uniform_rootfn(obs)
            rng_key, gmp_key = jax.random.split(rng_key)
            policy_output = gumbel_muzero_policy(
                params=(),
                rng_key=gmp_key,
                root=root,
                recurrent_fn=uniform_recurrentfn,
                num_simulations=FLAGS.num_simulations,
                max_depth=FLAGS.max_depth,
                max_num_considered_actions=FLAGS.max_num_considered_actions,
            )
            selected_action = policy_output.action[0]
            rng_key, act_key = jax.random.split(rng_key)
            if jax.random.uniform(act_key) < 0.2:
                selected_action = 2304
            env.apply(selected_action)
            actions.append(selected_action)
            rgb_frames.append(obs)
        
        print(f"Game {i} over with {len(rgb_frames)} frames.")
        result = env.await_result()
        print(f"Game {i} result: {result}")

        buffer.add_game(rgb_frames, actions, result)

    del env

    timestr = time.strftime("%m%d-%H%M")
    if FLAGS.host:
        timestr += '_H'

    with open(f'buffers/game_buffer_uniform_{timestr}.pkl', 'wb') as file_handle:
        pickle.dump(buffer, file_handle)

if __name__ == "__main__":
  app.run(main)