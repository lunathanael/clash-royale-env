import time
import pickle

import cv2
import jax
from mctx import gumbel_muzero_policy
from absl import app
from absl import flags

from envs import ClanClassicEnv, TrainerEnv
from network import uniform_recurrentfn, uniform_rootfn
import jax.numpy as jnp

import muax
from muax import nn

from config import MuZeroConfig


FLAGS = flags.FLAGS
flags.DEFINE_bool("host", None, "If the user is the host.")
flags.DEFINE_string("serial", "RFCWC04A2VY", "Serial ID of the Android Device")
flags.DEFINE_integer("buffer_size", 100, "Size of the buffer.")
flags.DEFINE_integer("buffer_warmup", 32, "Size for buffer warmup.")

flags.DEFINE_integer("seed", 429, "Random seed.")

flags.DEFINE_string("model_path", None, "Path of network weights.")
flags.DEFINE_string("buffer_path", None, "Path of buffer binary.")
flags.DEFINE_string("model_name", "default_model", "Name of model.")

flags.DEFINE_integer("epoch", 0, "epoch")
flags.DEFINE_integer("training_step", 0, "training_step")



def main(_):
    config = MuZeroConfig
    rng_key = jax.random.PRNGKey(FLAGS.seed)

    gradient_transform = muax.model.optimizer(init_value=config.init_value, peak_value=config.peak_value, 
        end_value=config.end_value, warmup_steps=config.warmup_steps, transition_steps=config.transition_steps)
    repr_fn = nn._init_ez_representation_func(nn.EZRepresentation, config.embedding_size)
    pred_fn = nn._init_ez_prediction_func(nn.EZPrediction, config.num_actions, config.full_support_size, config.output_init_scale)
    dy_fn = nn._init_ez_dynamic_func(nn.EZDynamic, config.embedding_size, config.num_actions, config.full_support_size, config.output_init_scale)
    model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='gumbel', discount=config.discount,
                        optimizer=gradient_transform, support_size=config.support_size)

    if FLAGS.model_path is None:
        sample_input = jnp.expand_dims(jnp.zeros((config.height, config.width, 3)), axis=0)
        rng_key, subkey = jax.random.split(rng_key)
        model.init(subkey, sample_input)
    else:
        model.load(FLAGS.model_path)


    if FLAGS.buffer_path is None:
        buffer = muax.TrajectoryReplayBuffer(FLAGS.buffer_size)
        warmup = True
    else:
        with open(FLAGS.buffer_path, 'rb') as file_handle: 
            buffer = pickle.load(file_handle)
        warmup = False


    tracer = muax.PNStep(10, config.discount, 0.5) 
    training_step = FLAGS.training_step
    max_training_steps_reached = False
    env = TrainerEnv(serial=FLAGS.serial)

    if warmup:
        print("buffer warm up stage...")
        while len(buffer) < FLAGS.buffer_warmup:
            trajectory = muax.Trajectory()
            temperature = config.temperature_fn(max_training_steps=config.max_training_steps, training_steps=training_step)
            env.reset()
            t_count = 0
            while True:
                obs = env.get_observation()
                obs = cv2.resize(obs, dsize=(config.width, config.height), interpolation=cv2.INTER_NEAREST)

                rng_key, subkey = jax.random.split(rng_key)
                # a, pi, v = model.act(subkey, obs, 
                #     with_pi=True, 
                #     with_value=True, 
                #     obs_from_batch=False,
                #     num_simulations=config.num_simulations,
                #     temperature=temperature)
                random_action = jax.random.randint(subkey, shape=(1,), minval=0, maxval=2305)[0]
                rng_key, subkey = jax.random.split(rng_key)
                random_prob = jax.random.uniform(rng_key, shape=(1,))[0]
                no_action_prob = 0.5
                t_count += 1
                a = random_action if (t_count % 8 != 7 or random_prob < no_action_prob) else 2304
                v = 0
                probability_of_other_values = (1 - no_action_prob) / 2304

                # Create the logits array
                pi = jnp.array([[probability_of_other_values] * 2304 + [no_action_prob]])

                env.apply(a)

                if not env.in_game():
                    break

                tracer.add(obs, a, 0.1, False, v=v, pi=pi)
                while tracer:
                    trans = tracer.pop()
                    trajectory.add(trans)
                time.sleep(float(random_prob) * 2)

            result = env.await_result()
            tracer.add(obs, a, result*1000, True, v=v, pi=pi)
            while tracer:
                trans = tracer.pop()
                trajectory.add(trans)
            trajectory.finalize()
            if len(trajectory) >= config.k_steps:
                buffer.add(trajectory, trajectory.batched_transitions.w.mean())

    print("Starting Training")
    for ep in range(FLAGS.epoch, config.max_episodes):
        trajectory = muax.Trajectory()
        temperature = config.temperature_fn(max_training_steps=config.max_training_steps, training_steps=training_step)
        env.reset()
        while True:
            obs = env.get_observation()
            obs = cv2.resize(obs, dsize=(288, 512), interpolation=cv2.INTER_NEAREST)
            rng_key, subkey = jax.random.split(rng_key)
            a, pi, v = model.act(subkey, obs, 
                with_pi=True, 
                with_value=True, 
                obs_from_batch=False,
                num_simulations=config.num_simulations,
                temperature=temperature)
            print(f"Pred v: {v}, {pi}", end='\r')
            env.apply(a)

            if not env.in_game():
                break

            tracer.add(obs, a, 0.1, False, v=v, pi=pi)
            while tracer:
                trans = tracer.pop()
                trajectory.add(trans)

        result = env.await_result()
        tracer.add(obs, a, result*1000, True, v=v, pi=pi)
        while tracer:
            trans = tracer.pop()
            trajectory.add(trans)
        trajectory.finalize()
        if len(trajectory) >= config.k_steps:
            buffer.add(trajectory, trajectory.batched_transitions.w.mean())


        with open(f'buffers/game_buffer_backup_{ep % 2}.pkl', 'wb') as file_handle:
            pickle.dump(buffer, file_handle)

        #Training
        print("Updating Network...")
        if max_training_steps_reached:
            break
        train_loss = 0
        for i in range(20):
            transition_batch = buffer.sample(num_trajectory=config.num_trajectory,
                                            sample_per_trajectory=config.sample_per_trajectory,
                                            k_steps=config.k_steps)
            loss_metric = model.update(transition_batch)
            train_loss += loss_metric['loss']
            training_step += 1
            if training_step >= config.max_training_steps:
                max_training_steps_reached = True
                break
            print(f'epoch: {ep:04d}, loss: {(train_loss/(i+1)):.8f}, training_step: {training_step}', end='\r')
        train_loss /= 20
        print(f'epoch: {ep:04d}, loss: {train_loss:.8f}, training_step: {training_step}')
        timestr = time.strftime("%m%d-%H%M")
        model.save(f'networks/{FLAGS.model_name}_epoch{ep}_step{training_step}_t{timestr}')



    del env # disconnect adb

if __name__ == "__main__":
    app.run(main)