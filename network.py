import jax.numpy as jnp
from mctx import RecurrentFnOutput, RootFnOutput

def uniform_policy(batch_size):
    return jnp.full([batch_size, 2305,], 1/2305, dtype=jnp.float32)

def uniform_recurrentfn(params, rng_key, action, embedding):
    del params, rng_key, action
    recurrent_fn_output = RecurrentFnOutput(
        reward = jnp.full(embedding.shape[0], 0, dtype=jnp.float32),
        discount=jnp.full(embedding.shape[0], -1, dtype=jnp.float32),
        prior_logits=uniform_policy(embedding.shape[0]),
        value=jnp.full(embedding.shape[0], 0.5, dtype=jnp.float32)
    )
    next_embedding = embedding + 1
    return recurrent_fn_output, next_embedding

def uniform_rootfn(input):
    del input
    return RootFnOutput(
      prior_logits=uniform_policy(1),
      value=jnp.full([1], 0.5, dtype=jnp.float32),
      embedding=jnp.zeros([1]),
    )
