class MuZeroConfig():
    discount = 0.997
    embedding_size = 128
    num_actions = 2305
    support_size = 2305
    full_support_size = 4611
    num_simulations = 5
    output_init_scale = 1.0

    num_trajectory = 32
    sample_per_trajectory = 1
    k_steps = 10

    max_training_steps = 100000
    max_episodes = 10000

    #optimizer
    init_value=0
    peak_value=0.02
    end_value=0.002
    warmup_steps=5_000
    transition_steps=50_000

    #scheduling
    def temperature_fn(max_training_steps, training_steps):
        if training_steps < 0.5 * max_training_steps:
            return 1.0
        elif training_steps < 0.75 * max_training_steps:
            return 0.5
        else:
            return 0.25
