"""
Minimal test for training script.
"""

import jax
from flax import nnx
import optax

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerConfig
from train import TrainingConfig, selfplay_single_episode
from mcts_integration import create_root_fn, create_recurrent_fn


def main():
    print("Testing training components...")

    config = TrainingConfig(
        vocab_size=15,
        max_formula_length=50,
        selfplay_batch_size=2,  # Very small for testing
        num_simulations=5,  # Very small for testing
    )

    # Initialize model
    print("\n1. Creating model...")
    rngs = nnx.Rngs(config.seed)
    model = BoolformerTransformer(
        rngs=rngs,
        num_variables=config.num_variables,
        vocab_size=config.vocab_size,
        max_formula_length=config.max_formula_length,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
    )
    print("  ✓ Model created")

    # Initialize environment
    print("\n2. Creating environment...")
    env_config = BoolformerConfig(
        num_variables=config.num_variables,
        vocab_size=config.vocab_size,
        max_formula_length=config.max_formula_length,
    )
    env = BoolformerEnv(model, env_config)
    print("  ✓ Environment created")

    # Test single episode
    print("\n3. Testing selfplay_single_episode...")
    rng_key = jax.random.key(config.seed)
    rng_key, subkey = jax.random.split(rng_key)

    truth_table = jax.random.bernoulli(subkey, 0.5, (1024,))

    root_fn = create_root_fn(model, env)
    recurrent_fn = create_recurrent_fn(model, env)

    rng_key, subkey = jax.random.split(rng_key)
    episode_data = selfplay_single_episode(
        model, env, root_fn, recurrent_fn, config, truth_table, subkey
    )

    print(f"  ✓ Episode data shapes:")
    print(f"    encoder_outputs: {episode_data[0].shape}")
    print(f"    formula_tokens: {episode_data[1].shape}")
    print(f"    positions: {episode_data[2].shape}")
    print(f"    action_weights: {episode_data[3].shape}")
    print(f"    rewards: {episode_data[4].shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
