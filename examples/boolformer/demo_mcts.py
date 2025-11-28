"""
Demo script for Boolformer with MCTS.

Shows how to use the model + environment + MCTS integration.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerConfig
from mcts_integration import generate_formula, run_mcts


def main():
    print("=" * 80)
    print("BOOLFORMER MCTS DEMO")
    print("=" * 80)

    # Create model
    print("\n1. Creating Boolformer model...")
    rngs = nnx.Rngs(42)
    n_embd = 512
    n_encoder_layers = 6
    n_decoder_layers = 6
    model = BoolformerTransformer(
        rngs=rngs,
        num_variables=10,
        vocab_size=15,
        max_formula_length=50,
        n_embd=n_embd,
        n_head=8,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
    )
    print(f"  ✓ Model created with {n_embd=}, {n_encoder_layers=}, {n_decoder_layers=}")

    # Create environment
    print("\n2. Creating environment...")
    config = BoolformerConfig(
        num_variables=10,
        truth_table_size=1024,
        vocab_size=15,
        max_formula_length=50,
    )
    env = BoolformerEnv(model, config)
    print(f"  ✓ Environment created with vocab_size={config.vocab_size}")

    # Generate random truth table
    print("\n3. Generating random truth table...")
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    truth_table = jax.random.bernoulli(subkey, 0.5, (1024,))
    print(f"  ✓ Truth table: {jnp.sum(truth_table)}/1024 ones")

    # Test single MCTS step
    print("\n4. Testing single MCTS step...")
    state = env.reset(key, truth_table)
    key, subkey = jax.random.split(key)

    print(f"  Initial state:")
    print(f"    - Position: {state.position}")
    print(f"    - First token: {state.formula_tokens[0]} (SOS)")

    # Run MCTS for next action
    action, weights = run_mcts(
        model=model,
        env=env,
        initial_state=state,
        num_simulations=10,  # Small number for demo
        temperature=1.0,
        rng_key=subkey,
    )

    print(f"  ✓ MCTS completed")
    print(f"    - Selected action: {action}")
    print(f"    - Action weights shape: {weights.shape}")
    print(f"    - Top 3 actions by weight: {jnp.argsort(weights)[-3:][::-1]}")

    # Generate complete formula (NOTE: This will be slow with untrained model)
    print("\n5. Generating complete formula with MCTS...")
    print(f"  (Using 5 simulations for demo - normally use 50+)")

    key, subkey = jax.random.split(key)
    formula = generate_formula(
        model=model,
        env=env,
        truth_table=truth_table,
        num_simulations=5,  # Very small for demo
        temperature=1.0,
        rng_key=subkey,
    )

    print(f"  ✓ Formula generated")
    print(f"    - Length: {jnp.sum(formula != config.pad_token)}")
    print(f"    - Tokens: {formula[:20]}...")  # Show first 20 tokens

    print("\n" + "=" * 80)
    print("DEMO COMPLETED ✓")
    print("=" * 80)
    print("\nNOTE: This model is untrained, so the generated formulas are random.")
    print("Next step: Implement training loop with MCTS self-play.")


if __name__ == "__main__":
    main()
