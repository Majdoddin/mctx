"""
Test script for Boolformer JAX model.

Tests:
1. Model instantiation
2. Forward pass with dummy data
3. Output shapes and value ranges
"""

import jax
import jax.numpy as jnp
from flax import nnx

from boolformer_jax_model import BoolformerTransformer


def test_model_creation():
    """Test that the model can be instantiated."""
    print("=" * 80)
    print("Test 1: Model Creation")
    print("=" * 80)

    rngs = nnx.Rngs(0)
    model = BoolformerTransformer(
        rngs=rngs,
        truth_table_size=1024,
        num_variables=10,
        vocab_size=20,
        max_formula_length=50,
        n_embd=512,
        n_head=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
    )

    print(f"✓ Model created successfully")
    print(f"  - Encoder layers: {len(model.encoder_blocks)}")
    print(f"  - Decoder layers: {len(model.decoder_blocks)}")
    print(f"  - Embedding dim: {model.n_embd}")
    print(f"  - RoPE cos shape: {model.cos.shape}")
    print(f"  - RoPE sin shape: {model.sin.shape}")

    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)

    batch_size = 4
    seq_len = 10

    # Create dummy truth table (batch, 1024) - binary values
    truth_table = jax.random.bernoulli(jax.random.key(1), 0.5, (batch_size, 1024)).astype(jnp.float32)

    # Create dummy formula tokens (batch, seq_len) - token indices [0, vocab_size)
    formula_tokens = jax.random.randint(jax.random.key(2), (batch_size, seq_len), 0, 20)

    print(f"Input shapes:")
    print(f"  - Truth table: {truth_table.shape}")
    print(f"  - Formula tokens: {formula_tokens.shape}")

    # Forward pass
    policy_logits, value = model(truth_table, formula_tokens)

    print(f"\nOutput shapes:")
    print(f"  - Policy logits: {policy_logits.shape}")
    print(f"  - Value: {value.shape}")

    # Check shapes
    expected_policy_shape = (batch_size, seq_len, 20)
    expected_value_shape = (batch_size,)

    assert policy_logits.shape == expected_policy_shape, \
        f"Policy shape mismatch: {policy_logits.shape} != {expected_policy_shape}"
    assert value.shape == expected_value_shape, \
        f"Value shape mismatch: {value.shape} != {expected_value_shape}"

    print(f"\n✓ Shapes correct")

    # Check value range [-1, 1]
    print(f"\nValue statistics:")
    print(f"  - Min: {jnp.min(value):.4f}")
    print(f"  - Max: {jnp.max(value):.4f}")
    print(f"  - Mean: {jnp.mean(value):.4f}")

    assert jnp.all(value >= -1.0) and jnp.all(value <= 1.0), \
        "Value should be in range [-1, 1]"

    print(f"✓ Value in correct range [-1, 1]")

    # Check policy logits (should be finite)
    assert jnp.all(jnp.isfinite(policy_logits)), "Policy logits contain NaN/Inf"

    print(f"\nPolicy logits statistics:")
    print(f"  - Min: {jnp.min(policy_logits):.4f}")
    print(f"  - Max: {jnp.max(policy_logits):.4f}")
    print(f"  - Mean: {jnp.mean(policy_logits):.4f}")

    print(f"✓ Policy logits are finite")

    return policy_logits, value


def test_batch_independence(model):
    """Test that different batch elements produce different outputs."""
    print("\n" + "=" * 80)
    print("Test 3: Batch Independence")
    print("=" * 80)

    batch_size = 2
    seq_len = 5

    # Create different truth tables
    truth_table1 = jnp.zeros((1, 1024))  # All zeros
    truth_table2 = jnp.ones((1, 1024))   # All ones
    truth_table = jnp.concatenate([truth_table1, truth_table2], axis=0)

    # Same formula tokens
    formula_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    policy_logits, value = model(truth_table, formula_tokens)

    # Check that outputs are different for different inputs
    policy_diff = jnp.abs(policy_logits[0] - policy_logits[1]).max()
    value_diff = jnp.abs(value[0] - value[1])

    print(f"Max policy difference between batch elements: {policy_diff:.4f}")
    print(f"Value difference between batch elements: {value_diff:.4f}")

    assert policy_diff > 1e-5, "Batch elements should produce different outputs"

    print(f"✓ Batch elements are independent")


def main():
    print("\n" + "=" * 80)
    print("BOOLFORMER JAX MODEL TEST SUITE")
    print("=" * 80 + "\n")

    # Test 1: Model creation
    model = test_model_creation()

    # Test 2: Forward pass
    policy_logits, value = test_forward_pass(model)

    # Test 3: Batch independence
    test_batch_independence(model)

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
