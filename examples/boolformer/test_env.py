"""Test Boolformer environment."""

import jax
import jax.numpy as jnp
from environment import BoolformerEnv, BoolformerConfig


def test_environment():
    """Test basic environment functionality."""
    print("=" * 80)
    print("Testing Boolformer Environment")
    print("=" * 80)

    config = BoolformerConfig(
        num_variables=10,
        truth_table_size=1024,
        vocab_size=20,
        max_formula_length=50,
    )

    env = BoolformerEnv(config)
    key = jax.random.key(0)

    # Test reset
    print("\n1. Testing reset...")
    state = env.reset(key)
    print(f"  ✓ Initial state created")
    print(f"    - Truth table shape: {state.truth_table.shape}")
    print(f"    - Formula tokens shape: {state.formula_tokens.shape}")
    print(f"    - Position: {state.position}")
    print(f"    - First token (SOS): {state.formula_tokens[0]}")
    print(f"    - Terminated: {state.terminated}")

    assert state.truth_table.shape == (1024,)
    assert state.formula_tokens.shape == (50,)
    assert state.position == 1
    assert state.formula_tokens[0] == config.sos_token
    assert not state.terminated

    # Test legal actions
    print("\n2. Testing legal actions...")
    legal = env.get_legal_actions(state)
    print(f"  ✓ Legal actions mask shape: {legal.shape}")
    print(f"    - Number of legal actions: {jnp.sum(legal)}")
    print(f"    - PAD legal: {legal[config.pad_token]}")
    print(f"    - SOS legal: {legal[config.sos_token]}")
    print(f"    - EOS legal: {legal[config.eos_token]}")

    assert legal.shape == (20,)
    assert not legal[config.pad_token], "PAD should not be legal"
    assert not legal[config.sos_token], "SOS should not be legal after position 0"

    # Test step
    print("\n3. Testing step...")
    action = jnp.int32(5)  # Some token
    next_state, reward, terminated = env.step(state, action)
    print(f"  ✓ Step executed with action {action}")
    print(f"    - Position advanced: {state.position} -> {next_state.position}")
    print(f"    - Token added at position: {next_state.formula_tokens[1]}")
    print(f"    - Reward: {reward}")
    print(f"    - Terminated: {terminated}")

    assert next_state.position == 2
    assert next_state.formula_tokens[1] == action
    assert not terminated

    # Test termination with EOS
    print("\n4. Testing termination with EOS...")
    eos_state, eos_reward, eos_terminated = env.step(next_state, config.eos_token)
    print(f"  ✓ EOS token triggers termination")
    print(f"    - Terminated: {eos_terminated}")
    print(f"    - Position: {eos_state.position}")

    assert eos_terminated

    # Test legal actions after termination
    print("\n5. Testing legal actions after termination...")
    legal_after = env.get_legal_actions(eos_state)
    print(f"  ✓ No legal actions after termination")
    print(f"    - Number of legal actions: {jnp.sum(legal_after)}")

    assert jnp.sum(legal_after) == 0, "No actions should be legal after termination"

    # Test max length termination
    print("\n6. Testing max length termination...")
    long_state = state
    for i in range(49):  # Fill up to max length
        long_state, _, _ = env.step(long_state, jnp.int32(5))

    print(f"  ✓ Max length reached")
    print(f"    - Position: {long_state.position}")
    print(f"    - Terminated: {long_state.terminated}")

    assert long_state.terminated, "Should terminate at max length"

    print("\n" + "=" * 80)
    print("ALL ENVIRONMENT TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_environment()
