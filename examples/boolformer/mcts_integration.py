"""
MCTS integration for Boolformer.

Connects the Boolformer model with the environment for mctx-based tree search.
"""

import jax
import jax.numpy as jnp
import mctx
from typing import Tuple

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerState


def create_root_fn(model: BoolformerTransformer, env: BoolformerEnv):
    """
    Create root function for mctx MCTS.

    The root function takes a state and returns initial prior logits and value
    for the root node of the search tree.

    Args:
        model: Boolformer transformer model
        env: Boolformer environment

    Returns:
        root_fn compatible with mctx
    """
    def root_fn(state: BoolformerState) -> mctx.RootFnOutput:
        """
        Compute policy and value for root state.

        Args:
            state: Current BoolformerState

        Returns:
            RootFnOutput with (prior_logits, value, embedding=state)
        """
        # Use cached encoder output (computed once in env.reset())
        # Add batch dimension: (1024, n_embd) -> (1, 1024, n_embd)
        encoder_output = state.encoder_output[None, :]

        # Get formula tokens up to current position
        # Add batch dimension: (max_len,) -> (1, max_len)
        formula_tokens = state.formula_tokens[None, :]

        # Decode only (encoder already ran in env.reset())
        policy_logits, value = model.decode_formula(formula_tokens, encoder_output, decode=False)
        # value shape: (batch, seq_len), policy_logits shape: (batch, seq_len, vocab_size)

        # Extract value and logits at current position (position is 1-indexed)
        current_value = value[:, state.position - 1]  # (batch,)
        next_token_logits = policy_logits[:, state.position - 1, :]  # (batch, vocab_size)

        # DEBUG: Print value prediction in root_fn
        # jax.debug.print("🔍 [root_fn] position={}, value={}", state.position, current_value[0])

        # Apply legal action mask
        legal_actions = env.get_legal_actions(state)
        # Mask illegal actions with large negative value
        # Add batch dimension to legal_actions for broadcasting
        masked_logits = jnp.where(
            legal_actions[None, :],  # Shape: (1, vocab_size)
            next_token_logits,
            jnp.full_like(next_token_logits, -1e9)
        )

        # DEBUG: Print action logits before MCTS (mcts_integration.py:66)
        # jax.debug.print("🔍 [root_fn] legal_actions={}", legal_actions)
        # jax.debug.print("🔍 [root_fn] raw_logits (before mask)={}", next_token_logits[0])
        # jax.debug.print("🔍 [root_fn] masked_logits={}", masked_logits[0])

        # Add batch dimension to all state fields for mctx embedding
        def add_batch_dim(arr):
            if arr.ndim == 0:  # scalar
                return arr[None]
            elif arr.ndim == 1:  # 1D array
                return arr[None, :]
            else:  # 2D array (encoder_output)
                return arr[None, :, :]

        batched_state = jax.tree.map(add_batch_dim, state)

        return mctx.RootFnOutput(
            prior_logits=masked_logits,  # Shape: (batch, vocab_size)
            value=current_value,  # Shape: (batch,) - extracted at current position
            embedding=batched_state  # Pass batched state as embedding
        )

    return root_fn


def create_recurrent_fn(model: BoolformerTransformer, env: BoolformerEnv):
    """
    Create recurrent function for mctx MCTS.

    The recurrent function takes a state and action, returns next state,
    reward, discount, and model predictions for the next state.

    Args:
        model: Boolformer transformer model
        env: Boolformer environment

    Returns:
        recurrent_fn compatible with mctx
    """
    def recurrent_fn(
        params,  # Model params (not used, model is already initialized)
        rng_key: jax.Array,
        action: jax.Array,
        embedding: BoolformerState  # This is the BATCHED state from mctx
    ) -> mctx.RecurrentFnOutput:
        """
        Compute transition and predictions for next state.

        Args:
            params: Model parameters (not used)
            rng_key: Random key (not used for deterministic environment)
            action: Action to take (token to add) - shape: (B,)
            embedding: Current batched state from mctx - all fields have shape (B, ...)

        Returns:
            RecurrentFnOutput with (reward, discount, prior_logits, value, embedding)
        """
        # Unbatch state (remove batch dimension from all fields)
        def remove_batch_dim(arr):
            return arr[0]

        state = jax.tree.map(remove_batch_dim, embedding)

        # Action is also batched, extract single action
        single_action = action[0] if action.ndim > 0 else action

        # Environment step (ignore is_perfect, only used for logging in selfplay)
        next_state, reward, terminated, _ = env.step(state, single_action)

        # Discount: 0 if terminated, 1 otherwise
        discount = jnp.where(terminated, jnp.float32(0.0), jnp.float32(1.0))

        # Get model predictions for next state using cached encoder output
        # Add batch dimension
        encoder_output = next_state.encoder_output[None, :, :]
        formula_tokens = next_state.formula_tokens[None, :]

        # Decode only (encoder already cached in state)
        policy_logits, value = model.decode_formula(formula_tokens, encoder_output, decode=False)
        # value shape: (batch, seq_len), policy_logits shape: (batch, seq_len, vocab_size)

        # Extract value and logits at the token we just placed (next_state.position - 1)
        current_value = value[:, next_state.position - 1]  # (batch,)
        next_token_logits = policy_logits[:, next_state.position - 1, :]  # (batch, vocab_size)

        # DEBUG: Print value prediction in recurrent_fn
        # jax.debug.print("🔍 [recurrent_fn] action={}, next_position={}, reward={}, value={}",
        #                 single_action, next_state.position, reward, current_value[0])

        # Apply legal action mask
        legal_actions = env.get_legal_actions(next_state)
        # Add batch dimension to legal_actions for broadcasting
        masked_logits = jnp.where(
            legal_actions[None, :],  # Shape: (1, vocab_size)
            next_token_logits,
            jnp.full_like(next_token_logits, -1e9)
        )

        # DEBUG: Print action logits in recurrent_fn (mcts_integration.py:161)
        # jax.debug.print("🔍 [recurrent_fn] legal_actions={}", legal_actions)
        # jax.debug.print("🔍 [recurrent_fn] masked_logits={}", masked_logits[0])

        # Add batch dimension to next_state for mctx embedding
        def add_batch_dim(arr):
            if arr.ndim == 0:  # scalar
                return arr[None]
            elif arr.ndim == 1:  # 1D array
                return arr[None, :]
            else:  # 2D array (encoder_output)
                return arr[None, :, :]

        batched_next_state = jax.tree.map(add_batch_dim, next_state)

        # Reward and discount need batch dimension too
        reward_batched = reward[None]  # scalar -> (1,)
        discount_batched = discount[None]  # scalar -> (1,)

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward_batched,  # Shape: (batch,)
            discount=discount_batched,  # Shape: (batch,)
            prior_logits=masked_logits,  # Shape: (batch, vocab_size)
            value=current_value,  # Shape: (batch,) - extracted at current position
        )

        # Return (output, next_embedding)
        return recurrent_output, batched_next_state

    return recurrent_fn


def run_mcts(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    initial_state: BoolformerState,
    num_simulations: int = 50,
    temperature: float = 1.0,
    rng_key: jax.Array = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Run MCTS tree search to select next action.

    Args:
        model: Boolformer transformer model
        env: Boolformer environment
        initial_state: Initial state to search from
        num_simulations: Number of MCTS simulations
        temperature: Temperature for action selection (1.0 = stochastic, 0.0 = greedy)
        rng_key: Random key for stochastic action selection

    Returns:
        (selected_action, action_weights) where action_weights are visit counts
    """
    if rng_key is None:
        rng_key = jax.random.key(0)

    # Create root and recurrent functions
    # Note: We use decode=False for MCTS since each simulation explores
    # different state paths, making KV caching inappropriate.
    root_fn = create_root_fn(model, env)
    recurrent_fn = create_recurrent_fn(model, env)

    # Get root
    root = root_fn(initial_state)

    # Run Gumbel MuZero MCTS (recommended for single-player games)
    policy_output = mctx.gumbel_muzero_policy(
        params=None,  # Model params not needed (model already initialized)
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=None,  # No depth limit
        qtransform=mctx.qtransform_completed_by_mix_value,  # Recommended for completed rewards
    )

    # Get action probabilities from visit counts
    action_weights = policy_output.action_weights

    # Sample action based on temperature
    if temperature == 0.0:
        # Greedy: select action with highest weight
        action = jnp.argmax(action_weights)
    else:
        # Stochastic: sample proportional to weights^(1/temperature)
        logits = jnp.log(action_weights + 1e-8) / temperature
        action = jax.random.categorical(rng_key, logits)

    return action, action_weights


def generate_formula(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    truth_table: jax.Array,
    num_simulations: int = 50,
    temperature: float = 1.0,
    rng_key: jax.Array = None,
) -> jax.Array:
    """
    Generate a complete formula using MCTS.

    Args:
        model: Boolformer transformer model
        env: Boolformer environment
        truth_table: Truth table to synthesize formula for (1024,)
        num_simulations: Number of MCTS simulations per action
        temperature: Temperature for action selection
        rng_key: Random key

    Returns:
        Generated formula tokens (max_length,)
    """
    if rng_key is None:
        rng_key = jax.random.key(0)

    # Reset environment with given truth table
    state = env.reset(rng_key, truth_table)

    # Generate tokens one by one until termination
    while not state.terminated:
        # Run MCTS to select next action
        rng_key, subkey = jax.random.split(rng_key)
        action, _ = run_mcts(model, env, state, num_simulations, temperature, subkey)

        # Take action
        state, _, _ = env.step(state, action)

    return state.formula_tokens
