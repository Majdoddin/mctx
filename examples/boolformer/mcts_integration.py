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

    Args:
        model: Boolformer transformer model
        env: Boolformer environment

    Returns:
        root_fn(state, encoder_output) compatible with mctx
    """
    def root_fn(state: BoolformerState, encoder_output: jax.Array) -> mctx.RootFnOutput:
        """
        Compute policy and value for root state.

        Args:
            state: Current BoolformerState (no encoder_output or points)
            encoder_output: Pre-computed encoder output (num_points, n_embd)

        Returns:
            RootFnOutput with (prior_logits, value, embedding=state)
        """
        # Add batch dimension: (num_points, n_embd) -> (1, num_points, n_embd)
        formula_tokens = state.formula_tokens[None, :]
        enc_out = encoder_output[None, :]

        # Decode only (encoder already ran once per episode)
        policy_logits, value = model.decode_formula(formula_tokens, enc_out, decode=False)

        # Extract value and logits at current position (position is 1-indexed)
        current_value = value[:, state.position - 1]  # (batch,)
        next_token_logits = policy_logits[:, state.position - 1, :]  # (batch, vocab_size)

        # Apply legal action mask
        legal_actions = env.get_legal_actions(state)
        masked_logits = jnp.where(
            legal_actions[None, :],
            next_token_logits,
            jnp.full_like(next_token_logits, -1e9)
        )

        # Add batch dimension to state fields for mctx embedding (stored per tree node)
        def add_batch_dim(arr):
            if arr.ndim == 0:  # scalar
                return arr[None]
            else:  # 1D array (formula_tokens, legal_action_mask)
                return arr[None, :]

        batched_state = jax.tree.map(add_batch_dim, state)

        return mctx.RootFnOutput(
            prior_logits=masked_logits,
            value=current_value,
            embedding=batched_state,
        )

    return root_fn


def create_recurrent_fn(model: BoolformerTransformer, env: BoolformerEnv):
    """
    Create recurrent function for mctx MCTS.

    params = (encoder_output, points) — constant per episode, NOT stored per tree node.

    Args:
        model: Boolformer transformer model
        env: Boolformer environment

    Returns:
        recurrent_fn compatible with mctx
    """
    def recurrent_fn(
        params,  # (encoder_output, points) — per-episode constants
        rng_key: jax.Array,
        action: jax.Array,
        embedding: BoolformerState  # Batched state from mctx (no encoder_output/points)
    ) -> mctx.RecurrentFnOutput:
        """
        Args:
            params: (encoder_output, points) — not stored in tree, just passed through
            action: (B,)
            embedding: Batched BoolformerState from mctx tree node
        """
        encoder_output, points = params

        # Unbatch state (remove batch dimension)
        state = jax.tree.map(lambda arr: arr[0], embedding)
        single_action = action[0] if action.ndim > 0 else action

        # Environment step — points passed explicitly
        next_state, reward, terminated, _ = env.step(state, single_action, points)

        # Discount: 0 if terminated, 1 otherwise
        discount = jnp.where(terminated, jnp.float32(0.0), jnp.float32(1.0))

        # Decode using encoder_output from params (not from state)
        formula_tokens = next_state.formula_tokens[None, :]
        enc_out = encoder_output[None, :]
        policy_logits, value = model.decode_formula(formula_tokens, enc_out, decode=False)

        current_value = value[:, next_state.position - 1]
        next_token_logits = policy_logits[:, next_state.position - 1, :]

        # Apply legal action mask
        legal_actions = env.get_legal_actions(next_state)
        masked_logits = jnp.where(
            legal_actions[None, :],
            next_token_logits,
            jnp.full_like(next_token_logits, -1e9)
        )

        # Batch next_state for mctx embedding (only lightweight fields)
        def add_batch_dim(arr):
            if arr.ndim == 0:
                return arr[None]
            else:
                return arr[None, :]

        batched_next_state = jax.tree.map(add_batch_dim, next_state)

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward[None],
            discount=discount[None],
            prior_logits=masked_logits,
            value=current_value,
        )

        return recurrent_output, batched_next_state

    return recurrent_fn


def run_mcts(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    initial_state: BoolformerState,
    encoder_output: jax.Array,
    points: jax.Array,
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
        encoder_output: Pre-computed encoder output (num_points, n_embd)
        points: Minority class points (num_points, num_variables)
        num_simulations: Number of MCTS simulations
        temperature: Temperature for action selection
        rng_key: Random key

    Returns:
        (selected_action, action_weights) where action_weights are visit counts
    """
    if rng_key is None:
        rng_key = jax.random.key(0)

    root_fn = create_root_fn(model, env)
    recurrent_fn = create_recurrent_fn(model, env)

    root = root_fn(initial_state, encoder_output)

    policy_output = mctx.gumbel_muzero_policy(
        params=(encoder_output, points),  # Passed through to recurrent_fn, NOT stored per node
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=None,
        qtransform=mctx.qtransform_completed_by_mix_value,
    )

    action_weights = policy_output.action_weights

    if temperature == 0.0:
        action = jnp.argmax(action_weights)
    else:
        logits = jnp.log(action_weights + 1e-8) / temperature
        action = jax.random.categorical(rng_key, logits)

    return action, action_weights
