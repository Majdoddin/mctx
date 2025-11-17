"""
Boolformer environment for MCTS-based formula synthesis.

Single-player environment where:
- State: (truth_table, formula_tokens_so_far)
- Action: Next token to add to formula
- Goal: Synthesize a boolean formula that matches the truth table
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from dataclasses import dataclass


@dataclass
class BoolformerConfig:
    """Configuration for Boolformer environment."""

    # Truth table config
    num_variables: int = 10
    truth_table_size: int = 1024  # 2^10

    # Formula config
    vocab_size: int = 20
    max_formula_length: int = 50

    # Special tokens
    sos_token: int = 0  # Start of sequence
    eos_token: int = 1  # End of sequence
    pad_token: int = 2  # Padding

    # Reward config
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    step_penalty: float = 0.0  # Optional penalty for longer formulas


class BoolformerState(NamedTuple):
    """State of the Boolformer environment."""

    truth_table: jax.Array  # (1024,) binary truth table
    formula_tokens: jax.Array  # (max_length,) token sequence so far
    position: jax.Array  # Current position in formula (scalar)
    terminated: jax.Array  # Whether episode is done (scalar bool)

    @property
    def current_length(self) -> jax.Array:
        """Number of tokens generated so far."""
        return self.position


class BoolformerEnv:
    """
    Boolformer environment for formula synthesis.

    Single-player environment compatible with mctx MCTS algorithms.
    """

    def __init__(self, config: BoolformerConfig = None):
        self.config = config or BoolformerConfig()

    def reset(self, key: jax.Array, truth_table: jax.Array = None) -> BoolformerState:
        """
        Reset environment with a new truth table.

        Args:
            key: JAX random key
            truth_table: Optional truth table (1024,). If None, generates random one.

        Returns:
            Initial state with SOS token
        """
        if truth_table is None:
            # Generate random truth table
            truth_table = jax.random.bernoulli(key, 0.5, (self.config.truth_table_size,))

        # Initialize formula with SOS token
        formula_tokens = jnp.full(
            (self.config.max_formula_length,),
            self.config.pad_token,
            dtype=jnp.int32
        )
        formula_tokens = formula_tokens.at[0].set(self.config.sos_token)

        return BoolformerState(
            truth_table=truth_table.astype(jnp.float32),
            formula_tokens=formula_tokens,
            position=jnp.int32(1),  # Next position after SOS
            terminated=jnp.bool_(False),
        )

    def step(
        self,
        state: BoolformerState,
        action: jax.Array
    ) -> Tuple[BoolformerState, jax.Array, jax.Array]:
        """
        Take a step by adding a token to the formula.

        Args:
            state: Current state
            action: Token to add (scalar in [0, vocab_size))

        Returns:
            (next_state, reward, terminated)
        """
        # Add token to formula
        formula_tokens = state.formula_tokens.at[state.position].set(action)
        position = state.position + 1

        # Check if EOS token or max length reached
        is_eos = action == self.config.eos_token
        is_max_length = position >= self.config.max_formula_length
        terminated = jnp.logical_or(is_eos, is_max_length)

        # Compute reward (only at end of episode)
        # TODO: Implement formula evaluation
        # For now, give 0 reward until termination
        reward = jnp.where(
            terminated,
            jnp.float32(0.0),  # Will be replaced with actual evaluation
            jnp.float32(self.config.step_penalty)
        )

        next_state = BoolformerState(
            truth_table=state.truth_table,
            formula_tokens=formula_tokens,
            position=position,
            terminated=terminated,
        )

        return next_state, reward, terminated

    def get_legal_actions(self, state: BoolformerState) -> jax.Array:
        """
        Get legal actions mask.

        Args:
            state: Current state

        Returns:
            Boolean mask (vocab_size,) indicating legal actions
        """
        # All tokens except PAD are legal
        # SOS is only legal at start (but we don't allow it after position 0)
        legal = jnp.ones(self.config.vocab_size, dtype=jnp.bool_)
        legal = legal.at[self.config.pad_token].set(False)
        legal = legal.at[self.config.sos_token].set(False)

        # If already terminated, no actions are legal
        legal = jnp.where(state.terminated, jnp.zeros_like(legal), legal)

        return legal

    @property
    def num_actions(self) -> int:
        """Number of possible actions (vocabulary size)."""
        return self.config.vocab_size


def create_recurrent_fn(env: BoolformerEnv):
    """
    Create recurrent function for mctx MCTS.

    This function takes a state and action, returns next state, reward, discount.
    Required signature for mctx.

    Returns:
        recurrent_fn compatible with mctx
    """
    def recurrent_fn(params, rng_key, action, state):
        """
        Args:
            params: Model parameters (not used in environment dynamics)
            rng_key: Random key (not used for deterministic environment)
            action: Action to take
            state: Current BoolformerState

        Returns:
            recurrent_fn_output with (reward, discount, next_state)
        """
        next_state, reward, terminated = env.step(state, action)

        # Discount is 0 if terminated, 1 otherwise (standard episodic setting)
        discount = jnp.where(terminated, jnp.float32(0.0), jnp.float32(1.0))

        # Return in mctx format
        # mctx expects: (prior_logits, value, state) for RootFnOutput
        # For recurrent_fn, we need: RecurrentFnOutput(reward, discount, prior_logits, value)
        # But prior_logits and value come from the model, not environment
        # So we just return the state changes here

        return reward, discount, next_state

    return recurrent_fn
