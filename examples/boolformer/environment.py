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
    """Configuration for Boolformer environment.

    Vocabulary (15 tokens):
        0: <SOS> - Start of sequence
        1: <PAD> - Padding
        2: ~     - Unary NOT operator
        3: &     - Binary AND operator
        4: |     - Binary OR operator
        5-14: x1-x10 - Boolean variables

    No EOS token - termination detected by Polish notation completeness check.
    """

    # Truth table config
    num_variables: int = 10
    truth_table_size: int = 1024  # 2^10

    # Formula config
    vocab_size: int = 15  # <SOS> + <PAD> + ~ + & + | + x1-x10
    max_formula_length: int = 50

    # Special tokens
    sos_token: int = 0
    pad_token: int = 1  # Must be 1 to match Boolformer's tokenization

    # Operator tokens
    not_token: int = 2   # Unary NOT (~)
    and_token: int = 3   # Binary AND (&)
    or_token: int = 4    # Binary OR (|)

    # Variable tokens (x1-x10)
    first_var_token: int = 5
    last_var_token: int = 14

    # Reward config
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    step_penalty: float = 0.0  # Optional penalty for longer formulas


class BoolformerState(NamedTuple):
    """State of the Boolformer environment."""

    points: jax.Array  # (num_points, 10) minority class variable combinations (for reward computation)
    encoder_output: jax.Array  # (num_points, n_embd) cached encoder output
    formula_tokens: jax.Array  # (max_length,) token sequence so far
    position: jax.Array  # Current position in formula (scalar)
    terminated: jax.Array  # Whether episode is done (scalar bool)
    legal_action_mask: jax.Array  # (vocab_size,) boolean mask of legal actions

    @property
    def current_length(self) -> jax.Array:
        """Number of tokens generated so far."""
        return self.position


def evaluate_polish_formula(
    tokens: jax.Array,
    position: jax.Array,
    variable_values: jax.Array,
    config: BoolformerConfig
) -> jax.Array:
    """
    Evaluate a Polish notation formula for given variable values.

    Uses stack-based evaluation with reversed iteration (right-to-left).

    Args:
        tokens: Formula tokens (max_length,), with SOS at position 0
        position: Formula length (including SOS)
        variable_values: Boolean values for x1-x10 (10,)
        config: BoolformerConfig

    Returns:
        Boolean result of evaluation
    """
    # Formula tokens are at tokens[1:position]
    # We'll process in reverse order, conditionally based on position

    # Stack for evaluation
    max_stack_size = config.max_formula_length
    stack = jnp.zeros(max_stack_size, dtype=jnp.bool_)

    def scan_fn(carry, i):
        """Process tokens in reverse (right-to-left)."""
        stack, sp = carry

        # Token index in original array (reversed, skipping SOS at position 0)
        # For formula at tokens[1:position], reversed:
        # i=0 -> tokens[position-1], i=1 -> tokens[position-2], ..., i=position-2 -> tokens[1]
        token_idx = position - 1 - i

        # Only process if this index is valid (>= 1, since SOS is at 0)
        should_process = token_idx >= 1

        token = tokens[token_idx]

        # Check token type
        is_and = token == config.and_token
        is_or = token == config.or_token
        is_not = token == config.not_token
        is_var = jnp.logical_and(token >= config.first_var_token, token <= config.last_var_token)

        # Binary operators: pop 2, push 1
        def binary_op():
            op1 = stack[sp - 1]
            op2 = stack[sp - 2]
            result = jnp.where(is_and, jnp.logical_and(op1, op2), jnp.logical_or(op1, op2))
            new_stack = stack.at[sp - 2].set(result)
            return new_stack, sp - 1

        # Unary operator: pop 1, push 1
        def unary_op():
            op1 = stack[sp - 1]
            result = jnp.logical_not(op1)
            new_stack = stack.at[sp - 1].set(result)
            return new_stack, sp

        # Variable: push value
        def push_var():
            var_idx = token - config.first_var_token
            value = variable_values[var_idx]
            new_stack = stack.at[sp].set(value)
            return new_stack, sp + 1

        # No-op for unknown tokens or out-of-range
        def no_op():
            return stack, sp

        # Process token if valid
        def process_token():
            # Conditional execution based on token type
            return jax.lax.cond(
                jnp.logical_or(is_and, is_or),
                binary_op,
                lambda: jax.lax.cond(
                    is_not,
                    unary_op,
                    lambda: jax.lax.cond(is_var, push_var, no_op)
                )
            )

        stack, sp = jax.lax.cond(should_process, process_token, no_op)

        return (stack, sp), None

    # Scan over all possible positions (fixed size)
    initial_carry = (stack, 0)
    (final_stack, final_sp), _ = jax.lax.scan(
        scan_fn,
        initial_carry,
        jnp.arange(config.max_formula_length)
    )

    # Result is at stack[0] (should have exactly 1 item)
    return final_stack[0]


def evaluate_formula_on_points(
    tokens: jax.Array,
    position: jax.Array,
    points: jax.Array,
    config: BoolformerConfig
) -> jax.Array:
    """Evaluate formula correctness: True for minority points, False for all others.

    Args:
        tokens: Formula tokens (max_length,)
        position: Formula length (including SOS)
        points: Minority class points (num_points, 10) with {-1, 0, +1}
        config: BoolformerConfig

    Returns:
        Boolean: True if formula outputs True for minority points and False for all others

    TODO: For training, return ratio of correct evaluations instead of binary 0/1
          to provide softer gradient signal
    """
    # Evaluate formula on all 1024 combinations
    def eval_row(row_idx):
        variable_values = jnp.array([(row_idx >> bit) & 1 for bit in range(config.num_variables)], dtype=jnp.bool_)
        return evaluate_polish_formula(tokens, position, variable_values, config)

    all_outputs = jax.vmap(eval_row)(jnp.arange(1024))

    # Check each point: should be in minority class (output=True) iff it's a non-padded point
    def check_point(point):
        is_padding = jnp.all(point == -1)
        # Convert point to row index: point values are {0, +1}, map to binary
        point_binary = (point == 1).astype(jnp.int32)
        row_idx = jnp.sum(point_binary * (2 ** jnp.arange(config.num_variables)))
        # Formula should output True for this minority point
        return jnp.where(is_padding, True, all_outputs[row_idx] == True)

    minority_correct = jnp.all(jax.vmap(check_point)(points))

    # Count True outputs - should equal number of non-padded points
    num_true_outputs = jnp.sum(all_outputs)
    num_minority_points = jnp.sum(~jnp.all(points == -1, axis=1))
    count_correct = num_true_outputs == num_minority_points

    return jnp.logical_and(minority_correct, count_correct)


def get_allowed_tokens(
    tokens: jax.Array,
    position: jax.Array,
    config: BoolformerConfig
) -> Tuple[jax.Array, jax.Array]:
    """
    Returns (is_complete, allowed_mask) for Polish notation formula generation.

    JAX port of Python reference implementation.

    Returns:
        is_complete: Boolean, True if formula is syntactically complete (open_slots == 0)
        allowed_mask: Boolean mask (vocab_size,) of valid next tokens

    Raises error if formula becomes syntactically incorrect (open_slots < 0).

    Usage:
        is_complete, mask = get_allowed_tokens(tokens, position, config)
        if is_complete:
            # Formula done, terminate
        else:
            # Use mask for legal actions

    Args:
        tokens: Token sequence (max_length,), with SOS at position 0
        position: Current position (next token goes here)
        config: BoolformerConfig
    """
    # Arity mapping: binary=2, unary=1, variables=0
    def get_arity(token):
        is_binary = jnp.logical_or(token == config.and_token, token == config.or_token)
        is_unary = token == config.not_token
        return jnp.where(is_binary, 2, jnp.where(is_unary, 1, 0))

    # 1. Calculate Open Slots (Stack Depth)
    # open_slots = open_slots - 1 + arity
    def scan_fn(open_slots, i):
        # Only process tokens at positions [1, position)
        should_process = jnp.logical_and(i >= 1, i < position)
        token = tokens[i]
        arity = get_arity(token)
        new_open_slots = open_slots - 1 + arity
        # Conditionally update based on whether this position is valid
        open_slots = jnp.where(should_process, new_open_slots, open_slots)
        return open_slots, open_slots

    # Scan over fixed range (all possible positions)
    # We process conditionally based on position value
    initial_open_slots = 1
    open_slots, all_open_slots = jax.lax.scan(
        scan_fn,
        initial_open_slots,
        jnp.arange(config.max_formula_length)
    )

    # Check if formula became invalid (open_slots went negative)
    # This should never happen with proper masking, but check anyway
    went_negative = jnp.any(all_open_slots < 0)
    # In JAX we can't raise errors, so we'll just return empty mask
    # The environment should never reach this state with proper masking

    # 2. Check if Complete
    is_complete = open_slots == 0

    # Calculate remaining space
    current_len = position - 1  # Exclude SOS
    remaining_space = config.max_formula_length - 1 - current_len  # -1 for SOS

    # If complete, no space, or went negative, no tokens allowed
    nothing_allowed = jnp.logical_or(
        jnp.logical_or(is_complete, remaining_space <= 0),
        went_negative
    )

    # Initialize all False
    allowed = jnp.zeros(config.vocab_size, dtype=jnp.bool_)

    # 3. Binary Operators (AND, OR)
    # Allowed if: (open_slots + 1) <= (remaining_space - 1)
    can_add_binary = jnp.logical_and(
        jnp.logical_not(nothing_allowed),
        (open_slots + 1) <= (remaining_space - 1)
    )
    allowed = allowed.at[config.and_token].set(can_add_binary)
    allowed = allowed.at[config.or_token].set(can_add_binary)

    # 4. Unary Operator (NOT)
    # Allowed if: open_slots <= (remaining_space - 1) AND prev != NOT
    can_add_unary = jnp.logical_and(
        jnp.logical_not(nothing_allowed),
        open_slots <= (remaining_space - 1)
    )
    # Rule: No consecutive NOTs
    prev_token = jnp.where(position > 1, tokens[position - 1], -1)
    prev_is_not = prev_token == config.not_token
    can_add_unary = jnp.logical_and(can_add_unary, jnp.logical_not(prev_is_not))
    allowed = allowed.at[config.not_token].set(can_add_unary)

    # 5. Variables (x1-x10)
    # Always allowed if not nothing_allowed
    can_add_var = jnp.logical_not(nothing_allowed)
    for var_token in range(config.first_var_token, config.last_var_token + 1):
        allowed = allowed.at[var_token].set(can_add_var)

    return is_complete, allowed


class BoolformerEnv:
    """
    Boolformer environment for formula synthesis.

    Single-player environment compatible with mctx MCTS algorithms.
    """

    def __init__(self, model, config: BoolformerConfig = None):
        """
        Args:
            model: BoolformerTransformer model for encoding truth tables
            config: BoolformerConfig
        """
        self.model = model
        self.config = config or BoolformerConfig()

    def reset(self, key: jax.Array, points: jax.Array, encoder_output: jax.Array) -> BoolformerState:
        """Reset environment with pre-computed encoder output.

        Args:
            key: JAX random key (unused)
            points: Points array (num_points, 10) with values in {-1, 0, +1}
            encoder_output: Pre-computed encoder output (num_points, n_embd)
        """
        formula_tokens = jnp.full((self.config.max_formula_length,), self.config.pad_token, dtype=jnp.int32)
        formula_tokens = formula_tokens.at[0].set(self.config.sos_token)

        position = jnp.int32(1)
        _, legal_action_mask = get_allowed_tokens(formula_tokens, position, self.config)

        return BoolformerState(
            points=points,
            encoder_output=encoder_output,
            formula_tokens=formula_tokens,
            position=position,
            terminated=jnp.bool_(False),
            legal_action_mask=legal_action_mask,
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
        # If already terminated, return state unchanged
        # (JAX will trace both branches, but this handles batched states correctly)

        # Add token to formula
        formula_tokens = state.formula_tokens.at[state.position].set(action)
        position = state.position + 1

        # Compute termination and legal actions for next state
        terminated, legal_action_mask = get_allowed_tokens(formula_tokens, position, self.config)

        # Compute reward only when terminated (formula syntactically complete)
        def eval_complete_formula():
            correct = evaluate_formula_on_points(formula_tokens, position, state.points, self.config)
            return jnp.where(correct,
                           jnp.float32(self.config.correct_reward),
                           jnp.float32(self.config.incorrect_reward))

        reward = jax.lax.cond(terminated, eval_complete_formula, lambda: jnp.float32(self.config.step_penalty))

        next_state = state._replace(
            formula_tokens=formula_tokens,
            position=position,
            terminated=terminated,
            legal_action_mask=legal_action_mask,
        )

        # If state was already terminated, return unchanged
        def terminated_branch():
            return state, jnp.float32(0.0), jnp.bool_(True)

        def active_branch():
            return next_state, reward, terminated

        return jax.lax.cond(
            state.terminated,
            terminated_branch,
            active_branch
        )

    def get_legal_actions(self, state: BoolformerState) -> jax.Array:
        """
        Get legal actions mask.

        Args:
            state: Current state

        Returns:
            Boolean mask (vocab_size,) indicating legal actions
        """
        # Return pre-computed legal action mask from state
        # Computed once in reset() and step() using get_allowed_tokens()
        return state.legal_action_mask

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
