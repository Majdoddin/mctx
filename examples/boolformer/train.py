"""
MCTS training for Boolformer.

Single-player AlphaZero-style training:
1. Selfplay: Generate formulas with MCTS
2. Train: Update model on (policy_logits, value) predictions
"""

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
import mctx
import optax

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerConfig, BoolformerState
from mcts_integration import create_root_fn, create_recurrent_fn
from generate_data import generate_formulas


# Training configuration
num_variables = 2
vocab_size = 5 + num_variables
max_formula_length = 4 + 1 #+1 for SOS

# Model architecture (matches Boolformer config/transformer/noiseless.py: 59M params)
# n_embd = 512  # D_MODEL (feedforward dim = 4 * n_embd = 2048 automatically)
# n_head = 16  # NUM_HEADS
# n_encoder_layers = 8  # NUM_ENCODER_LAYERS
# n_decoder_layers = 8  # NUM_DECODER_LAYERS

# CPU/test config:
n_embd=128
n_head=8
n_encoder_layers=2
n_decoder_layers=2

# Training
seed = 0
max_num_iters = 20
selfplay_batch_size = 3#128  # Formulas per iteration
num_simulations = 3#8  # MCTS simulations per action
max_train_formula_length = 4  # Filter out formulas longer than this (None = no filter)
# temperature = 1.0  # Not used (gumbel_muzero_policy uses Gumbel sampling, not temperature)
learning_rate = 0.0002  # Matches Boolformer LEARNING_RATE
training_batch_size = 512  # Minibatch size for training

# Checkpointing
checkpoint_interval = 1


class SelfplayData(NamedTuple):
    """Data collected during selfplay."""
    encoder_outputs: jax.Array  # (batch, max_steps, max_points, n_embd) - cached encoder output
    formula_tokens: jax.Array  # (batch, max_steps, max_len) - formula at each step
    positions: jax.Array  # (batch, max_steps) - position at each step
    action_weights: jax.Array  # (batch, max_steps, vocab_size) - MCTS visit counts
    rewards: jax.Array  # (batch, max_steps) - reward at each step


def selfplay_single_episode(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    root_fn,
    recurrent_fn,
    points: jax.Array,
    encoder_output: jax.Array,
    rng_key: jax.Array
):
    """
    Generate one episode with MCTS.

    Returns lists of (encoder_output, formula_tokens, position, action_weights, reward) for each step.
    """
    state = env.reset(rng_key, points, encoder_output)
    max_steps = max_formula_length

    # Storage for episode data
    data_encoder = []
    data_formula = []
    data_positions = []
    data_action_weights = []
    data_rewards = []

    def step_fn(carry, step_rng_key):
        """One step of episode."""
        state = carry

        # If already terminated, just pass through
        def active_step():
            # Get root
            root = root_fn(state)

            # Run MCTS
            policy_output = mctx.gumbel_muzero_policy(
                params=None,
                rng_key=step_rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                qtransform=mctx.qtransform_completed_by_mix_value,
            )

            # Take action
            action = policy_output.action[0]  # Remove batch dim
            action_weights = policy_output.action_weights[0]  # Remove batch dim

            # DEBUG: Print MCTS output (train.py:112)
            jax.debug.print("🎯 [MCTS] selected_action={}, action_weights={}", action, action_weights)

            next_state, reward, terminated = env.step(state, action)

            return next_state, action_weights, reward

        def terminated_step():
            # Return dummy values
            return state, jnp.zeros(vocab_size), jnp.float32(0.0)

        next_state, action_weights, reward = jax.lax.cond(
            state.terminated,
            terminated_step,
            active_step
        )

        # Store data
        # TODO: Optimize memory - encoder_output is duplicated across all steps
        #       Currently: (max_steps, max_points, n_embd) per episode
        #       Could store once: (max_points, n_embd) per episode
        step_data = (
            next_state.encoder_output,  # (max_points, n_embd)
            next_state.formula_tokens,  # (max_len,) - AFTER action
            state.position,  # scalar - position BEFORE action (where decision was made)
            action_weights,  # (vocab_size,)
            reward,  # scalar
        )

        return next_state, step_data

    # Generate RNG keys for all steps
    keys = jax.random.split(rng_key, max_steps)

    # Run episode
    _, episode_data = jax.lax.scan(step_fn, state, keys)

    return episode_data


def selfplay_episode(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    root_fn,
    recurrent_fn,
    rng_key: jax.Array
) -> SelfplayData:
    """
    Generate batch of selfplay episodes.

    Returns data for training: observations and MCTS targets at each step.
    """
    batch_size = selfplay_batch_size

    # Generate minority points using Boolformer formula generator
    # TODO: Convert to JAX for JIT compilation (currently uses Python/NumPy)
    max_gen_length = max_train_formula_length if max_train_formula_length is not None else 50  # TODO: Make default configurable
    points_array, polish_exprs = generate_formulas(batch_size, num_variables, max_gen_length)
    points = jnp.array(points_array)  # (batch_size, max_points, num_variables)

    # Batch encode all points at once: (batch_size, max_points, num_variables) -> (batch_size, max_points, n_embd)
    encoder_outputs = model.encode_points(points)

    # Generate keys for each episode
    episode_keys = jax.random.split(rng_key, batch_size)

    # Run episodes in parallel using vmap
    single_episode_fn = partial(selfplay_single_episode, model, env, root_fn, recurrent_fn)

    # vmap over (points, encoder_outputs, rng_key)
    # Episode i gets: points[i] (max_points, num_variables), encoder_outputs[i] (max_points, n_embd), episode_keys[i]
    batch_data = jax.vmap(single_episode_fn)(points, encoder_outputs, episode_keys)

    # batch_data is tuple of arrays with shape (batch, max_steps, ...)
    return SelfplayData(
        encoder_outputs=batch_data[0],  # (batch, max_steps, max_points, n_embd)
        formula_tokens=batch_data[1],  # (batch, max_steps, max_len)
        positions=batch_data[2],  # (batch, max_steps)
        action_weights=batch_data[3],  # (batch, max_steps, vocab_size)
        rewards=batch_data[4],  # (batch, max_steps)
    )


class TrainingSample(NamedTuple):
    """Training samples after processing selfplay data (batch of episodes)."""
    encoder_output: jax.Array  # (batch, num_points, n_embd)
    formula_tokens: jax.Array  # (batch, max_len)
    positions: jax.Array  # (batch, max_steps) - position at each step
    policy_targets: jax.Array  # (batch, max_steps, vocab_size) - action_weights from MCTS
    value_targets: jax.Array  # (batch, max_steps) - final reward for each step
    mask: jax.Array  # (batch, max_steps) bool - valid steps


def compute_training_samples(data: SelfplayData) -> TrainingSample:
    """
    Convert selfplay data to training samples.

    Keeps batch structure (batch, max_steps) for efficient training.
    - Input: (encoder_output, formula_tokens) per episode
    - Targets: (policy_targets, value_targets) for all steps
    """
    batch_size, max_steps = data.rewards.shape

    # Compute value target for each step (final reward, no bootstrapping)
    # Since rewards are sparse (only non-zero at termination), sum gives final reward
    # Shape: (batch,)
    final_rewards = jnp.sum(data.rewards, axis=1)
    value_targets = jnp.broadcast_to(final_rewards[:, None], (batch_size, max_steps))

    # Create mask for valid steps
    # We assume reward != 0 indicates termination
    terminated = data.rewards != 0.0
    terminated_cumsum = jnp.cumsum(terminated, axis=1)
    # Valid steps: before termination (cumsum == 0) or at termination (terminated == True)
    # After termination, cumsum stays at 1 but terminated is False, so we mask those out
    valid_mask = (terminated_cumsum == 0) | terminated

    # Extract final formula_tokens and encoder_output for each episode (last valid step)
    last_valid_idx = jnp.sum(valid_mask, axis=1) - 1  # (batch,)
    final_formula_tokens = data.formula_tokens[jnp.arange(batch_size), last_valid_idx]  # (batch, max_len)
    final_encoder_outputs = data.encoder_outputs[jnp.arange(batch_size), last_valid_idx]  # (batch, num_points, n_embd)

    return TrainingSample(
        encoder_output=final_encoder_outputs,  # (batch, num_points, n_embd)
        formula_tokens=final_formula_tokens,  # (batch, max_len)
        positions=data.positions,  # (batch, max_steps)
        policy_targets=data.action_weights,  # (batch, max_steps, vocab_size)
        value_targets=value_targets,  # (batch, max_steps)
        mask=valid_mask,  # (batch, max_steps)
    )


def loss_per_episode(
    model: BoolformerTransformer,
    encoder_output: jax.Array,  # (num_points, n_embd)
    formula_tokens: jax.Array,  # (max_len,)
    positions: jax.Array,  # (max_steps,) - position at each step
    policy_targets: jax.Array,  # (max_steps, vocab_size)
    value_targets: jax.Array,  # (max_steps,)
    mask: jax.Array,  # (max_steps,) bool
    failure_scale: jax.Array  # scalar
):
    """Compute loss for all steps in one episode."""
    # Forward pass: decode formula once to get all position outputs
    # vmap automatically handles batching when this is called with jax.vmap
    policy_logits, values = model.decode_formula(
        formula_tokens, encoder_output, decode=False
    )
    # policy_logits: (max_len, vocab_size), values: (max_len)

    # Extract predictions at each step's position
    # positions are 1-indexed, convert to 0-indexed
    indices = positions - 1  # (max_steps,)
    policy_preds = policy_logits[indices]  # (max_steps, vocab_size)
    value_preds = values[indices]  # (max_steps,)

    # Policy loss: cross-entropy with MCTS action weights
    policy_losses = jax.vmap(optax.softmax_cross_entropy)(policy_preds, policy_targets)  # (max_steps,)

    # Value loss: L2 loss with final reward
    # Broadcast value_targets if it's scalar to match value_preds shape
    value_targets_broadcast = jnp.broadcast_to(value_targets, value_preds.shape)
    value_losses = jax.vmap(optax.l2_loss)(value_preds, value_targets_broadcast)  # (max_steps,)

    # Scale failures by ratio to balance gradient contribution
    scales = jnp.where(value_targets_broadcast == -1.0, failure_scale, 1.0)  # (max_steps,)

    # Apply mask and scaling
    policy_losses = policy_losses * mask * scales
    value_losses = value_losses * mask * scales

    return policy_losses, value_losses


def loss_fn(
    model: BoolformerTransformer,
    samples: TrainingSample
):
    """
    Compute loss on batch of episodes.

    Returns:
        total_loss, (policy_loss_mean, value_loss_mean)
    """
    # Compute failure scale: num_failures / num_successes across all valid steps
    num_success = jnp.sum((samples.value_targets == 1.0) & samples.mask)
    num_failure = jnp.sum((samples.value_targets == -1.0) & samples.mask)

    # Scale = num_failures / num_successes (add epsilon to avoid division by zero)
    failure_scale = num_failure / (num_success + 1e-8)

    # DEBUG: Print gradient scaling
    jax.debug.print("⚖️  [loss_fn] num_success={}, num_failure={}, failure_scale={}",
                    num_success, num_failure, failure_scale)

    # Vmap over batch of episodes
    batch_loss_fn = jax.vmap(
        lambda e, f, p, pt, vt, m: loss_per_episode(model, e, f, p, pt, vt, m, failure_scale)
    )

    policy_losses, value_losses = batch_loss_fn(
        samples.encoder_output,
        samples.formula_tokens,
        samples.positions,
        samples.policy_targets,
        samples.value_targets,
        samples.mask,
    )
    # policy_losses, value_losses: (batch, max_steps)

    # Mean over all valid steps
    num_valid = jnp.sum(samples.mask) + 1e-8
    policy_loss_mean = jnp.sum(policy_losses) / num_valid
    value_loss_mean = jnp.sum(value_losses) / num_valid

    total_loss = policy_loss_mean + value_loss_mean

    return total_loss, (policy_loss_mean, value_loss_mean)


def train_step(
    model: BoolformerTransformer,
    optimizer: nnx.Optimizer,
    samples: TrainingSample
):
    """Single training step."""
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (policy_loss, value_loss)), grads = grad_fn(model, samples)

    # Update parameters (new API requires both model and grads)
    optimizer.update(model, grads)

    return loss, policy_loss, value_loss


def main():
    print("=" * 80)
    print("BOOLFORMER MCTS TRAINING")
    print("=" * 80)
    print(f"""
Config:
  num_variables={num_variables}
  vocab_size={vocab_size}
  max_formula_length={max_formula_length}
  n_embd={n_embd}, n_head={n_head}
  n_encoder_layers={n_encoder_layers}, n_decoder_layers={n_decoder_layers}
  seed={seed}
  max_num_iters={max_num_iters}
  selfplay_batch_size={selfplay_batch_size}
  num_simulations={num_simulations}
  learning_rate={learning_rate}
  checkpoint_interval={checkpoint_interval}
""")

    # Initialize model
    print("Initializing model...")
    rngs = nnx.Rngs(seed)
    model = BoolformerTransformer(
        rngs=rngs,
        num_variables=num_variables,
        vocab_size=vocab_size,
        max_formula_length=max_formula_length,
        n_embd=n_embd,
        n_head=n_head,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
    )

    # Reinitialize weights following nanochat scheme
    print("Reinitializing weights with nanochat scheme...")
    model.init_weights()

    # Initialize optimizer (use all model parameters)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.All(nnx.Param))

    # Initialize environment
    env_config = BoolformerConfig(
        num_variables=num_variables,
        vocab_size=vocab_size,
        max_formula_length=max_formula_length,
    )
    env = BoolformerEnv(model, env_config)

    # Create MCTS functions once (reused across all iterations)
    root_fn = create_root_fn(model, env)
    recurrent_fn = create_recurrent_fn(model, env)

    # Prepare checkpoint directory
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"boolformer_{now_str}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints: {ckpt_dir}")

    # Training loop
    rng_key = jax.random.key(seed)

    print("\nStarting training...\n")
    for iteration in range(max_num_iters):
        iter_start = time.time()

        # Selfplay
        print(f"[Iter {iteration:04d}] Running selfplay...")
        rng_key, subkey = jax.random.split(rng_key)
        selfplay_data = selfplay_episode(model, env, root_fn, recurrent_fn, subkey)

        # Track episode success (check if episode has +1 or -1 anywhere)
        num_success = jnp.sum(jnp.any(selfplay_data.rewards == 1.0, axis=1)).item()
        num_fail = jnp.sum(jnp.any(selfplay_data.rewards == -1.0, axis=1)).item()
        success_rate = num_success / selfplay_batch_size

        # Compute training samples
        samples = compute_training_samples(selfplay_data)
        num_valid = jnp.sum(samples.mask).item()
        print(f"  Episodes: {num_success}/{selfplay_batch_size} success ({success_rate:.1%}), {num_fail} fail")
        print(f"  Generated {num_valid} training samples")

        # Training
        print(f"  Training...")
        loss, policy_loss, value_loss = train_step(model, optimizer, samples)

        iter_time = time.time() - iter_start

        print(f"  Loss: {loss:.4f} (policy={policy_loss:.4f}, value={value_loss:.4f})")
        print(f"  Time: {iter_time:.2f}s\n")

        # Checkpoint
        if iteration % checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:06d}.ckpt")
            with open(ckpt_path, "wb") as f:
                state_dict = nnx.state(model)
                pickle.dump({
                    "model_state": state_dict,
                    "iteration": iteration,
                }, f)
            print(f"  Saved checkpoint: {ckpt_path}\n")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
