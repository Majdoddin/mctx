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
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import mctx
import optax

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerConfig
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
training_batch_size = 16  # Minibatch size for training
pool_size = 20  # Circular buffer size for sample pool

# Checkpointing
checkpoint_interval = 1




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
):
    """
    Generate batch of selfplay episodes.

    Returns (points, batch_data) where batch_data is tuple of:
        (encoder_outputs, formula_tokens, positions, action_weights, rewards)
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
    # batch_data[0]: encoder_outputs (batch, max_steps, max_points, n_embd)
    # batch_data[1]: formula_tokens (batch, max_steps, max_len)
    # batch_data[2]: positions (batch, max_steps)
    # batch_data[3]: action_weights (batch, max_steps, vocab_size)
    # batch_data[4]: rewards (batch, max_steps)
    return points, batch_data


class SamplePool:
    """Circular buffer for training samples using NumPy for mutability."""
    def __init__(self, pool_size: int, num_points: int, num_variables: int, max_len: int, vocab_size: int):
        self.pool_size = pool_size
        self.write_index = 0
        self.is_full = False

        # Use NumPy arrays (mutable)
        self.points = np.zeros((pool_size, num_points, num_variables), dtype=np.float32)
        self.formula_tokens = np.zeros((pool_size, max_len), dtype=np.int32)
        self.positions = np.zeros(pool_size, dtype=np.int32)
        self.policy_targets = np.zeros((pool_size, vocab_size), dtype=np.float32)
        self.value_targets = np.zeros(pool_size, dtype=np.float32)

    def add_samples(self, points, formula_tokens, positions, policy_targets, value_targets):
        """Add batch of samples to pool (circular overwrite)."""
        # Convert JAX → NumPy
        points = np.array(points)
        formula_tokens = np.array(formula_tokens)
        positions = np.array(positions)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)

        num_new = len(points)
        end_index = self.write_index + num_new

        # Compute two slices: before wraparound and after
        slice1_end = min(end_index, self.pool_size)
        slice1_len = slice1_end - self.write_index
        slice2_end = max(0, end_index - self.pool_size)

        # Write both chunks
        for dst_slice, src_slice in [
            (slice(self.write_index, slice1_end), slice(0, slice1_len)),
            (slice(0, slice2_end), slice(slice1_len, num_new))
        ]:
            self.points[dst_slice] = points[src_slice]
            self.formula_tokens[dst_slice] = formula_tokens[src_slice]
            self.positions[dst_slice] = positions[src_slice]
            self.policy_targets[dst_slice] = policy_targets[src_slice]
            self.value_targets[dst_slice] = value_targets[src_slice]

        self.write_index = end_index % self.pool_size
        if end_index >= self.pool_size:
            self.is_full = True

    def sample_batch(self, batch_size: int, rng_key: jax.Array, min_success_ratio: float = 0.25):
        """
        Sample random batch from pool with minimum success ratio.

        Args:
            batch_size: Number of samples to draw
            rng_key: Not used (using NumPy random)
            min_success_ratio: Minimum ratio of success samples: successes/total (default 0.25)
        """
        # Check pool composition
        num_successes_pool = np.sum(self.value_targets == 1.0)
        success_ratio_pool = num_successes_pool / self.pool_size

        # Use uniform sampling by default
        weights = None
        if success_ratio_pool < min_success_ratio:
            # Compute weight for successes to achieve minimum ratio
            # target_ratio = w * ratio / (w * ratio + (1-ratio))
            # Solving: w = target_ratio * (1-ratio) / (ratio * (1 - target_ratio))
            success_weight = (min_success_ratio * (1 - success_ratio_pool)) / (success_ratio_pool * (1 - min_success_ratio) + 1e-8)
            weights = np.where(self.value_targets == 1.0, success_weight, 1.0)
            weights = weights / weights.sum()  # Normalize

        # Sample with computed weights
        indices = np.random.choice(self.pool_size, size=batch_size, replace=False, p=weights)

        # Convert NumPy → JAX for training
        return (
            jnp.array(self.points[indices]),
            jnp.array(self.formula_tokens[indices]),
            jnp.array(self.positions[indices]),
            jnp.array(self.policy_targets[indices]),
            jnp.array(self.value_targets[indices]),
        )


def add_to_pool(points: jax.Array, batch_data: tuple, pool: SamplePool):
    """
    Flatten selfplay data and add to sample pool.

    Args:
        points: (batch, num_points, num_variables)
        batch_data: tuple of (encoder_outputs, formula_tokens, positions, action_weights, rewards)
        pool: SamplePool to add samples to
    """
    encoder_outputs, formula_tokens, positions, action_weights, rewards = batch_data
    batch_size, max_steps = rewards.shape

    # Compute value target for each step (final reward, no bootstrapping)
    final_rewards = jnp.sum(rewards, axis=1)  # (batch,)
    value_targets = jnp.broadcast_to(final_rewards[:, None], (batch_size, max_steps))  # (batch, max_steps)

    # Create mask for valid steps
    terminated = rewards != 0.0
    terminated_cumsum = jnp.cumsum(terminated, axis=1)
    valid_mask = (terminated_cumsum == 0) | terminated  # (batch, max_steps)

    # Extract final formula for each episode (last valid step)
    last_valid_idx = jnp.sum(valid_mask, axis=1) - 1  # (batch,)
    final_formulas = formula_tokens[jnp.arange(batch_size), last_valid_idx]  # (batch, max_len)

    # Repeat points and final formula for each step
    points_repeated = jnp.repeat(points[:, None, :, :], max_steps, axis=1)  # (batch, max_steps, num_points, num_variables)
    formulas_repeated = jnp.repeat(final_formulas[:, None, :], max_steps, axis=1)  # (batch, max_steps, max_len)

    # Flatten
    points_flat = points_repeated.reshape(-1, points.shape[1], points.shape[2])  # (batch*max_steps, num_points, num_variables)
    formula_tokens_flat = formulas_repeated.reshape(-1, final_formulas.shape[1])  # (batch*max_steps, max_len)
    positions_flat = positions.reshape(-1)  # (batch*max_steps,)
    policy_targets_flat = action_weights.reshape(-1, action_weights.shape[2])  # (batch*max_steps, vocab_size)
    value_targets_flat = value_targets.reshape(-1)  # (batch*max_steps,)
    mask_flat = valid_mask.reshape(-1)  # (batch*max_steps,)

    # Filter by mask - only add valid samples
    points_filtered = points_flat[mask_flat]
    formula_tokens_filtered = formula_tokens_flat[mask_flat]
    positions_filtered = positions_flat[mask_flat]
    policy_targets_filtered = policy_targets_flat[mask_flat]
    value_targets_filtered = value_targets_flat[mask_flat]

    # Add to pool
    pool.add_samples(
        points_filtered,
        formula_tokens_filtered,
        positions_filtered,
        policy_targets_filtered,
        value_targets_filtered,
    )


def loss_per_sample(
    model: BoolformerTransformer,
    points: jax.Array,  # (num_points, num_variables)
    formula_tokens: jax.Array,  # (max_len,)
    position: int,  # scalar
    policy_target: jax.Array,  # (vocab_size,)
    value_target: float,  # scalar
):
    """Compute loss for single sample."""
    # Encode points - add batch dim
    encoder_output = model.encode_points(points[None, ...])[0]  # (num_points, n_embd)

    # Forward pass - add batch dim
    policy_logits, values = model.decode_formula(
        formula_tokens[None, :], encoder_output[None, :, :], decode=False
    )
    # Remove batch dim: policy_logits: (max_len, vocab_size), values: (max_len)
    policy_logits = policy_logits[0]
    values = values[0]

    # Extract prediction at position (1-indexed → 0-indexed)
    policy_pred = policy_logits[position - 1]  # (vocab_size,)
    value_pred = values[position - 1]  # scalar

    # Policy loss: cross-entropy with MCTS action weights
    policy_loss = optax.softmax_cross_entropy(policy_pred, policy_target)

    # Value loss: L2 loss with final reward
    value_loss = optax.l2_loss(value_pred, value_target)

    return policy_loss, value_loss


def loss_fn(
    model: BoolformerTransformer,
    points: jax.Array,  # (batch, num_points, num_variables)
    formula_tokens: jax.Array,  # (batch, max_len)
    positions: jax.Array,  # (batch,)
    policy_targets: jax.Array,  # (batch, vocab_size)
    value_targets: jax.Array,  # (batch,)
):
    """
    Compute loss on batch of samples.

    Returns:
        total_loss, (policy_loss_mean, value_loss_mean)
    """
    # Compute failure scale: num_failures / num_successes
    num_success = jnp.sum(value_targets == 1.0)
    num_failure = jnp.sum(value_targets == -1.0)
    failure_scale = num_failure / (num_success + 1e-8)

    # DEBUG: Print gradient scaling
    jax.debug.print("⚖️  [loss_fn] num_success={}, num_failure={}, failure_scale={}",
                    num_success, num_failure, failure_scale)

    # Vmap over batch
    batch_loss_fn = jax.vmap(
        lambda p, f, pos, pt, vt: loss_per_sample(model, p, f, pos, pt, vt)
    )

    policy_losses, value_losses = batch_loss_fn(
        points, formula_tokens, positions, policy_targets, value_targets
    )
    # policy_losses, value_losses: (batch,)

    # Weighted mean to balance failures vs successes
    # weights = jnp.where(value_targets == -1.0, failure_scale, 1.0)  # (batch,)
    weights = jnp.ones_like(value_targets)  # Uncomment to disable failure scaling
    policy_loss_mean = jnp.sum(policy_losses * weights) / jnp.sum(weights)
    value_loss_mean = jnp.sum(value_losses * weights) / jnp.sum(weights)

    total_loss = policy_loss_mean + value_loss_mean

    return total_loss, (policy_loss_mean, value_loss_mean)


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

    # Initialize sample pool
    pool = SamplePool(
        pool_size=pool_size,
        num_points=2 ** (num_variables - 1),  # Max minority points
        num_variables=num_variables,
        max_len=max_formula_length,
        vocab_size=vocab_size,
    )
    print(f"Sample pool initialized with size {pool_size}")

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
        points, batch_data = selfplay_episode(model, env, root_fn, recurrent_fn, subkey)

        # Track episode success
        rewards = batch_data[4]
        num_success = jnp.sum(jnp.any(rewards == 1.0, axis=1)).item()
        num_fail = jnp.sum(jnp.any(rewards == -1.0, axis=1)).item()
        success_rate = num_success / selfplay_batch_size
        print(f"  Episodes: {num_success}/{selfplay_batch_size} success ({success_rate:.1%}), {num_fail} fail")

        # Add samples to pool
        add_to_pool(points, batch_data, pool)
        print(f"  Pool now has {pool.write_index} total samples added (full={pool.is_full})")

        # Training (only if pool is full)
        if not pool.is_full:
            print(f"  Skipping training (pool not full yet)\n")
            continue

        print(f"  Training on batch of {training_batch_size}...")
        rng_key, subkey = jax.random.split(rng_key)
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (policy_loss, value_loss)), grads = grad_fn(
            model, *pool.sample_batch(training_batch_size, subkey)
        )
        optimizer.update(model, grads)

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
