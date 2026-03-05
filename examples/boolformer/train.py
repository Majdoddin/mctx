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
from generate_data import generate_formulas, NUM_PHYSICAL_CORES


# Training configuration
num_variables = 2
vocab_size = 5 + num_variables
max_formula_length = 4 + 1 #+1 for SOS

# # CPU/test config: minimal settings for functionality testing
# n_embd = 16
# n_head = 2
# n_encoder_layers = 1
# n_decoder_layers = 1

# seed = 0
# max_num_iters = 5

# selfplay_batch_size = 4
# num_simulations = 2

# max_train_formula_length = 4
# length_distribution = [0.0, 0.0, 0.0, 0.0, 1]
# training_length_distribution = [0.0, 0.0, 0.0, 0.0, 1]
# min_success_ratio_per_length = [0.0, 0.0, 0.0, 0.0, 0.5]
# min_length_proportion = (0.0, 0.0, 0.0, 0.0, 1.0)

# learning_rate = 0.0002
# training_batch_size = 4
# training_steps_per_iter = 1
# pool_size = 20
# protect_l4_successes = False

# GPU config: RTX 3090(Ti) 24GB
# Model architecture (matches Boolformer config/transformer/noiseless.py: 59M params)
n_embd = 512           # D_MODEL (feedforward dim = 4 * n_embd = 2048 automatically)
n_head = 16            # NUM_HEADS
n_encoder_layers = 8   # NUM_ENCODER_LAYERS
n_decoder_layers = 8   # NUM_DECODER_LAYERS

seed = 0
max_num_iters = 1000

# Self-play: ~256 formulas × 2.5 samples/formula = ~640 samples/iter
selfplay_batch_size = 256
num_simulations = 8

max_train_formula_length = 4
length_distribution = [0.0, 0.0, 0.0, 0.0, 1]
training_length_distribution = [0.0, 0.0, 0.0, 0.0, 1]
min_success_ratio_per_length = [0.0, 0.0, 0.0, 0.0, 0.5]
min_length_proportion = (0.0, 0.0, 0.0, 0.0, 1.0)

# Training: 640 samples / 128 batch = 5 steps/iter → replay ratio 1.0
learning_rate = 0.0002
training_batch_size = 128
training_steps_per_iter = 5
pool_size = 6400
protect_l4_successes = True

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

    encoder_output and points are passed via mctx params (not stored per tree node).
    Returns (formula_tokens, position, action_weights, reward, is_perfect) per step.
    """
    state = env.reset(rng_key)
    max_steps = max_formula_length

    def step_fn(carry, step_rng_key):
        """One step of episode."""
        state = carry

        def active_step():
            # encoder_output passed to root_fn explicitly, and as params to mctx
            root = root_fn(state, encoder_output)

            policy_output = mctx.gumbel_muzero_policy(
                params=(encoder_output, points),  # NOT stored per tree node
                rng_key=step_rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                qtransform=mctx.qtransform_completed_by_mix_value,
            )

            action = policy_output.action[0]
            action_weights = policy_output.action_weights[0]

            next_state, reward, done, is_perfect = env.step(state, action, points)

            return next_state, action_weights, reward, is_perfect, done

        def terminated_step():
            # Already terminated: done=False (not a new termination, just padding)
            return state, jnp.zeros(vocab_size), jnp.float32(0.0), jnp.bool_(False), jnp.bool_(False)

        next_state, action_weights, reward, is_perfect, done = jax.lax.cond(
            state.terminated,
            terminated_step,
            active_step
        )

        step_data = (
            next_state.formula_tokens,  # (max_len,) - AFTER action
            state.position,  # scalar - position BEFORE action (where decision was made)
            action_weights,  # (vocab_size,)
            reward,  # scalar
            is_perfect,  # scalar bool
            done,  # scalar bool - whether this step ended the episode
        )

        return next_state, step_data

    keys = jax.random.split(rng_key, max_steps)
    _, episode_data = jax.lax.scan(step_fn, state, keys)

    return episode_data


@nnx.jit(static_argnums=(1, 2, 3))
def selfplay_jit(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    root_fn,
    recurrent_fn,
    points: jax.Array,
    rng_key: jax.Array
):
    """JIT-compiled selfplay: encode + vmapped MCTS. Compiled once, reused every iteration."""
    # Encode
    encoder_outputs = model.encode_points(points)

    # Generate keys for each episode
    episode_keys = jax.random.split(rng_key, points.shape[0])

    # Run episodes in parallel using vmap
    single_episode_fn = partial(selfplay_single_episode, model, env, root_fn, recurrent_fn)
    batch_data = jax.vmap(single_episode_fn)(points, encoder_outputs, episode_keys)

    return encoder_outputs, batch_data


def selfplay_episode(
    model: BoolformerTransformer,
    env: BoolformerEnv,
    root_fn,
    recurrent_fn,
    rng_key: jax.Array
):
    """
    Generate batch of selfplay episodes.

    Returns (points, encoder_outputs, batch_data, polish_exprs) where batch_data is tuple of:
        (formula_tokens, positions, action_weights, rewards, is_perfect)
    """
    batch_size = selfplay_batch_size

    # Generate minority points (Python/NumPy — outside JIT)
    max_gen_length = max_train_formula_length if max_train_formula_length is not None else 50
    points_array, polish_exprs = generate_formulas(batch_size, num_variables, max_gen_length, length_distribution)
    points = jnp.array(points_array)

    # JIT-compiled: encode + MCTS selfplay
    encoder_outputs, batch_data = selfplay_jit(model, env, root_fn, recurrent_fn, points, rng_key)

    return points, encoder_outputs, batch_data, polish_exprs


class SamplePool:
    """Circular buffer for training samples using NumPy for mutability."""
    def __init__(self, pool_size: int, num_points: int, num_variables: int, max_len: int, vocab_size: int):
        self.pool_size = pool_size
        self.write_index = 0
        self.total_written = 0  # Track total samples written
        self.is_full = False

        # Use NumPy arrays (mutable)
        self.points = np.zeros((pool_size, num_points, num_variables), dtype=np.float32)
        self.formula_tokens = np.zeros((pool_size, max_len), dtype=np.int32)
        self.target_polish_exprs = [None] * pool_size  # Target formula from generator (list of token strings)
        self.positions = np.zeros(pool_size, dtype=np.int32)
        self.policy_targets = np.zeros((pool_size, vocab_size), dtype=np.float32)
        self.value_targets = np.zeros(pool_size, dtype=np.float32)

    def add_samples(self, points, formula_tokens, target_polish_exprs, positions, policy_targets, value_targets):
        """Add batch of samples to pool (circular overwrite)."""
        # Convert JAX → NumPy
        points = np.array(points)
        formula_tokens = np.array(formula_tokens)
        positions = np.array(positions)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets)

        num_new = len(points)

        # Write samples one by one, skipping protected slots if enabled
        write_idx = self.write_index
        for src_idx in range(num_new):
            # Skip protected slots: length-4 successes (infinite-loops if pool is all L4 successes)
            if protect_l4_successes:
                while (self.target_polish_exprs[write_idx] is not None and
                       len(self.target_polish_exprs[write_idx]) == 4 and
                       self.value_targets[write_idx] == 1.0):
                    write_idx = (write_idx + 1) % self.pool_size

            # Write sample
            self.points[write_idx] = points[src_idx]
            self.formula_tokens[write_idx] = formula_tokens[src_idx]
            self.target_polish_exprs[write_idx] = target_polish_exprs[src_idx]
            self.positions[write_idx] = positions[src_idx]
            self.policy_targets[write_idx] = policy_targets[src_idx]
            self.value_targets[write_idx] = value_targets[src_idx]

            write_idx = (write_idx + 1) % self.pool_size

        self.write_index = write_idx
        self.total_written += num_new
        if self.total_written >= self.pool_size:
            self.is_full = True

    def sample_batch(self, batch_size: int, rng_key: jax.Array, length_distribution, min_success_ratio_per_length):
        """
        Sample batch from pool with length distribution and per-length success ratios.

        Args:
            batch_size: Number of samples to draw
            rng_key: Not used (using NumPy random)
            length_distribution: tuple of proportions per length (e.g., (0.0, 0.2, 0.3, 0.25, 0.25))
            min_success_ratio_per_length: tuple of min success ratios per length (e.g., (0.0, 0.2, 0.3, 0.2, 0.1))
        """
        # Compute formula lengths (count non-padding tokens, exclude SOS at position 0)
        # <PAD> = 1, <SOS> = 0
        formula_lengths = np.sum(self.formula_tokens[:, 1:] != 1, axis=1)

        max_len = len(length_distribution)

        # Pass 1: Gather statistics per length
        total_count = np.zeros(max_len, dtype=int)
        success_count = np.zeros(max_len, dtype=int)
        success_ratio = np.zeros(max_len, dtype=float)

        for length in range(1, max_len):
            length_mask = (formula_lengths == length)
            total_count[length] = length_mask.sum()
            if total_count[length] > 0:
                success_count[length] = ((self.value_targets == 1.0) & length_mask).sum()
                success_ratio[length] = success_count[length] / total_count[length]

        # Pass 2: Compute weights per sample
        weights = np.array(length_distribution)[formula_lengths]  # Start with length distribution

        # Boost successes for lengths that need more successes
        # NOTE: Boosting successes increases total weight for that length, breaking exact length distribution.
        # To maintain exact distribution, would need to proportionally reduce failures.
        for length in range(1, max_len):
            if success_ratio[length] < min_success_ratio_per_length[length] and success_ratio[length] > 0:
                # Compute boost factor to achieve target success ratio
                boost = (min_success_ratio_per_length[length] * (1 - success_ratio[length])) / (success_ratio[length] * (1 - min_success_ratio_per_length[length]) + 1e-8)

                # Apply boost to successes of this length
                length_and_success = (formula_lengths == length) & (self.value_targets == 1.0)
                weights[length_and_success] *= boost

        # Normalize and sample
        weights = weights / weights.sum()
        indices = np.random.choice(self.pool_size, size=batch_size, replace=False, p=weights)

        # Convert NumPy → JAX for training
        return (
            jnp.array(self.points[indices]),
            jnp.array(self.formula_tokens[indices]),
            jnp.array(self.positions[indices]),
            jnp.array(self.policy_targets[indices]),
            jnp.array(self.value_targets[indices]),
        )

    def add_from_batch(self, points: jax.Array, batch_data: tuple, polish_exprs: list):
        """
        Flatten selfplay data and add to sample pool.

        Args:
            points: (batch, num_points, num_variables)
            batch_data: tuple of (formula_tokens, positions, action_weights, rewards, is_perfect)
            polish_exprs: list of target polish expressions (one per episode)
        """
        formula_tokens, positions, action_weights, rewards, is_perfect, terminated_flags = batch_data
        batch_size, max_steps = rewards.shape

        # Compute value target for each step (final reward, no bootstrapping)
        final_rewards = jnp.sum(rewards, axis=1)  # (batch,)
        value_targets = jnp.broadcast_to(final_rewards[:, None], (batch_size, max_steps))  # (batch, max_steps)

        # Create mask for valid steps using terminated flag (not rewards != 0,
        # which breaks with adjusted IoU where terminated episodes can have reward 0)
        terminated_cumsum = jnp.cumsum(terminated_flags, axis=1)
        valid_mask = (terminated_cumsum == 0) | terminated_flags  # (batch, max_steps)

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

        # Repeat polish_exprs: each episode's expr repeated for its valid samples
        polish_exprs_filtered = [polish_exprs[i] for i in range(batch_size) for _ in range(int(jnp.sum(valid_mask[i])))]

        # Debug: print how many samples from each episode
        samples_per_episode = jnp.sum(valid_mask, axis=1)
        print(f"  DEBUG: total samples_per_episode = {jnp.sum(samples_per_episode)}")

        # Add to pool
        self.add_samples(
            points_filtered,
            formula_tokens_filtered,
            polish_exprs_filtered,
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
    # Vmap over batch
    batch_loss_fn = jax.vmap(
        lambda p, f, pos, pt, vt: loss_per_sample(model, p, f, pos, pt, vt)
    )

    policy_losses, value_losses = batch_loss_fn(
        points, formula_tokens, positions, policy_targets, value_targets
    )
    # policy_losses, value_losses: (batch,)

    # Uniform weighting for continuous rewards
    policy_loss_mean = jnp.mean(policy_losses)
    value_loss_mean = jnp.mean(value_losses)

    total_loss = policy_loss_mean + value_loss_mean

    return total_loss, (policy_loss_mean, value_loss_mean)


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

def update_curriculum(success_counts, total_counts):
    """
    Update length_distribution and training_length_distribution based on success rates.

    Generation distribution: Higher need (lower success) → higher proportion
    Training distribution: Only include lengths with enough success samples
    """
    max_len = len(length_distribution)
    new_gen_dist = [0.0] * max_len
    new_train_dist = [0.0] * max_len

    # Compute success rates
    success_rates = np.zeros(max_len)
    for length in range(1, max_len):
        if total_counts[length] > 0:
            success_rates[length] = success_counts[length] / total_counts[length]

    # Update generation distribution
    for length in range(1, max_len):
        need = 1.0 - success_rates[length]  # 0.0 if perfect, 1.0 if failing
        new_gen_dist[length] = min_length_proportion[length] + need * 0.3

    # Normalize generation distribution
    total = sum(new_gen_dist[1:])
    if total > 0:
        for length in range(1, max_len):
            new_gen_dist[length] /= total

    # Update training distribution
    for length in range(1, max_len):
        # Must have successes if min_success_ratio > 0
        if min_success_ratio_per_length[length] > 0 and success_counts[length] == 0:
            new_train_dist[length] = 0.0
        else:
            need = 1.0 - success_rates[length]
            new_train_dist[length] = min_length_proportion[length] + need * 0.3

    # Normalize training distribution
    total_train = sum(new_train_dist[1:])
    if total_train > 0:
        for length in range(1, max_len):
            new_train_dist[length] /= total_train

    return new_gen_dist, new_train_dist

# Training loop
import logging
logging.basicConfig()
jax.log_compiles()
rng_key = jax.random.key(seed)

print(f"\nStarting training... (formula gen: {NUM_PHYSICAL_CORES} physical cores)\n")
for iteration in range(max_num_iters):
    iter_start = time.time()

    # Selfplay
    selfplay_start = time.time()
    print(f"[Iter {iteration:04d}] Running selfplay...")
    rng_key, subkey = jax.random.split(rng_key)
    points, encoder_outputs, batch_data, polish_exprs = selfplay_episode(model, env, root_fn, recurrent_fn, subkey)

    # Track episode success using is_perfect flag
    rewards = batch_data[3]  # balanced accuracy values
    is_perfect = batch_data[4]  # perfect match flags
    episode_success = jnp.any(is_perfect, axis=1)  # (batch,) - any step was perfect
    num_success = jnp.sum(episode_success).item()
    success_rate = num_success / selfplay_batch_size

    # Count successes, totals, overshoot, and mean IoU per length
    success_counts = np.zeros(len(length_distribution), dtype=int)
    total_counts = np.zeros(len(length_distribution), dtype=int)
    overshoot_sum = np.zeros(len(length_distribution), dtype=float)
    iou_sum = np.zeros(len(length_distribution), dtype=float)

    # Get generated formula lengths from final step and max rewards per episode
    formula_tokens = batch_data[0]
    generated_lengths = jnp.sum(formula_tokens[:, -1, 1:] != 1, axis=1)
    max_rewards = jnp.max(rewards, axis=1)  # Best mIoU achieved in episode

    for i, expr in enumerate(polish_exprs):
        length = len(expr)
        total_counts[length] += 1
        iou_sum[length] += float(max_rewards[i])
        if episode_success[i]:
            success_counts[length] += 1
            overshoot_sum[length] += int(generated_lengths[i]) - length

    # Print overall and per-length stats with success/fail, mean IoU, and overshoot
    stats = []
    for i in range(1, len(length_distribution)):
        if total_counts[i] > 0:
            mean_iou = iou_sum[i] / total_counts[i]
            s = f"L{i}:{success_counts[i]}s/{total_counts[i]-success_counts[i]}f"
            s += f"(iou={mean_iou:.2f}"
            if success_counts[i] > 0:
                s += f",+{overshoot_sum[i]/success_counts[i]:.1f}"
            s += ")"
            stats.append(s)
    print(f"  Episodes: {num_success}/{selfplay_batch_size} ({success_rate:.1%}) | {' | '.join(stats)}")

    # Check value prediction accuracy
    # encoder_outputs: (batch, num_points, n_embd) — from selfplay_episode, computed once
    formula_tokens_batch = batch_data[0]  # (batch, max_steps, max_len)
    positions_batch = batch_data[1]  # (batch, max_steps)
    actual_rewards = batch_data[3]  # (batch, max_steps)

    # Get value predictions for all steps in all episodes
    # Flatten batch to (batch*max_steps, ...)
    batch_size, max_steps = actual_rewards.shape
    # Repeat encoder_outputs for each step: (batch, num_points, n_embd) → (batch*max_steps, num_points, n_embd)
    flat_encoder = jnp.repeat(encoder_outputs, max_steps, axis=0)
    flat_tokens = formula_tokens_batch.reshape(-1, formula_tokens_batch.shape[2])
    flat_positions = positions_batch.reshape(-1)
    flat_rewards = actual_rewards.reshape(-1)

    # Get predictions
    _, values = model.decode_formula(flat_tokens, flat_encoder, decode=False)

    # Value at SOS (position 0, which is position 1 in 1-indexed)
    # Reshape back to (batch, max_steps) to get first step per episode
    values_reshaped = values.reshape(batch_size, max_steps, -1)
    value_sos = values_reshaped[:, 0, 0]  # (batch,) - value at position 0 (SOS) for each episode
    mean_value_sos = jnp.mean(value_sos)

    # Value at final positions (completed formulas)
    batch_indices = jnp.arange(len(flat_positions))
    value_preds = values[batch_indices, flat_positions - 1]

    # Compute MAE for completed formulas (non-zero rewards)
    mask = flat_rewards > 0.0
    if jnp.sum(mask) > 0:
        value_mae_final = jnp.mean(jnp.abs(value_preds[mask] - flat_rewards[mask]))
        print(f"  Value: SOS={mean_value_sos:.3f}, Final MAE={value_mae_final:.4f}")
    else:
        print(f"  Value: SOS={mean_value_sos:.3f}")

    selfplay_time = time.time() - selfplay_start

    # Add samples to pool
    pool.add_from_batch(points, batch_data, polish_exprs)
    print(f"  Pool: {pool.total_written} total written, write_index={pool.write_index}, full={pool.is_full}")

    # Training (only if pool is full)
    if not pool.is_full:
        iter_time = time.time() - iter_start
        print(f"  Skipping training (pool not full yet) | Time: {iter_time:.2f}s (selfplay={selfplay_time:.2f}s)\n")
        continue

    # Update curriculum based on pool statistics (using target formula lengths)
    pool_expr_lengths = np.array([len(expr) for expr in pool.target_polish_exprs])
    pool_total_counts = np.zeros(len(length_distribution), dtype=int)
    pool_success_counts = np.zeros(len(length_distribution), dtype=int)
    for length in range(1, len(length_distribution)):
        length_mask = (pool_expr_lengths == length)
        pool_total_counts[length] = length_mask.sum()
        pool_success_counts[length] = ((pool.value_targets == 1.0) & length_mask).sum()

    # length_distribution[:], training_length_distribution[:] = update_curriculum(pool_success_counts, pool_total_counts)
    print(f"  Curriculum: gen={[f'{x:.2f}' for x in length_distribution[1:]]}, train={[f'{x:.2f}' for x in training_length_distribution[1:]]}")

    # Multiple training steps per iteration (like pgx AlphaZero)
    train_start = time.time()
    print(f"  Training: {training_steps_per_iter} steps × {training_batch_size} batch...")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    policy_losses, value_losses = [], []

    for step in range(training_steps_per_iter):
        rng_key, subkey = jax.random.split(rng_key)
        (loss, (policy_loss, value_loss)), grads = grad_fn(
            model, *pool.sample_batch(training_batch_size, subkey, training_length_distribution, min_success_ratio_per_length)
        )
        optimizer.update(model, grads)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

    # Average losses across training steps
    avg_policy_loss = sum(policy_losses) / len(policy_losses)
    avg_value_loss = sum(value_losses) / len(value_losses)
    avg_loss = avg_policy_loss + avg_value_loss

    train_time = time.time() - train_start
    iter_time = time.time() - iter_start

    print(f"  Loss: {avg_loss:.4f} (policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f})")
    print(f"  Time: {iter_time:.2f}s (selfplay={selfplay_time:.2f}s, train={train_time:.2f}s)\n")

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
