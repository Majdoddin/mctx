"""
Diagnostic script: Run one selfplay + training iteration and inspect all intermediate values.
Self-contained — does NOT import from train.py (which has module-level training loop).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from functools import partial
import mctx

from boolformer_jax_model import BoolformerTransformer
from environment import BoolformerEnv, BoolformerConfig, evaluate_formula_balanced_accuracy
from mcts_integration import create_root_fn, create_recurrent_fn
from generate_data import generate_formulas

# ── Config (CPU) ──
num_variables = 2
vocab_size = 5 + num_variables  # 7
max_formula_length = 4 + 1  # +1 for SOS = 5
n_embd = 16
n_head = 2
n_encoder_layers = 1
n_decoder_layers = 1
selfplay_batch_size = 8
num_simulations = 2
pool_size = 40
training_batch_size = 8
length_distribution = [0.0, 0.0, 0.0, 0.0, 1]

print("=" * 80)
print("BOOLFORMER DIAGNOSTIC")
print("=" * 80)

# ── 1. Model initialization ──
print("\n[1] MODEL INITIALIZATION")
rngs = nnx.Rngs(0)
model = BoolformerTransformer(
    rngs=rngs, num_variables=num_variables, vocab_size=vocab_size,
    max_formula_length=max_formula_length, n_embd=n_embd, n_head=n_head,
    n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
)
model.init_weights()

print(f"  policy_head kernel: shape={model.policy_head.kernel.value.shape}, "
      f"all_zero={bool(jnp.allclose(model.policy_head.kernel.value, 0))}")
print(f"  value_fc2 kernel: shape={model.value_fc2.kernel.value.shape}, "
      f"all_zero={bool(jnp.allclose(model.value_fc2.kernel.value, 0))}")
print(f"  formula_embedding: shape={model.formula_embedding.embedding.value.shape}, "
      f"std={float(jnp.std(model.formula_embedding.embedding.value)):.4f}")

# ── 2. Forward pass on dummy data ──
print("\n[2] FORWARD PASS (dummy data)")
dummy_points = jnp.ones((1, 2**(num_variables-1), num_variables))
dummy_tokens = jnp.array([[0, 1, 1, 1, 1]])  # SOS + PAD

encoder_out = model.encode_points(dummy_points)
print(f"  encoder_output: shape={encoder_out.shape}, "
      f"mean={float(jnp.mean(encoder_out)):.4f}, std={float(jnp.std(encoder_out)):.4f}")

policy_logits, values = model.decode_formula(dummy_tokens, encoder_out, decode=False)
print(f"  policy_logits: shape={policy_logits.shape}, "
      f"mean={float(jnp.mean(policy_logits)):.6f}, std={float(jnp.std(policy_logits)):.6f}")
print(f"  values (all positions): {np.array(values[0])}")
print(f"  -> At init: policy logits ~0 (zero init), values ~0.5 (sigmoid(0))")

policy_probs = jax.nn.softmax(policy_logits[0, 0])
print(f"  softmax(policy[0,0]): {np.array(policy_probs)}")
print(f"  -> Should be uniform (~{1/vocab_size:.4f} each)")

# ── 3. Generate formulas and run selfplay ──
print("\n[3] SELFPLAY")
env_config = BoolformerConfig(
    num_variables=num_variables, vocab_size=vocab_size, max_formula_length=max_formula_length,
)
env = BoolformerEnv(model, env_config)
root_fn = create_root_fn(model, env)
recurrent_fn = create_recurrent_fn(model, env)

rng_key = jax.random.key(42)
rng_key, subkey = jax.random.split(rng_key)

points_array, polish_exprs = generate_formulas(selfplay_batch_size, num_variables, 4, length_distribution)
points = jnp.array(points_array)

print(f"  Generated {len(polish_exprs)} formulas")
print(f"  points shape: {points.shape}")
for i in range(min(3, len(polish_exprs))):
    print(f"    [{i}] expr={polish_exprs[i]}, points={np.array(points[i])}")


# Inline selfplay (copied from train.py to avoid import)
def selfplay_single_episode(model, env, root_fn, recurrent_fn, points, encoder_output, rng_key):
    state = env.reset(rng_key)
    max_steps = max_formula_length

    def step_fn(carry, step_rng_key):
        state = carry

        def active_step():
            root = root_fn(state, encoder_output)
            policy_output = mctx.gumbel_muzero_policy(
                params=(encoder_output, points),
                rng_key=step_rng_key, root=root, recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                qtransform=mctx.qtransform_completed_by_mix_value,
            )
            action = policy_output.action[0]
            action_weights = policy_output.action_weights[0]
            next_state, reward, done, is_perfect = env.step(state, action, points)
            return next_state, action_weights, reward, is_perfect, done

        def terminated_step():
            return state, jnp.zeros(vocab_size), jnp.float32(0.0), jnp.bool_(False), jnp.bool_(False)

        next_state, action_weights, reward, is_perfect, done = jax.lax.cond(
            state.terminated, terminated_step, active_step)

        step_data = (next_state.formula_tokens, state.position, action_weights, reward, is_perfect, done)
        return next_state, step_data

    keys = jax.random.split(rng_key, max_steps)
    _, episode_data = jax.lax.scan(step_fn, state, keys)
    return episode_data


@nnx.jit(static_argnums=(1, 2, 3))
def selfplay_jit(model, env, root_fn, recurrent_fn, points, rng_key):
    encoder_outputs = model.encode_points(points)
    episode_keys = jax.random.split(rng_key, points.shape[0])
    single_fn = partial(selfplay_single_episode, model, env, root_fn, recurrent_fn)
    batch_data = jax.vmap(single_fn)(points, encoder_outputs, episode_keys)
    return encoder_outputs, batch_data


encoder_outputs, batch_data = selfplay_jit(model, env, root_fn, recurrent_fn, points, subkey)
formula_tokens, positions, action_weights, rewards, is_perfect, done_flags = batch_data

print(f"\n  batch_data shapes:")
print(f"    formula_tokens: {formula_tokens.shape}")
print(f"    positions:      {positions.shape}")
print(f"    action_weights: {action_weights.shape}")
print(f"    rewards:        {rewards.shape}")
print(f"    is_perfect:     {is_perfect.shape}")
print(f"    done_flags:     {done_flags.shape}")

# ── 4. Episode details ──
print("\n[4] EPISODE DETAILS")
TOKEN_NAMES = ['SOS', 'PAD', '~', '&', '|'] + [f'x{i}' for i in range(num_variables)]

for ep in range(min(4, selfplay_batch_size)):
    ep_done = done_flags[ep]
    term_step = int(jnp.argmax(ep_done)) if jnp.any(ep_done) else -1
    final_reward = float(rewards[ep, term_step]) if term_step >= 0 else 0.0
    perf = bool(is_perfect[ep, term_step]) if term_step >= 0 else False

    print(f"\n  Episode {ep}: target={polish_exprs[ep]}, reward={final_reward:.4f}, perfect={perf}")

    for step in range(max_formula_length):
        if step > term_step and term_step >= 0:
            break
        pos = int(positions[ep, step])
        aw = action_weights[ep, step]
        tok = formula_tokens[ep, step]
        r = float(rewards[ep, step])
        d = bool(done_flags[ep, step])

        # Show formula so far
        formula_str = " ".join([TOKEN_NAMES[int(tok[j])] for j in range(pos + 1)])
        # Top actions
        top_k = jnp.argsort(aw)[-3:][::-1]
        top_str = ", ".join([f"{TOKEN_NAMES[int(t)]}={float(aw[t]):.3f}" for t in top_k])
        print(f"    step {step}: pos={pos}, formula=[{formula_str}], "
              f"done={d}, reward={r:.3f}, top=[{top_str}]")

# ── 5. Valid mask logic ──
print("\n[5] VALID MASK & VALUE TARGETS")
terminated_cumsum = jnp.cumsum(done_flags, axis=1)
valid_mask = (terminated_cumsum == 0) | done_flags

for ep in range(min(4, selfplay_batch_size)):
    valid = valid_mask[ep]
    print(f"  Episode {ep}: valid_steps={int(jnp.sum(valid))}, mask={np.array(valid, dtype=int)}")

final_rewards = jnp.sum(rewards, axis=1)
print(f"\n  final_rewards (value targets): {np.array(final_rewards)}")
episode_perfect = jnp.any(is_perfect, axis=1)
print(f"  episode_perfect: {np.array(episode_perfect)}")

# ── 6. Pool add & sample ──
print("\n[6] POOL LOGIC")

# Minimal pool implementation (avoid train.py import)
class DiagPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.points = np.zeros((pool_size, 2**(num_variables-1), num_variables), dtype=np.float32)
        self.formula_tokens = np.zeros((pool_size, max_formula_length), dtype=np.int32)
        self.positions = np.zeros(pool_size, dtype=np.int32)
        self.policy_targets = np.zeros((pool_size, vocab_size), dtype=np.float32)
        self.value_targets = np.zeros(pool_size, dtype=np.float32)
        self.is_perfect = np.zeros(pool_size, dtype=bool)
        self.count = 0

    def add_from_batch(self, points, batch_data, polish_exprs):
        formula_tokens, positions, action_weights, rewards, is_perfect, done_flags = batch_data
        batch_size, max_steps = rewards.shape

        final_rewards = jnp.sum(rewards, axis=1)
        value_targets = jnp.broadcast_to(final_rewards[:, None], (batch_size, max_steps))

        terminated_cumsum = jnp.cumsum(done_flags, axis=1)
        valid_mask = (terminated_cumsum == 0) | done_flags

        last_valid_idx = jnp.sum(valid_mask, axis=1) - 1
        final_formulas = formula_tokens[jnp.arange(batch_size), last_valid_idx]

        episode_perfect = jnp.any(is_perfect, axis=1)

        for ep in range(batch_size):
            for step in range(max_steps):
                if not valid_mask[ep, step]:
                    continue
                idx = self.count % self.pool_size
                self.points[idx] = np.array(points[ep])
                self.formula_tokens[idx] = np.array(final_formulas[ep])
                self.positions[idx] = int(positions[ep, step])
                self.policy_targets[idx] = np.array(action_weights[ep, step])
                self.value_targets[idx] = float(value_targets[ep, step])
                self.is_perfect[idx] = bool(episode_perfect[ep])
                self.count += 1

pool = DiagPool(pool_size)
pool.add_from_batch(points, batch_data, polish_exprs)
print(f"  Added {pool.count} samples")

# Run second batch to fill pool
rng_key, subkey = jax.random.split(rng_key)
points2, exprs2 = generate_formulas(selfplay_batch_size, num_variables, 4, length_distribution)
points2 = jnp.array(points2)
_, batch_data2 = selfplay_jit(model, env, root_fn, recurrent_fn, points2, subkey)
pool.add_from_batch(points2, batch_data2, exprs2)
print(f"  After 2nd batch: {pool.count} total samples")

n = min(pool.count, pool_size)
vt = pool.value_targets[:n]
print(f"\n  Value targets: min={vt.min():.4f}, max={vt.max():.4f}, mean={vt.mean():.4f}")
print(f"    zeros: {(vt == 0).sum()}, nonzero: {(vt > 0).sum()}, perfect(>=0.99): {(vt >= 0.99).sum()}")
print(f"  is_perfect: {pool.is_perfect[:n].sum()} / {n}")

# ── 7. Loss computation ──
print("\n[7] LOSS COMPUTATION")

def loss_per_sample(model, points, formula_tokens, position, policy_target, value_target):
    encoder_output = model.encode_points(points[None, ...])[0]
    policy_logits, values = model.decode_formula(formula_tokens[None, :], encoder_output[None, :, :], decode=False)
    policy_logits = policy_logits[0]
    values = values[0]
    policy_pred = policy_logits[position - 1]
    value_pred = values[position - 1]
    policy_loss = optax.softmax_cross_entropy(policy_pred, policy_target)
    value_loss = optax.l2_loss(value_pred, value_target)
    return policy_loss, value_loss


# Sample a batch
indices = np.random.choice(n, size=training_batch_size, replace=False)
s_points = jnp.array(pool.points[indices])
s_tokens = jnp.array(pool.formula_tokens[indices])
s_positions = jnp.array(pool.positions[indices])
s_policy = jnp.array(pool.policy_targets[indices])
s_values = jnp.array(pool.value_targets[indices])

# Per-sample losses
batch_policy, batch_value = jax.vmap(
    lambda p, f, pos, pt, vt: loss_per_sample(model, p, f, pos, pt, vt)
)(s_points, s_tokens, s_positions, s_policy, s_values)

weighted = batch_policy * s_values

print(f"  Per-sample results:")
print(f"    {'idx':>3} {'pos':>3} {'val_tgt':>8} {'pol_loss':>9} {'val_loss':>9} {'weighted':>9} {'perfect':>7}")
for i in range(training_batch_size):
    print(f"    {i:3d} {int(s_positions[i]):3d} {float(s_values[i]):8.4f} "
          f"{float(batch_policy[i]):9.4f} {float(batch_value[i]):9.6f} "
          f"{float(weighted[i]):9.4f} {bool(pool.is_perfect[indices[i]]):>7}")

# Aggregated
policy_loss_mean = float(jnp.mean(weighted))
value_loss_mean = float(jnp.mean(batch_value))
total = policy_loss_mean + 10.0 * value_loss_mean
print(f"\n  Aggregated:")
print(f"    policy_loss (reward-weighted mean): {policy_loss_mean:.4f}")
print(f"    value_loss (mean):                  {value_loss_mean:.6f}")
print(f"    total (policy + 10*value):           {total:.4f}")

nonzero_mask = np.array(s_values) > 0
if nonzero_mask.any():
    print(f"\n  SUCCESS samples: policy_loss mean={float(jnp.mean(batch_policy[nonzero_mask])):.4f}")
if (~nonzero_mask).any():
    print(f"  FAILED  samples: policy_loss mean={float(jnp.mean(batch_policy[~nonzero_mask])):.4f}")
    print(f"  -> If similar, MCTS targets from failures are as 'confident' as from successes")

# ── 8. Value head predictions ──
print("\n[8] VALUE HEAD PREDICTIONS")
all_points_j = jnp.array(pool.points[:n])
all_tokens_j = jnp.array(pool.formula_tokens[:n])
all_positions_j = jnp.array(pool.positions[:n])

enc_all = model.encode_points(all_points_j)
_, val_all = model.decode_formula(all_tokens_j, enc_all, decode=False)

batch_idx = jnp.arange(n)
pred_values = val_all[batch_idx, all_positions_j - 1]

print(f"  Predicted: min={float(jnp.min(pred_values)):.4f}, max={float(jnp.max(pred_values)):.4f}, "
      f"mean={float(jnp.mean(pred_values)):.4f}, std={float(jnp.std(pred_values)):.4f}")
print(f"  Targets:   min={vt.min():.4f}, max={vt.max():.4f}, mean={vt.mean():.4f}")

sm = pool.is_perfect[:n]
if sm.sum() > 0:
    print(f"\n  SUCCESS: pred_mean={float(jnp.mean(pred_values[sm])):.4f}, target_mean={vt[sm].mean():.4f}")
if (~sm).sum() > 0:
    print(f"  FAILED:  pred_mean={float(jnp.mean(pred_values[~sm])):.4f}, target_mean={vt[~sm].mean():.4f}")
print(f"  -> At init all preds should be ~0.5. After training should diverge.")

# ── 9. Gradient analysis ──
print("\n[9] GRADIENT ANALYSIS")

def loss_fn(model, points, formula_tokens, positions, policy_targets, value_targets):
    batch_loss_fn = jax.vmap(
        lambda p, f, pos, pt, vt: loss_per_sample(model, p, f, pos, pt, vt))
    policy_losses, value_losses = batch_loss_fn(points, formula_tokens, positions, policy_targets, value_targets)
    weighted_policy = policy_losses * value_targets
    p_mean = jnp.mean(weighted_policy)
    v_mean = jnp.mean(value_losses)
    total = p_mean + 10.0 * v_mean
    return total, (p_mean, v_mean)

grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
(loss, (p_loss, v_loss)), grads = grad_fn(model, s_points, s_tokens, s_positions, s_policy, s_values)

# Gradient norms for key layers
state = nnx.state(grads)
print(f"  Key gradient norms:")
for path, value in jax.tree_util.tree_leaves_with_path(state):
    path_str = "/".join(str(p) for p in path)
    if hasattr(value, 'shape') and ('kernel' in path_str or 'embedding' in path_str):
        gn = float(jnp.sqrt(jnp.sum(value ** 2)))
        if gn > 0:
            print(f"    {path_str}: {gn:.6f} (shape={value.shape})")

# ── 10. One training step ──
print("\n[10] ONE TRAINING STEP")
optimizer = nnx.Optimizer(model, optax.adam(0.0002), wrt=nnx.All(nnx.Param))

# Before
_, val_before = model.decode_formula(s_tokens[:1], model.encode_points(s_points[:1]), decode=False)
v_before = float(val_before[0, int(s_positions[0]) - 1])

# Step
(loss, (p_loss, v_loss)), grads = grad_fn(model, s_points, s_tokens, s_positions, s_policy, s_values)
optimizer.update(model, grads)

# After
_, val_after = model.decode_formula(s_tokens[:1], model.encode_points(s_points[:1]), decode=False)
v_after = float(val_after[0, int(s_positions[0]) - 1])

print(f"  Sample 0: value_target={float(s_values[0]):.4f}")
print(f"    before={v_before:.6f}, after={v_after:.6f}, delta={v_after - v_before:.6f}")
print(f"  -> Should move toward target")

# ── 11. Reward sanity ──
print("\n[11] REWARD SANITY")
cfg = env_config
token_map = {'~': cfg.not_token, '&': cfg.and_token, '|': cfg.or_token}
for i in range(num_variables):
    token_map[f'x_{i}'] = cfg.first_var_token + i
    token_map[f'x{i+1}'] = cfg.first_var_token + i  # generate_data uses x1,x2,... not x_0,x_1

# Test correct formula
test_expr = polish_exprs[0]
test_points = points[0]
tokens = jnp.full((max_formula_length,), cfg.pad_token, dtype=jnp.int32)
tokens = tokens.at[0].set(cfg.sos_token)
for i, tok_str in enumerate(test_expr):
    tokens = tokens.at[i + 1].set(token_map[tok_str])
position = jnp.int32(len(test_expr) + 1)

miou, is_perf = evaluate_formula_balanced_accuracy(tokens, position, test_points, cfg)
print(f"  Correct formula {test_expr} -> mIoU={float(miou):.4f}, perfect={bool(is_perf)}")
print(f"  -> Should be 1.0 / True")

# Test single variable formula
tokens_x0 = jnp.full((max_formula_length,), cfg.pad_token, dtype=jnp.int32)
tokens_x0 = tokens_x0.at[0].set(cfg.sos_token).at[1].set(cfg.first_var_token)
miou_x0, perf_x0 = evaluate_formula_balanced_accuracy(tokens_x0, jnp.int32(2), test_points, cfg)
print(f"  Just 'x_0' -> mIoU={float(miou_x0):.4f}, perfect={bool(perf_x0)}")
print(f"  -> Should be ~0 (baseline-adjusted)")

# Show selfplay reward distribution
all_r = np.array(rewards).flatten()
nonzero_r = all_r[all_r > 0]
print(f"\n  Selfplay rewards: {len(nonzero_r)} nonzero / {len(all_r)} total")
if len(nonzero_r) > 0:
    print(f"    nonzero: min={nonzero_r.min():.4f}, max={nonzero_r.max():.4f}, mean={nonzero_r.mean():.4f}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
