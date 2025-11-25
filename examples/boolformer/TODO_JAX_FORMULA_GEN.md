# TODO: JAX Formula Generation

Currently `generate_formulas()` in `generate_data.py` uses Python/NumPy and cannot be JIT-compiled.

## What needs to be ported to JAX:

### 1. Formula Tree Generation
**Current**: `generate_formula(config)` uses Python recursion
```python
# From src/formula/generation_formula.py
def generate_formula(config):
    # Recursive tree building in Python
    # Returns Formula object
```

**JAX version needed**:
- Iterative tree construction (JAX doesn't support recursion well)
- Fixed-size tree representation
- Use `jax.lax.scan` or `jax.lax.while_loop`

### 2. Formula Evaluation
**Current**: `formula.evaluate_pts(pts)` uses PyTorch
```python
evals = formula.evaluate_pts(pts)  # PyTorch tensor operations
tt = evals[:, -1].numpy()
```

**JAX version**: ✅ Already exists!
- `evaluate_polish_formula()` in `environment.py` (lines 74-173)
- Just need to vmap over all 1024 truth table entries

### 3. Token Conversion
**Current**: String tokens → IDs
```python
token_to_id('&') → 3
token_to_id('x1') → 5
```

**JAX version**:
- Use integer arrays directly (no string conversion needed)
- Map tree nodes to token IDs in-place

### 4. Negation Logic
**Current**: Count 0s vs 1s, conditionally negate
```python
if count_zeros < count_ones:
    tt = 1 - tt
    polish_expr = ['~'] + polish_expr
```

**JAX version**: Easy!
```python
count_ones = jnp.sum(tt)
should_negate = count_ones > 512  # More 1s than 0s
tt = jnp.where(should_negate, 1 - tt, tt)
tokens = jnp.where(should_negate,
                   jnp.concatenate([jnp.array([not_token]), tokens]),
                   tokens)
```

## Priority:

**High**: Formula tree generation (main bottleneck)
**Medium**: Token conversion (minor overhead)
**Low**: Negation logic (already fast in NumPy, JAX won't help much)

## Benefits of JAX version:

1. **JIT compilation**: Entire training loop becomes compilable
2. **GPU acceleration**: Formula generation on GPU
3. **Batching**: Generate many formulas in parallel
4. **No Python overhead**: No GIL, no Python objects

## Estimated effort:

- Formula tree generation: **Hard** (complex recursive logic)
- Evaluation: **Easy** (already done)
- Token conversion: **Easy** (simple mapping)
- Negation: **Trivial** (5 lines)

Start with JAXing the tree generation algorithm!
