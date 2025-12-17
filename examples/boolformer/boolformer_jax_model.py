"""
Boolformer Encoder-Decoder Transformer in JAX Flax nnx.

Architecture closely follows nanochat encoder branch:
- No bias in linear layers (use_bias=False)
- RMSNorm without learnable parameters (use_scale=False)
- ReLU² activation in MLP
- Rotary positional embeddings (RoPE)
- Standard multi-head attention (no MQA - n_kv_head == n_head by default)
- Pre-norm architecture (norm before attention/MLP)
- Cross-attention in decoder blocks

Compatible with mctx for MCTS-based training.

Based on:
- nanochat encoder branch architecture (PyTorch)
- Flax Gemma example for RoPE implementation
- JAX machine translation tutorial for encoder-decoder structure
- Flax nnx built-in components (MultiHeadAttention, RMSNorm)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention, apply_rotary_emb
from typing import Optional, Tuple
import math


def norm(x):
    # Purely functional rmsnorm with no learnable params (matches nanochat gpt.py:36-38)
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    return x / rms


# TODO: bump base theta more, e.g. 100K is more common more recently
# FIXME: consider using bfloat16 for dtype (currently float32)
def _precompute_rotary_embeddings(seq_len, head_dim, base=10000, device='cpu', dtype=jnp.float32):
    # stride the channels
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = jnp.arange(seq_len, dtype=jnp.float32)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = jnp.outer(t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    cos, sin = cos.astype(dtype), sin.astype(dtype)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims for later broadcasting
    # move to specified device (JAX equivalent of .to(device))
    cos, sin = jax.device_put(cos, jax.devices(device)[0]), jax.device_put(sin, jax.devices(device)[0])
    return cos, sin


class MLP(nnx.Module):
    """
    Feed-forward network with ReLU² activation.
    Matches nanochat MLP architecture (lines 173-183).
    """

    def __init__(self, n_embd: int, rngs: nnx.Rngs):
        """
        Args:
            n_embd: Embedding dimension
            rngs: RNG state for initialization
        """
        # 4x expansion WITHOUT bias (matches nanochat line 176-177: bias=False)
        self.c_fc = nnx.Linear(n_embd, 4 * n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with ReLU² activation.
        Matches nanochat line 181: F.relu(x).square()

        Args:
            x: Input tensor (batch, seq_len, n_embd)
        Returns:
            Output tensor (batch, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = jax.nn.relu(x) ** 2  # ReLU² activation
        x = self.c_proj(x)
        return x


class TransformerBlock(nnx.Module):
    """
    Transformer block supporting both encoder and decoder layers.
    Matches nanochat Block architecture with pre-norm design (lines 186-224).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        cross_attention: bool,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
            cross_attention: If True, includes cross-attention (decoder)
            rngs: RNG state for initialization
        """
        self.cross_attention = cross_attention

        # Self-attention with RoPE and QK RMSNorm
        # use_bias=False, normalize_qk=True matches nanochat
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_head,
            in_features=n_embd,
            use_bias=False,
            normalize_qk=True,
            attention_fn=dot_product_attention,
            rngs=rngs,
        )

        # Cross-attention (decoder only) - no RoPE for cross-attention
        if cross_attention:
            self.cross_attn = nnx.MultiHeadAttention(
                num_heads=n_head,
                in_features=n_embd,
                use_bias=False,
                normalize_qk=True,
                attention_fn=dot_product_attention,
                rngs=rngs,
            )

        # Feed-forward
        self.mlp = MLP(n_embd, rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        cos_sin: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        context: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        decode: bool = False,
    ) -> jnp.ndarray:
        """
        Forward pass with pre-norm architecture.
        Matches nanochat lines 218, 221, 223: norm before attention/MLP, not after.

        Args:
            x: Input tensor (batch, seq_len, n_embd)
            cos_sin: (cos, sin) tuple for RoPE
            context: Optional encoder output for cross-attention (decoder only)
            mask: Optional attention mask
        Returns:
            Output tensor with same shape as x
        """
        # Self-attention with pre-norm (matches nanochat line 218)
        x = x + self.attn(norm(x), mask=mask, cos_sin=cos_sin, decode=decode)

        # Cross-attention if this is a decoder layer (matches nanochat lines 220-221)
        if self.cross_attention and context is not None:
            x = x + self.cross_attn(norm(x), context, context, mask=None, cos_sin=None, decode=decode)

        # MLP with pre-norm (matches nanochat line 223)
        x = x + self.mlp(norm(x))

        return x


class BoolformerTransformer(nnx.Module):
    """
    Complete encoder-decoder transformer for Boolformer.

    Architecture:
    - Encoder: processes truth table (1024 entries x 10 variables)
    - Decoder: generates formula tokens autoregressively
    - Policy head: next token probabilities (vocab_size=20)
    - Value head: success prediction scalar in [-1, 1]

    Compatible with mctx for MCTS-based training.
    Returns (policy_logits, value) matching mctx.RootFnOutput requirements.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        # Config
        num_variables: int,
        vocab_size: int,  # <SOS> + <PAD> + ~ + & + | + x1-xn
        max_formula_length: int,

        # Model config
        n_embd: int,
        n_head: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
    ):
        """
        Args:
            num_variables: Number of boolean variables (10)
            vocab_size: Size of formula token vocabulary (15 tokens: <SOS> + <PAD> + ~ + & + | + x1-x10)
            max_formula_length: Maximum formula sequence length (50)
            n_embd: Embedding dimension (512)
            n_head: Number of attention heads (8)
            n_encoder_layers: Number of encoder blocks (6)
            n_decoder_layers: Number of decoder blocks (6)
            rngs: RNG state for initialization
        """
        self.num_variables = num_variables
        self.vocab_size = vocab_size
        self.max_formula_length = max_formula_length
        self.n_embd = n_embd

        # Truth table projection: each row (10 binary values) -> n_embd
        # WITHOUT bias (matches nanochat)
        self.truth_table_proj = nnx.Linear(
            num_variables, n_embd, use_bias=False, rngs=rngs
        )

        # Formula token embedding
        self.formula_embedding = nnx.Embed(vocab_size, n_embd, rngs=rngs)

        # Encoder blocks (self-attention only, no cross-attention)
        self.encoder_blocks = nnx.List([
            TransformerBlock(n_embd, n_head, cross_attention=False, rngs=rngs)
            for _ in range(n_encoder_layers)
        ])

        # Decoder blocks (self-attention + cross-attention)
        self.decoder_blocks = nnx.List([
            TransformerBlock(n_embd, n_head, cross_attention=True, rngs=rngs)
            for _ in range(n_decoder_layers)
        ])

        # Output heads
        # Policy head: predicts next token WITHOUT bias
        # Matches nanochat line 235: nn.Linear(..., bias=False)
        self.policy_head = nnx.Linear(n_embd, vocab_size, use_bias=False, rngs=rngs)

        # Value head: predicts success probability
        # Two-layer MLP: n_embd -> n_embd -> 1
        self.value_fc1 = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.value_fc2 = nnx.Linear(n_embd, 1, use_bias=False, rngs=rngs)

        # Precompute RoPE cos/sin for decoder sequences (matches nanochat gpt.py:336-341)
        # Only decoder needs positional encoding (formula tokens), encoder processes unordered point set
        head_dim = n_embd // n_head
        self.cos, self.sin = _precompute_rotary_embeddings(max_formula_length, head_dim)

    def encode_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """
        Encode minority class points through encoder.

        Args:
            points: Variable combinations where output is 1 (batch, num_points, 10)
                    where num_points <= 512
        Returns:
            Encoder output (batch, num_points, n_embd)
        """
        # Points are already in the correct format: (batch, num_points, 10)
        # Each row is [v0, v1, v2, ..., v9] for that variable combination

        # Project to embedding space
        x = self.truth_table_proj(points)  # (batch, num_points, n_embd)

        # Pass through encoder blocks (no RoPE for encoder - points have no positional order)
        # Encoder always uses decode=False (processes full sequence in parallel, not autoregressive)
        for block in self.encoder_blocks:
            x = block(x, cos_sin=None, context=None, mask=None, decode=False)

        return x

    def decode_formula(
        self,
        formula_tokens: jnp.ndarray,
        encoder_output: jnp.ndarray,
        decode: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Decode formula tokens with cross-attention to encoder output.

        Args:
            formula_tokens: Token indices (batch, seq_len)
            encoder_output: Encoder memory (batch, 1024, n_embd)
        Returns:
            (policy_logits, value):
                - policy_logits: (batch, seq_len, vocab_size) next token logits
                - value: (batch, seq_len) success estimate in [0, 1]
        """
        B, T = formula_tokens.shape

        # Embed formula tokens
        x = self.formula_embedding(formula_tokens)  # (batch, seq_len, n_embd)

        # Create causal mask for decoder self-attention
        # Lower triangular matrix: can attend to current and previous positions
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        causal_mask = causal_mask.reshape(1, 1, T, T)  # (1, 1, T, T) for broadcast

        # Get cos/sin for decoder sequence length
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Pass through decoder blocks with RoPE and cross-attention to encoder output
        for block in self.decoder_blocks:
            x = block(x, cos_sin=cos_sin, context=encoder_output, mask=causal_mask, decode=decode)

        # TODO: Optimize - computing policy/value heads for all positions but only using one per sample
        #       Could extract x[batch_indices, positions-1] first. Also consider relu^2 activation.
        policy_logits = self.policy_head(x)  # (batch, seq_len, vocab_size)

        value = self.value_fc1(x)  # (batch, seq_len, n_embd) - Linear applies to last dim
        value = jax.nn.gelu(value)
        value = self.value_fc2(value)  # (batch, seq_len, 1)
        value = jax.nn.sigmoid(value).squeeze(-1)  # (batch, seq_len) in range [0, 1]

        return policy_logits, value

    def __call__(
        self,
        points: jnp.ndarray,
        formula_tokens: jnp.ndarray,
        decode: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Full forward pass: encode minority points, decode formula.

        Args:
            points: Variable combinations where output is 1 (batch, num_points, 10)
                    where num_points <= 512
            formula_tokens: Formula token indices (batch, seq_len)
            decode: Whether to use KV-caching (True for MCTS generation, False for training)
        Returns:
            (policy_logits, value):
                - policy_logits: (batch, seq_len, vocab_size)
                - value: (batch,) in [-1, 1]

        Output format compatible with mctx:
        - policy_logits[:, -1, :] can be used as prior_logits for mctx.RootFnOutput
        - value can be used directly as value for mctx.RootFnOutput
        """
        encoder_output = self.encode_points(points)
        policy_logits, value = self.decode_formula(formula_tokens, encoder_output, decode=decode)
        return policy_logits, value

    def init_weights(self):
        """
        Initialize weights following nanochat's scheme (gpt.py:157-183).

        Nanochat uses untied weights (line 6: "untied weights for token embedding and lm_head"):
        1. Input embeddings: std=1.0
        2. Hidden Linear layers: custom variance scaling std = (1/√fan_in) × min(1.0, √(fan_out/fan_in))
        3. Output layers: zero init (lm_head, MLP c_proj, attention c_proj)

        Our mapping:
        - truth_table_proj (Linear, input projection): std=1.0 (like embedding layer)
        - formula_embedding (Embed, input): std=1.0 (like nanochat wte)
        - policy_head (Linear, output): zero init (like nanochat lm_head)
        - value_fc1 (Linear, hidden): custom variance scaling
        - value_fc2 (Linear, output): zero init (like MLP c_proj)
        """
        import jax

        # Generate array of random keys for all layer initializations
        master_key = jax.random.PRNGKey(0)
        keys = jax.random.split(master_key, num=1000)
        i = 0

        def next_key():
            nonlocal i
            key = keys[i]
            i += 1
            return key

        def apply_custom_init(lin):
            """Apply nanochat's custom variance scaling (gpt.py:174-179)."""
            shape = lin.kernel.value.shape
            # Handle both Linear (2D) and LinearGeneral (3D+)
            # fan_in = first dimension, fan_out = product of remaining dimensions
            fan_in = int(shape[0])
            fan_out = int(jnp.prod(jnp.array(shape[1:])))
            std = (1.0 / jnp.sqrt(fan_in)) * min(1.0, jnp.sqrt(fan_out / fan_in))
            lin.kernel.value = jax.random.normal(next_key(), shape) * std

        # Input layers: std=1.0 (like nanochat embeddings)
        self.truth_table_proj.kernel.value = jax.random.normal(next_key(), self.truth_table_proj.kernel.value.shape) * 1.0
        self.formula_embedding.embedding.value = jax.random.normal(next_key(), self.formula_embedding.embedding.value.shape) * 1.0

        # Transformer blocks (encoder + decoder)
        all_blocks = list(self.encoder_blocks) + list(self.decoder_blocks)
        for block in all_blocks:
            # MLP layers
            apply_custom_init(block.mlp.c_fc)     # Hidden layer
            block.mlp.c_proj.kernel.value = jnp.zeros_like(block.mlp.c_proj.kernel.value)  # Zero init (nanochat gpt.py:163)

            # Attention layers: nnx.MultiHeadAttention has query, key, value, out
            for attn in [block.attn] + ([block.cross_attn] if block.cross_attention else []):
                # Query, Key, Value projections: custom variance scaling
                apply_custom_init(attn.query)
                apply_custom_init(attn.key)
                apply_custom_init(attn.value)
                # Output projection: zero init (nanochat gpt.py:164)
                attn.out.kernel.value = jnp.zeros_like(attn.out.kernel.value)

        # Value head
        apply_custom_init(self.value_fc1)        # Hidden layer
        self.value_fc2.kernel.value = jnp.zeros_like(self.value_fc2.kernel.value)  # Zero init

        # Policy head: zero init (untied from input, like nanochat lm_head)
        self.policy_head.kernel.value = jnp.zeros_like(self.policy_head.kernel.value)


# Example usage for mctx integration:
def create_boolformer_for_mctx(
    batch_size: int = 1,
    seed: int = 0,
) -> BoolformerTransformer:
    """
    Create Boolformer model for mctx training.

    Example:
        >>> model = create_boolformer_for_mctx(batch_size=32)
        >>> truth_table = jnp.ones((32, 1024))  # Dummy input
        >>> formula_tokens = jnp.zeros((32, 10), dtype=jnp.int32)  # Start tokens
        >>> policy_logits, value = model(truth_table, formula_tokens)
        >>>
        >>> # For mctx.RootFnOutput:
        >>> prior_logits = policy_logits[:, -1, :]  # Last position logits
        >>> root = mctx.RootFnOutput(
        ...     prior_logits=prior_logits,
        ...     value=value,
        ...     embedding=truth_table,  # Or encoded state
        ... )
    """
    rngs = nnx.Rngs(seed)
    model = BoolformerTransformer(rngs=rngs)
    return model
