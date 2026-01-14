from typing import Sequence

import jax
import jax.numpy as jnp
# from datasets import load_dataset
from flax import linen as nn
from typing import Optional, Sequence, List, Tuple, Union, Any

# import transformer_engine.jax as te
# import transformer_engine.jax.flax as te_flax

import warnings
warnings.filterwarnings('ignore')

class JaxSinusoidalPositionalEncoding(nn.Module):
    """
    PyTorch 버전의 SinusoidalPositionalEncoding을 JAX로 구현
    """

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor with shape (Batch, Seq_Len, D1, D_Model)
        """
        seq_len = x.shape[1]
        d_model = x.shape[3]

        # 1. Create position indices [0, 1, ..., max_len-1]
        position = jnp.arange(0, seq_len, dtype=jnp.float32)[:, jnp.newaxis]
        
        # 2. Calculate the division term
        div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / d_model))
        
        # 3. Create PE matrix
        pe = jnp.zeros((seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        # 4. Add batch dimension: [1, max_len, d_model]
        pe = pe[jnp.newaxis, ...]

        # 5. Reshape for broadcasting
        # PE should become: (1, Seq_Len, 1, D_Model)
        pe = jnp.expand_dims(pe, axis=2)

        return x + pe


class JaxAxialAttentionBlock(nn.Module):
    """
    Time-Axis Attention -> Entity-Axis Attention -> MLP
    """
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Input x shape: (Batch, L, Entities, D)
        B, L, E, D = x.shape

        # --- 1. Time Axis Attention ---
        # Reshape to (B*E, L, D) so attention is over Time
        # PyTorch: x.permute(0, 2, 1, 3).reshape(B * E, L, D)
        x_time = x.transpose((0, 2, 1, 3)).reshape((B * E, L, D))
        
        # Multi-Head Attention
        # Flax inputs: (inputs_q, inputs_kv) -> here self-attention, so both are x_time
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            use_bias=True
        )(x_time, x_time, deterministic=deterministic)
        
        x_time = x_time + attn_out
        
        # Restore shape: (B, E, L, D) -> (B, L, E, D)
        x = x_time.reshape((B, E, L, D)).transpose((0, 2, 1, 3))
        x = nn.LayerNorm()(x)

        # --- 2. Entity Axis Attention ---
        # Reshape to (B*L, E, D) so attention is over Entities
        x_entity = x.reshape((B * L, E, D))
        
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            use_bias=True
        )(x_entity, x_entity, deterministic=deterministic)
        
        x_entity = x_entity + attn_out
        
        # Restore shape: (B, L, E, D)
        x = x_entity.reshape((B, L, E, D))
        x = nn.LayerNorm()(x)

        # --- 3. MLP ---
        # Pre-norm style used in MLP input in original code: x + mlp(norm(x))
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_model * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        
        if not deterministic:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
            
        return x + y


class JaxAxialTransformerEncoder(nn.Module):
    """
    Inputs: Dict with 'history' and 'static_features'
    Output: latent representation (B, E, D_MODEL)
    """
    history_dim: int
    static_dim: int
    model_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, static_features, deterministic: bool = True):
        # --- Embedding Stage ---
        # (B, L, E, H) -> (B, L, E, D_MODEL)
        emb_dyn = nn.Dense(self.model_dim)(history)
        
        # (B, E, S) -> (B, E, D_MODEL) -> (B, 1, E, D_MODEL)
        emb_stat = nn.Dense(self.model_dim)(static_features)
        emb_stat = jnp.expand_dims(emb_stat, axis=1)
        
        # Add Embeddings
        x = emb_dyn + emb_stat
        
        # Add Time Positional Encoding
        x = JaxSinusoidalPositionalEncoding()(x)

        # --- Transformer Layers ---
        for _ in range(self.num_layers):
            x = JaxAxialAttentionBlock(
                d_model=self.model_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        
        return x


class JaxAxialDecoderBlock(nn.Module):
    """
    Time Self-Attention (causal)
    → Time Cross-Attention (encoder memory)
    → Entity Attention
    → MLP
    """
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, memory, deterministic: bool = True):
        """
        x:      (B, T_pred, E, D)
        memory: (B, H, E, D)
        """
        B, T, E, D = x.shape
        _, H, _, _ = memory.shape

        # =====================================================
        # 1. Time Self-Attention (Decoder causal)
        # =====================================================
        x_time = x.transpose((0, 2, 1, 3)).reshape((B * E, T, D))

        # causal mask: (T, T)
        causal_mask = jnp.tril(jnp.ones((T, T)))

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(
            x_time,
            x_time,
            mask=causal_mask,
            deterministic=deterministic,
        )

        x_time = x_time + attn_out
        x = x_time.reshape((B, E, T, D)).transpose((0, 2, 1, 3))
        x = nn.LayerNorm()(x)

        # =====================================================
        # 2. Time Cross-Attention (Decoder → Encoder)
        # =====================================================
        q = x.transpose((0, 2, 1, 3)).reshape((B * E, T, D))
        kv = memory.transpose((0, 2, 1, 3)).reshape((B * E, H, D))

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(
            q,
            kv,
            deterministic=deterministic,
        )

        q = q + attn_out
        x = q.reshape((B, E, T, D)).transpose((0, 2, 1, 3))
        x = nn.LayerNorm()(x)

        # =====================================================
        # 3. Entity Axis Attention
        # =====================================================
        x_entity = x.reshape((B * T, E, D))

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x_entity, x_entity, deterministic=deterministic)

        x_entity = x_entity + attn_out
        x = x_entity.reshape((B, T, E, D))
        x = nn.LayerNorm()(x)

        # =====================================================
        # 4. MLP
        # =====================================================
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_model * 4)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)

        if not deterministic:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)

        return x + y


class JaxDynamicsPredictor(nn.Module):
    """
    Encoder + Prediction Head
    PyTorch의 observation_space에서 가져오던 치수(dimension) 정보들은
    객체 생성 시 인자로 직접 전달해야 합니다.
    """
    model_dim: int
    state_dim: int
    
    # Hyperparameters
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, history, static_features, future_actions, deterministic: bool = True):
        """
        history:         (B, T_history, E, X+A)
        static_features: (B, E, S)
        """
        B, H, E, history_dim = history.shape
        static_dim = static_features.shape[-1]
        
        # Instantiate Encoder
        encoder = JaxAxialTransformerEncoder(
            history_dim=history_dim,
            static_dim=static_dim,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )
        
        # memory: (B, T_history, E, D_MODEL)
        memory = encoder(history, static_features, deterministic=deterministic)
        
        # Action embedding for decoder input
        # (B, T_pred, E, A) -> (B, T_pred, E, D_MODEL)
        x = nn.Dense(self.model_dim, name="action_embedding")(future_actions)

        # Decoder Blocks
        for i in range(self.num_layers):
            x = JaxAxialDecoderBlock(
                d_model=self.model_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"decoder_block_{i}",
            )(x, memory, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        # Prediction Head (B, T_pred, E, X)
        pred = nn.Dense(self.state_dim, name="pred_head")(x)
        
        return pred


class JaxKroneckerFactorizedAttention(nn.Module):
    """
    Kronecker Factorized Attention Block as described in the Higher Order Transformer paper.
    It decomposes the high-order attention into sums or products of mode-wise attentions.
    """
    num_heads: int
    d_model: int
    dropout_rate: float = 0.0
    use_bias: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        # Validation: d_model must be divisible by num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}).")
        self.head_dim = self.d_model // self.num_heads

    @nn.compact
    def __call__(self, 
                 inputs_q, 
                 inputs_kv= None, 
                 masks = None, 
                 deterministic: bool = True):
        """
        Applies Kronecker Factorized Attention.

        Args:
            inputs_q: Query input tensor. 
                      Shape: (Batch, N_1, ..., N_k, D)
            inputs_kv: Key/Value input tensor. If None, performs self-attention.
                       Shape: (Batch, M_1, ..., M_k, D) 
                       Note: In self-attention, M_i = N_i. In cross-attn, they can differ.
                       However, for Kronecker structure to hold simply, usually k is same.
            masks: List of masks for each spatial/temporal dimension.
                   Each mask should correspond to a dimension N_i.
                   Shape of masks[i]: (Batch, 1, 1, N_i) or (Batch, 1, N_i, N_i) broadcastable.
            deterministic: If true, dropout is disabled.

        Returns:
            output: Transformed tensor. Shape: (Batch, N_1, ..., N_k, D)
        """
        # 0. Handle Inputs and Shapes
        # -------------------------------------------------------------------------
        if inputs_kv is None:
            inputs_kv = inputs_q
        
        # Assertion: Check last dimension matches d_model
        assert inputs_q.shape[-1] == self.d_model, f"Input Q last dim {inputs_q.shape[-1]} must match d_model {self.d_model}"
        assert inputs_kv.shape[-1] == self.d_model, f"Input KV last dim {inputs_kv.shape[-1]} must match d_model {self.d_model}"
        
        # Shape: (Batch, N_1, ..., N_k, D)
        q_shape = inputs_q.shape
        kv_shape = inputs_kv.shape
        batch_size = q_shape[0]
        
        # Number of spatial/temporal dimensions (k)
        # Exclude Batch (0) and Channel (-1)
        k_dim = len(q_shape) - 2 
        
        # Validate D matches
        input_dim = q_shape[-1]
        
        # 1. Project Global Value Tensor
        # -------------------------------------------------------------------------        
        # Reshape to separate heads: (Batch, M_1, ..., M_k, num_heads, head_dim)
        new_v_shape = kv_shape[:-1] + (self.num_heads, self.head_dim)
        v_heads = inputs_kv.reshape(new_v_shape)
        
        # 2. Compute Mode-wise Attention Factors (S^(i))
        # -------------------------------------------------------------------------
        # For each mode i in [1, ..., k], we compute an attention matrix S^(i).
        # We need to pool the inputs to isolate mode i, then project to Q and K.
        
        attention_factors = []
        
        # Loop over each spatial/temporal dimension
        for i in range(k_dim):
            # Actual axis index in the tensor (skipping Batch)
            # q_shape is (B, N1, N2, ..., D). So mode 1 is axis 1.
            axis_idx = i + 1
            
            # --- 2.a Pooling ---
            # We pool over all spatial dimensions EXCEPT the current axis `axis_idx`.
            # We also keep Batch (0) and Channel (-1).
            pool_axes_q = [ax for ax in range(1, k_dim + 1) if ax != axis_idx]
            pool_axes_kv = [ax for ax in range(1, k_dim + 1) if ax != axis_idx] # Assumes aligned axes for now
            
            # Pooling function g_pool (Summation as per paper)
            # Input: (Batch, N_1, ..., N_k, D)
            # Output: (Batch, N_i, D)
            q_pooled = jnp.sum(inputs_q, axis=tuple(pool_axes_q))
            k_pooled = jnp.sum(inputs_kv, axis=tuple(pool_axes_kv))
            
            # --- 2.b Mode-specific Projections ---
            # Separate projections for each mode and each head.
            # We project D -> num_heads * head_dim
            # Paper Eq 5 & 6: \tilde{Q}_i^h = g_pool(X) W_query^{i,h}
            w_q_mode = nn.Dense(self.d_model, use_bias=self.use_bias, name=f'query_proj_mode_{i}')
            w_k_mode = nn.Dense(self.d_model, use_bias=self.use_bias, name=f'key_proj_mode_{i}')
            
            q_mode = w_q_mode(q_pooled) # (Batch, N_i, H*D_h)
            k_mode = w_k_mode(k_pooled) # (Batch, M_i, H*D_h)
            
            # Reshape for heads: (Batch, N_i, H, D_h)
            q_mode = q_mode.reshape((batch_size, -1, self.num_heads, self.head_dim))
            k_mode = k_mode.reshape((batch_size, -1, self.num_heads, self.head_dim))
            
            # --- 2.c Compute Attention Scores ---
            # Score = Q * K^T / sqrt(D_h)
            # Shape: (Batch, H, N_i, M_i)
            # Einsum: b (batch), n (q_len), m (k_len), h (heads), d (dim)
            scores = jnp.einsum('bnhd,bmhd->bhnm', q_mode, k_mode)
            scores = scores * (self.head_dim ** -0.5)
            
            # --- 2.d Masking ---
            if masks is not None and i < len(masks) and masks[i] is not None:
                mask = masks[i]
                # Assuming mask shape broadcastable to (Batch, H, N_i, M_i)
                # Standard causal or padding mask: 1 is keep, 0 is mask
                min_value = jnp.finfo(scores.dtype).min
                scores = jnp.where(mask > 0, scores, min_value)
            
            # Softmax
            attn_weights = nn.softmax(scores, axis=-1)
            
            # Dropout
            if not deterministic:
                attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)

            # ** SOWING ATTENTION MAPS **
            # Store in 'intermediates' collection for debugging/analysis
            self.sow('intermediates', f'attention_map_mode_{i}', attn_weights)
            
            attention_factors.append(attn_weights) # List of (Batch, H, N_i, M_i)

        # 3. Apply Attention to Value Tensor (Kronecker Structure)
        # -------------------------------------------------------------------------
        # V_heads shape: (Batch, M_1, ..., M_k, H, D_h)
        # We need to apply the attention matrices. 
        # Since we use Kronecker Product (or Sum), we apply them sequentially along axes.
        
        curr_v = v_heads # (Batch, M_1, ..., M_k, H, D_h)
        
        # ndim = 1 (Batch) + k (Spatial) + 1 (Head) + 1 (Dim)
        v_ndim = curr_v.ndim

        batch_idx = 0
        head_idx = v_ndim - 2
        dim_idx = v_ndim - 1
        spatial_indices = list(range(1, k_dim + 1))

        next_new_id = v_ndim
        
        # Formula: (A (x) B) v = vec(B V A^T) - generalized to n-modes
        # For product attention: V x_1 S^{(1)} x_2 S^{(2)} ...
        # We iterate through each mode and contract the corresponding attention matrix.
        for i, attn in enumerate(attention_factors):
            # attn shape: (Batch, H, N_i, M_i)
            # curr_v shape: (Batch, M_1, ..., M_i, ..., H, D_h)
            # We want to contract dimension M_i (axis i+1) with the last dim of attn.
            # Output dimension will be N_i at axis i+1.
            
            # target_old_id -> target_new_id by attention
            target_old_id = spatial_indices[i]
            target_new_id = next_new_id
            next_new_id += 1

            # curr_v index list [0, 1, 2, ..., H, D]
            input_indices = [batch_idx] + spatial_indices + [head_idx, dim_idx]

            # Attention(attn) index list: [Batch, Head, New, Old]
            attn_indices = [batch_idx, head_idx, target_new_id, target_old_id]

            # Output index list
            # Change 'old' part of input index to 'new'
            output_indices = list(input_indices)
            output_indices[1 + i] = target_new_id
            
            curr_v = jnp.einsum(curr_v, input_indices, attn, attn_indices, output_indices)
            spatial_indices[i] = target_new_id
            
        # 4. Final Reshape (No Output Projection)
        # -------------------------------------------------------------------------
        # curr_v shape: (Batch, N_1, ..., N_k, H, D_h)
        # We simply flatten H and D_h back to d_model.
        
        final_shape = curr_v.shape[:-2] + (self.d_model,)
        output = curr_v.reshape(final_shape)
        
        # Assertion: Check output shape
        assert output.shape[-1] == self.d_model
        
        return output


class JaxHOTLayer(nn.Module):
    """
    Single Transformer Layer using HOT attention.
    Norm -> Attn -> Norm -> MLP
    """
    num_heads: int
    d_model: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs_q, inputs_kv= None, masks=None, deterministic=True):
        """
        Args:
            inputs_q: Query input tensor. 
                      Shape: (Batch, N_1, ..., N_k, D)
            inputs_kv: Key/Value input tensor. If None, performs self-attention.
                       Shape: (Batch, M_1, ..., M_k, D) 
                       Note: In self-attention, M_i = N_i. In cross-attn, they can differ.
                       However, for Kronecker structure to hold simply, usually k is same.
            masks: List of masks for each spatial/temporal dimension.
                   Each mask should correspond to a dimension N_i.
                   Shape of masks[i]: (Batch, 1, 1, N_i) or (Batch, 1, N_i, N_i) broadcastable.
            deterministic: If true, dropout is disabled.
        """
        if inputs_kv is None:
            inputs_kv = inputs_q
        
        # 1. Attention Block
        inputs_q_norm = nn.LayerNorm()(inputs_q)
        inputs_kv_norm = nn.LayerNorm()(inputs_kv)
        y = JaxKroneckerFactorizedAttention(
            num_heads=self.num_heads,
            d_model=self.d_model,
            dropout_rate=self.dropout_rate
        )(inputs_q_norm, inputs_kv_norm, masks=masks, deterministic=deterministic)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        # Feed Forward the output of LN & Attention
        x = inputs_q + y
        
        # 2. MLP Block
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.d_model)(y) # Project back to d_model
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        
        # Residual Connection
        output = x + y
        
        return output


class JaxHOTDynamicsPredictor(nn.Module):
    """
    Full Higher Order Transformer Model.
    Stack of HOTLayers.
    """
    d_model: int
    mlp_dim: int
    state_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, history, static_features, future_actions, masks=None, deterministic=True):
        """
        Args:
            history:         (B, T_history, E, X+A)
            static_features: (B, E, S)
            future_actions:  (B, T_future, E, A)
            masks: List of masks per dimension.
            return_attention_map: If true, returns list of attention maps from all layers.
        """

        # Project history to model dimension (B, T_history, E, D)
        history_emb = nn.Dense(self.d_model, name="history_embedding")(history)

        # Project static feature to model dimension (B, E, D)
        static_features_emb = nn.Dense(self.d_model, name="static_features_embedding")(static_features)

        # Combine history and static_features (B, T_future, E, D)
        encoded_latent = history_emb + static_features_emb[:, None, :, :]

        for i in range(self.num_layers):
            encoded_latent = JaxHOTLayer(
                num_heads=self.num_heads,
                d_model=self.d_model,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'Encoding_HOT_Layer_{i}'
            )(encoded_latent, masks=masks, deterministic=deterministic)

        # Project future actions to model dimension (B, T_future, E, D)
        future_action_emb = nn.Dense(self.d_model, name="future_action_embedding")(future_actions)

        decoded_latent = future_action_emb
        for i in range(self.num_layers):
            decoded_latent = JaxHOTLayer(
                num_heads=self.num_heads,
                d_model=self.d_model,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'Decoding_HOT_Layer_{i}'
            )(decoded_latent, encoded_latent, masks=masks, deterministic=deterministic)
        
        output = nn.Dense(self.state_dim, name="state_projection")(decoded_latent)

        return output


class JaxLearnedPositionalEncoding(nn.Module):
    d_model: int
    max_len: int
    dropout_rate: float
    flip: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=False):
        pos_enc = self.param('pos_enc', lambda key, shape: jax.random.uniform(key, shape, self.dtype),
                             (self.max_len, self.d_model))
        if self.flip:
            x = x + jnp.flip(pos_enc, axis=0)
        else:
            x = x + pos_enc

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x


class TmpMLP(nn.Module):
    model_dim: int
    state_dim: int
    
    # Hyperparameters
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, history, static_features, future_actions, deterministic: bool = True):
        """
        history:         (B, T_history, E, X+A)
        static_features: (B, E, S)
        """
        B, H, E, history_dim = history.shape
        static_dim = static_features.shape[-1]
        pred_dim = future_actions.shape[1]

        x = nn.Dense(self.model_dim)(history)
        x = nn.Dense(self.model_dim*4)(x)
        x = nn.Dense(self.model_dim)(x)
        pred = nn.Dense(self.state_dim)(x)
        pred = pred[:, :pred_dim, :, :]
        
        return pred

# class JaxTransformerEncoder(nn.Module):
#     state_dim: int
#     action_dim: int
#     output_dim: int
#     latent_dim: int
#     num_heads: int
#     num_layers: int
#     dropout_rate: float
#     history_length: int
#     action_length: int
#     dtype: jnp.dtype = jnp.float32
#     # attn_mask_type: str = 'causal'

#     @nn.compact
#     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         action_encoder = te_flax.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
#         history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
#         history_emb = history_emb.at[:, ::2].set(te_flax.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_state')(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
#         history_emb = history_emb.at[:, 1::2].set(action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        
#         history_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.history_length * 2 - 1, self.dropout_rate, flip=True, dtype=self.dtype)(history_emb, deterministic=deterministic)

#         action_emb = action_encoder(action)
#         action_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.action_length, self.dropout_rate, dtype=self.dtype)(action_emb, deterministic=deterministic)

#         decoder_output = action_emb
#         for i in range(self.num_layers):
#             te_Decoder = partial(
#                 te_flax.TransformerLayer,
#                 hidden_size=self.latent_dim,
#                 mlp_hidden_size=self.latent_dim * 4,
#                 num_attention_heads=self.num_heads,
#                 hidden_dropout=self.dropout_rate,
#                 attention_dropout=self.dropout_rate,
#                 intermediate_dropout=self.dropout_rate,
#                 dropout_rng_name='dropout',
#                 mlp_activations=('gelu',),
#                 layer_type=te_flax.TransformerLayerType.ENCODER,
#                 self_attn_mask_type="causal",
#                 enable_relative_embedding=False,
#                 dtype=self.dtype,
#                 name=f'transformer_layer_{i}',
#             )
#             decoder_output = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)

#         output = te_flax.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
#         return output
    
     
# class JaxTransformerDecoder(nn.Module):
#     state_dim: int
#     action_dim: int
#     output_dim: int
#     latent_dim: int
#     num_heads: int
#     num_layers: int
#     dropout_rate: float
#     history_length: int
#     action_length: int
#     dtype: jnp.dtype = jnp.float32
#     # attn_mask_type: str = 'causal'

#     @nn.compact
#     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         action_encoder = te_flax.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
#         history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
#         history_emb = history_emb.at[:, ::2].set(
#             te_flax.DenseGeneral(
#                 self.latent_dim, 
#                 dtype=self.dtype, 
#                 name='linear_input_state'
#             )(history[:, :, :self.state_dim])
#         ) # shape: [batch_size, seq_length, latent_dim]
#         history_emb = history_emb.at[:, 1::2].set(
#             action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])
#         ) # shape: [batch_size, seq_length-1, latent_dim]
        
#         history_emb = JaxLearnedPositionalEncoding(
#             self.latent_dim, 
#             self.history_length * 2 - 1, 
#             self.dropout_rate, 
#             flip=True, 
#             dtype=self.dtype
#         )(history_emb, deterministic=deterministic)

#         action_emb = action_encoder(action)
#         action_emb = JaxLearnedPositionalEncoding(
#             self.latent_dim, 
#             self.action_length, 
#             self.dropout_rate, 
#             dtype=self.dtype
#         )(action_emb, deterministic=deterministic)

#         decoder_output = action_emb
#         for i in range(self.num_layers):
#             te_Decoder = partial(
#                 te_flax.TransformerLayer,
#                 hidden_size=self.latent_dim,
#                 mlp_hidden_size=self.latent_dim * 4,
#                 num_attention_heads=self.num_heads,
#                 hidden_dropout=self.dropout_rate,
#                 attention_dropout=self.dropout_rate,
#                 intermediate_dropout=self.dropout_rate,
#                 dropout_rng_name='dropout',
#                 mlp_activations=('gelu',),
#                 layer_type=te_flax.TransformerLayerType.DECODER,
#                 self_attn_mask_type="causal",
#                 enable_relative_embedding=False,
#                 dtype=self.dtype,
#                 name=f'transformer_layer_{i}',
#             )
#             decoder_output = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)

#         output = te_flax.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
#         return output
    
# class JaxTransformerDecoderVis(nn.Module):
#     state_dim: int
#     action_dim: int
#     output_dim: int
#     latent_dim: int
#     num_heads: int
#     num_layers: int
#     dropout_rate: float
#     history_length: int
#     action_length: int
#     dtype: jnp.dtype = jnp.float32
#     # attn_mask_type: str = 'causal'

#     @nn.compact
#     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         action_encoder = te_flax.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
#         history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
#         history_emb = history_emb.at[:, ::2].set(te_flax.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_state')(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
#         history_emb = history_emb.at[:, 1::2].set(action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        
#         history_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.history_length * 2 - 1, self.dropout_rate, flip=True, dtype=self.dtype)(history_emb, deterministic=deterministic)

#         action_emb = action_encoder(action)
#         action_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.action_length, self.dropout_rate, dtype=self.dtype)(action_emb, deterministic=deterministic)

#         decoder_output = action_emb
#         all_attention_weights = []
#         for i in range(self.num_layers):
#             te_Decoder = partial(
#                 te_flax.TransformerLayer,
#                 hidden_size=self.latent_dim,
#                 mlp_hidden_size=self.latent_dim * 4,
#                 num_attention_heads=self.num_heads,
#                 hidden_dropout=self.dropout_rate,
#                 attention_dropout=self.dropout_rate,
#                 intermediate_dropout=self.dropout_rate,
#                 dropout_rng_name='dropout',
#                 mlp_activations=('gelu',),
#                 layer_type=te_flax.TransformerLayerType.DECODER,
#                 self_attn_mask_type="causal",
#                 enable_relative_embedding=False,
#                 dtype=self.dtype,
#                 name=f'transformer_layer_{i}',
#             )
#             decoder_output, attn_weights = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)
#             all_attention_weights.append(attn_weights)
#         mean_attention_weights = jnp.mean(jnp.stack(all_attention_weights), axis=0)


#         output = te_flax.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
#         return output, mean_attention_weights
    
class JaxMLP(nn.Module):
    hidden_sizes: Sequence[int]
    output_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        x = jnp.concatenate([history.reshape(bs, -1), action.reshape(bs, -1)], axis=1)
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(self.output_dim * action.shape[1])(x)
        return x.reshape((x.shape[0], -1, self.output_dim))
    
class JaxCNN(nn.Module):
    conv_layers: Sequence[int]  # List of filter sizes for convolutional layers
    kernel_sizes: Sequence[int]  # Corresponding kernel sizes
    pool_sizes: Sequence[int]  # Corresponding pool sizes
    output_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        x = jnp.concatenate([history.reshape(bs, -1), action.reshape(bs, -1)], axis=1)  # Concatenate history and action
        x = x[:, :, None]
        # Apply the convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(zip(self.conv_layers[:-1], self.kernel_sizes[:-1], self.pool_sizes[:-1])):
            x = nn.Conv(features=filters, kernel_size=(kernel_size,), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(pool_size, ), strides=(pool_size, ), padding='SAME')
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        x = x.reshape((bs, -1))
        x = nn.Dense(self.output_dim * action.shape[1])(x)

        return x.reshape((x.shape[0], -1, self.output_dim))


# class JaxGRU(nn.Module):
#     hidden_size: int
#     output_dim: int
#     num_layers: int
#     dropout_rate: float

#     @nn.compact
#     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         bs = history.shape[0]
        
#         actions = action

#         x = history.reshape(bs, -1)

#         # Initialize GRU layers
#         gru_cell = nn.GRUCell(features = self.hidden_size, name=f'gru_layer')
        
#         # Initialize hidden state to zeros
#         hidden_state = nn.Dense(self.hidden_size)(x)
#         hidden_state = nn.relu(hidden_state)

#         # Apply GRU layers iteratively
#         outputs = []
#         for t in range(actions.shape[1]):
#             hidden_state, output = gru_cell(hidden_state, actions[:, t, :])
#             outputs.append(output)
        
#         x = jnp.stack(outputs, axis=1)  # Stack the outputs along the time dimension

#         x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

#         x = nn.Dense(self.output_dim)(x)
        
#         return x

# class JaxLSTM(nn.Module):
#     hidden_size: int
#     output_dim: int
#     dropout_rate: float

#     @nn.compact
#     def __call__(self,  history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         bs = history.shape[0]
#         # Initialize the LSTM cell
#         lstm_cell = nn.LSTMCell(features=self.hidden_size)
        
#         # Initialize carry (hidden and cell states)
#         carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (action.shape[0],))

#         carry = (nn.relu(nn.Dense(self.hidden_size)(history.reshape(bs, -1))), nn.relu(nn.Dense(self.hidden_size)(history.reshape(bs, -1))))
        
#         outputs = []
        
#         # Iterate through each time step
#         for t in range(action.shape[1]):
#             # Update carry and compute the output for each time step
#             carry, output = lstm_cell(carry, action[:, t, :])
#             outputs.append(output)
        
#         # Stack the outputs along the time axis
#         x = jnp.stack(outputs, axis=1)

#         x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        
#         # Apply a final dense layer to map to output dimension
#         x = nn.Dense(self.output_dim)(x)
        
#         return x
    

# # class JaxLSTM(nn.Module):
# #     hidden_size: int
# #     output_dim: int
# #     num_layers: int
# #     dropout_rate: float

# #     @nn.compact
# #     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
# #         bs = history.shape[0]
        
# #         actions = action

# #         x = history.reshape(bs, -1)

# #         # Initialize LSTM layers
# #         lstm_cells = [nn.LSTMCell(features=self.hidden_size, name=f'lstm_layer_{i}') for i in range(self.num_layers)]
        
# #         # Initialize hidden state (h_t) and cell state (c_t) to history
# #         # hidden_state = nn.Dense(self.hidden_size)(x)
# #         # hidden_state = nn.relu(hidden_state)

# #         # cell_state = nn.Dense(self.hidden_size)(x)
# #         # cell_state = nn.relu(cell_state)
        
# #         # Apply LSTM layers iteratively
# #         for lstm_cell in lstm_cells:
# #             outputs = []
# #             carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (action.shape[0],))
# #             for t in range(actions.shape[1]):
# #                 # carry = (hidden_state, cell_state)
# #                 carry = lstm_cell(carry, actions[:, t, :])
# #                 hidden_state, cell_state = carry
# #                 outputs.append(hidden_state)  # Only append the hidden state (h_t)

# #             x = jnp.stack(outputs, axis=1)  # Stack the outputs along the time dimension

# #             # Apply dropout after each LSTM layer (optional)
# #             x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

# #         x = x.reshape((bs, -1))

# #         x = nn.Dense(self.output_dim * action.shape[1])(x)
        
# #         return x.reshape((x.shape[0], -1, self.output_dim))

# # class JaxGRU(nn.Module):
#     hidden_size: int
#     output_dim: int
#     num_layers: int
#     dropout_rate: float

#     @nn.compact
#     def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
#         bs = history.shape[0]
        
#         actions = action

#         x = history.reshape(bs, -1)

#         # Initialize GRU layers
#         gru_cells = [nn.GRUCell(features = self.hidden_size, name=f'gru_layer_{i}') for i in range(self.num_layers)]
        
#         # Initialize hidden state to zeros
#         hidden_state = nn.Dense(self.hidden_size)(x)
#         hidden_state = nn.relu(hidden_state)

#         # Apply GRU layers iteratively
#         for gru_cell in gru_cells:
#             outputs = []
#             for t in range(actions.shape[1]):
#                 hidden_state, output = gru_cell(hidden_state, actions[:, t, :])
#                 outputs.append(output)
#             x = jnp.stack(outputs, axis=1)  # Stack the outputs along the time dimension

#             # Apply dropout after each GRU layer (optional)
#             x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

#         x = x.reshape((bs, -1))

#         x = nn.Dense(self.output_dim * action.shape[1])(x)
        
#         return x.reshape((x.shape[0], -1, self.output_dim))
    