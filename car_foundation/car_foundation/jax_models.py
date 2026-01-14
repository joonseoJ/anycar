import argparse
from functools import partial
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
# from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state

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
    