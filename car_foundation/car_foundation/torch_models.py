import torch
import torch.nn as nn
import numpy as np
import gym
from gym import spaces
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: The embedding dimension (last dimension of input).
            max_len: The maximum sequence length.
        """
        super().__init__()
        
        # 1. Create a placeholder for positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create position indices [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Calculate the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. Add a batch dimension at the beginning
        # Shape: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 6. Register as a buffer (part of state_dict but no gradients)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (Batch, Seq_Len, D1, D_Model)
        
        Returns:
            Tensor with added positional encoding, same shape as x.
        """
        # x.size(1) corresponds to the Sequence Length
        seq_len = x.size(1)
        
        # 1. Slice the pre-computed PE to the current sequence length
        # Current Shape: [1, Seq_Len, D_Model]
        current_pe = self.pe[:, :seq_len, :]
        
        # 2. Reshape for broadcasting
        # We need to align dimensions: (Batch, Seq_Len, D1, D_Model)
        # PE should become:            (1,     Seq_Len, 1,  D_Model)
        # This allows PE to be added to every element in the D1 dimension.
        current_pe = current_pe.unsqueeze(2)
        
        # 3. Add positional encoding to input (Broadcasting happens here)
        return x + current_pe


class AxialAttentionBlock(nn.Module):
    """
    Time-Axis Attention -> Entity-Axis Attention -> MLP
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.entity_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Input x shape: (Batch, L, Entities, D)
        B, L, E, D = x.shape

        # --- 1. Time Axis Attention ---
        # Reshape to (B*E, L, D) so attention is over Time
        x_time = x.permute(0, 2, 1, 3).reshape(B * E, L, D)
        attn_out, _ = self.time_attn(x_time, x_time, x_time)
        x_time = x_time + attn_out
        # Restore shape: (B, E, L, D) -> (B, L, E, D)
        x = x_time.view(B, E, L, D).permute(0, 2, 1, 3)
        x = self.norm1(x)

        # --- 2. Entity Axis Attention ---
        # Reshape to (B*L, E, D) so attention is over Entities
        x_entity = x.reshape(B * L, E, D)
        attn_out, _ = self.entity_attn(x_entity, x_entity, x_entity)
        x_entity = x_entity + attn_out
        # Restore shape: (B, L, E, D)
        x = x_entity.view(B, L, E, D)
        x = self.norm2(x)

        # --- 3. MLP ---
        x = x + self.mlp(self.norm3(x))
        return x


class AxialTransformerEncoder(nn.Module):
    """
    Inputs: Dict with 'history' and 'static_features'
        'history' (history Length, num_Entities, History_dim)
        'static_features' (num_Entities, Static_dim)
    Output: latent representation (B, E, D_MODEL)
    """
    def __init__(
            self, 
            observation_space: spaces.Dict, 
            model_dim: int,
            num_heads: int,
            num_layers: int,
            dropout_rate: float
        ):
        super().__init__()

        # Calculate feature dim: Output will be (Entities * D_MODEL)
        self.history_length = observation_space['history'].shape[0]
        self.num_entities = observation_space['history'].shape[1]
        self.history_dim = observation_space['history'].shape[2]
        self.static_dim = observation_space['static_features'].shape[1]
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # 1. Embeddings
        self.dynamic_proj = nn.Linear(self.history_dim, self.model_dim)
        self.static_proj = nn.Linear(self.static_dim, self.model_dim)
        
        # 2. Positional Encoding for Time (Learnable)
        self.time_pos_embed = SinusoidalPositionalEncoding(d_model=self.model_dim)

        # 3. Axial Transformer Layers
        self.layers = nn.ModuleList([
            AxialAttentionBlock(self.model_dim, self.num_heads, self.dropout_rate) 
            for _ in range(self.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.model_dim)

    def forward(self, observations):
        # Extract inputs
        # history: (Batch, L, E, H)
        # static_features: (Batch, E, S)
        x_dynamic = observations['history'] 
        x_static = observations['static_features']

        # --- Embedding Stage ---
        # (B, L, E, H) -> (B, L, E, D_MODEL)
        emb_dyn = self.dynamic_proj(x_dynamic)
        
        # (B, E, S) -> (B, E, D_MODEL) -> (B, 1, E, D_MODEL) (Broadcast over time)
        emb_stat = self.static_proj(x_static).unsqueeze(1)
        
        # Add Embeddings: Dynamic + Static (acting as Entity Positional Encoding)
        x = emb_dyn + emb_stat
        
        # Add Time Positional Encoding
        x = self.time_pos_embed(x)

        # --- Transformer Layers ---
        for layer in self.layers:
            x = layer(x) # (B, L, E, D_MODEL)

        x = self.final_norm(x)

        # --- Output Aggregation ---
        # We take the state at the LAST time step (prediction based on most recent context)
        # but the attention has already aggregated past info.
        # x shape: (B, L, E, D_MODEL) -> Take index -1 -> (B, E, D_MODEL)
        representation = x[:, -1, :, :] 
        
        # Flatten for SB3 Policy Head: (B, E*D_MODEL)
        return representation


class DynamicsPredictor(nn.Module):
    def __init__(self, observation_space: spaces.Dict, model_dim: int):
        super().__init__()
        
        #0. Dimensions
        self.num_entities = observation_space['history'].shape[1]
        self.state_dim = observation_space['current_state'].shape[1]
        self.model_dim = model_dim

        # 1. Encoder (Pre-trained Architecture)
        self.encoder = AxialTransformerEncoder(
            observation_space, 
            self.model_dim, 
            num_heads=4, 
            num_layers=4,
            dropout_rate=0.1,
        )
        self.d_model = self.model_dim
        self.num_entities = self.num_entities
        
        # 2. Decoder (Prediction Head)
        # Input: Latent (D_MODEL)
        # Output: Next State (Pose 7 + Twist 6) = 13
        
        self.head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim*4),
            nn.GELU(),
            nn.Linear(self.model_dim*4, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.state_dim)
        )

    def forward(self, history, static_features):
        # 1. Encode History
        obs_dict = {'history': history, 'static_features': static_features}
        
        # Encoder returns flattened (B, E, D_MODEL)
        latent = self.encoder(obs_dict)
        
        # 3. Predict Next State Delta or Absolute State
        # (B, E, STATE_DIM)
        prediction = self.head(latent)
        
        return prediction