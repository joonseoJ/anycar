import jax
import jax.numpy as jnp
import numpy as np
from car_foundation.jax_models import JaxHOTDynamicsPredictor, JaxHOTLayer, JaxKroneckerFactorizedAttention

def test_hot_dynamics_predictor():
    print("=== Testing JaxHOTDynamicsPredictor ===")
    
    # 1. Hyperparameters & Dimensions
    B = 2           # Batch size
    T_hist = 20     # History length
    T_fut = 10      # Future length (Different from history to test CrossAttn shape change)
    E = 5           # Number of entities/nodes
    X = 13   # Output state dimension (X)
    S = 12 # Static features (S)
    A = 6 # Future action features (A)
    H = X+A   # History features (X+A)
    
    D_model = 32
    Num_heads = 4
    
    # 2. Create Dummy Data
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    history = jax.random.normal(k1, (B, T_hist, E, H))
    static_features = jax.random.normal(k2, (B, E, S))
    future_actions = jax.random.normal(k3, (B, T_fut, E, A))
    
    print(f"Input History Shape: {history.shape}")
    print(f"Input Static Shape: {static_features.shape}")
    print(f"Input Future Action Shape: {future_actions.shape}")

    # 3. Initialize Model
    model = JaxHOTDynamicsPredictor(
        num_layers=2,
        num_heads=Num_heads,
        d_model=D_model,
        mlp_dim=64,
        state_dim=X,
        dropout_rate=0.1
    )
    
    # Init variables
    variables = model.init(k4, history, static_features, future_actions, deterministic=True)
    
    # 4. Forward Pass Test
    output = model.apply(variables, history, static_features, future_actions, deterministic=True)
    
    print(f"Output Shape: {output.shape}")
    
    # Check Output Shape: Should be (B, T_fut, E, State_dim)
    expected_shape = (B, T_fut, E, X)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✅ Forward pass shape verification passed.")
    
    # 5. JIT Compilation Test
    print("\nTesting JIT compilation...")
    @jax.jit
    def forward_jit(h, s, f):
        return model.apply(variables, h, s, f, deterministic=True)
    
    # Warmup
    _ = forward_jit(history, static_features, future_actions)
    print("✅ JIT compilation successful.")
    
    # 6. Attention Map Extraction Test (Debugging)
    print("\nTesting Attention Map Extraction...")
    _, state = model.apply(variables, history, static_features, future_actions, 
                           deterministic=True, mutable=['intermediates'])
    
    intermediates = state['intermediates']
    
    # Inspect Decoding Layer 0, Mode 0 (Time dimension)
    # This should show attention between Future (Query) and History (Key)
    # Path might vary slightly based on Flax version, but usually:
    # intermediates['Decoding_HOT_Layer_0']['JaxKroneckerFactorizedAttention_0']['attention_map_mode_0']
    
    try:
        dec_layer_0 = intermediates['Decoding_HOT_Layer_0']
        # The key for attention module might be auto-generated index, usually the first submodule
        attn_module_name = [k for k in dec_layer_0.keys() if 'JaxKroneckerFactorizedAttention' in k][0]
        attn_maps = dec_layer_0[attn_module_name]
        
        # Mode 0: Time attention
        time_attn = attn_maps['attention_map_mode_0'][0] # Tuple of outputs
        print(f"Decoder Time Attention Map Shape: {time_attn.shape}")
        
        # Expected: (Batch, Heads, T_fut, T_hist)
        expected_attn_shape = (B, Num_heads, T_fut, T_hist)
        assert time_attn.shape == expected_attn_shape, f"Attn Map shape mismatch! Expected {expected_attn_shape}, got {time_attn.shape}"
        
        # Mode 1: Entity attention
        entity_attn = attn_maps['attention_map_mode_1'][0]
        print(f"Decoder Entity Attention Map Shape: {entity_attn.shape}")
        
        # Expected: (Batch, Heads, E, E) -> Entity dimension stays same
        expected_ent_shape = (B, Num_heads, E, E)
        assert entity_attn.shape == expected_ent_shape, f"Entity Attn Map shape mismatch! Expected {expected_ent_shape}, got {entity_attn.shape}"
        
        print("✅ Attention map shapes verified.")
        
    except KeyError as e:
        print(f"⚠️ Could not navigate intermediates structure perfectly: {e}")
        print("Available keys:", intermediates.keys())

if __name__ == "__main__":
    test_hot_dynamics_predictor()