import os
import glob
import pickle
import random
import datetime
import numpy as np
from tqdm import tqdm, trange

# JAX / Flax / Optax imports
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint
from flax.training import orbax_utils

# PyTorch (Data Loading 용도)
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
# Tensorboard (JAX와 함께 사용 가능)
from torch.utils.tensorboard import SummaryWriter

# Project imports
from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR, CAR_FOUNDATION_LOG_DIR
# jax_models.py에서 DynamicsPredictor를 import 한다고 가정
from car_foundation.jax_models import JaxDynamicsPredictor 

# --- Random Seed ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def load_data_from_folder(folder_path):
    """
    Loads all .pkl files and merges them into a single TensorDataset.
    (PyTorch 로직 재사용)
    """
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {folder_path}")
    
    # Random sampling for debugging speed (remove min limit for full training)
    n = min(10, len(pkl_files)) 
    selected_files = random.sample(pkl_files, n)

    print(f"Randomly sampled {len(selected_files)} dataset files. Loading...")

    all_history = []
    all_static = []
    all_current_state = []

    for f_path in tqdm(selected_files, desc="Loading Files"):
        try:
            with open(f_path, 'rb') as f:
                data = pickle.load(f)
                # 데이터 구조에 따라 data.data_logs 또는 data 딕셔너리 사용
                if hasattr(data, 'data_logs'):
                    data = data.data_logs
            
            if 'history' not in data or 'static_features' not in data or 'current_state' not in data:
                print(f"Warning: Missing keys in {f_path}. Skipping.")
                continue

            all_history.append(torch.FloatTensor(data['history']))
            all_static.append(torch.FloatTensor(data['static_features']))
            all_current_state.append(torch.FloatTensor(data['current_state']))
            
        except Exception as e:
            print(f"Error loading {f_path}: {e}")

    if not all_history:
        raise ValueError("No valid data loaded.")

    print("Merging data tensors...")
    combined_history = torch.cat(all_history, dim=0)
    combined_static = torch.cat(all_static, dim=0)
    combined_current_state = torch.cat(all_current_state, dim=0)

    print(f"Total Samples: {combined_history.shape[0]}")
    
    # DataLoader용 TensorDataset 생성
    return TensorDataset(combined_history, combined_static, combined_current_state)

# --- JAX Training Step ---
@jax.jit
def train_step(state, history, static, target, target_scale):
    """
    history: (B, T, E, F)
    static: (B, E, S)
    target: (B, E, F) - Target State at t
    """
    # GT Delta 계산
    # history의 마지막 step (state at t-1) 추출. 인덱스 13까지가 state라고 가정
    state_t_minus_1 = history[:, -1, :, :13] 
    delta_gt = target - state_t_minus_1
    
    # Scaling
    scaled_delta_gt = delta_gt * target_scale

    def loss_fn(params):
        # Forward Pass (determinictic=False if dropout needed, but usually True for dynamics unless using MC dropout)
        # 이전 변환 코드에서 init 인자가 history_dim 등을 받았으므로 forward는 입력만 받음
        pred_scaled_delta = state.apply_fn(
            {'params': params}, 
            history, 
            static, 
            deterministic=False, 
            rngs={'dropout': jax.random.fold_in(state.rng, state.step)}
        )
        # MSE Loss
        loss = jnp.mean((pred_scaled_delta - scaled_delta_gt) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Update weights
    state = state.apply_gradients(grads=grads)
    return state, loss

# --- JAX Eval Step ---
@jax.jit
def eval_step(state, history, static, target, target_scale):
    state_t_minus_1 = history[:, -1, :, :13]
    delta_gt = target - state_t_minus_1
    scaled_delta_gt = delta_gt * target_scale
    
    pred_scaled_delta = state.apply_fn(
        {'params': state.params}, 
        history, 
        static, 
        deterministic=True
    )
    
    loss = jnp.mean((pred_scaled_delta - scaled_delta_gt) ** 2)
    return loss

# --- Custom TrainState to hold RNG key ---
class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey

def train_model():
    # --- Configuration ---
    DATA_FOLDER = os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_debugging')
    save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
    MODEL_SAVE_PATH = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')
    LOG_DIR = os.path.join(CAR_FOUNDATION_LOG_DIR, f'{save_model_folder_prefix}-model_log')
    
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 50
    TARGET_SCALE = 100.0
    
    # 1. Load Data (PyTorch logic)
    full_dataset = load_data_from_folder(DATA_FOLDER)
    
    # Extract dimensions from the first sample to initialize JAX model
    sample_hist, sample_static, sample_target = full_dataset[0]
    # sample_hist shape: (T, Entities, 19)
    # sample_static shape: (Entities, 6)
    
    HISTORY_DIM = sample_hist.shape[-1]
    STATIC_DIM = sample_static.shape[-1]
    STATE_DIM = 13 # output dimension (delta)
    MODEL_DIM = 128
    NUM_ENTITIES = sample_hist.shape[1]

    # Split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # 2. Initialize JAX Model
    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)
    
    # Instantiate Model (이전에 정의한 Flax Class)
    model = JaxDynamicsPredictor(
        model_dim=MODEL_DIM,
        output_dim=STATE_DIM
    )
    
    # Dummy Input for Initialization
    dummy_hist = jnp.ones((1, *sample_hist.shape)) # Add batch dim
    dummy_static = jnp.ones((1, *sample_static.shape))
    
    variables = model.init(init_rng, dummy_hist, dummy_static)
    params = variables['params']
    
    # 3. Optimizer & State
    optimizer = optax.adamw(learning_rate=LR)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng
    )

    # 4. Checkpoint Manager (Orbax)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=1) # Save every step(epoch) called
    checkpoint_manager = orbax.checkpoint.CheckpointManager(MODEL_SAVE_PATH, orbax_checkpointer, options)

    writer = SummaryWriter(LOG_DIR)
    best_val_loss = float('inf')

    print(f"Starting JAX training on {jax.devices()}...")

    # 5. Training Loop
    for epoch in trange(EPOCHS, desc="Epochs"):
        # --- Train ---
        train_loss_acc = 0.0
        count = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for batch in loop:
            # Torch Tensor -> Numpy -> JAX Array
            hist_cpu, static_cpu, target_cpu = [b.numpy() for b in batch]
            hist_jax = jnp.array(hist_cpu)
            static_jax = jnp.array(static_cpu)
            target_jax = jnp.array(target_cpu)
            
            # Update Step
            state, loss = train_step(state, hist_jax, static_jax, target_jax, TARGET_SCALE)
            
            train_loss_acc += loss.item()
            count += 1
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_acc / count
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # --- Validation ---
        val_loss_acc = 0.0
        val_count = 0
        
        for batch in val_loader:
            hist_cpu, static_cpu, target_cpu = [b.numpy() for b in batch]
            hist_jax = jnp.array(hist_cpu)
            static_jax = jnp.array(static_cpu)
            target_jax = jnp.array(target_cpu)

            loss = eval_step(state, hist_jax, static_jax, target_jax, TARGET_SCALE)
            val_loss_acc += loss.item()
            val_count += 1
            
        avg_val_loss = val_loss_acc / val_count
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Checkpointing (Save Best) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            ckpt = {'model': state, 'config': {'target_scale': TARGET_SCALE, 'dims': (HISTORY_DIM, STATIC_DIM)}}
            save_args = orbax_utils.save_args_from_target(ckpt)
            
            # Epoch을 step으로 사용하여 저장
            checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
            print(f"  >>> Best model saved at step {epoch} (Val Loss: {best_val_loss:.6f})")

    # 6. Final Test
    print("\nEvaluating on Test Set (using Best Model)...")
    
    # Restore Best Model
    # best_val_loss 갱신되었을 때 저장된 마지막 step 복원
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        restored = checkpoint_manager.restore(latest_step)
        state = state.replace(params=restored['model']['params'])
        print(f"Restored checkpoint from step {latest_step}")

    test_loss_acc = 0.0
    test_count = 0
    for batch in tqdm(test_loader, desc="Testing"):
        hist_cpu, static_cpu, target_cpu = [b.numpy() for b in batch]
        hist_jax = jnp.array(hist_cpu)
        static_jax = jnp.array(static_cpu)
        target_jax = jnp.array(target_cpu)

        loss = eval_step(state, hist_jax, static_jax, target_jax, TARGET_SCALE)
        test_loss_acc += loss.item()
        test_count += 1
            
    avg_test_loss = test_loss_acc / test_count
    print(f"Final Test MSE Loss (Scaled): {avg_test_loss:.6f}")
    writer.add_scalar('Loss/Test', avg_test_loss, 0)
    
    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    train_model()