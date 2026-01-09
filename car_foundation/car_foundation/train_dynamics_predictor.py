import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import datetime
from car_foundation.car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR, CAR_FOUNDATION_LOG_DIR
import pickle
import random

# --- Import your specific modules ---
# Adjust the import paths according to your project structure
from torch_models import DynamicsPredictor
from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar

def load_data_from_folder(folder_path):
    """
    Loads all .pkl files from the specified folder and merges them into a single TensorDataset.
    
    Expected Dictionary Structure in .pkl:
        - 'history': (N, T, NumEntities, 19) -> Contains State(13) + Action(6) from t-N+1 to t-1
        - 'static_features': (N, NumEntities, 6)
        - 'current_state': (N, NumEntities, 19) -> State at time t (Target)
    """
    pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {folder_path}")
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
                data = data.data_logs
            
            # Check for required keys
            if 'history' not in data or 'static_features' not in data or 'current_state' not in data:
                print(f"Warning: Missing keys in {f_path}. Skipping.")
                continue

            all_history.append( torch.FloatTensor(data['history']) )
            all_static.append( torch.FloatTensor(data['static_features']) )
            all_current_state.append( torch.FloatTensor(data['current_state']) )
            
        except Exception as e:
            print(f"Error loading {f_path}: {e}")

    if not all_history:
        raise ValueError("No valid data loaded.")

    print("Merging data tensors...")
    # Concatenate all lists into single tensors
    combined_history = torch.cat(all_history, dim=0)
    combined_static = torch.cat(all_static, dim=0)
    combined_current_state = torch.cat(all_current_state, dim=0)

    print(f"Total Samples: {combined_history.shape[0]}")
    
    # Create TensorDataset
    full_dataset = TensorDataset(combined_history, combined_static, combined_current_state)
    return full_dataset

def train_model():
    # --- Hyperparameters & Configuration ---
    DATA_FOLDER = os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_debugging')       # Path to your .pkl files
    save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
    MODEL_SAVE_PATH = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')   # Path to save the best model
    LOG_DIR = os.path.join(CAR_FOUNDATION_LOG_DIR, f'{save_model_folder_prefix}-model_log')     # TensorBoard log directory
    
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 50
    HISTORY_LENGTH = 10
    
    # [Scaling Factor]
    # Since the delta (next_state - current_state) is very small (e.g., ~0.001),
    # we multiply it by a large factor (e.g., 100 or 1000) to make the loss gradients significant.
    # The model learns to predict (Delta * SCALE). During inference, divide output by SCALE.
    TARGET_SCALE = 100.0 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load and Merge Data
    full_dataset = load_data_from_folder(DATA_FOLDER)
    
    # 2. Split Dataset (7:2:1)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    # Use manual_seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset Split -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 3. Create DataLoaders
    # num_workers and pin_memory for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    # Temporary env to get observation space shapes
    tmp_env = MuJoCoCar({'is_render': False})
    model = DynamicsPredictor(
        observation_space=tmp_env.observation_space,
        model_dim=128
    ).to(DEVICE)
    tmp_env.close()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    writer = SummaryWriter(LOG_DIR)
    
    best_val_loss = float('inf')
    print(f"Starting training on {DEVICE} with Target Scale: {TARGET_SCALE}...")

    # 5. Training Loop
    for epoch in trange(EPOCHS, desc="Epochs"):
        model.train()
        train_loss_acc = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in loop:
            # Unpack batch: history, static_features, current_state (target at t)
            history, static, target_state_t = [b.to(DEVICE) for b in batch]

            state_t_minus_1 = history[:, -1, :, :13] 
            delta_gt = target_state_t - state_t_minus_1
            scaled_delta_gt = delta_gt * TARGET_SCALE

            pred_scaled_delta = model(history, static)
            
            loss = criterion(pred_scaled_delta, scaled_delta_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_acc / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # 6. Validation Loop
        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                history, static, target_state_t = [b.to(DEVICE) for b in batch]
                
                state_t_minus_1 = history[:, -1, :, :13]
                
                delta_gt = target_state_t - state_t_minus_1
                scaled_delta_gt = delta_gt * TARGET_SCALE
                
                pred_scaled_delta = model(history, static)
                loss = criterion(pred_scaled_delta, scaled_delta_gt)
                
                val_loss_acc += loss.item()

        avg_val_loss = val_loss_acc / len(val_loader)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best model saved (Val Loss: {best_val_loss:.6f})")

    # 7. Final Test Evaluation
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # Load best weights
    model.eval()
    test_loss_acc = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            history, static, target_state_t = [b.to(DEVICE) for b in batch]
            
            state_t_minus_1 = history[:, -1, :, :13]
            
            delta_gt = target_state_t - state_t_minus_1
            scaled_delta_gt = delta_gt * TARGET_SCALE
            
            pred_scaled_delta = model(history, static)
            loss = criterion(pred_scaled_delta, scaled_delta_gt)
            test_loss_acc += loss.item()
            
    avg_test_loss = test_loss_acc / len(test_loader)
    print(f"Final Test MSE Loss (Scaled): {avg_test_loss:.6f}")
    writer.add_scalar('Loss/Test', avg_test_loss, 0)
    
    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    train_model()