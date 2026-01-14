################################################################################
# CHAPTER 1: IMPORTS AND GLOBAL SETUP
# Import necessary libraries for PyTorch (DataLoaders), JAX/Flax (Modeling), 
# and other utilities.
################################################################################
from functools import partial
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import orbax_utils
# from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state

# Custom modules for car foundation models and datasets
from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
from car_foundation.dataset import DynamicsDataset, IssacSimDataset, MujocoDataset
from car_foundation.models import TorchMLP, TorchTransformer, TorchTransformerDecoder, TorchGPT2
from car_foundation.jax_models import JaxDynamicsPredictor, JaxMLP, JaxCNN
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax
import datetime
import os
import glob
import time
import math
import random
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pytorch_warmup as warmup

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Keys for dictionary access
PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"

# Set random seeds for reproducibility across PyTorch, NumPy, and Python
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)


################################################################################
# CHAPTER 2: CONFIGURATION AND HYPERPARAMETERS
# Define all training parameters, model settings, and paths.
################################################################################

# Sequence lengths
history_length = 100
prediction_length = 50
delays = None
teacher_forcing = False

# Training mode flags
FINE_TUNE = False
ATTACK = False

# Hyperparameter setup based on mode (Fine-tuning vs. Scratch)
if FINE_TUNE:
    lr_begin = 5e-5
    warmup_period = 2
    num_epochs = 400
    load_checkpoint = True
    resume_model_checkpint = 20
    resume_model_name = "RESUME-MODEL-PATH"
else:
    lr_begin = 5e-4
    warmup_period = 500
    num_epochs = 400
    load_checkpoint = False
    resume_model_checkpint = 0
    resume_model_name = ""

val_every = 20          # Validation interval (epochs)
batch_size = 128
lambda_l2 = 1e-4        # Weight decay
dataset_path = os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_debugging')
comment = 'jax'

# Checkpoint paths
resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}")
num_workers = 6

# Model dimensions
state_dim = 13
action_dim = 6
static_dim = 9
num_entities = 5
latent_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1

# Setup save directory with timestamp
save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')


################################################################################
# CHAPTER 3: MODEL SELECTION AND INITIALIZATION
# Select the architecture (Decoder, MLP, or CNN) and instantiate the JAX model.
################################################################################

architecture = 'transformer'
# architecture = 'mlp'
# architecture = 'cnn'

# model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)

if architecture == 'transformer':
    model = JaxDynamicsPredictor(latent_dim, state_dim, num_heads, num_layers, dropout, name=architecture)
# elif architecture == 'mlp':
#     model = JaxMLP([256, 256, 256, 256, 256], state_dim, 0.1, name=architecture)
# elif architecture == 'cnn':
#     model = JaxCNN([32, 64, 128, 256], state_dim, 0.1, name=architecture)


################################################################################
# CHAPTER 4: DATASET LOADING AND PREPARATION
# Load file paths, split into train/val/test, and create PyTorch DataLoaders.
# Note: We use PyTorch DataLoaders to feed data into the JAX training loop.
################################################################################

binary_mask = False # type(model) == TorchGPT2
dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl'))
random.shuffle(dataset_files)

# Split dataset: 70% Train, 20% Val, 10% Test
total_len = len(dataset_files)
split_70 = int(total_len * 0.7)
split_20 = int(total_len * 0.9)
data_70 = dataset_files[:split_70]
data_20 = dataset_files[split_70:split_20]
data_10 = dataset_files[split_20:total_len+1]

# Create Dataset instances
train_dataset = MujocoDataset(data_70, history_length, prediction_length, attack=ATTACK)
print("train data length", len(train_dataset))

val_dataset = MujocoDataset(data_20, history_length, prediction_length, attack=ATTACK)
test_dataset = MujocoDataset(data_10, history_length, prediction_length, attack=ATTACK)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_steps_per_epoch = len(train_loader)


################################################################################
# CHAPTER 5: LOGGING AND RANDOM KEYS
# Initialize Weights & Biases (WandB) and JAX Random Number Generators (RNG).
################################################################################

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer-sequence-prediction",
    name=architecture,

    # track hyperparameters and run metadata
    config={
        "history_length": history_length,
        "prediction_length": prediction_length,
        "delays": delays,
        "teacher_forcing": teacher_forcing,

        "learning_rate": lr_begin,
        "warmup_period": warmup_period,
        "architecture": architecture,
        "dataset": "even_dist_data",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lambda_l2": lambda_l2,
        "dataset_path": dataset_path.split('/')[-1],
        "comment": comment,

        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "implementation": "jax",
        "model_path": save_model_folder_path,
        "resume": load_checkpoint,
        "resume_checkpoint_path": resume_model_folder_path,
        "resume_checkpoint": resume_model_checkpint,
        "attack": ATTACK,
    }
)
print(wandb.config)
# print(f"total params: {sum(p.numel() for p in model.parameters())}")

# Initialize JAX PRNG Keys
rng = jax.random.PRNGKey(3407)
rng, params_rng = jax.random.split(rng)
rng, dropout_rng = jax.random.split(rng)
init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}
global_rngs = init_rngs


################################################################################
# CHAPTER 6: OPTIMIZER, STATE, AND CHECKPOINT MANAGER
# Setup Learning Rate Schedule, TrainState, and Orbax Checkpointing.
################################################################################

# Create dummy inputs to initialize model parameters (JAX lazy initialization)
jax_history_input = jnp.ones((batch_size, history_length-1, num_entities, state_dim + action_dim), dtype=jnp.float32)
jax_history_mask = jnp.ones((batch_size, (history_length-1) * 2 - 1), dtype=jnp.float32)
jax_static_features_input = jnp.ones((batch_size, num_entities, static_dim), dtype=jnp.float32)
jax_prediction_input = jnp.ones((batch_size, prediction_length, num_entities, action_dim), dtype=jnp.float32)
jax_prediction_mask = jnp.ones((batch_size, prediction_length), dtype=jnp.float32)

def create_learning_rate_fn():
    """Create a learning rate schedule with warmup and exponential decay."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_period]
    )
    return schedule_fn

learning_rate_fn = create_learning_rate_fn()

# Initialize model variables (parameters)
global_var = model.init(init_rngs, jax_history_input, jax_static_features_input, jax_prediction_input)

# Create Optimizer (AdamW)
tx = optax.adamw(learning_rate_fn, weight_decay=lambda_l2)

# Create TrainState (holds params, optimizer state, and apply_fn)
global_state = train_state.TrainState.create(
    apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
)

# Setup Orbax for saving/loading checkpoints
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(create=True, save_interval_steps=val_every)
checkpoint_manager = orbax.checkpoint.CheckpointManager(save_model_folder_path, orbax_checkpointer, options)

# Handle Checkpoint Loading (Resume or Fine-tune)
if FINE_TUNE:
    restored = checkpoint_manager.restore(resume_model_folder_path)
    global_var['params'] = restored['model']['params']
    # Re-create state with loaded params
    global_state = train_state.TrainState.create(
            apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
    )

################################################################################
# CHAPTER 7: CORE FUNCTIONS (INFERENCE & LOSS)
# Define the forward pass logic, coordinate transforms, and JIT-compiled Loss.
################################################################################

def apply_batch(var_collect, history, static_features, action, y, rngs):
    """
    Run inference (prediction) on a batch.
    Normalizes input, predicts relative changes, and transforms back to world coordinates.
    """
    # x = history[:, 1:, :, :]
    x = history
    
    # Forward pass
    y_pred = model.apply(var_collect, x, static_features, action, rngs=rngs, deterministic=True)
    # print(attn_mask.shape)
    
    return y_pred

# JIT compile the loss function for speed
# static_argnums=(7,) means the 8th argument (deterministic) is not traced
@partial(jax.jit, static_argnums=(7,))
def loss_fn(state, var_collect, history, static_features, action, y, rngs, deterministic=False):
    """
    Calculate MSE loss between prediction and ground truth.
    Includes normalization and stop_gradient on targets.
    """    
    # Stop gradients for targets to prevent backprop into data
    history = jax.lax.stop_gradient(history)
    static_features = jax.lax.stop_gradient(static_features)
    y = jax.lax.stop_gradient(y)
    action = jax.lax.stop_gradient(action)

    # Forward pass via TrainState
    y_pred = state.apply_fn(var_collect, history, static_features, action, rngs=rngs, deterministic=deterministic)
    
    # MSE Loss
    loss = jnp.mean((y_pred - y) ** 2)
    return loss


################################################################################
# CHAPTER 8: VALIDATION AND VISUALIZATION HELPERS
# Functions to evaluate model performance and plot trajectories.
################################################################################

def val_episode(var_collect, episode_num, rngs):
    """Run inference on a single validation episode."""    
    # Fetch data from dataset
    history, static_features, action, y = val_dataset[episode_num:episode_num+1]
    
    # Convert to JAX arrays
    history = jnp.array(history.numpy())
    static_features = jnp.array(static_features.numpy())
    action = jnp.array(action.numpy())
    y = jnp.array(y.numpy())
    action_padding_mask = jnp.array(action_padding_mask.numpy())
    
    # Predict
    predicted_states = apply_batch(var_collect, history, static_features, action, y, rngs=rngs)
    return np.array(predicted_states)
    

def val_loop(state, var_collect, val_loader, rngs):
    """Compute average loss over the entire validation dataset."""
    val_loss = 0.0
    t_val = tqdm.tqdm(val_loader)
    for i, (history, static_features, action, y) in enumerate(t_val):
        # Convert PyTorch tensors to JAX arrays
        history = jnp.array(history.numpy())
        static_features = jnp.array(static_features.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())
        
        # Accumulate loss
        val_loss += loss_fn(state, var_collect, history, static_features, action, y, rngs=global_rngs, deterministic=True).item()
        t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
        t_val.refresh()
        
    val_loss /= len(val_loader)
    return val_loss

def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs):
    """
    Visualize ground truth vs predicted trajectory and log to WandB.
    This temporarily restores the specific checkpoint to visualize.
    """
    # Re-initialize collection
    val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    # Load specific checkpoint for visualization
    raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{epoch_num}", "default"))
    val_collect['params'] = raw_restored['model']['params']
    
    # Predict
    predicted_states = val_episode(val_collect, episode_num, rngs)
    episode = val_dataset.get_episode(episode_num)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # 1. Trajectory (X-Y)
    axs[0, 0].plot(episode[:, 0, 0], episode[:, 0, 1], label='Ground Truth', marker='o', markersize=5)
    axs[0, 0].plot(predicted_states[0, :, 0, 0], predicted_states[0, :, 0, 1], label='Predicted', marker='x', markersize=5)
    axs[0, 0].legend()
    axs[0, 0].axis('equal')

    # 2. Velocity (Vx, Vy)
    predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
    axs[0, 1].plot(episode[:, 0, 7], label='Ground Truth vx')
    axs[0, 1].plot(episode[:, 0, 8], label='Ground Truth vy')
    axs[0, 1].plot(predict_x, predicted_states[0, :, 0, 7], label='Predicted vx')
    axs[0, 1].plot(predict_x, predicted_states[0, :, 0, 8], label='Predicted vy')
    axs[0, 1].legend()

    # 3. Angular Velocity (Yaw Rate)
    axs[1, 1].plot(episode[:, 0, 12], label='Ground Truth v_yaw')
    axs[1, 1].plot(predict_x, predicted_states[0, :, 0, 12], label='Predicted v_yaw')
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig('episode.png')
    plt.close(fig)
    
    # Log image to WandB
    wandb.log({"episode": wandb.Image('episode.png')})


################################################################################
# CHAPTER 9: MAIN TRAINING LOOP
# Iterate through epochs, calculate gradients, update weights, and save checkpoints.
################################################################################

train_losses = []
val_losses = []
val_epoch_nums = []

for epoch in tqdm.tqdm(range(num_epochs), desc="Epoch"):
    running_loss = 0.0
    t = tqdm.tqdm(train_loader)
    
    # --- Batch Loop ---
    for i, (history, static_features, action, y) in enumerate(t):
        # Convert Data to JAX
        history = jnp.array(history.numpy())
        static_features = jnp.array(static_features.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())

        # Define wrapper for gradient calculation
        def this_loss_fn(var_collect, history, static_features, action, y):
            return loss_fn(global_state, var_collect, history, static_features, action, y, global_rngs)
        
        # Compute Gradients using value_and_grad
        grad_fn = jax.value_and_grad(this_loss_fn, has_aux=False)
        loss, grads = grad_fn(global_var, history, static_features, action, y)
        
        loss_item = loss.item()
        running_loss += loss_item

        # Update Progress Bar
        t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_state.step):.6f}')
        t.refresh()

        # Update Model Parameters
        global_state = global_state.apply_gradients(grads=grads["params"])
        global_var['params'] = global_state.params

    # --- End of Epoch Logging ---
    running_loss /= len(train_loader)
    train_losses.append(running_loss)
    wandb.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(global_state.step)})
    print(save_model_folder_path)
    
    # --- Save Checkpoint ---
    # Construct checkpoint dictionary
    ckpt = {'model': global_state, 'description': 'Inference future states (B,T,E,X) without normalizing'}
    save_args = orbax_utils.save_args_from_target(ckpt)
    
    # Save using step (epoch)
    checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})

    # --- Validation Interval ---
    if (epoch + 1) % val_every == 0:
        visualize_episode(epoch + 1, 1, val_dataset, global_rngs) # Visualize specific episode
        val_loss = val_loop(global_state, global_var, val_loader, global_rngs) # Compute full val loss
        
        val_losses.append(val_loss)
        val_epoch_nums.append(epoch + 1)
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})


################################################################################
# CHAPTER 10: EVALUATION AND FINISH
# Plot loss curves, evaluate on test set, and finalize logging.
################################################################################

# Plot Training vs Validation Loss
train_epoch_nums = list(range(1, num_epochs + 1))
plt.figure()
plt.plot(train_epoch_nums, train_losses, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig('train_val_loss.png')
# plt.show()

# Final Evaluation on Test Set
visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
test_loss = val_loop(global_state, global_var, test_loader, global_rngs)
print(f'Test Loss: {test_loss:.4f}')

# Save artifacts to WandB
wandb.save('model_checkpoint/')

# Finish WandB run
wandb.finish()