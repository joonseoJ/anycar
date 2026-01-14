from collections import OrderedDict
import numpy as np
import torch
import jax
import jax.numpy as jnp
from car_foundation.data_config import MujocoDataConfig

class HyperparameterManager:
    def __init__(self):
        self.hyperparameters = OrderedDict()

    def get_hyperparameter(self, key):
        return self.hyperparameters[key]

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def __str__(self):
        return str(self.hyperparameters)

    def __repr__(self):
        return str(self.hyperparameters)
    
    def __getitem__(self, key):
        return self.hyperparameters[key]
    
    def __setitem__(self, key, value):
        self.hyperparameters[key] = value
    
    def __iter__(self):
        return iter(self.hyperparameters)
    
    def __len__(self):
        return len(self.hyperparameters)
    
    def __contains__(self, key):
        return key in self.hyperparameters
    
    def __delitem__(self, key):
        del self.hyperparameters[key]
        
@jax.jit
def differentiate_state(history: jnp.ndarray):
    """
    절대 상태 시퀀스를 Base State와 Delta Sequence로 변환합니다.
    
    Args:
        history: (B, T_history+1, E, X+A) 
                 X=13 (Pos 3, Quat 4, LinVel 3, AngVel 3)
                 A=Action Dim (Raw values expected)
                 
    Returns:
        base_state: (B, E, X) - t=0 시점의 절대 상태 (복원 기준점)
        deltas: (B, T_history, E, X_delta + A) 
                X_delta=12 (Pos 3, so3 3, LinVel 3, AngVel 3)
                Action은 Delta가 아닌 t+1 시점의 값을 그대로 유지 (Dataset 규칙 따름)
    """
    
    # 1. Base State 추출 (t=0)
    # Action 부분은 제외하고 State(0~12)만 추출
    base_state = history[:, 0, :, :MujocoDataConfig.HISTORY_STATE_END] # (B, E, 13)
    
    # 2. Slice for Calculation
    # Prev: t=0 ~ T-1
    # Curr: t=1 ~ T
    
    # State parts
    state_prev = history[:, :-1, :, :MujocoDataConfig.HISTORY_STATE_END]
    state_curr = history[:, 1:, :, :MujocoDataConfig.HISTORY_STATE_END]
    
    # Action parts
    actions_curr = history[:, 1:, :, MujocoDataConfig.HISTORY_ACTION_START:] 
    
    # 3. Compute Deltas
    
    # (1) Position Delta
    d_pos = state_curr[..., 0:3] - state_prev[..., 0:3]
    
    # (2) Rotation Delta (Quaternion -> so3)
    q_prev = state_prev[..., 3:7]
    q_curr = state_curr[..., 3:7]
    
    # q_curr = q_prev * q_rel  =>  q_rel = q_prev^-1 * q_curr
    q_rel = q_multiply_jax(q_inverse_jax(q_prev), q_curr)
    d_so3 = q_to_so3_jax(q_rel)
    
    # (3) Velocity Delta
    d_lin_vel = state_curr[..., 7:10] - state_prev[..., 7:10]
    d_ang_vel = state_curr[..., 10:13] - state_prev[..., 10:13]
    
    # 4. Concatenate
    # State Delta: 3 + 3 + 3 + 3 = 12 dim
    state_delta = jnp.concatenate([d_pos, d_so3, d_lin_vel, d_ang_vel], axis=-1)
    
    # Final Output: [Delta_State, Raw_Action]
    # Shape: (B, T, E, 12 + A)
    deltas = jnp.concatenate([state_delta, actions_curr], axis=-1)
    
    return base_state, deltas

@jax.jit
def integrate_state(base_state: jnp.ndarray, pred_deltas: jnp.ndarray):
    """
    Delta Sequence를 Base State에 적분하여 전체 궤적(Trajectory)을 복원합니다.
    
    Args:
        base_state: (B, E, 13) - 초기 상태 (t=0 기준, 혹은 t-1 기준)
                    [Pos(3), Quat(4), LinVel(3), AngVel(3)]
        pred_deltas: (B, T, E, 12) - 변화량 시퀀스
                    [dPos(3), dSo3(3), dLinVel(3), dAngVel(3)]
                    
    Returns:
        traj_states: (B, T, E, 13) - 복원된 절대 상태 시퀀스
    """
    
    # --- 1. Euclidean States Integration (Pos, LinVel, AngVel) ---
    # 위치와 속도는 단순히 덧셈이므로 cumsum을 사용하여 병렬 처리 (매우 빠름)
    
    # (B, T, E, 3)
    d_pos = pred_deltas[..., 0:3]
    d_lin_vel = pred_deltas[..., 6:9]
    d_ang_vel = pred_deltas[..., 9:12]
    
    # Base state 확장 (Broadcasting을 위해 Time 차원 추가: B, 1, E, 3)
    base_pos = base_state[..., 0:3][:, None, ...]
    base_lin_vel = base_state[..., 7:10][:, None, ...]
    base_ang_vel = base_state[..., 10:13][:, None, ...]
    
    # 누적합(CumSum) + 초기값
    traj_pos = base_pos + jnp.cumsum(d_pos, axis=1)
    traj_lin_vel = base_lin_vel + jnp.cumsum(d_lin_vel, axis=1)
    traj_ang_vel = base_ang_vel + jnp.cumsum(d_ang_vel, axis=1)

    # --- 2. Quaternion Integration (Rotation) ---
    # 회전은 교환법칙이 성립하지 않으므로 순서대로 곱해야 함 -> scan 사용
    
    base_quat = base_state[..., 3:7]    # (B, E, 4)
    delta_so3_seq = pred_deltas[..., 3:6] # (B, T, E, 3)

    # scan은 첫 번째 차원(Time)을 따라 반복하므로, (B, T, ...) -> (T, B, ...)로 변환
    delta_so3_seq_T = jnp.swapaxes(delta_so3_seq, 0, 1)

    def scan_body(current_quat, d_so3):
        # current_quat: (B, E, 4) - 현재 스텝의 절대 회전
        # d_so3: (B, E, 3) - 이번 스텝의 회전 변화량
        
        # (1) Convert so3 vector to quaternion (Exponential map)
        theta = jnp.linalg.norm(d_so3, axis=-1, keepdims=True)
        # 0으로 나누기 방지 (Small angle approximation)
        scale = jnp.where(theta < 1e-7, 0.5, jnp.sin(theta/2) / theta)
        
        w_delta = jnp.cos(theta/2)
        xyz_delta = d_so3 * scale
        dq = jnp.concatenate([w_delta, xyz_delta], axis=-1) # (B, E, 4)
        
        # (2) Quaternion Multiplication: q_new = q_old * dq
        # (Global frame 기준 회전이면 q_new = dq * q_old 여야 할 수도 있으나, 
        #  보통 Body frame delta인 경우 q_old * dq)
        w1, x1, y1, z1 = jnp.split(current_quat, 4, axis=-1)
        w2, x2, y2, z2 = jnp.split(dq, 4, axis=-1)
        
        w_new = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x_new = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y_new = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z_new = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        q_new = jnp.concatenate([w_new, x_new, y_new, z_new], axis=-1)
        
        # (3) Normalization (누적 오차 방지)
        q_new = q_new / jnp.linalg.norm(q_new, axis=-1, keepdims=True)
        
        return q_new, q_new # carry, output

    # jax.lax.scan 실행
    # carry: 마지막 상태 유지, output_seq: 중간 과정 기록
    _, traj_quat_T = jax.lax.scan(scan_body, base_quat, delta_so3_seq_T)
    
    # 다시 (B, T, E, 4)로 복구
    traj_quat = jnp.swapaxes(traj_quat_T, 0, 1)

    # --- 3. Concatenate All ---
    # 순서: Pos, Quat, LinVel, AngVel
    traj_states = jnp.concatenate([traj_pos, traj_quat, traj_lin_vel, traj_ang_vel], axis=-1)
    
    return traj_states



def quaternion_to_euler(q):
    if q.ndim == 1:
        q = q.reshape(1,-1)
    # Normalize quaternion
    norm = np.linalg.norm(q, axis=1)[:, np.newaxis]
    q = q / norm
    
    # Extract the values from Q
    q_w, q_x, q_y, q_z = q[:,0], q[:,1], q[:,2], q[:,3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x**2 + q_y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1,
                    np.sign(sinp) * np.pi / 2,  # use 90 degrees if out of range
                    np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y**2 + q_z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_geodesic_loss(q_pred, q_gt, eps=1e-6):
    """
    q_pred, q_gt: (..., 4)  in (qw, qx, qy, qz)
    """
    # normalize
    q_pred = q_pred / jnp.linalg.norm(q_pred, axis=-1, keepdims=True)
    q_gt   = q_gt   / jnp.linalg.norm(q_gt,   axis=-1, keepdims=True)

    dot = jnp.sum(q_pred * q_gt, axis=-1)
    dot = jnp.clip(jnp.abs(dot), -1.0 + eps, 1.0 - eps)

    theta = 2.0 * jnp.arccos(dot)   # (...,)
    return jnp.mean(theta ** 2)

def generate_subsequences(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (-inf for padding, 0 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.fill_(torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device), float('-inf'))

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 0.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def generate_subsequences_hf(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.
    HuggingFace convention: mask is 1 for data and 0 for padding.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (0 for padding, 1 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device)

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 1.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    return d_yaw_aligned + yaw_2

def align_yaw_jax(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    return d_yaw_aligned + yaw_2

EPS = 1e-7

def q_inverse(q):
    """Quaternion inverse: [w, x, y, z] -> [w, -x, -y, -z]"""
    # q: (Batch, 4)
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)

def q_multiply(q1, q2):
    """Quaternion multiplication"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def q_to_so3(q):
    """
    Convert Quaternion to so(3) rotation vector (log map).
    q_rel = [cos(theta/2), u*sin(theta/2)]
    so3 = theta * u
    """
    # q: (..., 4) [w, x, y, z]
    w, v = q[..., 0], q[..., 1:]
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    
    # theta = 2 * atan2(|v|, w)
    # axis = v / |v|
    # so3 = theta * axis = 2 * atan2(|v|, w) * (v / |v|)
    
    # 수치 안정성을 위해 norm_v가 매우 작을 때(회전이 거의 없을 때) 처리
    # limit(|v|->0) of (2*atan2(|v|, w) / |v|) is 2/w * sign(w) => roughly 2 (if w~1)
    # 여기서는 안전하게 계산
    angle = 2 * torch.atan2(norm_v, w.unsqueeze(-1))
    
    scale = torch.where(
        norm_v < EPS,
        torch.tensor(2.0, device=q.device), # Small angle approximation
        angle / norm_v
    )
    return scale * v

def q_inverse_jax(q):
    w = q[..., 0:1]
    xyz = q[..., 1:]
    return jnp.concatenate([w, -xyz], axis=-1)


def q_multiply_jax(q1, q2):
    w1, x1, y1, z1 = jnp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(q2, 4, axis=-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return jnp.concatenate([w, x, y, z], axis=-1)


def q_to_so3_jax(q):
    w = q[..., 0:1]
    v = q[..., 1:]
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)

    angle = 2.0 * jnp.arctan2(norm_v, w)
    scale = jnp.where(
        norm_v < EPS,
        2.0,
        angle / (norm_v + EPS),
    )
    return scale * v