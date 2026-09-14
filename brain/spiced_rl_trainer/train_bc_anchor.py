import os
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V_MAX = 0.8
KAPPA_MAX = 1.25
def row_has_value(row, key):
    return row.get(key) not in (None, "")
def row_float(row, key, default=None):
    value = row.get(key)
    if value in (None, ""):
        if default is None:
            raise KeyError(key)
        return default
    return float(value)
def normalized_action(row):
    if row_has_value(row, "action_v_norm") and row_has_value(row, "action_kappa_norm"):
        return np.array([
            np.clip(row_float(row, "action_v_norm"), 0.0, 1.0),
            np.clip(row_float(row, "action_kappa_norm"), -1.0, 1.0),
        ], dtype=np.float32)
    cmd_v = row_float(row, "cmd_v")
    cmd_w = row_float(row, "cmd_w")
    norm_v = np.clip(cmd_v / V_MAX, 0.0, 1.0)
    raw_kappa = cmd_w / max(abs(cmd_v), 0.01)
    norm_kappa = np.clip(raw_kappa / KAPPA_MAX, -1.0, 1.0)
    return np.array([norm_v, norm_kappa], dtype=np.float32)
def row_speed(row):
    if row_has_value(row, "current_v"):
        return abs(row_float(row, "current_v"))
    if row_has_value(row, "cmd_v"):
        return abs(row_float(row, "cmd_v"))
    return abs(row_float(row, "action_v_norm", 0.0) * V_MAX)
class SpicedDataset(Dataset):
    def __init__(self, dataset_dir):
        self.inputs = []
        self.actions = []
        csv_pattern = os.path.join(dataset_dir, "spice_run_*.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        if not csv_files:
            raise FileNotFoundError(f"No purified CSV files found in {dataset_dir}")
        print(f"Loading and rebuilding {len(csv_files)} files with 15D sliding window...")
        for file_path in csv_files:
            rows = []
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if len(rows) < 300:
                continue
            max_vel = max(row_speed(r) for r in rows)
            if max_vel < 0.20:
                continue
            start_idx = 0
            for idx, r in enumerate(rows):
                if row_speed(r) > 0.05:
                    start_idx = idx
                    break
            end_idx = len(rows) - 1
            for idx in range(len(rows) - 1, -1, -1):
                r = rows[idx]
                if row_speed(r) > 0.05 or row_float(r, 'local_goal_dist') > 0.20:
                    end_idx = idx
                    break
            if end_idx - start_idx < 150:
                continue
            valid_rows = rows[start_idx:end_idx + 1]
            for t in range(4, len(valid_rows)):
                obs_15 = []
                for d in range(5):
                    r_hist = valid_rows[t - (4 - d)]
                    l_goal_x = row_float(r_hist, 'local_goal_x') * 0.20
                    l_goal_y = row_float(r_hist, 'local_goal_y') * 0.20
                    l_goal_dist = row_float(r_hist, 'local_goal_dist') * 0.20
                    obs_15.extend([l_goal_x, l_goal_y, l_goal_dist])
                act = normalized_action(valid_rows[t])
                self.inputs.append(np.array(obs_15, dtype=np.float32))
                self.actions.append(act)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        if len(self.inputs) == 0:
            raise ValueError("No usable Spice samples after purification filters.")
        print(f"Sliding window dataset built. Total training samples: {len(self.inputs)}")
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.actions[idx])
class GaussianPolicy(nn.Module):
    def __init__(self, input_dim=15, action_dim=2):
        super(GaussianPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, x):
        features = self.fc(x)
        raw_mean = self.mean_head(features)
        mean_v = torch.sigmoid(raw_mean[:, 0:1])
        mean_w = torch.tanh(raw_mean[:, 1:2])
        mean = torch.cat([mean_v, mean_w], dim=1)
        return mean, self.log_std.expand_as(mean)
class SpicedBrainInference(nn.Module):
    def __init__(self, policy):
        super(SpicedBrainInference, self).__init__()
        self.policy = policy
    def forward(self, x):
        mean, _ = self.policy(x)
        return mean
def train_bc():
    dataset_dir = os.path.join(REPO_ROOT, "dataset", "purified")
    model_dir = os.path.join(REPO_ROOT, "model")
    anchor_path = os.path.join(REPO_ROOT, "bc_anchor.pth")
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    full_dataset = SpicedDataset(dataset_dir)
    if len(full_dataset) < 2:
        raise ValueError("Need at least two usable Spice samples for train/validation split.")
    train_size = int(0.9 * len(full_dataset))
    train_size = min(max(train_size, 1), len(full_dataset) - 1)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    policy = GaussianPolicy(input_dim=15).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    best_val_loss = float('inf')
    epochs = 300
    print("Beginning epoch-loop training (15D Sliding Window & Clean Decoupled)...")
    for epoch in range(1, epochs + 1):
        policy.train()
        train_loss = 0.0
        for obs, act in train_loader:
            obs, act = obs.to(device), act.to(device)
            optimizer.zero_grad()
            mean, log_std = policy(obs)
            steer_weight = torch.where(torch.abs(act[:, 1]) > 0.02, 5.0, 1.0)
            var = torch.exp(2 * log_std)
            nll_elementwise = 0.5 * (((act - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi)).sum(dim=1)
            nll = (nll_elementwise * steer_weight).mean()
            nll.backward()
            optimizer.step()
            train_loss += nll.item() * obs.size(0)
        train_loss /= len(train_loader.dataset)
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, act in val_loader:
                obs, act = obs.to(device), act.to(device)
                mean, log_std = policy(obs)
                steer_weight = torch.where(torch.abs(act[:, 1]) > 0.02, 5.0, 1.0)
                var = torch.exp(2 * log_std)
                nll_elementwise = 0.5 * (((act - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi)).sum(dim=1)
                nll = (nll_elementwise * steer_weight).mean()
                val_loss += nll.item() * obs.size(0)
        val_loss /= len(val_loader.dataset)
        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(policy.state_dict(), anchor_path)
    print("Optimization completed. Training 15D BC Anchor Coach succeeded.")
    print("Compiling Spiced Brain model for ONNX export...")
    policy.load_state_dict(torch.load(anchor_path, map_location=device))
    policy.eval()
    inference_model = SpicedBrainInference(policy).to("cpu")
    inference_model.eval()
    dummy_input = torch.zeros(1, 15, dtype=torch.float32)
    onnx_path = os.path.join(model_dir, "spiced_brain.onnx")
    torch.onnx.export(
        inference_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_state'],
        output_names=['action_output'],
        dynamic_axes={'input_state': {0: 'batch_size'}, 'action_output': {0: 'batch_size'}}
    )
    print(f"SUCCESS: 15D Spiced Brain exported to {onnx_path}")
if __name__ == "__main__":
    train_bc()
