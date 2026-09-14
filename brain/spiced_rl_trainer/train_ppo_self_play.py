import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
from env.fsd_env import FSDCarGymEnv
from train_bc_anchor import GaussianPolicy, SpicedBrainInference
class ActorCritic(nn.Module):
    def __init__(self, policy, input_dim=15):
        super(ActorCritic, self).__init__()
        self.actor = policy
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def get_value(self, x):
        return self.critic(x)
    def get_action_and_value(self, x, action=None):
        mean, log_std = self.actor(x)
        std = torch.exp(log_std)
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
            action[:, 0:1] = torch.clamp(action[:, 0:1], 0.0, 1.0)
            action[:, 1:2] = torch.clamp(action[:, 1:2], -1.0, 1.0)
        log_prob = probs.log_prob(action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)
        value = self.critic(x)
        return action, log_prob, entropy, value

def evaluate_actor(actor, env, device, episodes=3):
    actor.eval()
    rewards = []
    action_v_values = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done and steps < env.max_steps:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _ = actor(obs_t)
                action = torch.cat([
                    torch.clamp(action[:, 0:1], 0.0, 1.0),
                    torch.clamp(action[:, 1:2], -1.0, 1.0),
                ], dim=1)
                act_np = action.cpu().numpy().flatten()
                action_v_values.append(float(act_np[0]))
                obs, reward, terminated, truncated, _ = env.step(act_np)
                total_reward += reward
                done = bool(terminated or truncated)
                steps += 1
            rewards.append(total_reward)
    actor.train()
    mean_reward = float(np.mean(rewards)) if rewards else float("-inf")
    mean_action_v = float(np.mean(action_v_values)) if action_v_values else 0.0
    return mean_reward, mean_action_v

def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPO training starting on device: {device}")
    env = FSDCarGymEnv()
    anchor = GaussianPolicy(input_dim=15).to(device)
    anchor_candidates = [
        os.path.join(REPO_ROOT, "bc_anchor.pth"),
        os.path.join(REPO_ROOT, "model", "bc_anchor.pth"),
        os.path.join(SCRIPT_DIR, "bc_anchor.pth"),
    ]
    anchor_path = next((path for path in anchor_candidates if os.path.exists(path)), anchor_candidates[0])
    if os.path.exists(anchor_path):
        anchor.load_state_dict(torch.load(anchor_path, map_location=device))
        print(f"Loaded frozen 15D BC Anchor Coach from: {anchor_path}")
    else:
        print("Warning: BC Anchor weights missing. Training from scratch without prior.")
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad = False
    ppo_policy = GaussianPolicy(input_dim=15).to(device)
    if os.path.exists(anchor_path):
        ppo_policy.load_state_dict(torch.load(anchor_path, map_location=device))
    ac_model = ActorCritic(ppo_policy, input_dim=15).to(device)
    optimizer = optim.Adam(ac_model.parameters(), lr=1e-4)
    model_dir = os.path.join(REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_actor_path = os.path.join(model_dir, "ppo_best_actor.pth")
    num_iterations = 200
    num_steps = 128
    batch_size = num_steps
    minibatch_size = 32
    update_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    entropy_coef = 0.001
    vf_coef = 2.0
    kl_lambda = 0.075
    eval_interval = 20
    min_eval_action_v = 0.08
    obs_buffer = torch.zeros((num_steps, 15)).to(device)
    action_buffer = torch.zeros((num_steps, 2)).to(device)
    logprob_buffer = torch.zeros(num_steps).to(device)
    reward_buffer = torch.zeros(num_steps).to(device)
    done_buffer = torch.zeros(num_steps).to(device)
    value_buffer = torch.zeros(num_steps).to(device)
    next_obs, _ = env.reset()
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = 0.0
    eval_env = FSDCarGymEnv()
    best_eval_reward, best_eval_action_v = evaluate_actor(ac_model.actor, eval_env, device)
    torch.save(ac_model.actor.state_dict(), best_actor_path)
    print(
        f"Initial BC anchor eval | Reward: {best_eval_reward:.2f} | "
        f"Mean action_v: {best_eval_action_v:.3f} | saved as baseline best"
    )
    print("Beginning Spiced PPO 15D self-play exploration loop...")
    print("-" * 80)
    for iteration in range(1, num_iterations + 1):
        episode_reward = 0.0
        step_reward_sum = 0.0
        for step in range(num_steps):
            obs_buffer[step] = next_obs_t
            done_buffer[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = ac_model.get_action_and_value(next_obs_t.unsqueeze(0))
                value_buffer[step] = value.flatten()
            action_buffer[step] = action.flatten()
            logprob_buffer[step] = logprob.flatten()
            act_np = action.cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            step_reward_sum += reward
            done = float(terminated or truncated)
            reward_buffer[step] = reward
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
            next_done = done
            if done:
                next_obs, _ = env.reset()
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
                episode_reward = step_reward_sum
                step_reward_sum = 0.0
        b_obs = obs_buffer
        b_actions = action_buffer
        b_logprobs = logprob_buffer
        with torch.no_grad():
            next_value = ac_model.get_value(next_obs_t.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(reward_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buffer[t + 1]
                    nextvalues = value_buffer[t + 1]
                delta = reward_buffer[t] + gamma * nextvalues * nextnonterminal - value_buffer[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + value_buffer
        b_advantages = advantages
        b_returns = returns
        for epoch in range(update_epochs):
            b_inds = np.arange(batch_size)
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = ac_model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()
                with torch.no_grad():
                    mean_anchor, log_std_anchor = anchor(b_obs[mb_inds])
                mean_pi, log_std_pi = ac_model.actor(b_obs[mb_inds])
                std_anchor = torch.exp(log_std_anchor)
                std_pi = torch.exp(log_std_pi)
                kl_div = torch.sum(
                    log_std_pi - log_std_anchor + 
                    (std_anchor**2 + (mean_anchor - mean_pi)**2) / (2.0 * std_pi**2) - 0.5,
                    dim=1
                ).mean()
                loss = pg_loss - entropy_coef * entropy.mean() + v_loss * vf_coef + kl_div * kl_lambda
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac_model.parameters(), 1.0)
                optimizer.step()
        if iteration % eval_interval == 0 or iteration == 1:
            eval_reward, eval_action_v = evaluate_actor(ac_model.actor, eval_env, device)
            if eval_action_v >= min_eval_action_v and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_action_v = eval_action_v
                torch.save(ac_model.actor.state_dict(), best_actor_path)
                checkpoint_status = "saved_best"
            elif eval_action_v < min_eval_action_v:
                checkpoint_status = "rejected_velocity_collapse"
            else:
                checkpoint_status = "kept_previous_best"
            print(
                f"Iter: {iteration:03d} | Pg Loss: {pg_loss.item():.4f} | "
                f"Val Loss: {v_loss.item():.4f} | KL Div: {kl_div.item():.4f} | "
                f"Ep Reward: {episode_reward:.2f} | Eval Reward: {eval_reward:.2f} | "
                f"Eval action_v: {eval_action_v:.3f} | {checkpoint_status}"
            )
    print("-------------------------------------------------------------------------------------")
    print("Spiced PPO self-play 15D optimization completed successfully.")
    print("Compiling final 15D PPO Brain for ONNX deployment...")
    ac_model.actor.load_state_dict(torch.load(best_actor_path, map_location=device))
    print(
        f"Exporting guarded best actor from {best_actor_path} "
        f"(reward={best_eval_reward:.2f}, mean_action_v={best_eval_action_v:.3f})"
    )
    ac_model.eval()
    inference_model = SpicedBrainInference(ac_model.actor).to("cpu")
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
    print(f"SUCCESS: Final 15D Spiced PPO Brain exported to {onnx_path}")
    eval_env.close()
    env.close()
if __name__ == "__main__":
    train_ppo()
