from types import SimpleNamespace
from pathlib import Path
import re

import ale_py
import gymnasium as gym
from huggingface_hub import HfApi, hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import ActorCriticNetwork


HF_REPO_ID = "Marcorico/diamond"
RUN_ID = 1
NUM_EPISODES = 10
ENV_ID = "ALE/Pong-v5"
SEED = 42
N_FRAME_SKIP = 4
MAX_NOOP = 30


if __name__ == "__main__":
    files = HfApi().list_repo_files(HF_REPO_ID)
    pattern = rf"run_{RUN_ID}/epoch_(\d+)/models\.pt$"
    checkpoints = sorted((int(m[1]), str(f)) for f in files
                         if (m := re.fullmatch(pattern, f)) and int(m[1]) <= 1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(f"validation/run_{RUN_ID}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gym.register_envs(ale_py)
    env = gym.make(ENV_ID, frameskip=1)
    results = []
    model = ActorCriticNetwork(SimpleNamespace(action_space_dim=env.action_space.n, burn_in_len=4,)).to(device).eval()
    with torch.inference_mode():
        for epoch, filename in checkpoints:
            path = hf_hub_download(HF_REPO_ID, filename)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["actor_critic"])
            del checkpoint
            rewards, lengths = [], []
            for episode in range(NUM_EPISODES):
                torch.manual_seed(SEED + episode)
                obs, _ = env.reset(seed=SEED + episode)
                noops = env.np_random.integers(1, MAX_NOOP + 1)
                for _ in range(noops):
                    obs, _, terminated, truncated, _ = env.step(0)
                    if terminated or truncated:
                        obs, _ = env.reset()
                h = torch.zeros((1, model.lstm.hidden_size), device=device)
                c = torch.zeros_like(h)
                total_reward, length = 0.0, 0
                while True:
                    x = torch.from_numpy(np.ascontiguousarray(obs)).permute(2, 0, 1)[None].float()
                    x = torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear")
                    logits, _, h, c = model((x / 127.5 - 1).to(device), h, c)
                    action = torch.distributions.Categorical(logits=logits).sample().item()
                    frames = []
                    for _ in range(N_FRAME_SKIP):
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += float(reward)
                        frames.append(obs)
                        if terminated or truncated:
                            break
                    obs = np.maximum.reduce(frames[-2:])
                    length += 1
                    if terminated or truncated:
                        break
                rewards.append(total_reward)
                lengths.append(length)
                print(f"Epoch {epoch}, trajectory {episode + 1}: reward={total_reward:g}, "
                      f"length={length} actions{' (truncated)' if truncated else ''}", flush=True)

            mean_reward, mean_length = np.mean(rewards), np.mean(lengths)
            results.append((epoch, mean_reward, mean_length))
            print(f"Epoch {epoch}: mean reward={mean_reward:.3f}, "
                  f"mean length={mean_length:.1f} actions", flush=True)
    env.close()
    np.savetxt(output_dir / "rewards.csv", results, delimiter=",",
               header="epoch,mean_reward,mean_length", comments="", fmt=["%d", "%.6f", "%.6f"])

    plt.plot([row[0] for row in results], [row[1] for row in results], "o-")
    plt.xlabel("Checkpoint epoch")
    plt.ylabel("Average episode reward")
    plt.title(f"{ENV_ID} — run {RUN_ID}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "average_reward_vs_epoch.png", dpi=150)
    print(f"Scores and plot saved to {output_dir}")
    plt.show()
    plt.close()
