from argparse import Namespace
from typing import List, Dict
import random
import torch

class Dataset():
    def __init__(self, args: Namespace):

        self.capacity = 100_000
        self.size = 0

        self.data: Dict[str, torch.Tensor | List[int]] = {
            "observation": torch.empty(self.capacity, 3, 64, 64, device="cpu"),
            "action": torch.empty(self.capacity, dtype=torch.int64, device="cpu"),
            "reward": torch.empty(self.capacity, device="cpu"),
            "end": torch.empty(self.capacity, dtype=torch.bool, device="cpu"),
            "trajectory_index": [0]
        }
        self.batch_size = args.batch_size
        self.keys = ["observation", "action", "reward", "end"]

    def sample_batch(self, seq_len: int) -> Dict[str, torch.Tensor]:
        batch = []
        for _ in range(self.batch_size):
            traj_id = random.randint(1, len(self.data["trajectory_index"])-1)
            while self.data["trajectory_index"][traj_id] - self.data["trajectory_index"][traj_id-1] < seq_len:
                traj_id = random.randint(1, len(self.data["trajectory_index"])-1)
            start = self.data["trajectory_index"][traj_id-1]
            end = self.data["trajectory_index"][traj_id]
            j = random.randint(0, (end - start) - seq_len)
            sequence = {k: self.data[k][start+j:start+j+seq_len] for k in self.keys}
            batch.append(sequence)
        tensor_batch = {k: torch.stack([sequence[k] for sequence in batch]) for k in self.keys}
        return tensor_batch

    def add(self, trajectory: List[Dict[str, torch.Tensor]], end_of_traj: bool = False):
        if self.size >= self.capacity:
            raise OverflowError("Replay buffer capacity exceeded")

        trajectory = self._preprocess(trajectory)
        for key in self.keys:
            self.data[key][self.size] = trajectory[key]
        self.size += 1

        if end_of_traj:
            self.data["trajectory_index"].append(self.size)

    def save_dataset(self, checkpoint_dir: str = None):
        assert checkpoint_dir is not None, "Dataset checkpoint_dir is None !"
        data_to_save = {
            key: self.data[key][:self.size].clone()
            for key in self.keys if key != "observation"
        }
        observations = self.data["observation"][:self.size].clone()
        data_to_save["observation"] = ((observations + 1) * 127.5).to(torch.uint8)
        data_to_save["trajectory_index"] = self.data["trajectory_index"]
        torch.save(data_to_save, checkpoint_dir)

    def load_dataset(self, checkpoint_dir):
        data = torch.load(checkpoint_dir)
        size = len(data["observation"])

        for key in self.keys:
            self.data[key][:size] = data[key]
        self.data["observation"][:size] = self.data["observation"][:size].float() / 127.5 - 1
        self.data["trajectory_index"] = data["trajectory_index"]
        self.size = size

    def _preprocess(self, trajectory: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        observation = trajectory["observation"].squeeze(0).cpu()
        action = torch.tensor(trajectory["action"], dtype=torch.int64)
        reward = torch.tensor(trajectory["reward"])
        end = torch.tensor(trajectory["end"])
        return {
            "observation": observation,
            "action": action,
            "reward": reward,
            "end": end
        }

