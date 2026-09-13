from argparse import Namespace
from typing import List, Dict
import random
import torch

class Dataset():
    def __init__(self, args: Namespace):
        self.data: Dict[str, torch.Tensor] = {
            "observation": torch.empty(0, 3, 64, 64, device="cpu"),
            "action": torch.empty(0, dtype=torch.int64, device="cpu"),
            "reward": torch.empty(0, device="cpu"),
            "end": torch.empty(0, dtype=torch.bool, device="cpu"),
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
        trajectory = self._preprocess(trajectory)
        self.data.update({k: torch.concatenate([self.data[k], trajectory[k].unsqueeze(0)], dim=0) for k in self.keys})
        if end_of_traj:
            self.data["trajectory_index"].append(len(self.data["observation"]))

    def save_dataset(self, checkpoint_dir: str = None):
        assert checkpoint_dir is not None, "Dataset checkpoint_dir is None !"
        data_to_save = self.data.copy()
        data_to_save["observation"] = ((data_to_save["observation"] + 1) * 127.5).to(torch.uint8)
        torch.save(data_to_save, checkpoint_dir)

    def load_dataset(self, checkpoint_dir):
        self.data = torch.load(checkpoint_dir)
        self.data["observation"] = (self.data["observation"].to(torch.float32) / 127.5 - 1)

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


