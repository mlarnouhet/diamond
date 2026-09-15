import logging
from logging import Logger
from typing import Dict, List, Tuple
from pathlib import Path
import shutil
from argparse import Namespace
import random
import numpy as np
import wandb
import torch
from torch.distributions import Categorical


class Metrics():
    def __init__(self, args: Namespace, logger: Logger):
        self.use_wandb = args.setup_wandb
        self.logger = logger
        self.metrics_dict: Dict[List] = {
            "diffusion_loss": [],
            "reward_loss": [],
            "critic_loss": [],
            "policy_loss": [],
            "entropy_loss": [],
            "diffusion_grad_norm": [],
            "reward_grad_norm": [],
            "actor_critic_grad_norm": [],
            "gpu/memory_allocated_gb": [],
            "gpu/memory_reserved_gb": [],
            "gpu/max_memory_allocated_gb": [],
            "gpu/max_memory_reserved_gb": [],
        }

    def reset_metrics(self):
        self.temp_metrics_dict = {
            "diffusion_loss": [],
            "reward_loss": [],
            "critic_loss": [],
            "policy_loss": [],
            "entropy_loss": [],
            "diffusion_grad_norm": [],
            "reward_grad_norm": [],
            "actor_critic_grad_norm": [],
            "gpu/memory_allocated_gb": 0,
            "gpu/memory_reserved_gb": 0,
            "gpu/max_memory_allocated_gb": 0,
            "gpu/max_memory_reserved_gb": 0
        }

    def add_metric(self, loss: float | None, key: str, model=None):
        if model is not None:
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None], norm_type=2.0)
            self.temp_metrics_dict[key + "_grad_norm"].append(grad_norm.item())
        if loss is not  None:
            self.temp_metrics_dict[key + "_loss"].append(loss)

    def log_metrics(self, epoch: int):
        self.logger.info(f"Epoch {epoch} metrics:")
        for key, value in self.temp_metrics_dict.items():
            self.logger.info(f"{key}: {value}")
        if self.use_wandb:
            wandb.log({"epoch": epoch, **self.temp_metrics_dict})

    def update_metrics(self):
        self.temp_metrics_dict = {k: sum(v)/len(v) if isinstance(v, List) else v for (k,v) in self.temp_metrics_dict.items()}
        for (k,v) in self.temp_metrics_dict.items():
            self.metrics_dict[k].append(v)

    def compute_perf_metrics(self):
        self.temp_metrics_dict["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3,
        self.temp_metrics_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3,
        self.temp_metrics_dict["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3,
        self.temp_metrics_dict["gpu/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3,
        

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_dirs(args: Namespace):
    run_dirs = [
        Path(f"checkpoints/run_{args.run_id}"),
        Path(f"samples/run_{args.run_id}"),
    ]

    for run_dir in run_dirs:
        if not args.resume and run_dir.exists():
            shutil.rmtree(run_dir)

        run_dir.mkdir(parents=True, exist_ok=True)

def setup_logs(args: Namespace) -> Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_{args.run_id}.log"

    if not args.resume and log_file.exists():
        log_file.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

    return logging.getLogger(__name__)

def setup_wnb(args: Namespace) -> wandb.Run:
    if args.setup_wandb:
        run = wandb.init(
        project=f"Diamond",
        name=f"run__{args.task_name}_{args.run_id}",
        config=vars(args)
    )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        return run

def preprocess(obs: np.array) -> torch.Tensor:
    obs = np.ascontiguousarray(obs)
    obs = torch.from_numpy(obs).permute(2, 0, 1)
    obs = torch.nn.functional.interpolate(
        obs.unsqueeze(0).float(),
        size=(64, 64),
        mode="bilinear",
    ).squeeze(0)
    obs = obs / 127.5 - 1.0
    obs = obs.to("cuda").unsqueeze(0)
    return obs

def policy_loss_fn(logits: torch.Tensor, actions: torch.Tensor, lambda_returns: torch.Tensor, values: torch.Tensor, eta: float = 1e-3, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor]:
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(actions)        
    advantage = (lambda_returns - values).detach()  
    entropy = dist.entropy()  
    policy_objective = log_prob * advantage    
    entropy_objective = eta * entropy

    if mask is not None:
        policy_objective = policy_objective.masked_fill(~mask, 0.0)
        entropy_objective = entropy_objective.masked_fill(~mask, 0.0)

    policy_loss = -policy_objective.sum(dim=1).mean()
    entropy_loss = -entropy_objective.sum(dim=1).mean()
    return policy_loss, entropy_loss

def display(image: np.array):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def tensor_hist(x: torch.Tensor):
    import matplotlib.pyplot as plt
    plt.hist(x.flatten().detach().cpu().numpy(), bins=100)
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.title("conv1 weight distribution")
    plt.show()

