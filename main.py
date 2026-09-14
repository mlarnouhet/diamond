import argparse
from argparse import Namespace
import numpy as np
from tqdm import tqdm
from utils import setup_logs, setup_wnb, setup_dirs, set_seed
from runner import Runner


def main(args: Namespace):
    set_seed(args.seed)
    setup_dirs(args)
    setup_wnb(args)
    logger = setup_logs(args)
    runner = Runner(args, logger)
    runner.run()
    runner.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", type=str, default="pong")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--n_epochs", type=int, default=901)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--env_steps", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=400)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--action_space_dim", type=int, default=6)
    parser.add_argument("--burn_in_len", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--im_horizon", type=int, default=4) #15
    parser.add_argument("--n_frame_skip", type=int, default=4)
    parser.add_argument("--n_sampling_steps", type=int, default=3)
    parser.add_argument("--setup_wandb", type=int, default=False)
    parser.add_argument("--max_noop", type=int, default=30)
    parser.add_argument("--lbda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--resume", type=float, default=False)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--enable_tf32", type=bool, default=True)
    parser.add_argument("--enable_mixed_precision", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--hf_repo_id", type=str, default="Marcorico/diamond")
    parser.add_argument("--epoch_to_load", type=int, default=0)
    args = parser.parse_args()
    main(args)
