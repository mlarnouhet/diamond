from argparse import Namespace
import os
from logging import Logger
from typing import Tuple, List, Dict
from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import gymnasium as gym
from huggingface_hub import HfApi
import ale_py
from models import ActorCriticNetwork, RewardEndNetwork
from diffusion import EDMDiffusionModel
from dataset import Dataset
from utils import preprocess, policy_loss_fn, Metrics


class Runner(nn.Module):
    def __init__(self, args: Namespace, logger: Logger):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.wandb_run = None
        self.wandb_run_id = ""
        self.run_id = args.run_id
        self.env_id = args.env_id
        self.n_epochs = args.n_epochs
        self.env_steps = args.env_steps
        self.train_steps = args.train_steps
        self.burn_in_len = args.burn_in_len
        self.im_horizon = args.im_horizon
        self.n_frame_skip = args.n_frame_skip
        self.L = args.L
        self.batch_size = args.batch_size
        self.max_noop = args.max_noop
        self.lbda = args.lbda
        self.gamma = args.gamma
        self.entropy_weight = args.entropy_weight
        self.save_interval = args.save_interval
        self.hf_repo_id = args.hf_repo_id
        self.log_every = args.log_every

        self.mixed_prec_dtype = torch.bfloat16 
        self.start_epoch = 1 if args.debug else 1
        self.metrics = Metrics(args, logger)

        if args.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._setup_env(args)
        self._init_models(args)
        self._setup_optims(args)
        self.dataset = Dataset(args)
        self._collection_state = None
        self._collection_has_open_trajectory = False
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        if args.resume:
            self._load_checkpoint(args)
            logger.info(f"Resuming run {args.run_id} on env {args.env_id} at epoch {self.start_epoch}")
            for key, value in vars(args).items():
                self.logger.info(f"{key}: {value}")
        else:
            self.wandb_run = self._setup_wnb(args)
            self.wandb_run_id = self.wandb_run.id
            logger.info(f"Starting run {args.run_id} on env {args.env_id}")
            for key, value in vars(args).items():
                self.logger.info(f"{key}: {value}")

        self.logger.info("Using device {self.device}")

    def run(self):
        for epoch in tqdm(range(self.start_epoch, self.n_epochs), desc="Training"):
            collect_steps = 10000 if (epoch == 0) else self.env_steps
            wm_steps = 10000 if (epoch == 0) else self.train_steps
            ac_steps = 5000 if (epoch == 0) else self.train_steps
            self.metrics.reset_metrics()

            self._collect_experience(collect_steps)
            for i in tqdm(range(wm_steps), desc="Training diffusion model"):
                 self._update_diffusion_model(log_grad_norm = ((i % 50) == 0))
            for i in tqdm(range(wm_steps), desc="Training reward model"):
                self._update_reward_end_model(log_grad_norm = ((i % 50) == 0))
            for i in tqdm(range(ac_steps), desc="Training actor critic"):
                self._update_actor_critic(log_grad_norm = ((i % 50) == 0))

            if ((epoch % self.log_every) == 0):
                self.metrics.compute_perf_metrics()
                self.metrics.update_metrics()
                self.metrics.log_metrics(epoch)
            
            if ((epoch % self.save_interval) == 0) and (epoch > self.start_epoch):
                self._save_checkpoint(epoch)
                 
    def _collect_experience(self, collect_steps=100):
        if self._collection_has_open_trajectory:
            self.dataset.data["trajectory_index"].pop()
            self._collection_has_open_trajectory = False

        for step in tqdm(range(collect_steps), desc="Collecting experience"):
            if self._collection_state is None:
                h = torch.zeros((1, 512), device=self.device)
                c = torch.zeros((1, 512), device=self.device)
                observation = self._reset_env()
            else:
                observation, h, c = self._collection_state

            observation = preprocess(observation)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
                    logits, _, h, c = self.actor_critic_model(observation, h, c)
            action = Categorical(logits=logits).sample().item()

            next_observation, reward, end, truncated, _ = self._step_env(action)
            episode_done = end or truncated
            end_of_traj = episode_done or step == collect_steps - 1

            self.dataset.add({
                "observation": observation,
                "action": action,
                "reward": reward,
                "end": end
            }, end_of_traj=end_of_traj)

            self._collection_state = (
                None if episode_done else (next_observation, h.detach(), c.detach())
            )
            self._collection_has_open_trajectory = end_of_traj and not episode_done

    def _step_env(self, action: int):
        total_reward = 0.0
        previous_frame = None
        last_frame = None
        for _ in range(self.n_frame_skip):
            frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            previous_frame, last_frame = last_frame, frame.copy()
            if terminated or truncated:
                break

        if previous_frame is not None:
            last_frame = np.maximum(previous_frame, last_frame)
        return last_frame, total_reward, terminated, truncated, info

    def _update_diffusion_model(self, log_grad_norm = False):
        batch = self.dataset.sample_batch(self.L+1)
        past_frames, target, actions = self._process_batch(batch, mode="diffusion")
        log_sigma = -0.4 + 1.2 * torch.randn((self.batch_size), device=self.device)
        sigma_tau = log_sigma.exp()
        noised_target = target + sigma_tau.view(-1, 1, 1, 1) * torch.randn_like(target)
        with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
            prediction, c_out = self.diffusion_model(noised_target, past_frames, sigma_tau, actions)
            loss = self.mse_loss(prediction / c_out, target / c_out)
        self.diffusion_optimizer.zero_grad()
        loss.backward()
        self.metrics.add_metric(loss, "diffusion", self.diffusion_model if log_grad_norm else None)
        self.diffusion_optimizer.step()

    def _update_reward_end_model(self, log_grad_norm = False):
        batch = self.dataset.sample_batch(self.im_horizon+self.burn_in_len)
        burn_in_obs, obs, burn_in_actions, actions, target_reward, target_end = self._process_batch(batch, mode="reward")

        loss = 0
        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
                h, c = self._burn_in(self.reward_end_model, burn_in_obs, burn_in_actions)

        with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
            for i in range(self.im_horizon):
                pred_reward, pred_end, h, c = self.reward_end_model(obs[:, i, :, :, :], actions[:, i], h, c)
                loss += self.ce_loss(pred_reward, target_reward[:, i]) + self.bce_loss(pred_end.squeeze(1), target_end[:, i])

        self.reward_optimizer.zero_grad()
        loss.backward()
        self.metrics.add_metric(loss, "reward", self.reward_end_model if log_grad_norm else None)
        self.reward_optimizer.step()

    def _update_actor_critic(self, log_grad_norm = False):
        batch = self.dataset.sample_batch(self.burn_in_len+1)
        burn_in_obs, obs, burn_in_actions = self._process_batch(batch, mode="actor_critic")
        past_frames = burn_in_obs.detach().clone()
        past_actions = burn_in_actions.detach().clone()
        pred_reward_list = []
        pred_end_list = []
        value_list = []
        logits_list = []
        action_list = []
        loss_masks = []
        alive = torch.ones(obs.size(0), dtype=torch.bool, device=obs.device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
                h_1, c_1 = self._burn_in(self.actor_critic_model, burn_in_obs)
                h_2, c_2 = self._burn_in(self.reward_end_model, burn_in_obs, burn_in_actions)

        with torch.autocast("cuda", dtype=self.mixed_prec_dtype):
            for i in range(self.im_horizon):
                loss_masks.append(alive.clone())
                logits, value, h_1, c_1 = self.actor_critic_model(obs, h_1, c_1, compute_value=True)
                action = Categorical(logits=logits).sample()
                with torch.no_grad():
                    reward_logits, end_logits, h_2, c_2 = self.reward_end_model(obs, action, h_2, c_2)
                    pred_reward = Categorical(logits=reward_logits).sample() - 1
                    pred_end = Bernoulli(logits=end_logits).sample()

                pred_reward_list.append(pred_reward)
                pred_end_list.append(pred_end)
                value_list.append(value)
                logits_list.append(logits)
                action_list.append(action)

                alive = alive & ~pred_end.squeeze(1).bool()
                if not alive.any():
                    break

                with torch.no_grad():
                    past_frames = torch.concatenate([past_frames[:, 1:, :, :, :], obs.unsqueeze(1)], dim=1)
                    past_actions = torch.concatenate([past_actions[:, 1:], action.unsqueeze(1)], dim=1)
                    obs = self.diffusion_model.sample(past_frames, past_actions)

            with torch.no_grad():
                if alive.any():
                    _, bootstrap_value, _, _ = self.actor_critic_model(obs, h_1, c_1, compute_value=True)
                    bootstrap_value = bootstrap_value.masked_fill(~alive.unsqueeze(1), 0.0)
                else:
                    bootstrap_value = torch.zeros_like(value_list[-1])
                value_list.append(bootstrap_value)

                lambda_list = [bootstrap_value]
                for t in reversed(range(len(pred_reward_list))):
                    lambda_return = pred_reward_list[t].unsqueeze(1) + self.gamma*(1-pred_end_list[t]) * ((1 - self.lbda) * value_list[t+1] + self.lbda * lambda_list[-1])
                    lambda_list.append(lambda_return.masked_fill(~loss_masks[t].unsqueeze(1), 0.0))
                lambdas = torch.concatenate(list(reversed(lambda_list[1:])), dim=1)

            values = torch.concatenate(value_list[:-1], dim=1)
            action_logits = torch.stack(logits_list, dim=1)
            actions = torch.stack(action_list, dim=1)
            loss_mask = torch.stack(loss_masks, dim=1)
            critic_loss = ((values - lambdas)**2).masked_fill(~loss_mask, 0.0).sum(dim=-1).mean()
            policy_loss, entropy_loss = policy_loss_fn(action_logits, actions, lambdas, values, self.entropy_weight, mask=loss_mask)
            loss = policy_loss + entropy_loss + critic_loss

        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        self.metrics.add_metric(None, "actor_critic", self.actor_critic_model if log_grad_norm else None)
        self.metrics.add_metric(policy_loss, "policy")
        self.metrics.add_metric(critic_loss, "critic")
        self.metrics.add_metric(entropy_loss, "entropy")
        self.actor_critic_optimizer.step()

    def _process_batch(self, batch: torch.Tensor, mode: str | None = None) -> Tuple[torch.Tensor | None]:
        match mode:
            case "diffusion":
                past_frames = batch["observation"][:, :-1, :, :, :].to(self.device)
                target = batch["observation"][:, -1, :, :, :].to(self.device)
                actions = batch["action"][:, :-1].to(self.device)
                return past_frames, target, actions
            case "reward":
                burn_in_obs = batch["observation"][:, :self.burn_in_len, :, :, :].to(self.device)
                burn_in_actions = batch["action"][:, :self.burn_in_len].to(self.device)
                obs = batch["observation"][:, self.burn_in_len:, :, :, :].to(self.device)
                actions = batch["action"][:, self.burn_in_len:].to(self.device)
                target_reward = batch["reward"][:, self.burn_in_len:].sign().long().to(self.device) + 1
                target_end = batch["end"][:, self.burn_in_len:].float().to(self.device)
                return burn_in_obs, obs, burn_in_actions, actions, target_reward, target_end
            case "actor_critic":
                burn_in_obs = batch["observation"][:, :-1, :, :, :].to(self.device)
                obs = batch["observation"][:, -1, :, :, :].to(self.device)
                burn_in_actions = batch["action"][:, :-1].to(self.device)
                return burn_in_obs, obs, burn_in_actions
            case _:
                raise Exception(f"Mode {mode} not supported")

    def _setup_env(self, args: Namespace):
        gym.register_envs(ale_py)
        self.env = gym.make(args.env_id, frameskip=1)
        self.env.reset(seed=args.seed)

    def _init_models(self, args: Namespace):
        assert args.action_space_dim == self.env.action_space.n, f"action space dim is {args.action_space_dim} =! {self.env.action_space.n}"
        self.actor_critic_model = ActorCriticNetwork(args)
        self.reward_end_model = RewardEndNetwork(args)
        self.diffusion_model = EDMDiffusionModel(args)

        self.actor_critic_model.to(self.device)
        self.reward_end_model.to(self.device)
        self.diffusion_model.to(self.device)

        if not args.debug:
            self._compile_models()

    def _setup_optims(self, args: Namespace):
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=args.lr,
            eps=args.eps,
            weight_decay=args.wd,
        )

        self.reward_optimizer = torch.optim.AdamW(
            self.reward_end_model.parameters(),
            lr=args.lr,
            eps=args.eps,
            weight_decay=args.wd,
        )

        self.actor_critic_optimizer = torch.optim.AdamW(
            self.actor_critic_model.parameters(),
            lr=args.lr,
            eps=args.eps,
            weight_decay=0.0,
        )

    def _burn_in(self, model: nn.Module, x: torch.Tensor, actions: torch.Tensor = None) -> torch.Tensor:
        h = torch.zeros((self.batch_size, 512), device=self.device)
        c = torch.zeros((self.batch_size, 512), device=self.device)
        if actions is not None:
            h, c = model.burn_in(x, actions, h, c)
        else:
            h, c = model.burn_in(x, h, c)
        return h, c

    def _reset_env(self) -> torch.Tensor:
        observation, _ = self.env.reset()
        n = self.env.np_random.integers(1, self.max_noop+1)
        for _ in range(n):
            observation, _, terminated, truncated, _ = self.env.step(0)  
            if terminated or truncated: 
                observation, _ = self.env.reset()
        return observation

    def _save_checkpoint(self, epoch: int):
            self.logger.info(f"Saving checkpoint for run {self.run_id} on env {self.env_id} at epoch {epoch}")
            checkpoint_dir = f"checkpoints/run_{self.run_id}/checkpoint_epoch_{epoch}/models.pt"
            dataset_dir = f"checkpoints/run_{self.run_id}/checkpoint_epoch_{epoch}/dataset.pt"
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "diffusion_model": self.diffusion_model.state_dict(),
                "reward_end_model": self.reward_end_model.state_dict(),
                "actor_critic": self.actor_critic_model.state_dict(),
                "diffusion_optimizer": self.diffusion_optimizer.state_dict(),
                "reward_optimizer": self.reward_optimizer.state_dict(),
                "actor_critic_optimizer": self.actor_critic_optimizer.state_dict(),
                "metrics_dict": self.metrics.metrics_dict,
                "wandb_run_id": self.wandb_run_id
            }, checkpoint_dir)

            self.dataset.save_dataset(dataset_dir)
            self.logger.info(f"Saved checkpoint {checkpoint_dir} and dataset {dataset_dir} at epoch {epoch}")
            self._save_to_hf(epoch, checkpoint_dir, dataset_dir)

    def _load_checkpoint(self, args: Namespace):
        checkpoint_main_dir = f"checkpoints/run_{self.run_id}/checkpoint_epoch_{args.epoch_to_load}"
        checkpoint_dir = os.path.join(checkpoint_main_dir, "models.pt")
        dataset_dir = os.path.join(checkpoint_main_dir, "dataset.pt")

        self.logger.info(f"Loading checkpoint {checkpoint_dir} and dataset {dataset_dir}")
        checkpoint = torch.load(checkpoint_dir)
        epoch = checkpoint["epoch"]
        self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        self.actor_critic_model.load_state_dict(checkpoint["actor_critic"])
        self.reward_end_model.load_state_dict(checkpoint["reward_end_model"])
        self.diffusion_optimizer.load_state_dict(checkpoint["diffusion_optimizer"])
        self.reward_optimizer.load_state_dict(checkpoint["reward_optimizer"])
        self.actor_critic_optimizer.load_state_dict(checkpoint["actor_critic_optimizer"])
        self.metrics.metrics_dict = checkpoint["metrics_dict"]
        self.wandb_run_id = checkpoint["wandb_run_id"]

        self.dataset.load_dataset(dataset_dir)
        self.start_epoch = epoch + 1

        self.wandb_run = self._setup_wnb(args)
        self.wandb_run_id = self.wandb_run.id
        self.logger.info(f"Loaded checkpoint for run {self.run_id} on env {self.env_id} at epoch {epoch}")

    def _save_to_hf(self, epoch: int, checkpoint_dir: str, dataset_dir: str):
        api = HfApi()
        api.create_repo(repo_id=self.hf_repo_id, private=True, exist_ok=True)
        upload_future_model = api.upload_file(
            repo_id=self.hf_repo_id,
            path_or_fileobj=checkpoint_dir,
            path_in_repo=f"run_{self.run_id}/epoch_{epoch}/models.pt",
            commit_message=f"Checkpoint: run {self.run_id}, epoch {epoch}",
            run_as_future=True,
        )
        upload_future_data = api.upload_file(
            repo_id=self.hf_repo_id,
            path_or_fileobj=dataset_dir,
            path_in_repo=f"run_{self.run_id}/epoch_{epoch}/dataset.pt",
            commit_message=f"Checkpoint: run {self.run_id}, epoch {epoch}",
            run_as_future=True,
        )
        upload_future_model.result()
        upload_future_data.result()

    def _setup_wnb(self, args: Namespace) -> wandb.Run | None:
        if args.setup_wandb:
            if args.resume:
                run = wandb.init(
                project=f"Diamond",
                entity=os.getenv("WANDB_ENTITY"),
                name=f"run__{args.task_name}_{args.run_id}",
                id=self.wandb_run_id,
                config=vars(args),
                resume="must",
            )
            else:
                run = wandb.init(
                project=f"Diamond",
                name=f"run__{args.task_name}_{args.run_id}",
                config=vars(args),
            )
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            return run
        else:
            return None

    def _compile_models(self):
        for model in (
            self.actor_critic_model,
            self.reward_end_model,
            self.diffusion_model,
        ):
            model.compile()

        self.actor_critic_model.burn_in = torch.compile(
            self.actor_critic_model.burn_in
        )
        self.reward_end_model.burn_in = torch.compile(
            self.reward_end_model.burn_in
        )
        self.diffusion_model.sample = torch.compile(
            self.diffusion_model.sample
        )

    def finish(self):
        self.env.close()
        if self.wandb_run is not None:
            self.run.finish()
