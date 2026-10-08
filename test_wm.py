from types import SimpleNamespace
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from diffusion import EDMDiffusionModel

HF_REPO_ID = "Marcorico/diamond"
RUN_ID = 1
EPOCH = 750
TRAJECTORY_INDEX = -3

if __name__ == "__main__":
    path = hf_hub_download(HF_REPO_ID, f"run_{RUN_ID}/epoch_{EPOCH}/models.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EDMDiffusionModel(SimpleNamespace(n_sampling_steps=3, batch_size=1, action_space_dim=6)).cuda().eval()
    model.load_state_dict(checkpoint["diffusion_model"])
    del checkpoint

    path = hf_hub_download(HF_REPO_ID, f"run_{RUN_ID}/epoch_{EPOCH}/dataset.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    start = data["trajectory_index"][:-1][TRAJECTORY_INDEX]
    end = data["trajectory_index"][1:][TRAJECTORY_INDEX]
    observations = data["observation"][start:end]
    actions = data["action"][start:end].cuda().unsqueeze(0)
    history = observations[:4].float().cuda().unsqueeze(0) / 127.5 - 1
    torch.manual_seed(42)

    output = f"wm_run_{RUN_ID}_epoch_{EPOCH}_trajectory_{TRAJECTORY_INDEX}.mp4"
    video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 15, (1024, 512))
    with torch.inference_mode():
        for t in range(4, len(observations)):
            prediction = model.sample(history, actions[:, t - 4:t])
            history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)
            real = observations[t].permute(1, 2, 0).numpy()
            generated = ((prediction[0] + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            frame = cv2.resize(np.concatenate([real, generated], axis=1), (1024, 512), interpolation=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, "Real", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Generated", (528, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video.write(frame)
    video.release()
    print(f"Saved {output}")
