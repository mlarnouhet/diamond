import argparse
import cv2
from huggingface_hub import hf_hub_download
import torch

HF_REPO_ID = "Marcorico/diamond"
FPS = 15

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_index", type=int)
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1000)
    args = parser.parse_args()
    index = args.trajectory_index
    path = hf_hub_download(HF_REPO_ID, f"run_{args.run_id}/epoch_{args.epoch}/dataset.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    start, end = data["trajectory_index"][index:index + 2]
    frames = data["observation"][start:end].permute(0, 2, 3, 1).numpy()

    output = f"run_{args.run_id}_epoch_{args.epoch}_trajectory_{index}.mp4"
    video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
    for frame in frames:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Saved {output}")
