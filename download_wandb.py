import wandb

api = wandb.Api()

run = api.run("/Matthieu12/Diamond/runs/5i9kzqhi")

for file in run.files():
    file.download(root="./downloaded_run", replace=True)
