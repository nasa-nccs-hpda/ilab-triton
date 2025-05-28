import os
import sys
import torch
import subprocess
import numpy as np
import urllib.request
import torch.nn as nn
from datetime import datetime

# ------------------------------------------------------------------------------------
# 1. Automatically download the model from huggingface
# ------------------------------------------------------------------------------------

# setuo url addresses
model_url = "https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-finetuned.ckpt"

# download the model into the filesystem
output_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/" + \
    "weather_aurora_demo_model/1"
model_output_path = os.path.join(
    output_dir, "aurora-0.25-finetuned.ckpt")

# download request execution
if not os.path.exists(model_output_path):
    urllib.request.urlretrieve(model_url, model_output_path)
    print(f"Downloaded to {model_output_path}")
else:
    print(f"Model {model_output_path} already exists.")

# ------------------------------------------------------------------------------------
# 2. Load the checkpoint of the model
# ------------------------------------------------------------------------------------

# download satvision-toa repo dependencies
repo_url = "https://github.com/microsoft/aurora"
repo_target_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/" + \
    "weather_aurora_demo_model/aurora"

if not os.path.exists(repo_target_dir):
    subprocess.run(["git", "clone", repo_url, repo_target_dir], check=True)
    print(f"Cloned {repo_url} into {repo_target_dir}")
else:
    print("Repository already exists.")

# setting up the path and dependencies
sys.path.append(repo_target_dir)

from aurora import Aurora, rollout, AuroraSmall, Batch, Metadata

# model = Aurora(use_lora=False)
# model.load_checkpoint_local(model_output_path, use_lora=False)
model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval().cuda()

# print(f'Attempting to load checkpoint from {config.MODEL.PRETRAINED}')
# checkpoint = torch.load(config.MODEL.PRETRAINED)
# model.load_state_dict(checkpoint['module'])
# print('Successfully applied checkpoint')

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

prediction = model.forward(batch)

print(prediction.surf_vars["2t"])
print(type(prediction))
