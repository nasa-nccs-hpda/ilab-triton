import os
import sys
import torch
import subprocess
import urllib.request

# ------------------------------------------------------------------------------------
# 1. Automatically download the model from huggingface
# ------------------------------------------------------------------------------------

# setuo url addresses
model_url = "https://huggingface.co/nasa-cisto-data-science-group/" + \
    "satvision-toa-giant-patch8-window8-128/resolve/main/mp_rank_00_model_states.pt"
config_url = "https://huggingface.co/nasa-cisto-data-science-group/" + \
    "satvision-toa-giant-patch8-window8-128/resolve/main/" + \
    "mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml"

# download the model into the filesystem
output_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/satvision_toa_model/1"
model_output_path = os.path.join(
    output_dir, "mp_rank_00_model_states.pt")
config_output_path = os.path.join(
    output_dir, "mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml")

# download request execution
if not os.path.exists(model_output_path):
    urllib.request.urlretrieve(model_url, model_output_path)
    print(f"Downloaded to {model_output_path}")
else:
    print(f"Model {model_output_path} already exists.")
if not os.path.exists(config_output_path):
    urllib.request.urlretrieve(config_url, config_output_path)
    print(f"Downloaded to {config_output_path}")
else:
    print(f"Config {config_output_path} already exists.")

# ------------------------------------------------------------------------------------
# 2. Load the checkpoint of the model
# ------------------------------------------------------------------------------------

# download satvision-toa repo dependencies
repo_url = "https://github.com/nasa-nccs-hpda/satvision-toa"
repo_target_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/" + \
    "satvision_toa_model/satvision-toa"

if not os.path.exists(repo_target_dir):
    subprocess.run(["git", "clone", repo_url, repo_target_dir], check=True)
    print(f"Cloned {repo_url} into {repo_target_dir}")
else:
    print("Repository already exists.")

# setting up the path and dependencies
sys.path.append(repo_target_dir)
from satvision_toa.models.mim import build_mim_model
from satvision_toa.transforms.mim_modis_toa import MimTransform
from satvision_toa.configs.config import _C, _update_config_from_file

# load model config
config = _C.clone()
_update_config_from_file(config, config_output_path)

config.defrost()
config.MODEL.PRETRAINED = model_output_path
config.freeze()

# load model weights from checkpoint
print('Building un-initialized model')
model = build_mim_model(config)
print('Successfully built uninitialized model')

print(f'Attempting to load checkpoint from {config.MODEL.PRETRAINED}')
checkpoint = torch.load(config.MODEL.PRETRAINED)
model.load_state_dict(checkpoint['module'])
print('Successfully applied checkpoint')
model.eval()

# 3. Quick test with dummy input
# 4. Save the new model

