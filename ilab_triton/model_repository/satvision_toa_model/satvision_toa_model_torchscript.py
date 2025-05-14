import os
import sys
import torch
import subprocess
import numpy as np
import urllib.request

# ---------------------------------------------
# 1. Download model + config from Hugging Face
# ---------------------------------------------
model_url = "https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128/resolve/main/mp_rank_00_model_states.pt"
config_url = "https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128/resolve/main/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml"

output_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/satvision_toa_model/1"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "mp_rank_00_model_states.pt")
config_path = os.path.join(output_dir, "mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml")

if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Downloaded model to {model_path}")
if not os.path.exists(config_path):
    urllib.request.urlretrieve(config_url, config_path)
    print(f"Downloaded config to {config_path}")

# ---------------------------------------------
# 2. Clone repo and import model
# ---------------------------------------------
repo_url = "https://github.com/nasa-nccs-hpda/satvision-toa"
repo_path = "/raid/ilab/ilab-triton/ilab_triton/model_repository/satvision_toa_model/satvision-toa"

if not os.path.exists(repo_path):
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
sys.path.append(repo_path)

from satvision_toa.models.mim import build_mim_model
from satvision_toa.transforms.mim_modis_toa import MimTransform
from satvision_toa.configs.config import _C, _update_config_from_file

# ---------------------------------------------
# 3. Load and initialize model
# ---------------------------------------------
config = _C.clone()
_update_config_from_file(config, config_path)
config.defrost()
config.MODEL.PRETRAINED = model_path
config.TRAIN.USE_CHECKPOINT = False
config.freeze()

model = build_mim_model(config)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['module'])
model.eval()

# ---------------------------------------------
# 4. Clean TorchScript blockers
# ---------------------------------------------
from timm.models.layers import DropPath

def replace_drop_path(module):
    for name, child in module.named_children():
        if isinstance(child, DropPath):
            setattr(module, name, torch.nn.Identity())
        else:
            replace_drop_path(child)

def unwrap_compile(mod):
    if hasattr(mod, '_orig_mod'):
        return unwrap_compile(mod._orig_mod)
    for name, child in mod.named_children():
        unwrapped = unwrap_compile(child)
        if unwrapped is not child:
            setattr(mod, name, unwrapped)
    return mod

replace_drop_path(model)
model = unwrap_compile(model)

# ---------------------------------------------
# 5. Dummy input for scripting
# ---------------------------------------------
dummy_input = torch.randn(1, 14, 128, 128)
dummy_mask = torch.zeros(1, 32, 32)

# ---------------------------------------------
# 6. Export model.encoder as TorchScript
# ---------------------------------------------
scripted_encoder = torch.jit.trace(model.encoder, (dummy_input, dummy_mask))
scripted_encoder.save(os.path.join(output_dir, "encoder.pt"))
print("TorchScript encoder.pt saved successfully.")
