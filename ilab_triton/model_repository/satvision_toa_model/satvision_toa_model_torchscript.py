"""
import torch
from satvision_toa import SatVisionTOAModel  # your actual class
from deepspeed.runtime.checkpoint_engine import load_torch_checkpoint

# Instantiate model architecture
model = SatVisionTOAModel()

# Load the DeepSpeed checkpoint
checkpoint = torch.load("mp_rank_00_model_states.pt", map_location="cpu")

# Extract model weights
model.load_state_dict(checkpoint['module'], strict=False)

# Set to eval mode
model.eval()
"""
import os
import urllib.request

# 1. Automatically download the model from huggingface
model_url = "https://huggingface.co/nasa-cisto-data-science-group/" + \
    "satvision-toa-giant-patch8-window8-128/resolve/main/mp_rank_00_model_states.pt"
config_url = "https://huggingface.co/nasa-cisto-data-science-group/" + \
    "satvision-toa-giant-patch8-window8-128/resolve/main/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml"

# download the model into the filesystem
output_dir = "/raid/ilab/ilab-triton/ilab_triton/model_repository/satvision_toa_model/1/"
model_output_path = os.path.join(output_dir, "mp_rank_00_model_states.pt")
config_output_path = os.path.join(output_dir, "mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml")

# download request execution
urllib.request.urlretrieve(model_url, model_output_path)
print(f"Downloaded to {model_output_path}")
urllib.request.urlretrieve(config_url, config_output_path)
print(f"Downloaded to {config_output_path}")


# 2. Load the checkpoint of the model
# 3. Quick test with dummy input
# 4. Save the new model

