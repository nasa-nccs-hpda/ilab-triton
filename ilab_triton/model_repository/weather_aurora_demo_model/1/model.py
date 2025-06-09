# aurora python backend
import os
import sys
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from datetime import datetime

# Add Aurora module to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), "aurora"))
from aurora import Aurora, Batch, Metadata


class TritonPythonModel:
    def initialize(self, args):

        # define model directory
        self.model_dir = os.path.dirname(__file__)
        self.ckpt_path = os.path.join(
            self.model_dir, "aurora-0.25-finetuned.ckpt")

        # The pretrained version does not use LoRA.
        self.model = Aurora(use_lora=False)
        self.model.load_checkpoint(
            "microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        
        # the local function does not parse the use lora option
        # might need to add a commit to their repo to fix this
        # self.model.load_checkpoint_local(ckpt_path)
        self.model.cuda()
        self.model.eval()

    def execute(self, requests):

        responses = []

        for request in requests:
            def get_tensor(name, squeeze=False):
                tensor = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
                print(tensor)
                return torch.tensor(tensor).cuda() if not squeeze else torch.tensor(tensor).squeeze(0).cuda()

            # Get surf vars
            surf_vars = {k: get_tensor(f"surf_vars_{k}") for k in ["2t", "10u", "10v", "msl"]}

            # Get static vars
            static_vars = {k: get_tensor(f"static_vars_{k}", squeeze=True) for k in ["lsm", "z", "slt"]}

            # Get atmos vars
            atmos_vars = {k: get_tensor(f"atmos_vars_{k}") for k in ["z", "u", "v", "t", "q"]}


            # Metadata
            lat = get_tensor("metadata_lat", squeeze=True)
            lon = get_tensor("metadata_lon", squeeze=True)
            time_val = get_tensor("metadata_time", squeeze=True)
            levels = get_tensor("metadata_atmos_levels", squeeze=True)

            metadata = Metadata(
                lat=lat,
                lon=lon,
                time=datetime.fromtimestamp(float(time_val.item())), #(datetime.utcfromtimestamp(float(time_val.item())),),
                atmos_levels=tuple(float(l.item()) for l in levels)
            )

            # print("Received request:", request)
            # Just echo one dummy output for now
            dummy_output = pb_utils.Tensor("surf_vars_2t", np.zeros((1, 2, 721, 1440), dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[dummy_output]))

        return responses

