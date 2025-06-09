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
        self.model_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(self.model_dir, "aurora-0.25-finetuned.ckpt")

        print("LOADED MODEL", ckpt_path)

        # self.model = Aurora(use_lora=False)
        # self.model.load_checkpoint_local(ckpt_path, use_lora=False)
        # self.model.eval().cuda()
        self.model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
        # self.model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
        self.model.load_checkpoint_local(ckpt_path)
        self.model.cuda()
        self.model.eval()

        print("LOADED MODEL TO CUDA")


    def execute(self, requests):

        responses = []

        for request in requests:
            def get_tensor(name, squeeze=False):
                tensor = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
                print(tensor)
                return torch.tensor(tensor).cuda() if not squeeze else torch.tensor(tensor).squeeze(0).cuda()
            
            print("Received request:", request)
            # Just echo one dummy output for now
            dummy_output = pb_utils.Tensor("surf_vars_2t", np.zeros((1, 2, 721, 1440), dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[dummy_output]))

            # Get surf vars
            #surf_vars = {k: get_tensor(f"surf_vars_{k}") for k in ["2t", "10u", "10v", "msl"]}
            # Get static vars
            #static_vars = {k: get_tensor(f"static_vars_{k}", squeeze=True) for k in ["lsm", "z", "slt"]}
            # Get atmos vars
            #atmos_vars = {k: get_tensor(f"atmos_vars_{k}") for k in ["z", "u", "v", "t", "q"]}

            # Metadata
            #lat = get_tensor("metadata_lat", squeeze=True)
            #lon = get_tensor("metadata_lon", squeeze=True)
            #time_val = get_tensor("metadata_time", squeeze=True)
            #levels = get_tensor("metadata_atmos_levels", squeeze=True)

            #metadata = Metadata(
            #    lat=lat,
            #    lon=lon,
            #    time=(datetime.utcfromtimestamp(float(time_val.item())),),
            #    atmos_levels=tuple(float(l.item()) for l in levels)
            #)

            #batch = Batch(
            #    surf_vars=surf_vars,
            #    static_vars=static_vars,
            #    atmos_vars=atmos_vars,
            #    metadata=metadata
            #)

            #with torch.no_grad():
            #    prediction = self.model(batch)

            # Prepare outputs
            #out_tensors = []
            #def output_tensor(name, data):
            #    arr = data.squeeze(0).cpu().numpy().astype(np.float32)
            #    return pb_utils.Tensor(name, arr)

            # Add all surf_vars, static_vars, atmos_vars to outputs
            #for name in ["2t", "10u", "10v", "msl"]:
            #    out_tensors.append(output_tensor(f"surf_vars_{name}", prediction.surf_vars[name]))
            #for name in ["lsm", "z", "slt"]:
            #    out_tensors.append(output_tensor(f"static_vars_{name}", prediction.static_vars[name]))
            #    print('lololol')
            #for name in ["z", "u", "v", "t", "q"]:
            #    out_tensors.append(output_tensor(f"atmos_vars_{name}", prediction.atmos_vars[name]))

            #responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))

        return responses

