import gc
import os
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from datetime import datetime
from aurora import Aurora, Batch, Metadata, rollout


class TritonPythonModel:
    def initialize(self, args):
        self.model_dir = os.path.dirname(__file__)
        self.ckpt_path = os.path.join(self.model_dir, "aurora-0.25-finetuned.ckpt")

        self.model = Aurora(use_lora=False)
        self.model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            def get_tensor(name, squeeze=False):
                tensor = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
                return torch.tensor(tensor, device=self.device) if not squeeze else torch.tensor(tensor, device=self.device).squeeze(0)

            # Get surf, static, atmos vars
            surf_vars = {k: get_tensor(f"surf_vars_{k}") for k in ["2t", "10u", "10v", "msl"]}
            static_vars = {k: get_tensor(f"static_vars_{k}", squeeze=True) for k in ["lsm", "z", "slt"]}
            atmos_vars = {k: get_tensor(f"atmos_vars_{k}") for k in ["z", "u", "v", "t", "q"]}

            # Metadata
            lat = get_tensor("metadata_lat", squeeze=True)
            lon = get_tensor("metadata_lon", squeeze=True)
            time_val = get_tensor("metadata_time", squeeze=True)
            levels = get_tensor("metadata_atmos_levels", squeeze=True)
            steps = get_tensor("metadata_steps", squeeze=True).item()

            metadata = Metadata(
                lat=lat,
                lon=lon,
                time=(datetime.fromtimestamp(float(time_val.item())),),
                atmos_levels=tuple(float(l.item()) for l in levels)
            )

            batch = Batch(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                metadata=metadata
            )

            with torch.inference_mode():
                preds = [pred.to("cpu") for pred in rollout(self.model, batch, steps=int(steps))]

            # Stack N-step output for surf and atmos
            def stack_preds(var_group, name):
                my_stack = torch.stack([getattr(pred, var_group)[name] for pred in preds])
                print('stack', name, my_stack.shape)
                return my_stack  # [N, 721, 1440]

            def to_tensor(name, tensor):
                return pb_utils.Tensor(name, tensor.numpy().astype(np.float32))

            out_tensors = []

            for name in ["2t", "10u", "10v", "msl"]:
                out_tensors.append(to_tensor(f"surf_vars_{name}", stack_preds("surf_vars", name)))

            for name in ["z", "u", "v", "t", "q"]:
                out_tensors.append(to_tensor(f"atmos_vars_{name}", stack_preds("atmos_vars", name)))

            for name in ["lsm", "z", "slt"]:
                out_tensors.append(to_tensor(f"static_vars_{name}", preds[0].static_vars[name].cpu()))

            forecast_times = np.array(
                [pred.metadata.time[0].timestamp() for pred in preds],
                dtype=np.float64
            )

            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))

            # cleanup
            del preds, batch
            torch.cuda.empty_cache()
            gc.collect()

        return responses
