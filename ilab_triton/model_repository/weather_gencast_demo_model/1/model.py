# gencast_model/1/model.py
import os
import numpy as np
import xarray as xr
import triton_python_backend_utils as pb_utils

from saving_gencast import run_forward
from graphcast import checkpoint, gencast


class TritonPythonModel:

    def initialize(self, args):
        self.params_path = "/raid/ilab/ilab-triton/ilab_triton/model_repository/weather_gencast_demo_model/1/gencast_params_GenCast_1p0deg_Mini_<2019.npz"
        with open(self.params_path, "rb") as f:
            self.params = checkpoint.load(f, gencast.init_params())

    def execute(self, requests):
        responses = []
        for request in requests:
            nc_bytes = pb_utils.get_input_tensor_by_name(request, "input_nc").as_numpy()[0]
            ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_bytes))

            result_ds = run_forward(ds, self.params)
            result_np = result_ds.to_array().values.astype(np.float32)

            output_tensor = pb_utils.Tensor("output_array", result_np)
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses
