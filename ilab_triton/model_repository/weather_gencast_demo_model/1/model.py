# gencast_model/1/model.py
import os
import numpy as np
import xarray as xr
import triton_python_backend_utils as pb_utils

from saving_gencast import run_forward
from graphcast import checkpoint, gencast

import os
import warnings
import dataclasses
import jax
import numpy as np
import xarray

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import gencast

from saving_gencast import grads_fn, run_forward

warnings.filterwarnings("ignore", category=FutureWarning)


class TritonPythonModel:

    def initialize(self, args):
        self.model_dir = os.path.dirname(__file__)
        self.params_path = os.path.join(
            self.model_dir, "gencast_params_GenCast_1p0deg_Mini_2019.npz")
        with open(self.params_path, "rb") as f:
            # self.params = checkpoint.load(f, gencast.init_params())
            self.ckpt = checkpoint.load(f, gencast.CheckPoint)
            self.params = self.ckpt.params
            self.state = {}
            self.task_config = self.ckpt.task_config
        print("LOADED GENCAST")
        self.dataset_path = '/raid/ilab/ilab-triton/ilab_triton/model_repository/weather_gencast_demo_model/gencast_dataset_source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc'

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
