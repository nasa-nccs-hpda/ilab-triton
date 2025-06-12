# gencast_model/1/model.py
import os
import gc
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
        self.dataset_path = os.path.join(
            self.model_dir, "gencast_dataset_source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc")

        # nc_bytes = pb_utils.get_input_tensor_by_name(request, "input_nc").as_numpy()[0]
        # ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_bytes))
        with open(self.dataset_path, "rb") as f:
            example_batch = xarray.load_dataset(f).compute()

        # 2 for input, >=1 for targets
        assert example_batch.dims["time"] >= 3
        for var_name, var in example_batch.data_vars.items():
            print(f"{var_name} shape: {var.shape}")

        # only 1AR training
        #train_itf = data_utils.extract_inputs_targets_forcings(
        #    example_batch, target_lead_times=slice("12h", "12h"),
        #    **dataclasses.asdict(self.task_config))
        #train_inputs, train_targets, train_forcings = train_itf

        # all but 2 timeslices
        eval_itf = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice(
                "12h", f"{(example_batch.dims['time']-2)*12}h"),
            **dataclasses.asdict(self.task_config))
        self.eval_inputs, self.eval_targets, self.eval_forcings = eval_itf

        # GST orig below
        # grads_fn_jitted = jax.jit(grads_fn)
        run_forward_jitted = jax.jit(
            lambda rng, i, t, f: run_forward.apply(
                self.params, self.state, rng, i, t, f)[0]
        )

        # We also produce a pmapped version for running in parallel.
        # This trains the model over each subdataset of the input data
        self.run_forward_pmap = xarray_jax.pmap(
            run_forward_jitted, dim="sample")

        # The number of ensemble members should be a multiple of the number of devices.
        print(f"Number of local devices {len(jax.local_devices())}")
        print("Inputs:  ", self.eval_inputs.dims.mapping)
        print("Targets: ", self.eval_targets.dims.mapping)
        print("Forcings:", self.eval_forcings.dims.mapping)

        self.num_ensemble_members = 1
        rng = jax.random.PRNGKey(0)

        # We fold-in the ensemble member, this way the first N members should always
        # match across different runs which use take the same inputs, regardless of
        # total ensemble size.
        self.rngs = np.stack(
            [jax.random.fold_in(rng, i) for i in range(self.num_ensemble_members)], axis=0)

    def execute(self, requests):
        responses = []
        for request in requests:

            chunks = []

            chunked_pred = rollout.chunked_prediction_generator_multiple_runs(
                predictor_fn=self.run_forward_pmap,
                rngs=self.rngs,
                inputs=self.eval_inputs,
                targets_template=self.eval_targets*np.nan,
                forcings=self.eval_forcings,
                num_steps_per_chunk=1,
                num_samples=self.num_ensemble_members,
                pmap_devices=jax.local_devices())

            for chunk in chunked_pred:
                chunks.append(chunk)
                
            predictions = xarray.combine_by_coords(chunks)
            #result_ds = run_forward(ds, self.params)
            #result_np = result_ds.to_array().values.astype(np.float32)

            #output_tensor = pb_utils.Tensor("output_array", result_np)
            #responses.append(pb_utils.InferenceResponse([output_tensor]))
            # Return dummy array with required shape
            dummy_output = np.ones((8, 1, 1, 181, 360), dtype=np.float32)
            output_tensor = pb_utils.Tensor("output_array", dummy_output)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            responses.append(inference_response)
            # cleanup
            del predictions, chunked_pred, chunks
            gc.collect()
            jax.clear_backends()
        return responses
