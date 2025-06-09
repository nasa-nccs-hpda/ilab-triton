import os
import gevent.ssl
import numpy as np
import xarray as xr
from datetime import datetime
from huggingface_hub import snapshot_download
import tritonclient.http as httpclient
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

"""
# Connect to Triton Server
triton_server_url = "gs6n-dgx02.sci.gsfc.nasa.gov"
# Initialize the Triton client
ssl_context_factory = gevent.ssl._create_unverified_context
client = httpclient.InferenceServerClient(
    url=triton_server_url,
    ssl=True,
    insecure=True,
    ssl_context_factory=ssl_context_factory
)
"""
client = httpclient.InferenceServerClient(url="gs6n-dgx02.sci.gsfc.nasa.gov:8000")


# Model name (must match config.pbtxt name and model folder)
model_name = "weather_aurora_demo_model"
yyyy = 2024
mm = 12
dd = 2
t_stamp = f"{yyyy}-{mm:02d}-{dd:02d}"

# dataset url
hf_dataset_repo_id = "nasa-cisto-data-science-group/demo-qefm"

# inference data
inference_data_dir = snapshot_download(repo_id=hf_dataset_repo_id, allow_patterns="*.nc", repo_type='dataset')
inference_data_dir = os.path.join(inference_data_dir, 'aurora')
print(inference_data_dir)


# set dataset
static_file = os.path.join(inference_data_dir, "static.nc")
surf_file = os.path.join(inference_data_dir, f"{t_stamp}-surface-level.nc")
atmos_file = os.path.join(inference_data_dir, f"{t_stamp}-atmospheric.nc")

# read the data
static_vars_ds = xr.open_dataset(static_file)
surf_vars_ds = xr.open_dataset(surf_file)
atmos_vars_ds = xr.open_dataset(atmos_file)

# send the data to triton

"""
# surf vars
print("surf vars")
print("2t", surf_vars_ds["t2m"].values[:2][None].shape)
print("10u", surf_vars_ds["u10"].values[:2][None].shape)
print("10v", surf_vars_ds["v10"].values[:2][None].shape)
print("msl", surf_vars_ds["msl"].values[:2][None].shape)

# static vars
print("static vars")
print("z", static_vars_ds["z"].values[0].shape)
print("slt", static_vars_ds["slt"].values[0].shape)
print("lsm", static_vars_ds["lsm"].values[0].shape)

# atmos vars
print("atmos vars")
print("t", atmos_vars_ds["t"].values[:2][None].shape)
print("u", atmos_vars_ds["u"].values[:2][None].shape)
print("v", atmos_vars_ds["v"].values[:2][None].shape)
print("q", atmos_vars_ds["q"].values[:2][None].shape)
print("z", atmos_vars_ds["z"].values[:2][None].shape)

# metadata vars
print("metadata vars")
print("lat", surf_vars_ds.latitude.values.shape)
print("lon", surf_vars_ds.longitude.values.shape)
print("time", len((surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],)))
print("atmos_levels", len(tuple(int(level) for level in atmos_vars_ds.pressure_level.values)))

"""
# get triton tensor
def get_triton_tensor(name, data, dtype="FP32"):
    print(name, data.shape, dtype)
    tensor = InferInput(name, data.shape, dtype)
    tensor.set_data_from_numpy(data)
    return tensor

# Create input tensors
inputs = []

# get surf vars
surf_netcdf_convention = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
for name in surf_netcdf_convention.keys():
    print(surf_netcdf_convention[name], f"surf_vars_{name}")
    data = surf_vars_ds[surf_netcdf_convention[name]].values[:2][None]
    tensor = get_triton_tensor(f"surf_vars_{name}", data)
    inputs.append(tensor)

# get static vars
static_vars = ["z", "slt", "lsm"]
for name in static_vars:
    print(name, f"static_vars_{name}")
    data = static_vars_ds[name].values[0]
    tensor = get_triton_tensor(f"static_vars_{name}", data)
    inputs.append(tensor)

# get atmos vars
atmos_vars = ["t", "u", "v", "q", "z"]
for name in atmos_vars:
    print(name, f"atmos_vars_{name}")
    data = atmos_vars_ds[name].values[:2][None]
    tensor = get_triton_tensor(f"atmos_vars_{name}", data)
    inputs.append(tensor)

# metadata vars
metadata_vars = ["lat", "lon", "time", "atmos_levels"]
for name in metadata_vars:
    dtype = "FP64"
    if name == "lat":
        data = surf_vars_ds.latitude.values
    elif name == "lon":
        data = surf_vars_ds.longitude.values
    elif name == "time":
        datetime_tuple = (surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],)
        data = np.array([dt.timestamp() for dt in datetime_tuple], dtype=np.float64)
        print("TIME", data.shape)
        # print(surf_vars_ds.valid_time.values, datetime_tuple, timestamps, datetime.fromtimestamp(timestamps[0]))
    elif name == "atmos_levels":
        data = np.array(tuple(int(level) for level in atmos_vars_ds.pressure_level.values))
        dtype = "INT64"
    tensor = get_triton_tensor(f"metadata_{name}", data, dtype=dtype)
    inputs.append(tensor)

steps = 2
data = np.array([steps], dtype=np.int64)
tensor = get_triton_tensor("metadata_steps", data, dtype="INT64")
inputs.append(tensor)

# Define expected outputs
output_names = [
    "surf_vars_2t", "surf_vars_10u", "surf_vars_10v", "surf_vars_msl",
    "static_vars_lsm", "static_vars_z", "static_vars_slt",
    "atmos_vars_z", "atmos_vars_u", "atmos_vars_v", "atmos_vars_t", "atmos_vars_q",
    "metadata_time"
]

#output_names = ["surf_vars_2t"]
outputs = [InferRequestedOutput(name) for name in output_names]

# Send inference request
response = client.infer(model_name, inputs=inputs, outputs=outputs)
print(response)

# Print a few example outputs
for name in output_names:
    output = response.as_numpy(name)
    print(f"{name}: shape = {output.shape}, dtype = {output.dtype}")

