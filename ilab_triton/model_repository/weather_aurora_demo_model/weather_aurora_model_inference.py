import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

# Connect to Triton Server
client = InferenceServerClient("localhost:8000")

# Model name (must match config.pbtxt name and model folder)
model_name = "weather_aurora_demo_model"

# Create dummy inputs with correct shapes and dtypes
def create_input(name, shape):
    if name == "metadata_lat":
        data = np.linspace(90, -90, 17).astype(np.float32)
    elif name == "metadata_lon":
        data = np.linspace(0, 360, 33)[:-1].astype(np.float32)
    #
    else:
        data = np.random.rand(*shape).astype(np.float32)
    tensor = InferInput(name, data.shape, "FP32")
    tensor.set_data_from_numpy(data)
    print(tensor)
    print(data)
    return tensor, data  # return both to inspect later if needed

# Define all input tensors
input_defs = {
    "surf_vars_2t": (1, 2, 17, 32),
    "surf_vars_10u": (1, 2, 17, 32),
    "surf_vars_10v": (1, 2, 17, 32),
    "surf_vars_msl": (1, 2, 17, 32),
    "static_vars_lsm": (17, 32),
    "static_vars_z": (17, 32),
    "static_vars_slt": (17, 32),
    "atmos_vars_z": (1, 2, 4, 17, 32),
    "atmos_vars_u": (1, 2, 4, 17, 32),
    "atmos_vars_v": (1, 2, 4, 17, 32),
    "atmos_vars_t": (1, 2, 4, 17, 32),
    "atmos_vars_q": (1, 2, 4, 17, 32),
    "metadata_lat": (17,),
    "metadata_lon": (32,),
    "metadata_time": (1,),
    "metadata_atmos_levels": (4,)
}

# Create input tensors
inputs = []
for name, shape in input_defs.items():
    tensor, _ = create_input(name, shape)
    print(tensor)
    inputs.append(tensor)

# Define expected outputs
output_names = [
    "surf_vars_2t", "surf_vars_10u", "surf_vars_10v", "surf_vars_msl",
    "static_vars_lsm", "static_vars_z", "static_vars_slt",
    "atmos_vars_z", "atmos_vars_u", "atmos_vars_v", "atmos_vars_t", "atmos_vars_q"
]

print(inputs)

outputs = [InferRequestedOutput(name) for name in output_names]

# Send inference request
response = client.infer(model_name, inputs=inputs, outputs=outputs)

# Print a few example outputs
for name in output_names:
    output = response.as_numpy(name)
    print(f"{name}: shape = {output.shape}, dtype = {output.dtype}")

