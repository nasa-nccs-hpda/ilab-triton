# jax_client.py

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import tritonclient.http as httpclient

# Connect to Triton server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create dummy input
input_data = np.linspace(0, np.pi, 10).astype(np.float32)
inputs = [httpclient.InferInput("INPUT", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("OUTPUT")]

response = client.infer("jax_demo_model", inputs=inputs, outputs=outputs)
output = response.as_numpy("OUTPUT")
print(output)
