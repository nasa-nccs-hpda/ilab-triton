# identity_client.py
import numpy as np
import tritonclient.http as httpclient

# Connect to Triton server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create dummy input
input_data = np.random.randn(3, 224, 224).astype(np.float32)

# Set up input and output
input_tensor = httpclient.InferInput("INPUT__0", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

# Output name
output_tensor = httpclient.InferRequestedOutput("OUTPUT__0")

# Run inference
response = client.infer(
    model_name="identity_demo_model",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Get prediction
result = response.as_numpy("OUTPUT__0")
print("Output shape:", result.shape)
print("Same as input?", np.allclose(input_data, result))
