import numpy as np
import tritonclient.http as httpclient

# Initialize the Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Create dummy input data
input_data = np.random.rand(14, 128, 128).astype(np.float32)

# Prepare the input tensor
input_tensor = httpclient.InferInput("input", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

# Specify the output tensor
output_tensor = httpclient.InferRequestedOutput("output")

# Perform inference
response = client.infer(
    model_name="satvision_toa_model",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Retrieve and print the output
output_data = response.as_numpy("output")
print("Output shape:", output_data.shape)
