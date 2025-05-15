import numpy as np
import tritonclient.http as httpclient

# Initialize the Triton client
client = httpclient.InferenceServerClient(url="https://gs6n-dgx02.sci.gsfc.nasa.gov")

# Create dummy input data
image = np.random.rand(1, 14, 128, 128).astype(np.float32)
mask = np.random.randint(0, 2, size=(1, 32, 32)).astype(bool)

# Prepare input tensors
image_tensor = httpclient.InferInput("image", image.shape, "FP32")
image_tensor.set_data_from_numpy(image)

mask_tensor = httpclient.InferInput("mask", mask.shape, "BOOL")
mask_tensor.set_data_from_numpy(mask)

# Specify output tensor
output_tensor = httpclient.InferRequestedOutput("output")

# Perform inference
response = client.infer(
    model_name="satvision_toa_model",
    inputs=[image_tensor, mask_tensor],
    outputs=[output_tensor]
)

# Retrieve and print output
output = response.as_numpy("output")
print("Output shape:", output.shape)
