import numpy as np
import jax.numpy as jnp
from jax import jit
import triton_python_backend_utils as pb_utils

# JAX model function
@jit
def jax_model(x):
    return jnp.sin(x)

class TritonPythonModel:
    def initialize(self, args):
        print("JAX model initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_array = input_tensor.as_numpy()

            # Run the JAX model
            # output_array = jax_model(input_array).astype(np.float32)
            output_array = np.array(jax_model(input_array)).astype(np.float32)

            output_tensor = pb_utils.Tensor("OUTPUT", output_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        print("Cleaning up JAX model")

