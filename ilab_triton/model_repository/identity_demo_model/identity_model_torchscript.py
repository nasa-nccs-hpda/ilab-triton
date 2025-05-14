# identity_model.py
import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def forward(self, x):
        return x

model = IdentityModel()
example_input = torch.randn(1, 3, 224, 224)  # Example input
traced_model = torch.jit.trace(model, example_input)
traced_model.save(
    '/raid/ilab/ilab-triton/ilab_triton/model_repository/' + \
    'identity_demo_model/1/model.pt'
)
