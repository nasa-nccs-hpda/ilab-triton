# Arguments to pass to the image
ARG VERSION_DATE=25.04
ARG FROM_IMAGE=nvcr.io/nvidia/tritonserver

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive

# Sumlink python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install model dependencies
RUN pip --no-cache-dir install \
    timm \
    huggingface_hub \
    torch torchvision torchaudio  \
    scipy \
    einops \
    pydantic \
    xarray \
    netcdf4 \
    azure-storage-blob \
    "jax[cuda12_local]"

    # --index-url https://download.pytorch.org/whl/cu121
    #git+https://github.com/microsoft/aurora.git@main \
    # git+https://github.com/nasa-nccs-hpda/satvision-toa@main
    # satvision-toa
    # microsoft-aurora
