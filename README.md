# ilab-triton

ILAB Triton Server Configuration and Benchmarks

## Serving the Triton Server

```bash
docker run --rm --gpus all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /raid/ilab/ilab-triton/ilab_triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models
```

## Individual Models Setup

Setup of the individual models available in the system are documented below.

### Indentity Model Setup

#### 1. Generate torchscript model

```bash
docker run --rm --gpus all --shm-size=900g \
  -v /home/jacaraba:/home/jacaraba \
  -v /raid/jacaraba:/raid/jacaraba \
  -v /raid/ilab:/raid/ilab \
  nasanccs/hyperself:latest \
  python /raid/ilab/ilab-triton/ilab_triton/model_repository/identity_demo_model/identity_model_torchscript.py
```

#### 2. Run inference example

You might want to run the latest version of the container:

```bash
docker pull nasanccs/hyperself:latest
```

Proceed to run inference:

```bash
docker run --rm --gpus all --shm-size=900g --network=host \
  -v /home/jacaraba:/home/jacaraba \
  -v /raid/jacaraba:/raid/jacaraba \
  -v /raid/ilab:/raid/ilab \
  nasanccs/hyperself:latest \
  python /raid/ilab/ilab-triton/ilab_triton/model_repository/identity_demo_model/identity_model_inference.py
```

### SatVision-TOA Model Setup

#### 1. Generate torchscript model

```bash
```

#### 2. Run inference example

```bash
```
