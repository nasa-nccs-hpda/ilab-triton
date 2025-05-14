# ilab-triton

ILAB Triton Server Configuration and Benchmarks

## Serving the Server

```bash
docker run --rm --gpus all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /raid/ilab/ilab-triton/ilab_triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models
```
