# For Docker
docker build -t interweave-reasoning:cu124-py311 .

# NVIDIA Container Toolkit is required.
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace \
  interweave-reasoning:cu124-py311 bash
