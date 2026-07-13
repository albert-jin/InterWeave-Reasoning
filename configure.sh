conda create -n interweave python=3.11 -y && conda activate interweave
pip install --upgrade pip
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver
pip install "datasets<4.0.0"
pip install flash_attn --no-build-isolation

# Install SGLang (0.4.6.post1) patched for InterWeave latent-token injection.
# The directory keeps its historical name for backend compatibility.
cd sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ..

# Install LiveCodeBench (0.1.0) if you want to evaluate on LiveCodeBench.
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
mv LiveCodeBench LiveCodeBench_pkg
cd LiveCodeBench_pkg
pip install -e . --no-deps
cd ..

# Docker
docker build -t interweave-reasoning:cu124-py311 .
docker run --gpus all --rm -it \
  -v "$PWD":/workspace \
  interweave-reasoning:cu124-py311 bash

# Version checks
# python -V
# python -c "import torch; print(torch.__version__, torch.version.cuda)"
# python -c "import sglang; print(sglang.__version__)"
