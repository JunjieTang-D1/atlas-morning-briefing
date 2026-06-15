#!/bin/bash
# Setup script for trn2.48xlarge instance
# Run this after launching the instance

set -e

echo "=== HASA Setup for Trainium2 ==="

# 1. System packages
sudo apt-get update -y
sudo apt-get install -y python3-pip git htop tmux

# 2. Neuron SDK (2.22+)
pip install --upgrade \
    torch-neuronx \
    neuronx-cc \
    neuronx-distributed \
    transformers-neuronx \
    aws-neuronx-runtime-discovery

# 3. ML dependencies
pip install \
    torch \
    transformers \
    safetensors \
    datasets \
    accelerate \
    numpy \
    tqdm \
    matplotlib \
    pandas

# 4. Benchmark datasets
pip install \
    lm-eval  # for standard benchmarks

# 5. Clone our code
# (assuming SCP or git clone already done)

# 6. Download Llama-3.1-70B weights (requires HF token)
echo "Download model weights separately: huggingface-cli download meta-llama/Llama-3.1-70B-Instruct"

# 7. Verify Neuron devices
neuron-ls
echo "=== Setup complete ==="
