#!/bin/bash

MODEL_DIR="model"
MODEL_FILE="stt_hi_conformer_ctc_medium.nemo"
URL="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/stt_hi_conformer_ctc_medium/1.6.0/files?redirect=true&path=${MODEL_FILE}"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download using wget or curl
if command -v wget &> /dev/null; then
    echo "Downloading model using wget..."
    wget --content-disposition "$URL" -O "${MODEL_DIR}/${MODEL_FILE}"
elif command -v curl &> /dev/null; then
    echo "Downloading model using curl..."
    curl -L "$URL" -o "${MODEL_DIR}/${MODEL_FILE}"
else
    echo "Error: wget or curl not found. Please install one and retry."
    exit 1
fi

echo "Model downloaded to ${MODEL_DIR}/${MODEL_FILE}"
