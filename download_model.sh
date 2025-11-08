#!/bin/bash
set -e

echo "==> Checking for model..."

MODEL_DIR="./models/saved_model_debertav3_anli_r2_tpu"

if [ -d "$MODEL_DIR" ]; then
    echo "==> Model already exists at $MODEL_DIR"
    ls -la "$MODEL_DIR"
else
    echo "==> Model not found. Downloading from Google Drive..."

    # Install gdown
    pip install gdown

    # Create models directory
    mkdir -p ./models
    cd ./models

    # Download using folder ID
    gdown --folder https://drive.google.com/drive/folders/1jhkLh1DYuC3vY5DWm92SaP1Pe_8LmEPu?usp=sharing -O ./

    echo "==> Download completed"

    # Check what was downloaded
    ls -la

    # Go back to root
    cd ..

    if [ -d "$MODEL_DIR" ]; then
        echo "==> Model downloaded successfully!"
        ls -la "$MODEL_DIR"
    else
        echo "==> ERROR: Model not found after download!"
        echo "==> Contents of models/:"
        ls -la ./models/
        exit 1
    fi
fi

echo "==> Ready!"