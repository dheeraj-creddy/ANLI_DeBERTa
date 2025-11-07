#!/bin/bash
set -e

echo "Checking for model..."
if [ ! -d "/app/models/anli_deberta_model" ]; then
    echo "Downloading model from Google Drive..."
    pip install gdown
    mkdir -p /app/models
    cd /app/models

    # Download using folder ID
    gdown --folder https://drive.google.com/drive/folders/1jhkLh1DYuC3vY5DWm92SaP1Pe_8LmEPu?usp=sharing -O ./

    echo "Model downloaded successfully!"
else
    echo "Model already exists, skipping download."
fi