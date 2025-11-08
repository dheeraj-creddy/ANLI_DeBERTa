#!/bin/bash
set -e

echo "=========================================="
echo "ANLI Model Downloader (Local Version)"
echo "=========================================="

# Configuration
MODEL_DIR="./models/saved_model_debertav3_anli_r2_tpu"
GDRIVE_FOLDER_ID="https://drive.google.com/drive/folders/1jhkLh1DYuC3vY5DWm92SaP1Pe_8LmEPu?usp=sharing" # Replace with your actual folder ID -O ./

# Check if model already exists
if [ -d "$MODEL_DIR" ]; then
    echo ""
    echo "✓ Model already exists at: $MODEL_DIR"
    echo ""
    echo "Contents:"
    ls -lh "$MODEL_DIR"
    echo ""
    echo "If you want to re-download, delete the folder first:"
    echo "  rm -rf $MODEL_DIR"
    exit 0
fi

echo ""
echo "Model not found. Starting download..."
echo ""

# Install gdown if not already installed
echo "Installing gdown..."
pip install -q gdown

# Create models directory
mkdir -p ./models

# Download from Google Drive
echo ""
echo "Downloading from Google Drive..."
echo "Folder ID: $GDRIVE_FOLDER_ID"
echo ""

cd ./models
gdown --folder "$GDRIVE_FOLDER_ID"

# Go back to root
cd ..

# Verify download
echo ""
echo "=========================================="
if [ -d "$MODEL_DIR" ]; then
    echo "✓ SUCCESS! Model downloaded to: $MODEL_DIR"
    echo ""
    echo "Model files:"
    ls -lh "$MODEL_DIR"
    echo ""
    FILE_COUNT=$(ls "$MODEL_DIR" | wc -l)
    echo "Total files: $FILE_COUNT"
    echo ""

    # Check for required files
    REQUIRED_FILES=("config.json" "model.safetensors" "tokenizer_config.json")
    echo "Checking required files:"
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$MODEL_DIR/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file MISSING!"
        fi
    done
else
    echo "✗ FAILED! Model directory not created."
    echo ""
    echo "Contents of ./models/:"
    ls -lh ./models/
fi
echo "=========================================="