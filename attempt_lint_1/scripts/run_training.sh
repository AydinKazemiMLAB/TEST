#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define paths relative to the project root
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
MAIN_SCRIPT="$PROJECT_ROOT/src/main.py"
PREPROCESS_SCRIPT="$PROJECT_ROOT/scripts/preprocess_data.py"

# Default configuration file
CONFIG_FILE="$PROJECT_ROOT/configs/base_model.yaml"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$PROJECT_ROOT/$2"
            shift # past argument
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift # past argument or value
done

echo "Using configuration file: $CONFIG_FILE"

# Step 1: Create necessary directories if they don't exist
echo "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/outputs/logs/training_runs"
mkdir -p "$PROJECT_ROOT/outputs/models/checkpoints"
mkdir -p "$PROJECT_ROOT/outputs/models/final_models"
mkdir -p "$PROJECT_ROOT/data/raw/wmt14" # Example for WMT14, adjust as needed
mkdir -p "$PROJECT_ROOT/data/processed/wmt14_en_de" # Example for processed data, adjust as needed
mkdir -p "$PROJECT_ROOT/data/processed/wmt14_en_de_big"
mkdir -p "$PROJECT_ROOT/data/processed/wmt14_en_fr_big"
mkdir -p "$PROJECT_ROOT/data/raw/wsj"
mkdir -p "$PROJECT_ROOT/data/processed/wsj_parsing"
mkdir -p "$PROJECT_ROOT/outputs/models/checkpoints_parsing"


# Step 2: Run data preprocessing (tokenizer training, etc.)
# This step is crucial for creating the tokenizer models and vocab files.
# You might need to manually place raw data files in data/raw/wmt14/ etc.
echo "Running data preprocessing..."
python "$PREPROCESS_SCRIPT" --config "$CONFIG_FILE"

# Step 3: Run the main training script
echo "Starting Transformer training..."
python "$MAIN_SCRIPT" --config "$CONFIG_FILE"

echo "Training script finished."