#!/bin/bash

# Purpose: Update all HF_TOKEN files with a user's own token.
# Usage:
#   ./hf_token.sh <your_hf_token>

# Check if an argument was provided.
if [ -z "$1" ]; then
  echo "❌ Error: Missing argument."
  echo "Usage: $0 <your_hf_token>"
  exit 1
fi

# Store the string argument.
input_string="$1"

# Replace all occurrences of HF_TOKEN="YOUR_HF_TOKEN" with user's token.
sed -i "s/HF_TOKEN=\"<YOUR_HF_TOKEN>\"/HF_TOKEN=\"$input_string\"/" \
./mixtral_8x7b/import_mixtral_8x7b.slurm \
./llama31_8b/import_llama31_8b.slurm