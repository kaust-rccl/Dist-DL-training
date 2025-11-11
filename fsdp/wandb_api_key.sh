#!/bin/bash

# Purpose: Update all env_vars.sh files with a new WANDB_API_KEY.
# Usage:
#   ./wandb_update.sh <your_wandb_api_key>
# Example:
#   ./wandb_update.sh abcd1234xyz+789==

# Check if an argument was provided.
if [ -z "$1" ]; then
  echo "❌ Error: Missing argument."
  echo "Usage: $0 <your_wandb_api_key>"
  exit 1
fi

# Store the string argument.
input_string="$1"

# Replace all occurrences of WANDB_API_KEY="your_wandb_api_key" with user's key.
sed -i "s/WANDB_API_KEY=\"your_wandb_api_key\"/WANDB_API_KEY=\"$input_string\"/" \
bloom/baseline/env_vars.sh \
bloom/multi_gpu/2_gpus/env_vars.sh \
bloom/multi_gpu/4_gpus/env_vars.sh \
bloom/multi_gpu/8_gpus/env_vars.sh \
bloom/multi_node/2_nodes/env_vars.sh \
bloom/multi_node/4_nodes/env_vars.sh \
bloom/multi_node/8_nodes/env_vars.sh \
custom_model/single_node/env_vars.sh \
custom_model/multi_gpu/2_gpus/env_vars.sh \
custom_model/multi_gpu/4_gpus/env_vars.sh \
custom_model/multi_gpu/8_gpus/env_vars.sh \
custom_model/multi_node/2_nodes/env_vars.sh \
custom_model/multi_node/4_nodes/env_vars.sh \
custom_model/multi_node/8_nodes/env_vars.sh \
custom_model/weak_scaling/2_gpus/env_vars.sh \
custom_model/weak_scaling/4_gpus/env_vars.sh \
custom_model/weak_scaling/8_gpus/env_vars.sh \
custom_model/weak_scaling/16_gpus/env_vars.sh \

