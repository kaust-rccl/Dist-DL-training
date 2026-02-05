# Weights & Biases (W&B) Account & API Key Setup Guide
To run the workshop training scripts, each participant must have a W&B API key.
This key allows the training jobs to log metrics, loss curves, GPU usage, and artifacts to each participant’s personal W&B account.

Follow the steps below to create an account and obtain your API key.

## 1. Create a Weights & Biases Account 
#### (if you don't already have one)

   1. Open:
   https://wandb.ai/siteClick 

   2. Click Sign Up.
   3. Sign up with your KAUST email.
   4. Complete the initial registration steps.

You now have a W&B account.

## 2. Retrieve Your W&B API Key

Your W&B API key is what authenticates your account inside the training scripts.
Get the API Key from W&B Website

1. Go to: https://wandb.ai/authorize

2. You will see a page displaying your personal API key:

    ![Wandb API Key](./wandb_api_key.jpg "Wandb API Key")

3. Copy the key.

## 3.Add Your API Key to Your Environment

Inside the cluster environment, export your key before starting training
To export it in all scripts at once run:
```commandline
cd ../../fsdp
./wandb_api_key.sh <YOUR_WANDB_API_KEY>
```