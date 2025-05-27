from transformers import TrainingArguments

TRAINING_ARGS = TrainingArguments(
    output_dir="./bloom-qa-finetuned",          # Save checkpoints, logs, etc. in this directory
    eval_strategy="epoch",                      # Run evaluation at the end of each epoch
    save_strategy="epoch",                      # Save a model checkpoint at the end of each epoch
    per_device_train_batch_size=4,              # Batch size per device during training
    per_device_eval_batch_size=4,               # Batch size per device during evaluation
    gradient_accumulation_steps=4,              # Simulate larger batch size by accumulating gradients over 4 steps
    num_train_epochs=3,                         # Number of full training passes over the dataset
    learning_rate=5e-5,                         # Learning rate for the Adam optimizer (5×10⁻⁵)
    weight_decay=0.01,                          # Regularization to prevent overfitting
    fp16=True,                                  # Use mixed precision training (requires compatible hardware)
    gradient_checkpointing=False,               # Do not use gradient checkpointing (saves memory if True but slows compute)
    push_to_hub=False,                          # Do not upload model or logs to the Hugging Face Hub
    deepspeed="./ds_config.json",               # <--- Links the DeepSpeed configuration file

)