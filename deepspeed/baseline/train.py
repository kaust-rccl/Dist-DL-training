from transformers import Trainer, DataCollatorForLanguageModeling
from model import load_model, save_model
from data_loader import load_squad
from config import TRAINING_ARGS

# Load model and tokenizer
model, tokenizer = load_model()

# Load preprocessed dataset
tokenized_datasets = load_squad(subset_size=500)

# Data collator for Causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Run training
trainer.train()

# Save the fine-tuned model
save_model(model, tokenizer)
