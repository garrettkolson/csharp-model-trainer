# scripts/train.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

MODEL_NAME = "Qwen/Qwen3.5-9B"
DATA_PATH = "../data/csharp_code.txt"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=DATA_PATH,
    block_size=512,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Load training args
import json
with open("../configs/training_args.json") as f:
    train_args_dict = json.load(f)
training_args = TrainingArguments(**train_args_dict)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model
trainer.save_model("../outputs/qwen-sharp")
