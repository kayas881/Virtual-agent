# scripts/fine_tune.py

import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# --- 1. Configuration ---

# Model and tokenizer names
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- UPDATED ---
# Point to the new, larger, and smarter dataset
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_tasks_v3.json') 
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'adapters', 'lora-task-adapter-v3') 

# --- 2. Data Preparation ---

def format_training_prompt(example):
    """
    Formats a single data example into a structured prompt for training.
    --- UPDATED ---
    - Uses the correct key 'command' instead of 'instruction'.
    - Converts the 'output' dictionary to a JSON string.
    """
    # The user's input is now under the 'command' key
    instruction = example['command']
    # The model's expected output is a dictionary, which we format as a JSON string
    response = json.dumps(example['output'])
    
    return (
        f"### INSTRUCTION:\n{instruction}\n\n"
        f"### RESPONSE:\n{response}"
    )

print("Loading and formatting dataset...")
# Load the dataset from the new JSON file
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

print(f"Dataset loaded with {len(dataset)} examples.")
print("Sample entry after formatting:\n", format_training_prompt(dataset[0]))


# --- 3. Model & Tokenizer Loading with QLoRA ---

print("\nLoading model and tokenizer...")

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model with the quantization config
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", 
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model and tokenizer loaded successfully.")


# --- 4. PEFT (LoRA) Configuration ---

print("\nConfiguring PEFT (LoRA)...")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


# --- 5. SFTConfig Configuration ---

print("\nSetting up SFTConfig...")

# SFTConfig now only contains the arguments that are shared with Transformers' TrainingArguments
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)


# --- 6. SFTTrainer Initialization and Training ---

print("\nInitializing SFTTrainer...")

# --- UPDATED ---
# Removed the 'tokenizer' argument to match the version in your environment.
# The trainer will infer the tokenizer from the model.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=format_training_prompt,
    args=sft_config,
)

# Note: To control sequence length, set 'max_seq_length' in your data preprocessing or use 'max_length' in the tokenizer config. SFTTrainer does not accept 'max_seq_length' as an argument in your TRL version.

print("\n--- Starting Training ---")
trainer.train()
print("--- Training Finished ---")


# --- 7. Save the Final Adapter ---

print(f"\nSaving the fine-tuned adapter to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
print("Adapter saved successfully.")
