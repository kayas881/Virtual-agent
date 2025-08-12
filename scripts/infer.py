# scripts/infer.py

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# --- 1. Configuration ---
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), '..', 'adapters', 'lora-task-adapter-v3')

def setup_inference():
    """
    Loads the base model, tokenizer, and merges the LoRA adapter.
    This function is called once by the agent at the start.
    """
    print("Setting up inference model...")
    
    # Use the same 4-bit quantization configuration as during training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the fine-tuned PEFT model (the adapter)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Inference model ready.")
    return model, tokenizer

def understand_command(model, tokenizer, command: str):
    """
    Takes a command and uses the loaded model to predict the action.
    """
    # Format the command into the same prompt structure used for training
    prompt = (
        f"### INSTRUCTION:\n{command}\n\n"
        f"### RESPONSE:\n"
    )

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    
    # Decode the output, skipping special tokens and the original prompt
    response_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    return response_text.strip()

# --- Main block for direct testing (if you run this file itself) ---
if __name__ == "__main__":
    inference_model, inference_tokenizer = setup_inference()
    
    print("\n--- Jarvis Task Inference (Direct Test) ---")
    print("Enter a command, or type 'exit' to quit.")

    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'exit':
            break
        
        print("Thinking...")
        action = understand_command(inference_model, inference_tokenizer, user_input)
        print(f"Action: {action}")
