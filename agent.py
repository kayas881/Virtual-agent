# agent.py

import json
import os
import sys
from scripts.infer import setup_inference, understand_command
from scripts.action_executor import execute_action

# --- Configuration ---
MEMORY_FILE = os.path.join('memory', 'memory.json')

# --- Memory Functions ---

def load_memory():
    """
    Loads the memory file. Creates the directory and file if they don't exist.
    Handles permissions errors and corrupted files gracefully.
    """
    memory_dir = os.path.dirname(MEMORY_FILE)

    # Defensive Check: If a file named 'memory' exists, we can't create the directory.
    if os.path.exists(memory_dir) and not os.path.isdir(memory_dir):
        print(f"FATAL ERROR: A file named '{memory_dir}' exists where a directory is required.", file=sys.stderr)
        print("Please delete the 'memory' file in your project folder and restart the agent.", file=sys.stderr)
        sys.exit(1) # Exit the script because it cannot proceed.

    # Ensure the directory exists.
    try:
        os.makedirs(memory_dir, exist_ok=True)
    except OSError as e:
        print(f"FATAL ERROR: Could not create memory directory at '{memory_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # Now, handle the file itself.
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)
        return []
    
    # Handle empty file case and potential read/decode errors
    try:
        with open(MEMORY_FILE, 'r') as f:
            # Check if file is empty before trying to load JSON
            content = f.read()
            if not content:
                return [] 
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse memory file at '{MEMORY_FILE}'. It might be corrupted. Starting with fresh memory.")
        return []
    except PermissionError as e:
        print(f"FATAL ERROR: Permission denied when trying to read '{MEMORY_FILE}'. Check your folder permissions.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading memory: {e}", file=sys.stderr)
        return []


def save_memory(memory_data):
    """Saves the updated memory data to the file."""
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory_data, f, indent=4)
    except Exception as e:
        print(f"Error: Could not save to memory file: {e}", file=sys.stderr)


# --- Main Agent Loop ---

def main():
    """The main function to run the agent."""
    
    # 1. SETUP: Load the AI model once at the beginning.
    inference_model, tokenizer = setup_inference()
    
    # 2. MEMORY: Load past interactions.
    memory = load_memory()
    print(f"Agent initialized. Loaded {len(memory)} memories.")
    
    print("\n--- Jarvis is Online ---")
    print("Enter a command, or type 'exit' to shut down.")

    # 3. LOOP: Start the main interaction cycle.
    while True:
        user_command = input("\n> ")
        if user_command.lower() == 'exit':
            print("Jarvis shutting down. Goodbye.")
            break

        # --- THINK ---
        print("Thinking...")
        # predicted_action is now a JSON string, e.g., '{"action": "get_time", "parameters": {}}'
        predicted_action = understand_command(inference_model, tokenizer, user_command)
        print(f"   [Thought]: {predicted_action}")

        # --- ACT ---
        # The updated execute_action now correctly handles the JSON string
        success, result_message = execute_action(predicted_action)
        print(f"   [Action Result]: {result_message}")

        # --- LEARN (Log Interaction) ---
        # UPDATED: We must save the output as a dictionary, not a string, to match the training format.
        if success:
            try:
                # Convert the JSON string from the model back into a Python dictionary
                predicted_output_dict = json.loads(predicted_action)

                new_memory_entry = {
                    "command": user_command, # Changed 'instruction' to 'command' to match v3 data format
                    "output": predicted_output_dict # Save the dictionary
                }

                if new_memory_entry not in memory:
                    memory.append(new_memory_entry)
                    save_memory(memory)
                    print("   [Learned]: New successful interaction saved to memory.")
            
            except json.JSONDecodeError:
                # This handles rare cases where the model output was not valid JSON despite a successful action
                print("   [Learn Error]: Could not save to memory, model output was not valid JSON.")

if __name__ == "__main__":
    main()
