# scripts/generate_data.py

import json
import os
import re
import requests
import time
# NEW: Import jsonschema for robust validation of the LLM's output
from jsonschema import validate, ValidationError

# --- Configuration ---
OLLAMA_API_ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_tasks_v3.json')
# INCREASED: Generate more examples per task for a more robust dataset.
EXAMPLES_PER_TASK = 75

# --- Task Definitions ---
# The schemas are now more detailed to support automated validation.
TASK_DEFINITIONS = [
    {
        "task_name": "open_application",
        "description": "User wants to open a software application.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "open_application"},
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"]
                }
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "Can you open Chrome for me?", "output": {"action": "open_application", "parameters": {"app_name": "Google Chrome"}}},
            {"command": "Launch Spotify.", "output": {"action": "open_application", "parameters": {"app_name": "Spotify"}}},
        ]
    },
    {
        "task_name": "web_search",
        "description": "User wants to search for something on the internet.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "search_web"},
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "Search for the latest news on AI.", "output": {"action": "search_web", "parameters": {"query": "latest news on AI"}}},
            {"command": "Find me a recipe for lasagna.", "output": {"action": "search_web", "parameters": {"query": "recipe for lasagna"}}},
        ]
    },
    {
        "task_name": "send_message",
        "description": "User wants to send a message to a recipient.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "send_message"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "message": {"type": "string"}
                    },
                    "required": ["recipient", "message"]
                }
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "Tell Navya 'I'll be home in 10 minutes'", "output": {"action": "send_message", "parameters": {"recipient": "Navya", "message": "I'll be home in 10 minutes"}}},
        ]
    },
    {
        "task_name": "get_time",
        "description": "User wants to know the current time.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "get_time"},
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "what time is it?", "output": {"action": "get_time", "parameters": {}}},
        ]
    },
    {
        "task_name": "chitchat",
        "description": "User is making small talk, greeting, or being polite.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "chitchat"},
                "parameters": {
                    "type": "object",
                    "properties": {"response": {"type": "string"}},
                    "required": ["response"]
                }
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "how are you?", "output": {"action": "chitchat", "parameters": {"response": "I'm a computer program, but I'm running perfectly!"}}},
        ]
    },
    {
        "task_name": "clarify",
        "description": "User command is too vague, ambiguous, or incomplete to execute.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"const": "clarify"},
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"]
                }
            },
            "required": ["action", "parameters"]
        },
        "examples": [
            {"command": "open", "output": {"action": "clarify", "parameters": {"question": "What would you like me to open?"}}},
            {"command": "send a text", "output": {"action": "clarify", "parameters": {"question": "Who is the recipient and what should the message say?"}}},
            {"command": "search", "output": {"action": "clarify", "parameters": {"question": "What would you like to search for?"}}},
        ]
    }
]

def build_prompt(task_definition):
    """
    IMPROVED: Builds a more detailed few-shot prompt for the LLM.
    This new prompt explicitly asks for variety, edge cases, and realistic commands.
    """
    prompt = (
        f"You are a helpful assistant that generates synthetic training data for an AI agent. "
        f"Your task is to create varied user commands that match a given JSON schema.\n\n"
        f"--- TASK ---\n"
        f"Task Name: {task_definition['task_name']}\n"
        f"Description: {task_definition['description']}\n"
        f"JSON Schema: {json.dumps(task_definition['schema'])}\n\n"
        f"--- EXAMPLES ---\n"
    )

    for example in task_definition['examples']:
        prompt += f"Command: \"{example['command']}\"\n"
        prompt += f"Output: {json.dumps(example['output'])}\n"

    prompt += (
        f"\n--- INSTRUCTIONS ---\n"
        f"Now, generate {EXAMPLES_PER_TASK} new and varied user commands based on the task above. "
        f"Your goal is to create a robust dataset. Please include a mix of:\n"
        f"- **Simple & Direct Commands:** (e.g., 'launch chrome', 'search for news')\n"
        f"- **Natural Language Phrasing:** (e.g., 'could you please find out what time it is?')\n"
        f"- **Edge Cases & Unusual Phrasing:** (e.g., 'I need my code editor', 'find stuff on the web about dogs')\n"
        f"- **Negative/Malformed Examples:** For the 'clarify' task, provide commands that are intentionally ambiguous or incomplete.\n\n"
        f"Format your response as a single, valid JSON array of objects, where each object has a 'command' and 'output' key. "
        f"Do not include any other text, explanation, or markdown. Your entire response must be only the JSON array."
    )
    return prompt

def extract_json_from_response(response_text):
    """Finds and extracts the first valid JSON array from the model's raw output."""
    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            return None
    return None

def call_local_llm(prompt):
    """
    IMPROVED: Sends a prompt to the local LLM and gets the response.
    - Sets a lower temperature for more focused and reliable JSON output.
    - Increases timeout for longer generation tasks.
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.4  # Lower temperature for less random, more predictable output
            }
        }
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload, timeout=600)
        response.raise_for_status()
        response_text = response.json().get('response', '')
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"\nError calling local LLM: {e}")
        return None

def main():
    """Main function to generate and save the synthetic dataset."""
    print("Starting enhanced synthetic data generation with schema validation...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_generated_data = []

    for task in TASK_DEFINITIONS:
        print(f"\n--- Generating data for task: {task['task_name']} ---")
        prompt = build_prompt(task)
        task_schema = task['schema'] # Get the schema for validation
        
        retries = 3
        for i in range(retries):
            raw_response = call_local_llm(prompt)
            if not raw_response:
                print(f"Attempt {i+1}/{retries}: Failed to get a response. Retrying...")
                time.sleep(5)
                continue

            clean_json_str = extract_json_from_response(raw_response)
            if not clean_json_str:
                print(f"Attempt {i+1}/{retries}: Could not find a valid JSON array in the response.")
                time.sleep(2)
                continue

            try:
                generated_examples = json.loads(clean_json_str)
                if not isinstance(generated_examples, list):
                    generated_examples = [generated_examples]

                valid_examples = []
                for ex in generated_examples:
                    # NEW: Validate the 'output' part of the example against the task's JSON schema
                    try:
                        if 'command' in ex and 'output' in ex:
                            validate(instance=ex['output'], schema=task_schema)
                            ex['task_name'] = task['task_name']
                            valid_examples.append(ex)
                        else:
                            print(f"Warning: Skipping malformed entry (missing keys): {ex}")
                    except ValidationError as e:
                        # This is crucial for quality control. We log and discard invalid data.
                        print(f"Warning: Schema validation failed for entry. SKIPPING. Error: {e.message}")
                        print(f"   -> Offending data: {ex.get('output')}")


                print(f"Successfully generated and validated {len(valid_examples)} examples for '{task['task_name']}'.")
                all_generated_data.extend(valid_examples)
                break 
            except json.JSONDecodeError as e:
                print(f"Attempt {i+1}/{retries}: Could not parse JSON. Error: {e}")
                time.sleep(2)
        else:
            print(f"Failed to generate valid data for task '{task['task_name']}' after {retries} attempts.")

    print(f"\nTotal examples generated across all tasks: {len(all_generated_data)}")

    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_generated_data, f, indent=4)
        print(f"Successfully saved new synthetic data to {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error saving data to file: {e}")

if __name__ == "__main__":
    main()