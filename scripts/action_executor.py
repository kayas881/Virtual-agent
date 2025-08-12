# scripts/action_executor.py

import os
import webbrowser
import subprocess

import datetime
import json
import ast # Import the Abstract Syntax Tree module for safe parsing

# --- Configuration ---
PATH_CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'memory', 'app_paths.json')

# --- Platform-specific setup for detaching processes ---
if os.name == 'nt':
    DETACHED_PROCESS = 0x00000008
else:
    DETACHED_PROCESS = 0

# --- Caching Functions ---

def load_path_cache():
    """Loads the application path cache from a JSON file."""
    if not os.path.exists(PATH_CACHE_FILE):
        return {}
    try:
        with open(PATH_CACHE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_path_cache(cache):
    """Saves the application path cache to a JSON file."""
    try:
        os.makedirs(os.path.dirname(PATH_CACHE_FILE), exist_ok=True)
        with open(PATH_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
    except IOError as e:
        print(f"Error saving path cache: {e}")

# --- Helper function for Windows ---

def find_executable_on_windows(app_name, path_cache):
    """
    Searches common installation directories for an executable.
    Checks the cache first before searching the file system.
    """
    app_key = app_name.lower()
    if app_key in path_cache:
        cached_path = path_cache[app_key]
        if os.path.exists(cached_path):
            print(f"Found '{app_name}' in cache: {cached_path}")
            return cached_path
        else:
            print(f"Cached path for '{app_name}' is invalid. Re-searching...")
            del path_cache[app_key] # Remove invalid path

    search_paths = [
        os.environ.get('LOCALAPPDATA'),
        os.environ.get('APPDATA'),
        os.environ.get('ProgramFiles'),
        os.environ.get('ProgramFiles(x86)'),
    ]
    search_paths = [path for path in search_paths if path and os.path.isdir(path)]

    exe_name = f"{app_name.replace(' ', '').lower()}.exe"
    print(f"Searching for '{exe_name}' in {len(search_paths)} locations...")

    for path in search_paths:
        for root, dirs, files in os.walk(path):
            if exe_name in [f.lower() for f in files]:
                original_filename = [f for f in files if f.lower() == exe_name][0]
                full_path = os.path.join(root, original_filename)
                print(f"Found executable at: {full_path}")
                # Save the newly found path to the cache
                path_cache[app_key] = full_path
                save_path_cache(path_cache)
                return full_path
    return None

# --- Action Definitions ---

def open_app(app_name: str):
    """Opens an application by mapping its name to a command and searching for its executable."""
    print(f"ACTION: Opening application '{app_name}'...")
    path_cache = load_path_cache()

    try:
        if os.name == 'nt':  # Windows
            # This map translates the model's friendly name (key) to the actual system command (value).
            # It includes common aliases like 'vscode' for 'Visual Studio Code'.
            app_map = {
                'google chrome': 'chrome',
                'chrome': 'chrome',
                'visual studio code': 'code',
                'vscode': 'code',
                'spotify': 'spotify',
                'discord': 'discord',
                'calculator': 'calc',
                'notepad': 'notepad',
                'file explorer': 'explorer',
                'epic games launcher': 'epicgameslauncher' # Example for apps with no space
            }

            # Find the correct command name from the map using the app_name from the model.
            command_name = app_map.get(app_name.lower())

            # If the app is not in our map, we fall back to a generic name.
            if not command_name:
                print(f"'{app_name}' not found in the common application map. Trying a generic search...")
                # This fallback might work for apps like 'Blender' -> 'blender'
                command_name = app_name.replace(' ', '').lower()

            # Now, 'command_name' will be 'chrome', which is correct.
            full_path = find_executable_on_windows(command_name, path_cache)

            if full_path:
                subprocess.Popen(
                    [full_path],
                    creationflags=DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"Successfully launched '{app_name}' from its path: {full_path}")
                return True, f"Opened {app_name}"
            else:
                # This fallback is for system apps not found by search (e.g., 'explorer', 'calc')
                print(f"Could not find a specific path for '{app_name}'. Attempting to launch via 'start' command...")
                exit_code = os.system(f'start {command_name} > nul 2>&1')
                if exit_code == 0:
                    print(f"Successfully launched '{app_name}' using the 'start' command.")
                    return True, f"Opened {app_name}"

                print(f"Failed to launch '{app_name}'. The application could not be found.")
                return False, f"Could not open {app_name}"

        elif os.name == 'posix':  # macOS or Linux
            # This logic remains the same
            subprocess.Popen(['open', '-a', app_name])
            print(f"Successfully launched {app_name}.")
            return True, f"Opened {app_name}"

    except Exception as e:
        print(f"An error occurred while trying to open {app_name}: {e}")
        return False, f"Could not open {app_name}"


def web_search(query: str):
    """Performs a web search using the default browser."""
    print(f"ACTION: Searching the web for '{query}'...")
    try:
        url = f"https://www.google.com/search?q={query}"
        webbrowser.open(url)
        print("Search page opened successfully.")
        return True, f"Searched for '{query}'"
    except Exception as e:
        print(f"Error performing web search: {e}")
        return False, "Could not perform web search"

def send_message(recipient: str, message: str, app: str = None):
    """Placeholder for sending a message."""
    app_info = f" via {app}" if app else ""
    print(f"ACTION: Sending message to {recipient}{app_info}: '{message}'")
    return True, f"Message to {recipient} was prepared."

def set_reminder(task_description: str, time: str):
    """Placeholder for setting a reminder."""
    print(f"ACTION: Setting reminder for '{task_description}' at {time}.")
    return True, f"Reminder for '{task_description}' was set."
# --- NEW ACTIONS ---
def get_time():
    """Gets the current time and returns it as a formatted string."""
    print("ACTION: Getting the current time...")
    now = datetime.datetime.now()
    # You can change the format to your liking
    formatted_time = now.strftime("%I:%M %p on %A, %B %d, %Y") 
    print(f"The current time is {formatted_time}")
    # The agent doesn't need to speak this, but we'll return it for consistency
    return True, f"The time is {formatted_time}"

def chitchat(response: str):
    """Handles chitchat by simply printing the model's chosen response."""
    print(f"ACTION: Responding to chitchat...")
    # The model decided what to say, we just print it.
    print(response) 
    return True, "Chitchat response delivered."

def clarify(question: str):
    """Handles clarification by asking the user the model's question."""
    print(f"ACTION: Asking for clarification...")
    # The model decided what to ask, we just print it.
    print(question)
    return True, "Clarification question asked."
# --- Parser and Executor ---

FUNCTION_MAP = {
    'open_application': open_app,
    'search_web': web_search,
    'send_message': send_message,
    'get_time': get_time,
    'chitchat': chitchat,
    'clarify': clarify,
}

def execute_action(action_json: str):
    """
    Parses the model's JSON output string and executes the corresponding function.
    """
    try:
        action_data = json.loads(action_json)
        func_name = action_data.get("action")
        params = action_data.get("parameters", {})
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse model output as JSON: {e}")
        return False, "Invalid action format (not JSON)"

    if func_name in FUNCTION_MAP:
        try:
            func = FUNCTION_MAP[func_name]
            # Use keyword arguments to call the function
            success, result_message = func(**params) 
            return success, result_message

        except TypeError as e:
            print(f"ERROR: Failed to execute {func_name}. Mismatched parameters: {e}")
            return False, f"Execution failed for {func_name} due to incorrect parameters."
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during execution of {func_name}: {e}")
            return False, f"Execution failed for {func_name}."
    else:
        print(f"ERROR: Unknown function '{func_name}'")
        return False, f"Unknown function: {func_name}"

# --- Direct Execution for Testing ---
if __name__ == '__main__':
    print("--- Action Executor Test ---")
    
    test_cases = [
        "execute: open_app('Discord')",
        "execute: open_app('Epic Games Launcher')",
        "execute: open_app('Notepad')",
    ]

    for test in test_cases:
        print(f"\nExecuting: {test}")
        success, message = execute_action(test)
        print(f"Result: Success={success}, Message='{message}'")
