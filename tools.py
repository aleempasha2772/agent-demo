import datetime
import os

def get_time():
    return str(datetime.datetime.now())

def list_files(path="."):
    return ", ".join(os.listdir(path))

def save_note(text):
    with open("agent_notes.txt", "a") as f:
        f.write(text + "\n")
    return "Saved successfully"
