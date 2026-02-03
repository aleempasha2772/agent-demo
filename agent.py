from langchain_ollama import OllamaLLM
from tools import get_time, list_files, save_note
from dotenv import load_dotenv



load_dotenv()

llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

SYSTEM_PROMPT = """
You are a local AI agent.

You can use these tools:
- get_time
- list_files(path)
- save_note(text)

STRICT FORMAT:

Thought:
Action: tool_name(arguments)
Observation:
Final:

If no tool is needed, skip Action and Observation.
"""

def run_agent(user_input: str) -> str:
    prompt = SYSTEM_PROMPT + f"\nUser: {user_input}"
    response = llm.invoke(prompt).strip()

    print("\n--- MODEL OUTPUT ---\n", response)

    if "Action:" not in response:
        return response

    # Tool execution
    if "get_time" in response:
        observation = get_time()

    elif "list_files" in response:
        start = response.find("(") + 1
        end = response.find(")")
        path = response[start:end] or "."
        observation = list_files(path)

    elif "save_note" in response:
        start = response.find("(") + 1
        end = response.find(")")
        text = response[start:end]
        observation = save_note(text)

    else:
        observation = "Unknown action"

    followup = f"""
{response}
Observation: {observation}
Final:
"""
    return llm.invoke(followup).strip()
