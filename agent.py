from langchain_ollama import OllamaLLM
from tools import get_time, list_files, save_note
import re

llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)

SYSTEM_PROMPT = """You are a helpful AI agent. You have these tools:

1. get_time() - Get current date and time
2. list_files(path) - List files in a directory (default: ".")
3. save_note(text) - Save text to notes file

RESPOND IN THIS EXACT FORMAT:

Thought: <explain what you need to do>
Action: <tool_name(arguments)>

After you use a tool, I will give you:
Observation: <result>

Then respond with:
Final: <your answer to the user>

If no tool is needed, skip Action and Observation.

EXAMPLES:

User: What time is it?
Thought: I need to check the current time
Action: get_time()

User: Save "meeting at 3pm" 
Thought: I need to save this note
Action: save_note("meeting at 3pm")
"""

def run_agent(user_input: str) -> str:
    # First LLM call
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nYour response:"
    response = llm.invoke(prompt).strip()
    
    print("\n=== LLM Response ===")
    print(response)
    print("=" * 50)
    
    # Try to extract action using regex
    action_pattern = r'Action:\s*(\w+)\((.*?)\)'
    match = re.search(action_pattern, response, re.DOTALL)
    
    if not match:
        # No action needed, return response
        return response
    
    tool_name = match.group(1).strip()
    args_raw = match.group(2).strip()
    
    # Clean arguments (remove quotes)
    args = args_raw.strip('\'"') if args_raw else ""
    
    print(f"\n🔧 Tool: {tool_name}")
    print(f"📝 Args: {args}")
    
    # Execute tool
    try:
        if tool_name == "get_time":
            observation = get_time()
        elif tool_name == "list_files":
            observation = list_files(args if args else ".")
        elif tool_name == "save_note":
            observation = save_note(args)
        else:
            observation = f"ERROR: Unknown tool '{tool_name}'"
    except Exception as e:
        observation = f"ERROR: {str(e)}"
    
    print(f"👁️ Observation: {observation}\n")
    
    # Second LLM call with observation
    followup_prompt = f"""{response}

Observation: {observation}

Final:"""
    
    final_answer = llm.invoke(followup_prompt).strip()
    
    print("=== Final Answer ===")
    print(final_answer)
    print("=" * 50)
    
    return final_answer