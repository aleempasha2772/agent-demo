from agent import run_agent

print("Local Agent (Ollama only)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    answer = run_agent(user_input)
    print("\nAgent:", answer)
