import ollama
from redis_tools import load_summaries, store_summary

def summarize_message(message: str):
    prompt = f"Summarize the following message in 1–2 sentences:\n\n{message}\n"
    response = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def start_session(user_id: str):
    past = load_summaries(user_id)
    memory = "\n".join([f"- {s}" for s in past])

    print("Loaded memory:")
    print(memory)
    return memory


def chat_with_agent(user_id: str):
    memory_context = start_session(user_id)

    system_prompt = f"""
You are a helpful assistant.
Here is the summary of the user's last conversations:
{memory_context}

Use this to maintain continuity.
"""

    history = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        history.append({"role": "user", "content": user_input})

        response = ollama.chat(model="llama3.2", messages=history)
        assistant_msg = response["message"]["content"]

        print("\nAssistant:", assistant_msg, "\n")

        # Summarize and store
        summary = summarize_message(user_input + "\n" + assistant_msg)
        store_summary(user_id, summary)


if __name__ == "__main__":
    user_id = "123"
    chat_with_agent(user_id)