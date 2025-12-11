import os
import json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)
client = OpenAI()
api_key = os.getenv('OPENAI_API_KEY')
try:
    memory = json.load(open("memory.json"))
except:
    memory = {"history": []}


def save_memory():
    json.dump(memory, open("memory.json", "w"), indent=2)


while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    memory["history"].append({"role": "user", "content": user})

    response = client.responses.create(
        model="gpt-4.1",
        input=memory["history"]
    )

    bot_reply = response.output_text
    memory["history"].append({"role": "assistant", "content": bot_reply})

    print("Bot:", bot_reply)
    save_memory()
