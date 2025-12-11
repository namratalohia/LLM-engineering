import os
import json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)
client = OpenAI()
api_key = os.getenv('OPENAI_API_KEY')
import numpy as np

# 1. Load PDF text (already extracted)
kb_text = open("knowledge.txt").read()

# 2. Create embeddings
kb_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=kb_text
).data[0].embedding

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

while True:
    q = input("Ask: ")

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    score = similarity(kb_embedding, q_emb)

    response = client.responses.create(
        model="gpt-4.1",
        input=f"Use this information only if relevant:\n\n{kb_text}\n\nUser: {q}"
    )

    print("\nAnswer:", response.output_text)