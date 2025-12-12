# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import os                 # Provides access to operating-system functions (environment variables, file paths)
import openai             # OpenAI SDK for embeddings and chat completions
import PyPDF2             # Library for reading PDF files and extracting text
import random             # Used for random operations (not used in this script but imported)
import pinecone           # Pinecone vector database SDK

# Importing custom variables/functions from your own script folders
from Scripts.firstdemo import message, messages
from Scripts.second import client

from dotenv import load_dotenv  # Loads environment variables from .env file
load_dotenv(override=True)      # Load .env and override system variables if needed

# Import OpenAI class for new client
from openai import OpenAI
client = OpenAI()   # Create a new OpenAI client instance for API calls


# ---------------------------------------------------------
# STEP 1: LOAD PDF AND EXTRACT TEXT
# ---------------------------------------------------------
# This function opens a PDF file in binary mode, reads each page,
# extracts text using PyPDF2, and returns it as a single long string.

def load_pdf(file_path):
    text_from_PDF = ""   # Store extracted text here

    # Open PDF in RB = read-binary mode
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)  # Load PDF reader

        # Loop through all pages in PDF
        for page in pdf_reader.pages:
            page_text = page.extract_text()  # Extract text from each page

            if page_text:                    # Add only if text exists
                text_from_PDF += page_text

    return text_from_PDF  # Final combined text from PDF


# ---------------------------------------------------------
# STEP 2: CHUNK TEXT (WORD OR CHARACTER BASED)
# ---------------------------------------------------------
# We split large text into smaller chunks for embeddings.
# chunk_size = how big each chunk should be
# chunk_overlap = overlap between chunks to prevent information loss
# by = word or char-based splitting

def chunk_text(text, chunk_size=1500, chunk_overlap=100, by='word'):

    # Validate mode
    if by not in ['word', 'char']:
        raise ValueError('by must be "word" or "char"')

    chunks = []  # Store all chunks

    # Choose tokenization type
    if by == 'word':
        tokens = text.split()   # Split text into words
        total_len = len(tokens)
    else:   # Character-based
        tokens = text           # Treat each char as a token
        total_len = len(tokens)

    start = 0  # Beginning index of each chunk

    # Loop until we reach end of text
    while start < total_len:

        end = start + chunk_size

        if by == 'word':
            chunk = " ".join(tokens[start:end])  # Join tokens back into sentence
        else:
            chunk = tokens[start:end]            # Characters slice

        chunks.append(chunk)  # Save chunk

        # Move window ahead but allow overlap
        start += chunk_size - chunk_overlap

    return chunks  # Return all chunks


# ---------------------------------------------------------
# RUN THE PIPELINE
# ---------------------------------------------------------

# Load text from a PDF file
pdf_loaded = load_pdf("state_of_ai_docs.pdf")

# Chunk the text (character-based splitting)
chunks = chunk_text(pdf_loaded, by='char')

# Show summary
print("Total chunks:", len(chunks))
print("First chunk preview:", chunks[0][:500])


# ---------------------------------------------------------
# BUILDING A RAG SYSTEM (RETRIEVAL-AUGMENTED GENERATION)
# ---------------------------------------------------------

# --- PINECONE INIT ---
from pinecone import Pinecone

pinecone_api_key = os.getenv("PINECONE_API_KEY")        # Get API key from .env
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")        # Get environment name

# 1. Create Pinecone client
pc = Pinecone(pc_api_key=pinecone_api_key)

# 2. Connect to an existing Pinecone index
index = pc.Index("rag-test")   # Replace with your actual index name


# ---------------------------------------------------------
# INSERT CHUNK VECTORS INTO PINECONE
# ---------------------------------------------------------

for i in range(len(chunks)):

    # Create embedding vector for each chunk
    vector = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunks[i]
    )

    print(vector.data[0].embedding)   # Show embedding vector for debugging

    # Insert into Pinecone
    insert_stats = index.upsert(
        vectors=[
            (
                str(i),                       # Unique ID
                vector.data[0].embedding,     # Embedding vector
                {"org_text": chunks[i]}       # Metadata to store the original text
            )
        ]
    )

    break  # Insert only first chunk (remove this break to insert all)


# ---------------------------------------------------------
# USER QUESTION → FIND BEST MATCHING CHUNK
# ---------------------------------------------------------

user_input = input("Ask a question about a file:")

# Convert user question to embedding
user_vector = client.embeddings.create(
    model="text-embedding-3-large",
    input=user_input
).data[0].embedding

# Query Pinecone for closest match
matches = index.query(
    vector=user_vector,
    top_k=1,
    include_metadata=True
)

# ---------------------------------------------------------
# BUILD CHAT MESSAGE WITH RAG RESULT
# ---------------------------------------------------------

message = [
    {
        "role": "system",
        "content": """I want you to act as a support agent. 
Your name is "My Super Assistant". 
You will provide me answers only from the given info. 
If the answer is not included, say exactly "Ooops! I don't know that." and stop.
Refuse to answer anything not related to the info."""
    }
]

# Add retrieved chunk text into conversation
message.append({
    "role": "system",
    "content": matches['matches'][0]['metadata']['org_text']
})

# Add user question
message.append({
    "role": "system",
    "content": user_input
})


# ---------------------------------------------------------
# GENERATE FINAL ANSWER USING OPENAI CHAT COMPLETION
# ---------------------------------------------------------

chat_messages = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=message,
    temperature=0,
    max_tokens=100
)

# Print model response
print(chat_messages.choices[0].message["content"])
