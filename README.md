
# 🇲🇦 Rihla – AI-Powered Morocco Travel Assistant

Rihla is an AI-powered travel assistant that helps users explore Morocco by asking natural language questions.
It uses semantic search and Retrieval-Augmented Generation (RAG) to provide accurate and contextual travel recommendations.

# Features

AI chat for Morocco travel questions

Semantic search using vector embeddings

Travel itinerary generator

FastAPI backend

Pinecone vector database

HuggingFace LLM (Mistral)

# How It Works (Architecture)

The user asks a question from the frontend

The backend converts the question into a vector

Pinecone retrieves the most relevant destinations

The AI model generates a response using retrieved data

The answer is returned to the user

This follows the RAG (Retrieval-Augmented Generation) approach.

# Tech Stack

Backend: FastAPI (Python)

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

Vector Database: Pinecone

LLM: Mistral-7B (HuggingFace)

Data: JSON (Morocco destinations)

# Project Structure
RIHLA/
│── main.py

│── app.py

│── config.py

│── create_embeddings.py

│── morocco_destinations.json

│── requirements.txt

│── .gitignore

│── README.md

# Installation & Setup
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rihla-ai-travel-assistant.git
cd rihla-ai-travel-assistant

# Install dependencies
pip install -r requirements.txt

# Create a .env file
PINECONE_API_KEY=your_key_here

HUGGINGFACE_API_KEY=your_key_here

PINECONE_INDEX_NAME=rihla-morocco




# Run the Project
python main.py


API available at:

http://127.0.0.1:8000

Docs: http://127.0.0.1:8000/docs

# Key Concepts Used 

Semantic Search

Vector Embeddings

Retrieval-Augmented Generation (RAG)

REST API

# Academic Purpose 

This project was developed as an educational AI project to demonstrate how semantic search and AI can be combined to build intelligent applications.




