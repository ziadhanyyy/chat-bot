# main.py
# Final version with the updated and correct Groq AI model name.

import json
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# Your API Endpoint and Key are already set.
# ==============================================================================
NLU_API_URL = "https://api.groq.com/openai/v1/chat/completions"

NLU_API_KEY = os.getenv("GROQ_API_KEY")
# ==============================================================================

# --- In-Memory Cache (Populated from JSON at startup) ---
chatbot_db: Dict[str, List[Dict]] = {"rooms": []}

# --- Function to Load Data from JSON into Cache ---
def load_data_into_cache():
    print("Attempting to load data from rooms.json into cache...")
    try:
        with open('rooms.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            chatbot_db["rooms"] = data.get("rooms", [])
        print(f"Successfully loaded {len(chatbot_db['rooms'])} rooms into cache.")
    except FileNotFoundError:
        print("WARNING: rooms.json not found. The chatbot will have no room data.")
    except json.JSONDecodeError:
        print("WARNING: Could not decode rooms.json. Check for syntax errors.")

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data_into_cache()
    yield
    print("Shutting down.")

# --- FastAPI Application Instance ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- Pydantic Models ---
class UserMessage(BaseModel):
    text: str

class BotReply(BaseModel):
    reply: str

# --- Core Logic to Call the AI Model ---
async def get_ai_response(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {NLU_API_KEY}",
        "Content-Type": "application/json"
    }
    room_context = json.dumps(chatbot_db["rooms"])

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # <-- ### THE FIX IS HERE ###
        "messages": [
            {
                "role": "system",
                "content": f"""You are a helpful hotel booking assistant for a hotel named 'Bookify'.
                Your only job is to answer questions based on the user's query and the provided JSON data about available rooms.
                Keep your answers concise and friendly.
                Current room data: {room_context}"""
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(NLU_API_URL, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP ERROR: {e.response.status_code} - {e.response.text}")
        return "Sorry, I received an error from the AI service. Please check the terminal for details."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please check the server logs."

# --- API Endpoint ---
@app.post("/chat", response_model=BotReply)
async def handle_chat_message(message: UserMessage):
    response_text = await get_ai_response(message.text)
    return BotReply(reply=response_text)

# --- Serve the HTML UI ---
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')