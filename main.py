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

# --- In-Memory Cache (UPDATED to store the whole JSON structure) ---
# Initialize as an empty dictionary to store the full contents of rooms.json
chatbot_db: Dict[str, Any] = {}

# --- Function to Load Data from JSON into Cache (UPDATED) ---
def load_data_into_cache():
    print("Attempting to load data from rooms.json into cache...")
    try:
        with open('rooms.json', 'r', encoding='utf-8') as f:
            # Load the WHOLE JSON object into chatbot_db
            global chatbot_db
            chatbot_db = json.load(f)
        
        # Add a print statement to verify contents (assuming 'rooms' key exists)
        room_count = len(chatbot_db.get("Rooms", []))
        room_type_count = len(chatbot_db.get("RoomTypes", []))
        print(f"Successfully loaded {room_count} rooms and {room_type_count} room types into cache.")
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

# --- Core Logic to Call the AI Model (UPDATED) ---
async def get_ai_response(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {NLU_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Send the ENTIRE chatbot_db dictionary (which holds the whole JSON)
    full_context = json.dumps(chatbot_db)

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "system",
                "content": f"""You are a helpful hotel booking assistant for a hotel named 'Bookify'.
                Your only job is to answer questions based on the user's query and the provided JSON data about rooms and room types.
                Keep your answers concise and friendly.
                Current hotel data: {full_context}"""
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
