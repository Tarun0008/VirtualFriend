import torch
import pyttsx3
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS engine setup
_tts_engine = pyttsx3.init()
for voice in _tts_engine.getProperty("voices"):
    if "zira" in voice.id.lower():
        _tts_engine.setProperty("voice", voice.id)
        break

def speak(text):
    _tts_engine.say(text)
    _tts_engine.runAndWait()

# ChatBot Wrapper
class ChatBotService:
    def __init__(self):
        model_id = os.getenv("MODEL_ID")
        if not model_id:
            raise ValueError("Model ID is missing in environment variables")

        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_id)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_reply(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        reply_ids = self.model.generate(**inputs, max_length=50, do_sample=True, top_p=0.9)
        return self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# Initialize
_chatbot = ChatBotService()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty input is not allowed")
    bot_reply = _chatbot.get_reply(user_input)
    return {"reply": bot_reply}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
