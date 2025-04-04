import torch
import pyttsx3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow only your React app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load Model & Tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
)

# Set Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.backends.cudnn.benchmark = True

# Initialize Text-to-Speech (TTS)
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty("voices")

for voice in voices:
    if "zira" in voice.id.lower():  # Match Zira's voice ID
        tts_engine.setProperty("voice", voice.id)
        break


def speak(text):
    """Text-to-speech output (optional for API response)."""
    tts_engine.say(text)
    tts_engine.runAndWait()


class ChatRequest(BaseModel):
    user_input: str


@app.post("/chat/")
async def chat(request: ChatRequest):
    user_input = request.user_input.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty input is not allowed")

    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    reply_ids = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.9)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    return {"reply": bot_reply}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
