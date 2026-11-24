
# ==============================
# FASTAPI IMPORTS
# ==============================

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import re
from pathlib import Path

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# ==============================
# INITIALIZE FASTAPI APP
# ==============================

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# ==============================
# BASE DIRECTORY
# ==============================

BASE_DIR = Path(__file__).resolve().parent


# ==============================
# MODEL AND TOKENIZER
# ==============================

print("Loading T5 model...")

model = T5ForConditionalGeneration.from_pretrained(
    str(BASE_DIR)
)

tokenizer = AutoTokenizer.from_pretrained(
    str(BASE_DIR)
)

print("Model loaded successfully!")


# ==============================
# DEVICE CONFIGURATION
# ==============================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

print("Using device:", device)

model.to(device)
model.eval()


# ==============================
# HTML TEMPLATING
# ==============================

templates = Jinja2Templates(
    directory=str(BASE_DIR)
)


# ==============================
# INPUT SCHEMA
# ==============================

class DialogueInput(BaseModel):
    dialogue: str


# ==============================
# CLEAN DATA
# ==============================

def clean_data(text: str) -> str:

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip().lower()

    return text


# ==============================
# SUMMARIZATION FUNCTION
# ==============================

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter some text.")

    # Tokenize input
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = inputs.to(device)

    # Generate summary
    with torch.inference_mode():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode output
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# ==============================
# SUMMARIZE API ENDPOINT
# ==============================

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):

    try:

        summary = summarize_dialogue(
            dialogue_input.dialogue
        )

        return {
            "summary": summary
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Summarization error:", error)

        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(error)}"
        )



# ==============================
# HOME PAGE
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():

    return {
        "status": "running",
        "model": "T5",
        "device": str(device)
    }
