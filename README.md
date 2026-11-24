# Text Summarizer App

A FastAPI web application for text and dialogue summarization powered by a fine-tuned T5 model and Hugging Face Transformers.

## Features

- **T5-Powered Summarization**: Generates clean, concise summaries of input dialogues or text.
- **Interactive UI**: Built-in responsive HTML web interface using Jinja2 templates.
- **REST API**: FastAPI backend endpoint `/summarize` for automated integration.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <REPO_URL>
   cd "Text Summarizer App"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   cd saved_summary_model
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Open in browser**:
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000)
