"""Entry point for the Ollama Cloud FastAPI app."""

from fastapi import FastAPI

from routes.chat import router as chat_router

app = FastAPI(title="Ollama Cloud Chat API")
app.include_router(chat_router)

