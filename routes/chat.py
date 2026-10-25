"""Chat endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from services.ollama_service import OllamaConfigurationError, OllamaServiceError, ask_ollama

router = APIRouter()


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer = ask_ollama(request.question)
    except OllamaConfigurationError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except OllamaServiceError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return AskResponse(answer=answer)
