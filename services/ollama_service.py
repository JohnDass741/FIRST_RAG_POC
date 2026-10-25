"""Small Ollama Cloud client using Python's standard library."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://ollama.com/api/chat"
MODEL = "gpt-oss:20b-cloud"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"


class OllamaConfigurationError(Exception):
    """Raised when config.json does not contain a usable API key."""


class OllamaServiceError(Exception):
    """Raised when Ollama cannot return a usable chat response."""


def _read_api_key() -> str:
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as error:
        raise OllamaConfigurationError(
            "config.json was not found. Add OLLAMA_API_KEY to config.json."
        ) from error
    except (json.JSONDecodeError, OSError) as error:
        raise OllamaConfigurationError("config.json could not be read as valid JSON.") from error

    if not isinstance(config, dict):
        raise OllamaConfigurationError("config.json must contain a JSON object.")

    api_key = config.get("OLLAMA_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip() or api_key.strip() == "your_api_key":
        raise OllamaConfigurationError(
            'Add your Ollama API key to config.json under "OLLAMA_API_KEY".'
        )

    return api_key.strip()


def ask_ollama(question: str) -> str:
    """Send one question to Ollama Cloud and return its answer text."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_read_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise OllamaServiceError(f"Ollama returned HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        raise OllamaServiceError("Could not connect to Ollama Cloud or the request timed out.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaServiceError("Ollama returned an invalid response.") from error

    try:
        answer = result["message"]["content"]
    except (TypeError, KeyError) as error:
        raise OllamaServiceError("Ollama's response did not contain an answer.") from error

    if not isinstance(answer, str):
        raise OllamaServiceError("Ollama's response did not contain an answer.")

    return answer
