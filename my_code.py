"""Ask the local Ollama model a question."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def main():
    question = input(f"Ask {MODEL} a question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    # Ollama expects a chat message and a flag to return one complete response.
    request_data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    request = Request(
        API_URL,
        data=json.dumps(request_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        print("\nModel response:")
        print(result["message"]["content"])
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"Ollama returned HTTP {error.code}: {details}")
    except URLError as error:
        print(f"Could not connect to Ollama: {error.reason}")
    except (json.JSONDecodeError, KeyError) as error:
        print(f"Could not read Ollama's response: {error}")
    except TimeoutError:
        print("The request timed out. Please try again.")


if __name__ == "__main__":
    main()
