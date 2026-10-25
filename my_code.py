"""Ask Ollama Cloud a question using the key in config.json."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://ollama.com/api/chat"
MODEL = "gpt-oss:20b-cloud"
CONFIG_FILE = Path(__file__).with_name("config.json")


def main():
    # Read the API key from the local config file.
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError:
        print('config.json was not found. Create it with: {"OLLAMA_API_KEY": "your_api_key"}')
        return
    except json.JSONDecodeError:
        print("config.json is not valid JSON. Check its braces, quotes, and commas.")
        return

    api_key = config.get("OLLAMA_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key":
        print('Add your Ollama API key to config.json under "OLLAMA_API_KEY".')
        return

    question = input("Ask gpt-oss-20b a question: ").strip()
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
            "Authorization": f"Bearer {api_key}",
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
