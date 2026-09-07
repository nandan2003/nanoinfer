import json
import urllib.request
from typing import Iterator

class NanoInferClient:
    """
    Lightweight client SDK for interacting with the NanoInfer CPU serving engine.
    Supports both non-streaming and live SSE generator streaming.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def chat_stream(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> Iterator[str]:
        """
        Streams generated tokens live as an Iterator[str].
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    json_str = line[6:]
                    try:
                        data = json.loads(json_str)
                        token = data["choices"][0]["delta"]["content"]
                        yield token
                    except (json.JSONDecodeError, KeyError):
                        continue

    def chat(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
        """
        Convenience method that consumes the stream and returns the complete text string.
        """
        return "".join(self.chat_stream(prompt, max_tokens=max_tokens, temperature=temperature))

