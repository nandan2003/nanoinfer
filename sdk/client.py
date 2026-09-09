import json
import urllib.request
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class Telemetry:
    ttft_ms: float
    itl_mean_ms: float
    tps: float
    e2e_sec: float
    token_count: int

class NanoInferClient:
    """
    Lightweight client SDK for interacting with the NanoInfer CPU serving engine.
    Supports both non-streaming and live SSE generator streaming.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        self.last_telemetry: Optional[Telemetry] = None

    def chat_stream(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> Iterator[str]:
        """
        Streams generated tokens live as an Iterator[str].
        Populates self.last_telemetry upon stream completion.
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

        current_event = None
        with urllib.request.urlopen(req) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    current_event = None
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    payload_str = line[6:]
                    if current_event == "telemetry":
                        try:
                            t_data = json.loads(payload_str)
                            self.last_telemetry = Telemetry(
                                ttft_ms=t_data.get("ttft_ms", 0.0),
                                itl_mean_ms=t_data.get("itl_mean_ms", 0.0),
                                tps=t_data.get("tps", 0.0),
                                e2e_sec=t_data.get("e2e_sec", 0.0),
                                token_count=t_data.get("token_count", 0),
                            )
                        except (json.JSONDecodeError, KeyError):
                            pass
                        continue

                    try:
                        data = json.loads(payload_str)
                        token = data["choices"][0]["delta"]["content"]
                        yield token
                    except (json.JSONDecodeError, KeyError):
                        continue

    def chat(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> str:
        """
        Convenience method that consumes the stream and returns the complete text string.
        """
        return "".join(self.chat_stream(prompt, max_tokens=max_tokens, temperature=temperature))

    def list_models(self) -> list[str]:
        """
        Queries /v1/models and returns the list of model IDs available on the server.
        """
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [m["id"] for m in data.get("data", [])]

