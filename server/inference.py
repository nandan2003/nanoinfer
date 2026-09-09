from typing import Iterator, Any
from llama_cpp import Llama

class InferenceEngine:
    def __init__(self, model_path: str, n_ctx: int = 512, n_threads: int | None = None, verbose: bool = False):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    def tokenize(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"))

    def save_state(self) -> Any:
        return self.llm.save_state()

    def load_state(self, state: Any) -> None:
        self.llm.load_state(state)

    def reset(self) -> None:
        self.llm.reset()

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> Iterator[str]:
        stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            text = chunk["choices"][0]["text"]
            yield text 

