import unittest
from server.inference import InferenceEngine

MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(model_path=MODEL_PATH, n_ctx=256, verbose=False)

    def test_01_tokenize(self):
        text = "Hello, CPU inference!"
        tokens = self.engine.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        print(f"\n[PASS] Tokenized '{text}' -> {tokens}")

    def test_02_generate_streaming(self):
        prompt = "The color of the sky is"
        stream = self.engine.generate(prompt, max_tokens=10, temperature=0.1)

        chunks = []
        for token in stream:
            self.assertIsInstance(token, str)
            chunks.append(token)

        full_text = "".join(chunks)
        self.assertGreater(len(chunks), 0)
        print(f"\n[PASS] Streamed {len(chunks)} tokens: '{full_text.strip()}'")

    def test_03_save_and_load_state(self):
        self.engine.reset()
        prompt = "Explain gravity in one sentence:"
        for _ in self.engine.generate(prompt, max_tokens=5):
            pass

        state = self.engine.save_state()
        self.assertIsNotNone(state)

        self.engine.reset()
        self.engine.load_state(state)
        print("\n[PASS] Successfully saved and restored model state snapshot")

if __name__ == "__main__":
    unittest.main()

