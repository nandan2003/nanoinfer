import unittest
import asyncio
import threading
from server.http_server import HTTPServer
from sdk.client import NanoInferClient

SDK_TEST_PORT = 8091

class TestSDKIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(host="127.0.0.1", port=SDK_TEST_PORT)
        cls.loop = asyncio.new_event_loop()

        def run_server_loop():
            asyncio.set_event_loop(cls.loop)
            cls.loop.run_until_complete(cls.server.start())
            cls.loop.run_forever()

        cls.server_thread = threading.Thread(target=run_server_loop, daemon=True)
        cls.server_thread.start()

        cls.client = NanoInferClient(base_url=f"http://127.0.0.1:{SDK_TEST_PORT}")

    @classmethod
    def tearDownClass(cls):
        cls.loop.call_soon_threadsafe(cls.loop.stop)

    def test_01_sdk_non_streaming_chat(self):
        prompt = "The chemical symbol for water is"
        reply = self.client.chat(prompt, max_tokens=5, temperature=0.1)

        self.assertIsInstance(reply, str)
        self.assertGreater(len(reply.strip()), 0)
        print(f"\n[PASS] client.chat() returned: '{reply.strip()}'")

    def test_02_sdk_streaming_chat(self):
        prompt = "The largest ocean on Earth is the"
        stream = self.client.chat_stream(prompt, max_tokens=6, temperature=0.1)

        tokens = []
        for token in stream:
            self.assertIsInstance(token, str)
            tokens.append(token)

        full_reply = "".join(tokens)
        self.assertGreater(len(tokens), 0)
        self.assertIsNotNone(self.client.last_telemetry)
        print(f"\n[PASS] client.chat_stream() yielded {len(tokens)} tokens: '{full_reply.strip()}'")
        print(f"[PASS] Telemetry -> TTFT: {self.client.last_telemetry.ttft_ms} ms | TPS: {self.client.last_telemetry.tps}")

if __name__ == "__main__":
    unittest.main()

