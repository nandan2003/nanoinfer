import unittest
import asyncio
import json
from server.http_server import HTTPServer

TEST_PORT = 8089

class TestHTTPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(host="127.0.0.1", port=TEST_PORT)

    def test_01_http_streaming_sse(self):
        async def run_client():
            await self.server.start()
            await asyncio.sleep(0.5)

            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", TEST_PORT)

                payload = json.dumps({"prompt": "The capital of Japan is", "max_tokens": 5})
                request = (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{TEST_PORT}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                    f"{payload}"
                )
                writer.write(request.encode("utf-8"))
                await writer.drain()

                status_line = await reader.readline()
                self.assertIn(b"200 OK", status_line)

                headers = {}
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    line_str = line.decode("utf-8").strip()
                    if ":" in line_str:
                        k, v = line_str.split(":", 1)
                        headers[k.lower()] = v.strip()

                self.assertEqual(headers.get("content-type"), "text/event-stream")

                chunks = []
                current_event = None
                telemetry_received = False
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        current_event = None
                        continue
                    if line_str.startswith("event: "):
                        current_event = line_str[7:].strip()
                        continue
                    if line_str == "data: [DONE]":
                        break
                    if line_str.startswith("data: "):
                        json_str = line_str[6:]
                        data = json.loads(json_str)
                        if current_event == "telemetry":
                            telemetry_received = True
                            self.assertIn("ttft_ms", data)
                            self.assertIn("tps", data)
                            print(f"\n[PASS] Telemetry Event received: TTFT={data['ttft_ms']}ms, TPS={data['tps']}")
                            continue
                        if "choices" in data:
                            token = data["choices"][0]["delta"]["content"]
                            chunks.append(token)

                self.assertTrue(telemetry_received)
                full_reply = "".join(chunks)
                self.assertGreater(len(chunks), 0)
                print(f"[PASS] HTTP 200 OK + SSE Streaming: '{full_reply.strip()}'")

                writer.close()
                await writer.wait_closed()

            finally:
                await self.server.stop()

        asyncio.run(run_client())

if __name__ == "__main__":
    unittest.main()

