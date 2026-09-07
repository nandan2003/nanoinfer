import asyncio
import json
from typing import Optional
from server.scheduler import RingBufferQueue, RequestItem, WorkerPool
from server.inference import InferenceEngine
from server.cache import PrefixTrieCache

class HTTPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"):
        self.host = host
        self.port = port

        self.engine = InferenceEngine(model_path=model_path, n_ctx=512, verbose=False)
        self.cache = PrefixTrieCache(max_nodes=500)
        self.queue = RingBufferQueue(capacity=32)
        self.pool = WorkerPool(num_workers=2, queue=self.queue, engine=self.engine, cache=self.cache)

        self.server: Optional[asyncio.Server] = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handles a single incoming HTTP connection."""
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return

            line_str = request_line.decode("utf-8").strip()
            parts = line_str.split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]

            headers = {}
            while True:
                header_line = await reader.readline()
                if header_line in (b"\r\n", b"\n", b""):
                    break
                header_str = header_line.decode("utf-8").strip()
                if ":" in header_str:
                    key, val = header_str.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            if method != "POST" or path != "/v1/chat/completions":
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 13\r\n\r\n404 Not Found")
                await writer.drain()
                writer.close()
                return

            content_length = int(headers.get("content-length", 0))
            body_bytes = await reader.readexactly(content_length)
            data = json.loads(body_bytes.decode("utf-8"))

            prompt = data.get("prompt", "")
            max_tokens = int(data.get("max_tokens", 32))
            temperature = float(data.get("temperature", 0.7))

            loop = asyncio.get_running_loop()
            output_queue = asyncio.Queue()
            req = RequestItem(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                output_queue=output_queue,
                loop=loop,
            )

            if not self.queue.put(req, block=False):
                resp_429 = b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 30\r\n\r\n{\"error\": \"Server is busy\"}"
                writer.write(resp_429)
                await writer.drain()
                writer.close()
                return

            headers_resp = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Connection: keep-alive\r\n"
                b"Access-Control-Allow-Origin: *\r\n\r\n"
            )
            writer.write(headers_resp)
            await writer.drain()

            while True:
                token = await output_queue.get()
                if token is None:
                    break

                chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
                writer.write(f"data: {chunk}\n\n".encode("utf-8"))
                await writer.drain()

            writer.write(b"data: [DONE]\n\n")
            await writer.drain()

        except Exception as e:
            print(f"[Server Error]: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self) -> None:
        """Starts the worker pool and HTTP server."""
        self.pool.start()
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"Server listening on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully closes server and worker pool."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.pool.stop()


async def main():
    server = HTTPServer()
    await server.start()
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())

