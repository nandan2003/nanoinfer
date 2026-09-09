import os
import time
import asyncio
import json
from typing import Optional
from server.scheduler import RingBufferQueue, RequestItem, WorkerPool
from server.inference import InferenceEngine
from server.cache import PrefixTrieCache

class HTTPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", pin_cores: bool = True, num_workers: int = 2, parallel_engines: bool = False):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).replace(".gguf", "")

        threads_per_engine = 1 if pin_cores else None
        if parallel_engines and num_workers > 1:
            engines = [InferenceEngine(model_path=model_path, n_ctx=512, n_threads=threads_per_engine, verbose=False) for _ in range(num_workers)]
        else:
            engines = InferenceEngine(model_path=model_path, n_ctx=512, n_threads=threads_per_engine, verbose=False)

        self.engine = engines if not isinstance(engines, list) else engines[0]
        self.cache = PrefixTrieCache(max_nodes=500)
        self.queue = RingBufferQueue(capacity=32)
        self.pool = WorkerPool(num_workers=num_workers, queue=self.queue, engine=engines, cache=self.cache, pin_cores=pin_cores)

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

            if method == "GET" and path == "/v1/models":
                models_payload = json.dumps({
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "nanoinfer"
                        }
                    ]
                }).encode("utf-8")
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(models_payload)}\r\n\r\n".encode("utf-8")
                    + models_payload
                )
                await writer.drain()
                writer.close()
                return

            if method != "POST" or path != "/v1/chat/completions":
                err_body = json.dumps({
                    "error": {
                        "message": f"Invalid endpoint or method: {method} {path}",
                        "type": "invalid_request_error",
                        "code": 404
                    }
                }).encode("utf-8")
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n" + f"Content-Length: {len(err_body)}\r\n\r\n".encode("utf-8") + err_body)
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

            t_start = time.perf_counter()
            token_timestamps = []

            if not self.queue.put(req, block=False):
                err_429 = json.dumps({
                    "error": {
                        "message": "Server queue is full, request rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": 429
                    }
                }).encode("utf-8")
                writer.write(b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n" + f"Content-Length: {len(err_429)}\r\n\r\n".encode("utf-8") + err_429)
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
                t_now = time.perf_counter()
                if token is None:
                    break

                token_timestamps.append(t_now)
                chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
                writer.write(f"data: {chunk}\n\n".encode("utf-8"))
                await writer.drain()

            t_end = time.perf_counter()
            e2e_sec = round(t_end - t_start, 4)
            token_count = len(token_timestamps)

            if token_count > 0:
                ttft_ms = round((token_timestamps[0] - t_start) * 1000, 2)
                if token_count > 1:
                    decode_duration = token_timestamps[-1] - token_timestamps[0]
                    itl_mean_ms = round((decode_duration / (token_count - 1)) * 1000, 2)
                    tps = round((token_count - 1) / decode_duration, 2) if decode_duration > 0 else 0.0
                else:
                    itl_mean_ms = 0.0
                    tps = 0.0
            else:
                ttft_ms = 0.0
                itl_mean_ms = 0.0
                tps = 0.0

            telemetry_payload = json.dumps({
                "ttft_ms": ttft_ms,
                "itl_mean_ms": itl_mean_ms,
                "tps": tps,
                "e2e_sec": e2e_sec,
                "token_count": token_count,
            })
            writer.write(f"event: telemetry\ndata: {telemetry_payload}\n\n".encode("utf-8"))
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

