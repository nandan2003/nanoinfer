import threading
from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class RequestItem:
    """Represents a client request waiting for inference."""
    prompt: str
    max_tokens: int
    temperature: float
    output_queue: Any
    loop: Any  


class RingBufferQueue:
    """
    Thread-safe bounded circular queue using Condition Variables.
    Demonstrates zero-allocation bounded buffering.
    """
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0 
        self.tail = 0 
        self.count = 0 

        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Enqueues an item. If full and block=False, immediately returns False (HTTP 429 backpressure).
        """
        with self.lock:
            if self.count >= self.capacity:
                if not block:
                    return False
                    
                if not self.not_full.wait(timeout=timeout):
                    return False

            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.count += 1

            self.not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Dequeues an item. If queue is empty, worker sleeps until notified.
        """
        with self.lock:
            while self.count == 0:
                if not self.not_empty.wait(timeout=timeout):
                    return None

            item = self.buffer[self.head]
            self.buffer[self.head] = None 
            self.head = (self.head + 1) % self.capacity
            self.count -= 1

            self.not_full.notify()
            return item


class WorkerPool:
    """
    Manages a fixed pool of background OS threads.
    Pulls requests from the RingBufferQueue, checks the Prefix Trie Cache,
    runs CPU inference, and streams tokens back to the async event loop.
    """
    def __init__(self, num_workers: int, queue: RingBufferQueue, engine: Any, cache: Any):
        self.num_workers = num_workers
        self.queue = queue
        self.engine = engine
        self.cache = cache
        self.threads: list[threading.Thread] = []
        self.running = True

    def start(self) -> None:
        """Spawns the worker threads in daemon mode."""
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"Worker-{i}")
            t.start()
            self.threads.append(t)

    def _worker_loop(self) -> None:
        while self.running:
            item = self.queue.get(timeout=1.0)
            if item is None:
                continue

            try:
                tokens = self.engine.tokenize(item.prompt)

                matched_node, matched_len = self.cache.match_longest_prefix(tokens)
                if matched_node is not None and matched_node.state is not None:
                    self.engine.load_state(matched_node.state)
                else:
                    self.engine.reset()

                for token in self.engine.generate(item.prompt, max_tokens=item.max_tokens, temperature=item.temperature):
                    item.loop.call_soon_threadsafe(item.output_queue.put_nowait, token)

                new_state = self.engine.save_state()
                self.cache.insert(tokens, new_state)

            except Exception as e:
                item.loop.call_soon_threadsafe(item.output_queue.put_nowait, f"[Error: {e}]")
            finally:
                item.loop.call_soon_threadsafe(item.output_queue.put_nowait, None)

    def stop(self) -> None:
        """Gracefully shuts down worker threads."""
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
