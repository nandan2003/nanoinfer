import os
import unittest
import asyncio
from server.inference import InferenceEngine
from server.cache import PrefixTrieCache
from server.scheduler import RingBufferQueue, RequestItem, WorkerPool

MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

class TestSchedulerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine(model_path=MODEL_PATH, n_ctx=256, verbose=False)
        cls.cache = PrefixTrieCache(max_nodes=100)
        cls.queue = RingBufferQueue(capacity=16)
        cls.pool = WorkerPool(num_workers=2, queue=cls.queue, engine=cls.engine, cache=cls.cache)
        cls.pool.start()

    @classmethod
    def tearDownClass(cls):
        cls.pool.stop()

    def test_worker_cpu_affinity(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("os.sched_getaffinity not supported on this platform")

        for i, thread in enumerate(self.pool.threads):
            expected_core = self.pool.worker_cores[i]
            actual_mask = os.sched_getaffinity(thread.native_id)
            self.assertEqual(actual_mask, {expected_core})
            print(f"\n[PASS] Worker-{i} pinned to CPU core {expected_core} (affinity: {actual_mask})")

    def test_worker_end_to_end_streaming(self):
        async def run_request():
            loop = asyncio.get_running_loop()
            output_queue = asyncio.Queue()

            req = RequestItem(
                prompt="The planet Mars is known as the",
                max_tokens=8,
                temperature=0.1,
                output_queue=output_queue,
                loop=loop,
            )

            self.assertTrue(self.queue.put(req))

            received_tokens = []
            while True:
                token = await output_queue.get()
                if token is None:
                    break
                received_tokens.append(token)

            full_text = "".join(received_tokens)
            self.assertGreater(len(received_tokens), 0)
            print(f"\n[PASS] Worker generated & bridged {len(received_tokens)} tokens: '{full_text.strip()}'")

            tokens = self.engine.tokenize(req.prompt)
            node, matched_len = self.cache.match_longest_prefix(tokens)
            self.assertIsNotNone(node)
            self.assertGreater(matched_len, 0)
            print(f"[PASS] Prompt tokens successfully stored in Prefix Trie (matched: {matched_len})")

        asyncio.run(run_request())

if __name__ == "__main__":
    unittest.main()

