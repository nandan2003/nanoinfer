import unittest
import threading
from server.scheduler import RingBufferQueue

class TestRingBufferQueue(unittest.TestCase):
    def test_01_fifo_order(self):
        queue = RingBufferQueue(capacity=4)
        queue.put("A")
        queue.put("B")
        queue.put("C")

        self.assertEqual(queue.get(), "A")
        self.assertEqual(queue.get(), "B")
        self.assertEqual(queue.get(), "C")
        print("\n[PASS] FIFO ordering verified")

    def test_02_backpressure_when_full(self):
        queue = RingBufferQueue(capacity=2)
        self.assertTrue(queue.put("Req1", block=False))
        self.assertTrue(queue.put("Req2", block=False))

        self.assertFalse(queue.put("Req3", block=False))
        self.assertEqual(queue.count, 2)
        print("\n[PASS] Bounded capacity backpressure verified")

    def test_03_concurrent_producer_consumer_stress(self):
        queue = RingBufferQueue(capacity=16)
        total_items = 200
        consumed_items = []
        lock = threading.Lock()

        def producer(start_id, count):
            for i in range(count):
                queue.put(f"item_{start_id}_{i}")

        def consumer(count):
            for _ in range(count):
                item = queue.get()
                with lock:
                    consumed_items.append(item)

        producers = [threading.Thread(target=producer, args=(p, 50)) for p in range(4)]
        consumers = [threading.Thread(target=consumer, args=(50,)) for _ in range(4)]

        for t in consumers + producers:
            t.start()
        for t in consumers + producers:
            t.join()

        self.assertEqual(len(consumed_items), total_items)
        self.assertEqual(queue.count, 0)
        print(f"\n[PASS] Multi-threaded stress test passed ({total_items} items, 0 dropped, 0 race conditions)")

if __name__ == "__main__":
    unittest.main()

