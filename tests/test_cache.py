import unittest
from server.cache import PrefixTrieCache

class TestPrefixTrieCache(unittest.TestCase):
    def test_01_prefix_match(self):
        cache = PrefixTrieCache(max_nodes=100)

        cache.insert([101, 102, 103], state="KV_CHECKPOINT_A")

        node, matched_len = cache.match_longest_prefix([101, 102, 103, 201, 202])

        self.assertIsNotNone(node)
        self.assertEqual(node.state, "KV_CHECKPOINT_A")
        self.assertEqual(matched_len, 3)
        print(f"\n[PASS] Matched prefix of length {matched_len} with state: {node.state}")

    def test_02_prefix_miss(self):
        cache = PrefixTrieCache(max_nodes=100)
        cache.insert([10, 20, 30], state="STATE_X")

        node, matched_len = cache.match_longest_prefix([99, 98, 97])
        self.assertIsNone(node)
        self.assertEqual(matched_len, 0)
        print("\n[PASS] Handled cache miss correctly (0 tokens matched)")

    def test_03_lru_eviction(self):
        cache = PrefixTrieCache(max_nodes=3)

        cache.insert([1], state="STATE_1")
        cache.insert([2], state="STATE_2")
        cache.insert([3], state="STATE_3")
        self.assertEqual(cache.size, 3)

        cache.insert([4], state="STATE_4")
        self.assertEqual(cache.size, 3)

        node, _ = cache.match_longest_prefix([1])
        self.assertIsNone(node)

        node, _ = cache.match_longest_prefix([4])
        self.assertIsNotNone(node)
        print("\n[PASS] O(1) LRU Eviction successfully pruned oldest node")

    def test_04_subtree_pruning_on_eviction(self):
        cache = PrefixTrieCache(max_nodes=4)
        cache.insert([1, 2, 3], state="STATE_123")
        cache.insert([1, 2, 3, 4], state="STATE_1234")
        self.assertEqual(cache.size, 4)

        cache.insert([5], state="STATE_5")
        self.assertLessEqual(cache.size, cache.max_nodes)
        node, _ = cache.match_longest_prefix([1])
        self.assertIsNone(node)
        print("\n[PASS] Subtree pruning cleanly unlinked descendant nodes upon eviction")

if __name__ == "__main__":
    unittest.main()

