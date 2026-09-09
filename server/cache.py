from typing import Optional, Any

class TrieNode:
    def __init__(self, token_id: int, parent: Optional['TrieNode'] = None):
        self.token_id = token_id
        self.parent = parent
        self.children: dict[int, 'TrieNode'] = {}
        self.state: Any = None

        self.prev: Optional['TrieNode'] = None
        self.next: Optional['TrieNode'] = None


class PrefixTrieCache:
    def __init__(self, max_nodes: int = 1000):
        self.max_nodes = max_nodes
        self.size = 0

        self.root = TrieNode(token_id=-1)

        self.head = TrieNode(token_id=-1)
        self.tail = TrieNode(token_id=-1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: TrieNode) -> None:
        """Inserts node right after dummy head (Most Recently Used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: TrieNode) -> None:
        """Unlinks a node from its current position in the list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _touch(self, node: TrieNode) -> None:
        """Moves an accessed node to the Head (marks it Most Recently Used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _prune_subtree(self, node: TrieNode) -> int:
        """Recursively unlinks all descendant nodes from Trie and LRU list."""
        pruned = 0
        for child in list(node.children.values()):
            pruned += self._prune_subtree(child)
            self._remove_node(child)
            child.state = None
            child.parent = None
            pruned += 1
        node.children.clear()
        return pruned

    def _evict_lru(self) -> None:
        """Evicts the Least Recently Used node and unlinks any orphaned descendants."""
        if self.tail.prev == self.head:
            return  

        lru_node = self.tail.prev
        self._remove_node(lru_node)

        pruned_count = self._prune_subtree(lru_node)
        lru_node.state = None

        if lru_node.parent and lru_node.token_id in lru_node.parent.children:
            del lru_node.parent.children[lru_node.token_id]

        self.size -= (1 + pruned_count)

    def match_longest_prefix(self, tokens: list[int]) -> tuple[Optional[TrieNode], int]:
        """
        Walks the Trie to find the longest cached prefix match.
        Returns: (deepest_node_with_state, matched_token_count)
        Time Complexity: O(K) where K is len(tokens).
        """
        curr = self.root
        matched_count = 0
        last_state_node = None
        last_state_len = 0

        for token in tokens:
            if token in curr.children:
                curr = curr.children[token]
                matched_count += 1
                if curr.state is not None:
                    last_state_node = curr
                    last_state_len = matched_count
            else:
                break

        if last_state_node is not None:
            self._touch(last_state_node)
            return last_state_node, last_state_len

        return None, 0

    def insert(self, tokens: list[int], state: Any) -> TrieNode:
        """
        Inserts a token sequence into the Trie and attaches the KV state at the leaf.
        Evicts the oldest node if capacity is reached.
        """
        curr = self.root

        for token in tokens:
            if token not in curr.children:
                if self.size >= self.max_nodes:
                    self._evict_lru()

                new_node = TrieNode(token_id=token, parent=curr)
                curr.children[token] = new_node
                self._add_to_head(new_node)
                self.size += 1

            curr = curr.children[token]

        curr.state = state
        self._touch(curr)
        return curr

