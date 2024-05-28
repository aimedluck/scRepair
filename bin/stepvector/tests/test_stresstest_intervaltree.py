import numpy as np
from intervaltree import IntervalTree


def test_performance():
    N = 2_000

    L = 250_000

    rs = np.random.RandomState(42)

    tree = IntervalTree()
    for n in range(N):
        low = rs.randint(0, L)
        high = rs.randint(low + 1, L)
        tree.addi(low, high, {n})

    print(sum(1 for _ in tree[0:L]))
    print(list(tree[1000:2000])[:10])
    tree.split_overlaps()
    print(list(tree[1000:2000])[:10])

if __name__ == "__main__":
    test_performance()
