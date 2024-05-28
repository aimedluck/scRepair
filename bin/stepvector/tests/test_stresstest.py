import numpy as np
from stepvector import StepVector


def test_performance():
    #N = 100_000
    N = 2_000

    L = 250_000

    rs = np.random.RandomState(42)

    sv = StepVector(set)
    for n in range(N):
        low = rs.randint(0, L)
        high = rs.randint(low + 1, L)
        sv.add_value(low, high, {n})

    print(sum(1 for _ in sv[0:L]))
    print(list(sv[1000:2000])[:10])

if __name__ == "__main__":
    test_performance()
