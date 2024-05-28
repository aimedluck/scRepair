import sys

import numpy as np
from HTSeq.StepVector import StepVector as HTStepVector


def test_performance():
    N = 2_000

    L = 250_000

    rs = np.random.RandomState(42)

    sv = HTStepVector.create(typecode='O')
    sv[:] = set()
    for n in range(N):
        low = rs.randint(0, L)
        high = rs.randint(low + 1, L)
        segment = sv[low:high]
        def addvalue(x):
            stepvalue = x.copy()
            stepvalue.add(n)
            return stepvalue
        segment.apply(addvalue)

    print(sum(1 for _ in sv[0:L].get_steps()))
    print(list(sv[1000:2000].get_steps())[:10])

if __name__ == "__main__":
    test_performance()
