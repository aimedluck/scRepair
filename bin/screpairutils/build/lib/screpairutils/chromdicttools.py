# lol naming things is hard

from functools import reduce

import numpy as np


def subsample_prob(v, n, random_seed=None):
    """
    Probablistic subsampling
    """
    assert v.ndim == 1
    vs = v.sum()
    assert n <= vs
    rs = np.random.RandomState(random_seed)
    return rs.binomial(n, v / vs)


def subsample_chroms_prob(ds, n, random_seed=None):
    # transparently concatenate chroms and re-split after subsampling
    dschroms = sorted(ds)
    v = np.concatenate([ds[chrom] for chrom in dschroms])
    vs = subsample_prob(v, n, random_seed)
    dschrompos = np.array([ds[chrom].size for chrom in dschroms]).cumsum()[:-1]
    return dict(zip(dschroms, np.split(vs, dschrompos)))


# def negate_chroms(vs):
#     return {chrom: ~vs[chrom] for chrom in vs}

# embrace the functional approach:

def map_dict(func, *args):
    keys = reduce(set.intersection, (set(d.keys()) for d in args))
    return {k: func(*[d[k] for d in args]) for k in keys}
# --> negate_chroms(vs) =~ map_dict(np.logical_not, vs)


def mask_ds(ds, w, fillvalue=0):
    return np.choose(
        w,
        (ds, fillvalue)
    )


def reduce_dict(func, d):
    return reduce(func, d.values())


def sum_ds(ds):
    return reduce_dict(lambda a, b: a + b, map_dict(lambda v: v.sum(), ds))
