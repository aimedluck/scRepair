import logging
import hashlib

import numpy as np
import scipy.signal
import scipy.stats


log = logging.getLogger(__name__)


SMOOTH_STDDEV_SF = 2500e3


def normalized_gaussian_window(s):
    s = float(s)
    W = scipy.signal.windows.gaussian(int(np.ceil(3 * s)) * 2 + 1, s)
    return (W / W.sum())


def get_scaling_factor_kernel(binsize):
    return normalized_gaussian_window(SMOOTH_STDDEV_SF / binsize)


def ndarray_hash(v):
    """
    Return md5 hash of **contents of** v
    """

    dig = hashlib.md5()
    dig.update(v.data)
    return dig.digest()


def ds_convolve(ds, kernel):
    return {
        chrom: scipy.signal.convolve(
            ds[chrom].astype(float),
            kernel,
            mode='same',
        )
        for chrom in ds
    }


def calc_scale_factor(X, alpha=0.5, n_iter=100):
    """
    X: (Nbins, Nsamples)
    """

    _offset = 0.01

    assert X.ndim == 2
    assert (X.sum(axis=0) > 0).all()

    w = ((~np.isclose(X, 0)) & (X > 0)).any(axis=1)
    assert w.ndim == 1

    assert w.sum() > 0

    Xij = X[w]
    S = np.ones(len(Xij), dtype=bool)

    Xi = Xij.sum(axis=1)
    sj = Xij[S].sum(axis=0) / Xi[S].sum(axis=0)

    assert np.isclose(sj.sum(), 1.)

    Eij = np.outer(Xi, sj)
    GOFi = (((Xij - Eij) ** 2) / Eij).sum(axis=1)
    assert len(GOFi) == len(S)

    _GOF_low, _GOF_high = np.quantile(GOFi, np.array([_offset, min(1.0 - _offset, 1 - alpha + _offset)]))

    # add some cycle detection for fun and ... profit(???)
    cycles = set()
    converged = True
    for _ in range(n_iter):
        S_update = (GOFi >= _GOF_low) & (GOFi < _GOF_high)

        if (S == S_update).all():
            break

        S_update_hash = ndarray_hash(S_update)
        if S_update_hash in cycles:
            break
        cycles.add(S_update_hash)

        S = S_update.copy()

        sj = Xij[S].sum(axis=0) / Xi[S].sum(axis=0)

        Eij = np.outer(Xi, sj)
        GOFi = (((Xij - Eij) ** 2) / Eij).sum(axis=1)

        _GOF_low, _GOF_high = np.quantile(GOFi, np.array([_offset, min(1.0 - _offset, 1 - alpha + _offset)]))
    else:
        log.warn("Did not converge")
        converged = False

    return sj, converged


def calc_scale_factor_adapt(X, alpha=0.5):
    """
    X: (Nbins, Nsamples)
    """

    _offset = 0.01

    assert X.ndim == 2
    assert (X.sum(axis=0) > 0).all()

    w = ((~np.isclose(X, 0)) & (X > 0)).any(axis=1)
    assert w.ndim == 1

    assert w.sum() > 0

    Xij = X[w]
    S = np.ones(len(Xij), dtype=bool)

    Xi = Xij.sum(axis=1)
    sj = Xij[S].sum(axis=0) / Xi[S].sum(axis=0)

    assert np.isclose(sj.sum(), 1.)

    Eij = np.outer(Xi, sj)
    # Note: difference instead of squared distance!
    GOFi = ((Xij - Eij) / Eij).sum(axis=1)
    assert len(GOFi) == len(S)

    _GOF_low, _GOF_high = np.quantile(GOFi, np.array([_offset, min(1.0 - _offset, 1 - alpha + _offset)]))

    for _ in range(20):
        S_update = (GOFi >= _GOF_low) & (GOFi < _GOF_high)
        if (S == S_update).all():
            break

        S = S_update.copy()
        sj = Xij[S].sum(axis=0) / Xi[S].sum(axis=0)

        Eij = np.outer(Xi, sj)
        GOFi = (((Xij - Eij) ** 1) / Eij).sum(axis=1)

        _GOF_low, _GOF_high = np.quantile(GOFi, np.array([_offset, min(1.0 - _offset, 1 - alpha + _offset)]))
    else:
        log.warn("Did not converge")

    sj_final = sj.copy()

    return sj_final


# wrapper function:
def calc_sf(fg, bg, w_mapab, alpha, sf_kernel):
    total_bg = float(sum(v.sum() for v in bg.values()))
    assert fg.keys() == bg.keys()
    assert set(w_mapab.keys()).issuperset(set(fg.keys()))

    chroms = sorted(fg.keys())

    fg_c = ds_convolve(fg, sf_kernel)
    bg_c = ds_convolve(bg, sf_kernel)

    Xij = np.array([
        np.concatenate([fg_c[chrom][w_mapab[chrom]] for chrom in chroms]),
        np.concatenate([bg_c[chrom][w_mapab[chrom]] for chrom in chroms]),
    ]).T

    rel_sf, is_converged = calc_scale_factor(Xij, alpha=alpha)

    if is_converged:
        assert np.isclose(1., rel_sf.sum())

    sf = rel_sf[0] / (rel_sf[1] / total_bg)  # scale fg to bg
    if is_converged:
        assert sf > 0.

    return sf, is_converged
